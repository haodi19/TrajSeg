import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from datasets.lasot import build as build_lasot
from datasets.tnl2k import build as build_tnl2k

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, NEW_MODE_QUESTION_LIST, NEW_MODE_ANSWER_LIST
from datasets.util.box_ops import box_xyxy_to_cxcywh, box_clip_transform

def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class RVOTDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        args,
        base_video_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        rvot_data="tnl2k||lasot",
        # train_mode_rate=[1,0],
        answer_type='1',
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_video_dir = base_video_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = vision_tower

        self.short_question_list = SHORT_QUESTION_LIST

        self.answer_type = answer_type

        if answer_type == '1':
            self.answer_list = ANSWER_LIST          # it is [SEG]
        elif answer_type == '2':
            self.answer_list = NEW_MODE_ANSWER_LIST # [referring text] [SEG]


        self.data2list = {}
        self.data2classes = {}

        self.new_mode_question_list = NEW_MODE_QUESTION_LIST
        self.new_mode_answer_list = NEW_MODE_ANSWER_LIST

        self.rvot_datas = rvot_data.split("||")
        # self.train_mode_rate = train_mode_rate
        self.train_mode = 0

        # lasot
        if 'lasot' in self.rvot_datas:
            lasot_dataset = build_lasot('train', os.path.join(base_video_dir, 'LaSOT'), args)
            self.data2list['lasot'] = lasot_dataset
        # tnl2k
        if 'tnl2k' in self.rvot_datas:
            tnl2k_dataset = build_tnl2k('train', os.path.join(base_video_dir, 'TNL2K/TNL2K_train_subset/train_data'), args)
            self.data2list['tnl2k'] = tnl2k_dataset


    def __len__(self):
        return self.samples_per_epoch

    def set_train_mode(self, mode):
        self.train_mode = mode

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def preprocess_boxes(self, target, img_shape):
        h, w = img_shape
        boxes = target["boxes"]
        new_boxes = box_xyxy_to_cxcywh(boxes)
        new_boxes = new_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        target["boxes"] = new_boxes

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.rvot_datas) - 1)
        ds = self.rvot_datas[ds]

        if ds in ["lasot", "tnl2k"]:
            rvot_data = self.data2list[ds]
            # image, labels = self.data2list[ds]
            idx = random.randint(0, len(rvot_data.metas) - 1)
            imgs, target = rvot_data[idx]
            imgs = [np.array(img) for img in imgs]
            
            original_shapes = [img.shape[:2] for img in imgs]
            resized_shapes = [self.clip_image_processor.resize(img, self.clip_image_processor.size, self.clip_image_processor.resample).shape[:2] for img in imgs]
            target_size = tuple(self.clip_image_processor.crop_size.values())
            
            image_clips = torch.stack([self.clip_image_processor.preprocess(
                img, return_tensors="pt"
            )["pixel_values"][0] for img in imgs], 0)
            
            trajectoies = box_clip_transform(target['boxes'], original_shapes, resized_shapes, target_size)
            
            images = [self.transform.apply_image(img) for img in imgs]      # for sam (numpy as LISA)
            target['boxes'] = self.transform.apply_boxes_torch(target['boxes'], np.array(imgs[0]).shape[:2])      # for sam (numpy as LISA)
            resize = images[0].shape[:2]

        questions = []
        answers = []

        # only one object
        sampled_cls = target['caption']
        text = sampled_cls

        assert len(text.split("||")) == 1
        
        # different train modes
        if self.train_mode == 0:   # traditional mode
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            # answers.append(random.choice(self.answer_list))
            if self.answer_type == '1':
                answers.append(random.choice(self.answer_list))
            elif self.answer_type == '2':
                answer_template = random.choice(self.new_mode_answer_list)
                answers.append(answer_template.format(class_name=text.lower())) # with captions in answers
            train_mode = 0
        else:
            question_template = random.choice(self.new_mode_question_list)
            questions.append(question_template)                             # without captions in questions
            answer_template = random.choice(self.new_mode_answer_list)
            answers.append(answer_template.format(class_name=text.lower())) # with captions in answers
            train_mode = 1

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = torch.stack([self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()) for image in images], 0)
        self.preprocess_boxes(target, resize)
        
        masks = None
        boxes = target['boxes']

        return (
            None,
            boxes,
            image,
            image_clips,
            conversations,
            masks,
            masks,
            resize,
            questions,
            [text],
            train_mode,
            trajectoies
        )
