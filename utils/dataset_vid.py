import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from utils.multi_reason_seg_dataset import MultiReasonSegDataset
from utils.reason_seg_dataset import ReasonSegDataset
from utils.refer_seg_dataset import ReferSegDataset
from utils.sem_seg_dataset import SemSegDataset
from utils.vqa_dataset import VQADataset

if __name__ != "__main__":
    from model.llava import conversation as conversation_lib
    from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                    IMAGE_TOKEN_INDEX)
    from model.llava.mm_utils import tokenizer_image_token, tokenizer_trajectory_token
    from model.segment_anything.utils.transforms import ResizeLongestSide

    from .data_processing import get_mask_from_json
from .refer import REFER
from .rvos_dataset import RVOSDataset
from .rvot_dataset import RVOTDataset
from .refseg_dataset import RefSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, DEFAULT_SEG_END_TOKEN, DEFAULT_SEG_START_TOKEN, DEFAULT_TRAJ_END_TOKEN, DEFAULT_TRAJ_START_TOKEN, DEFAULT_TRAJECTORY_TOKEN, DEFAULT_VD_END_TOKEN, DEFAULT_VD_START_TOKEN)
from .dataset import collate_fn as collate_fn_img

def uniform_sample_frames(n, min_frames=8, max_frames=12):
    if n <= min_frames:
        return [i for i in range(n)]
    
    # 为帧数范围 [8,12] 指定权重，越接近 12，权重越高
    frame_options = list(range(min_frames, max_frames + 1))
    weights = [1, 2, 3, 4, 5]  # 8 对应 1 权重，12 对应 5 权重

    # 根据权重随机选择目标采样帧数
    target_frames = random.choices(frame_options, weights=weights, k=1)[0]

    # 计算理想采样间隔，并加入随机波动
    interval = n / target_frames
    sampled_indices = [0]
    
    # 使用随机偏差进行采样，确保随机性
    for i in range(1, target_frames):
        index = int(i * interval + random.uniform(0, interval))  # 在每个间隔内随机选择
        index = min(index, n - 1)  # 确保索引不超过n
        sampled_indices.append(index)

    # 去重并排序
    sampled_indices = sorted(set(sampled_indices))

    # 如果采样后帧数不足或过多，进行调整
    while len(sampled_indices) < min_frames:
        sampled_indices.append(random.choice(range(n)))  # 随机补充
        sampled_indices = sorted(set(sampled_indices))   # 去重排序

    while len(sampled_indices) > max_frames:
        sampled_indices.pop(random.randint(1, len(sampled_indices) - 1))  # 随机移除

    return sampled_indices

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, use_multuple_seg_token = False, llm_sample_mode = 'nosample', local_rank=-1
):
    image_path_list = []
    box_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    prompts = []
    train_mode_list = []
    trajectoies = []

    # Image sample
    if len(batch[0]) == 10:
        return collate_fn_img(batch, tokenizer= tokenizer, conv_type=conv_type, use_mm_start_end=use_mm_start_end, local_rank= local_rank, wrap_as_video=True)
    
    video_len = batch[0][2].shape[0]
    
    if llm_sample_mode == 'uniform':
        sampled_indices = uniform_sample_frames(video_len)
        training_type = -1
    elif llm_sample_mode == 'nosample':
        sampled_indices = None
        probabilities = [4/5, 1/5]
        training_type = np.random.choice([1,2], p=probabilities)
        
    if use_multuple_seg_token:
        remove_frame_num = np.random.randint(0, 8)
        remove_indices = np.random.choice(video_len, remove_frame_num, replace=False)
        all_numbers = set(range(video_len))
        remaining = all_numbers - set(remove_indices)
        keep_indices = list(remaining)
        new_video_len = len(keep_indices)
    
    for (
        image_path,
        boxes,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        train_mode,
        trajectory, 
        inference,
    ) in batch:
        
        is_video = True
        
        if len(images_clip.shape) == 3:
            is_video = False
            images_clip = images_clip.unsqueeze(0)
            images = images.unsqueeze(0)
            masks = masks.unsqueeze(0)
            
            images_clip = images_clip.repeat(2,1,1,1)
            images = images.repeat(2,1,1,1)
            masks = masks.repeat(2,1,1,1)
            
        for i in range(len(questions)):
            prompt = questions[i].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
            prompts.append([prompt])
            
        if llm_sample_mode == 'uniform':
            images_clip = images_clip[sampled_indices]
            trajectory = trajectory[sampled_indices]
           
        if use_multuple_seg_token: 
            images = images[keep_indices]
            images_clip = images_clip[keep_indices]
            masks = masks[keep_indices]
            if boxes is not None:
                boxes = boxes[keep_indices]

            for i in range(len(conversations)):
                conversations[i] = conversations[i].format(video_len=new_video_len)
        
        image_path_list.append(image_path)
        box_list.append(boxes)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        for i in range(len(conversations)):
            train_mode_list.append(train_mode)                  # 0: traditional (text to mask trajectories) 1: new mode
        trajectoies.append(trajectory)
        if label is not None:
            if not is_video:
                # image
                label_list.append(label)
                masks_list.append(masks.float())
            else:
                label_list.append(label[0])
                masks_list.append(masks.float().unsqueeze(1))   # len, h, w -> len, 1, h, w, (1 all the time since only 1 obj corresponds to the txt)
        else:
            label_list.append(None)
            masks_list.append(None)        
        
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            
            replace_traj_token = DEFAULT_TRAJECTORY_TOKEN
            
            
            replace_token_vid = (
                    DEFAULT_VD_START_TOKEN + replace_token + DEFAULT_VD_END_TOKEN
                )
            replace_token_img = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            
            replace_token_traj = (
                    DEFAULT_TRAJ_START_TOKEN + replace_traj_token + DEFAULT_TRAJ_END_TOKEN
                )
                
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token_vid
            )
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * images_clip_list[0].shape[0])
            
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token_img
            )
            
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_TRAJECTORY_TOKEN, replace_token_traj
            )
            
    if use_multuple_seg_token:
        for i in range(len(conversation_list)):
            replace_token = '[SEG]'
                
            replace_token_seg = (
                    DEFAULT_SEG_START_TOKEN + replace_token + DEFAULT_SEG_END_TOKEN
                )
                
            conversation_list[i] = conversation_list[i].replace('[SEG]', '[SEG]' * images_clip_list[0].shape[0])
            
            conversation_list[i] = conversation_list[i].replace(
                '[SEG]', replace_token_seg
            )
    
    # input_ids = [
    #     tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    #     for prompt in conversation_list
    # ]

    input_ids = []
    for train_mode, prompt in zip(train_mode_list, conversation_list):
        if train_mode == 0: # <image>
            input_ids.append(tokenizer_image_token(prompt, tokenizer, return_tensors="pt"))
        else:               # <trajectory>
            input_ids.append(tokenizer_trajectory_token(prompt, tokenizer, return_tensors="pt"))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for train_mode, conversation, target in zip(train_mode_list, conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # conv.sep2: </s>
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if train_mode == 0: # <image>
                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            else:               # <trajectory>
                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_trajectory_token(rou, tokenizer))
                    instruction_len = len(tokenizer_trajectory_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        # truncate_len = tokenizer.model_max_length - 255
        truncate_len = tokenizer.model_max_length * 2 
        
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    # labels 是GT，只有LLM的回答（即ASSISTANT:后的部分不被pad IGNORE_INDEX)
    # offset 是每个batch对应问题在整个question_list中的区间，如batch 0有1个问题， batch 1有3个问题，那offset是[0,1,4]


    # if images_clip_list[0].shape[0] != images_clip_list[1].shape[0]:
    #     print(33333333333333331)
    #     print(images_clip_list[0].shape)
    #     print(images_clip_list[1].shape)
    #     print('qq',torch.stack(images_clip_list, dim=0).shape)
    #     print(99999)
    
    return {
        # image_paths: list, len=bs, e.g. ['lisa/coco/train2017/000000094127.jpg', 'lisa/ade20k/images/training/ADE_train_00000359.jpg']
        "image_paths": image_path_list, 
        # images: torch.Size([bs, video_len, 3, 1024, 1024]), 用于SAM分割
        "images": torch.stack(images_list, dim=0),
        # images_clip: torch.Size([bs, video_len, 3, 224, 224]), 用于大模型
        "images_clip": torch.stack(images_clip_list, dim=0),
        # input_ids: torch.Size([qs, 234]), tokenize的文本
        "input_ids": input_ids,
        # labels: torch.Size([qs, 234]), 文本GT
        "labels": targets,
        # attention_masks: torch.Size([qs, 234]), 遮住模型不能看到的文本部分
        "attention_masks": attention_masks,
        # box_list: box 列表, list[len=bs], [torch.Size([video_len, 4]), torch.Size([video_len, 4])]
        "box_list": box_list,
        # masks_list: mask的GT, list[len=bs], 如[torch.Size([video_len, 3, 330, 440]), torch.Size([video_len, 3, 427, 640])]
        "masks_list": masks_list,
        # label_list: list[len=bs], e,g. [torch.Size([330, 440]), torch.Size([427, 640])], 原图像尺寸
        "label_list": label_list,
        # resize_list: list[len=bs], e,g. [[768, 1024], [683, 1024]], resize后的图像尺寸
        "resize_list": resize_list,
        # offset: tensor,长度为bs+1, e.g. tensor([0, 3, 6], device='cuda:0'), 用于确认每个视频对应文本input_ids的索引范围
        "offset": torch.LongTensor(offset_list),
        # questions_list: list[len=bs], 该参数在训练时没有用到, 每个图像/视频的问题列表, e.g.
        # [['<image>\nWho is in this image? Please respond with segmentation mask.',]
        #  ['<image>\nWhat is ceiling in this image? Please respond with segmentation mask.', '<image>\nWhat is cushion in this image? Please respond with segmentation mask.', '<image>\nPlease segment the lamp in this image.']]
        "questions_list": questions_list,
        # sampled_classes_list: list[len=bs], 该参数在训练时没有用到, 每个图像/视频的sampled_class列表
        "sampled_classes_list": sampled_classes_list,
        # inference: False
        "inference": inferences[0],
        # conversation_list: list[len=qs], 该参数在训练时没有用到, 所有完整文本列表(完整问答)
        "conversation_list": conversation_list,
        # prompts: list[len=qs], 每个文本对应一个prompt, 计算方式为question.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', ''))，具体见上面
        # e.g. '<image>\nWhat is ceiling in this image? Please respond with segmentation mask.' -> ['What is ceiling in this image? Please respond with segmentation mask.']
        "prompts": prompts,
        "train_mode_list": train_mode_list, # 列表，长度为 bs，0 表示文本到序列，1 表示序列到文本
        "trajectoies": trajectoies,
        "sampled_indices": sampled_indices,
        "training_type": training_type
    }

class HybridDatasetVid(torch.utils.data.Dataset):
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
        dataset="rvos",
        sample_rate=[2, 1],
        train_mode_rate=[1, 0],
        rvos_data="ytbvos||mevis",
        rvot_data="lasot||tnl2k",
        refseg_data="refcoco||refcoco+||refcocog",
        explanatory=0,
        answer_type='1',
        image_dataset = None,
        base_image_dir = None,
        image_sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        train_mode_rate = np.array(train_mode_rate)
        self.train_mode_rate = train_mode_rate / train_mode_rate.sum()

        self.base_video_dir = base_video_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        vision_tower = CLIPImageProcessor.from_pretrained(vision_tower)
        
        self.data_type = "video"
        self.data_cnt = 0

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "rvos":
                self.all_datasets.append(
                    RVOSDataset(
                        args,
                        base_video_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        rvos_data,
                        # train_mode_rate,
                        answer_type,
                    )
                )
            elif dataset == "rvot":
                self.all_datasets.append(
                    RVOTDataset(
                        args,
                        base_video_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        rvot_data,
                        # train_mode_rate,
                        answer_type,
                    )
                )
            elif dataset == "refseg":
                self.all_datasets.append(
                    RefSegDataset(
                        args,
                        base_video_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refseg_data,
                        # train_mode_rate,
                        answer_type,
                    )
                )
            elif dataset == "MUSE":
                self.all_datasets.append(
                    MultiReasonSegDataset(
                        base_video_dir, 
                        tokenizer, 
                        vision_tower, 
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        "MUSE",
                        )
                )
        
        if image_dataset is not None:
            self.image_datasets = image_dataset.split("||")
            self.image_sample_rate = image_sample_rate / np.array(image_sample_rate).sum()
            self.all_image_datasets = []
            
            self.explanatory = explanatory
            for image_dataset in self.image_datasets:
                if image_dataset == "sem_seg":
                    self.all_image_datasets.append(
                        SemSegDataset(
                            base_image_dir,
                            tokenizer,
                            vision_tower,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            sem_seg_data,
                        )
                    )
                elif image_dataset == "refer_seg":
                    self.all_image_datasets.append(
                        ReferSegDataset(
                            base_image_dir,
                            tokenizer,
                            vision_tower,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            refer_seg_data,
                        )
                    )
                elif image_dataset == "vqa":
                    self.all_image_datasets.append(
                        VQADataset(
                            base_image_dir,
                            tokenizer,
                            vision_tower,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            vqa_data,
                        )
                    )
                elif image_dataset == "reason_seg":
                    self.all_image_datasets.append(
                        ReasonSegDataset(
                            base_image_dir,
                            tokenizer,
                            vision_tower,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            reason_seg_data,
                            explanatory,
                        )
                    )
        # import threading
        # self.thread = threading.Thread(target=self.print_id)
        # self.thread.daemon = True  # 设置为守护线程，当主程序退出时线程也会退出
        # self.thread.start()

    def print_id(self):
        import time
        while True:
            print(f"{id(self)}, {self.data_type}, {id(self.data_type)}")
            time.sleep(4)  # 等待5秒
            
    def set_data_type(self, index):
        self.data_type = index  # 设置当前活跃的数据集索引

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if getattr(self, "image_datasets", None) is None:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
            mode_ind = np.random.choice([0,1], p=self.train_mode_rate)
            self.all_datasets[ind].set_train_mode(mode_ind)
            data = self.all_datasets[ind]
            inference = False
            return *data[0], inference
        
        if self.data_cnt % 2 == 0:
            self.data_type = np.random.choice(["video", "image"], p=[4/5, 1/5])
        
        self.data_cnt += 1

        if self.data_type == "video":
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
            mode_ind = np.random.choice([0,1], p=self.train_mode_rate)
            self.all_datasets[ind].set_train_mode(mode_ind)
            data = self.all_datasets[ind]
            inference = False
            return *data[0], inference
        elif self.data_type == "image":
            ind = np.random.choice(list(range(len(self.image_datasets))), p=self.image_sample_rate)
            data = self.all_image_datasets[ind]
            inference = False
            return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )


if __name__ == '__main__':
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        '/ssd-nvme1/gaomingqi/checkpoints/UNIFY/llama-2-7b/',
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )

    dataset = HybridDatasetVid(
        base_video_dir = '/ssd-nvme1/gaomingqi/datasets/unify',
        tokenizer = '',
        vision_tower = '',
        samples_per_epoch=500 * 8 * 2 * 10,
        precision = 'bf16',
        image_size = 512,
        num_classes_per_sample = 3,
        exclude_val = False,
        dataset = "rvos || rvot",
        sample_rate = [2, 1],
        rvos_data = "ytbvos||mevis",
        rvot_data = "lasot||tnl2k",
        explanatory = 0,
    )
    print('Initialised Done.')

