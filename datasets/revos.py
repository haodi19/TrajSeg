###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
"""
MeViS data loader
"""
from collections import defaultdict
from pathlib import Path

import torch
# # from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random


from pycocotools import mask as coco_mask


class ReVOSDataset(Dataset):
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    def __init__(self, img_folder: Path, ann_file: Path, transforms, return_masks: bool,
                 num_frames: int, max_skip: int):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks  # not used
        self.num_frames = num_frames
        self.max_skip = max_skip
        # create video meta data
        self.prepare_metas()

    def prepare_metas(self):
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        videos = list(subset_expressions_by_video.keys())  # d56a6ec78cfa, 377b1c5f365c, ...
        print('number of video in the datasets:{}'.format(len(videos)))
        self.metas = []
        
        mask_json = os.path.join(str(self.img_folder) + '/mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)

        # self.vid2metaid好像是VISA采样使用的, 我们应该没用到
        self.vid2metaid = defaultdict(list)
        for vid in videos:  # d56a6ec78cfa, 377b1c5f365c, ...
            # vid_data    = {'expressions': dict, 'vid_id': int, 'frames': List[int]}
            # expressions = {'0': {"exp": str, "obj_id": List[int], "anno_id": List[int]}, ...}
            vid_data   = subset_expressions_by_video[vid]  
            vid_frames = sorted(vid_data['frames'])  # 00000, 00001, ...
            vid_len    = len(vid_frames)
            if vid_len < 2:
                continue
            # if ('rgvos' in dataset_name) and vid_len > 80:
            #     continue
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video']    = vid  # 377b1c5f365c
                    meta['exp']      = exp_dict['exp']  # 4 lizards moving around
                    meta['obj_id']   = [int(x) for x in exp_dict['obj_id']]   # [0, 1, 2, 3, ]
                    meta['anno_id']  = [str(x) for x in exp_dict['anno_id']]  # [2, 3, 4, 5, ]
                    meta['frames']   = vid_frames  # ['00000', '00001', ...]
                    meta['frame_id'] = frame_id
                    meta['exp_id']   = exp_id  # '0'
                    meta['category'] = 0
                    meta['length']   = vid_len
                    self.metas.append(meta)
                    self.vid2metaid[vid].append(len(self.metas) - 1)

        print('\n video num: ', len(videos), ' clip num: ', len(self.metas))
        print('\n')
        
        
    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, anno_id, category, frames, frame_id = \
                meta['video'], meta['exp'], meta['anno_id'], meta['category'], meta['frames'], meta['frame_id']
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = 0
            vid_len = len(frames)

            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            if self.num_frames != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >= global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), video, frame_name + '.jpg')
                # mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                # h, w = img.shape
                mask = np.zeros(img.size[::-1], dtype=np.float32)
                for x in anno_id:
                    frm_anno = self.mask_dict[x][frame_indx]
                    if frm_anno is not None:
                        mask += coco_mask.decode(frm_anno)

                # create the target
                label = torch.tensor(category_id)

                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else:  # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0)
            boxes = torch.stack(boxes, dim=0)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)
            target = {
                'frames_idx': sample_indx,      # [T,]
                'labels': labels,               # [T,]
                'boxes': boxes,                 # [T, 4], xyxy
                'masks': masks,                 # [T, H, W]
                'valid': valid,                 # [T,]
                'caption': exp,
                'orig_size': [int(h), int(w)],
                'size': [int(h), int(w)],
                'dataset':'revos'
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            # imgs, target = self._transforms(imgs, target)
            # imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]

            # FIXME: handle "valid", since some box may be removed due to random crop
            if sum(target['valid']) > 1:  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target


def make_coco_transforms(image_set, max_size=640):
    # do nothing (only crop?) when combined with llama-vid + lisa=
    scales = [288, 320, 352, 392, 416, 448, 480, 512]
    return T.Compose([
            T.RandomResize([360], max_size=640),
            # normalize,
        ])

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])

    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, root, args):
    root = Path(root)
    assert root.exists(), f'provided ReVOS path {root} does not exist'
    PATHS = {
        "train": (root, root / "meta_expressions_train_.json"),
    }
    img_folder, ann_file = PATHS['train']
    dataset = ReVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), return_masks=args.masks,
                           num_frames=args.num_frames, max_skip=args.max_skip)
    return dataset


if __name__ == '__main__':
    root = Path('/hdd2/ljn/MUTR/data/MeVIS')
    image_set = 'train'
    PATHS = {
        "train": (root / "train", root / "train" / "meta_expressions.json"),
    }
    img_folder, ann_file = PATHS['train']

    dataset = MeViSDataset(img_folder, ann_file, transforms=T.ToTensor(), return_masks=True, num_frames=5, max_skip=3)

    img, meta = dataset[0]