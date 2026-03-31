"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
# from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

from datasets.categories import ytvos_category_dict as category_dict


class LaSOTDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """

    def __init__(self, root_path: str, transforms, return_masks: bool,
                 num_frames: int, max_skip: int):
        self.root_path = root_path
        self._transforms = transforms
        self.return_masks = return_masks  # not used
        self.num_frames = num_frames
        self.max_skip = max_skip
        # create video meta data
        self.prepare_videos()
        self.prepare_metas()

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_videos(self):
        training_list_path = os.path.join(self.root_path, "list_train.txt")
        with open(training_list_path, 'r') as f:
            training_videos = f.readlines()
            training_videos = [video.rstrip('\n')[:-2] for video in training_videos]
        self.videos = sorted(training_videos)

    def prepare_metas(self):
        self.metas = []
        for video_name in self.videos:
            video_category = video_name.split('-')[0]
            video_path = os.path.join(self.root_path, 'LaSOTBenchmark-full', video_category, video_name)
            imgs_path = os.path.join(video_path, "img")
            gt_path = os.path.join(video_path, "groundtruth.txt")
            caption_path = os.path.join(video_path, "nlp.txt")

            with open(caption_path, 'r') as cf:
                exp = cf.readline()
            with open(gt_path, 'r') as lf:
                # (x1,y1,width,hidth)
                gt = lf.readline().strip('\n').split(',')

            # 此处frames带有.jpg后缀
            frames = os.listdir(imgs_path)
            frames.sort()

            if 'Thumbs.db' in frames:
                frames.remove('Thumbs.db')

            vid_len = len(frames)
            for frame_id in range(0, vid_len, self.num_frames):
                meta = {}
                meta['video'] = video_name
                meta['exp'] = exp
                meta['frames'] = frames
                meta['frame_id'] = frame_id
                self.metas.append(meta)

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
        # 所有mask部分删除
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, frames, frame_id = \
                meta['video'], meta['exp'], meta['frames'], meta['frame_id']

            # clean up the caption
            exp = " ".join(exp.lower().split())
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
            imgs, boxes, valid = [], [], []

            video_category = video.split('-')[0]
            video_path = os.path.join(self.root_path, 'LaSOTBenchmark-full', video_category, video)
            imgs_path = os.path.join(video_path, "img")

            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(imgs_path, frame_name)
                img = Image.open(img_path).convert('RGB')

                gt_path = os.path.join(video_path, "groundtruth.txt")

                with open(gt_path, 'r') as lf:
                    # (x1,y1,width,hidth)
                    gts = lf.readlines()

                box = gts[frame_indx].rstrip().split(',')
                # create the target
                if box != ['0', '0', '0', '0']:
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]) + float(box[0]), float(box[3]) + float(
                        box[1])
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else:  # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)

                # append
                imgs.append(img)
                boxes.append(box)

            # transform
            w, h = img.size
            boxes = torch.stack(boxes, dim=0)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            
            target = {
                'frames_idx': sample_indx,      # [T,]
                'boxes': boxes,                 # [T, 4], xyxy
                'valid': valid,                 # [T,]
                'caption': exp,
                'orig_size': [int(h), int(w)],
                'size': [int(h), int(w)],
                'dataset': 'lasot'
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
    assert root.exists(), f'provided lasot path {root} does not exist'
    dataset = LaSOTDataset(root, transforms=make_coco_transforms(image_set, max_size=args.max_size),
                           return_masks=args.masks,
                           num_frames=args.num_frames, max_skip=args.max_skip)
    return dataset
