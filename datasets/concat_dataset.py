# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data

from torch.utils.data import Dataset, ConcatDataset
# from .refexp2seq import build as build_seq_refexp
from .ytvos import build as build_ytvs
# from .a2d import build as build_a2d
from .tnl2k import build as build_tnl2k
from .lasot import build as build_lasot
from .mevis import build as build_mevis


# def build(image_set, args):
#     concat_data = []

#     print('preparing coco2seq dataset ....')
#     coco_names = ["refcoco", "refcoco+", "refcocog"]
#     for name in coco_names:
#         coco_seq =  build_seq_refexp(name, image_set, args)
#         concat_data.append(coco_seq)

#     print('preparing ytvos dataset  .... ')
#     ytvos_dataset = build_ytvs(image_set, args)
#     concat_data.append(ytvos_dataset)

#     concat_data = ConcatDataset(concat_data)

#     return concat_data


def build(image_set, args):
    concat_data = []

    print('preparing coco2seq dataset ....')
    coco_names = ["refcoco", "refcoco+", "refcocog"]
    for name in coco_names:
        coco_seq =  build_seq_refexp(name, image_set, args)
        concat_data.append(coco_seq)

    print('preparing ytvos dataset  .... ')
    ytvos_dataset = build_ytvs(image_set, args)
    concat_data.append(ytvos_dataset)
    
    # print('preparing a2d dataset  .... ')
    # a2d_dataset = build_a2d(image_set, args)
    # concat_data.append(a2d_dataset)
    
    print('preparing mevis dataset  .... ')
    mevis_dataset = build_mevis(image_set, args)
    concat_data.append(mevis_dataset)
    
    print('preparing lasot dataset  .... ')
    lasot_dataset = build_lasot(image_set, args)
    concat_data.append(lasot_dataset)

    print('preparing tnl2k dataset  .... ')
    tnl2k_dataset = build_tnl2k(image_set, args)
    concat_data.append(tnl2k_dataset)
    

    concat_data = ConcatDataset(concat_data)

    return concat_data