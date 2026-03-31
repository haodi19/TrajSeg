"""
Utilities for bounding box manipulation and GIoU.
"""
from typing import Tuple
import torch
from torchvision.ops.boxes import box_area

def clip_iou(boxes1,boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,0] * wh[:,1]
    union = area1 + area2 - inter
    iou = (inter + 1e-6) / (union+1e-6)
    return iou

def multi_iou(boxes1, boxes2):
    lt = torch.max(boxes1[...,:2], boxes2[...,:2])
    rb = torch.min(boxes1[...,2:], boxes2[...,2:])
    wh = (rb - lt).clamp(min=0)
    wh_1 = boxes1[...,2:] - boxes1[...,:2]
    wh_2 = boxes2[...,2:] - boxes2[...,:2]
    inter = wh[...,0] * wh[...,1]
    union = wh_1[...,0] * wh_1[...,1] + wh_2[...,0] * wh_2[...,1] - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return iou

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = (inter+1e-6) / (union+1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - ((area - union) + 1e-6) / (area + 1e-6)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def box_clip_transform(boxes, orig_size, resized_shapes, target_size):
    new_boxes = adjust_bboxes_for_center_crop_non_normalized(boxes, orig_size, resized_shapes, target_size)
    return new_boxes

def adjust_bboxes_for_center_crop_non_normalized(batch_bboxes: torch.Tensor, 
                                                 batch_orig_sizes: list, 
                                                 resized_shapes,
                                                 target_size: tuple) -> torch.Tensor:
    """
    针对没有归一化的BBox坐标,首先进行归一化,然后调整批量归一化的BBox坐标以匹配
    图像经过中心裁剪操作后的变化。

    参数:
    - batch_bboxes (torch.Tensor): 批量没有归一化的BBox坐标,尺寸为[bs, 4]。
    - batch_orig_sizes (list): 每张图像的原始尺寸的列表,列表中的每个元素都是一个(h, w)元组。
    - target_size (tuple): 目标尺寸,格式为(width, height)。

    返回:
    - torch.Tensor: 调整后的批量归一化BBox坐标,尺寸为[bs, 4]。
    """
    _batch_bboxes = batch_bboxes.clone()
    bs = _batch_bboxes.size(0)
    target_height, target_width = target_size
    
    # 初始化新的BBoxes Tensor
    new_bboxes = torch.zeros_like(_batch_bboxes, dtype=torch.float32)
    
    for i in range(bs):
        orig_height, orig_width = batch_orig_sizes[i]
        resized_height, resized_width = resized_shapes[i]
        # 归一化BBox坐标
        norm_bboxes = _batch_bboxes[i].float()
        norm_bboxes[0] /= orig_width
        norm_bboxes[1] /= orig_height
        norm_bboxes[2] /= orig_width
        norm_bboxes[3] /= orig_height
        
        crop_width = min(resized_width, target_width)
        crop_height = min(resized_height, target_height)
        norm_start_x = ((resized_width - crop_width) // 2) / resized_width
        norm_start_y = ((resized_height - crop_height) // 2) / resized_height
        norm_end_x = ((resized_width - crop_width) // 2 + crop_width) / resized_width
        norm_end_y = ((resized_height - crop_height) // 2 + crop_height) / resized_height
        
        x_min, y_min, x_max, y_max = norm_bboxes
        
        new_x_min = (x_min - norm_start_x) / (norm_end_x - norm_start_x)
        new_y_min = (y_min - norm_start_y) / (norm_end_y - norm_start_y)
        new_x_max = (x_max - norm_start_x) / (norm_end_x - norm_start_x)
        new_y_max = (y_max - norm_start_y) / (norm_end_y - norm_start_y)
        
        # 确保坐标在[0, 1]范围内
        new_bboxes[i, 0] = max(0, min(new_x_min, 1))
        new_bboxes[i, 1] = max(0, min(new_y_min, 1))
        new_bboxes[i, 2] = max(0, min(new_x_max, 1))
        new_bboxes[i, 3] = max(0, min(new_y_max, 1))
    
    return new_bboxes