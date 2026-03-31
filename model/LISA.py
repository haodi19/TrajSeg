from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel
from einops import rearrange

from model.fuse_modules import BiAttentionBlock
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, IMAGE_TOKEN_INDEX, TRAJECTORY_TOKEN_INDEX)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from datasets.util import box_ops
import copy

from .sam2.modeling.sam2_base import NO_OBJ_SCORE

# sam 2
# from .sam2 import build_sam2_video_predictor
from .sam2 import build_sam2_video_predictor_refine

# def qqq(
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     num_masks: float,
#     scale=1000,  # 100000.0,
#     eps=1e-6,
# ):
#     """
#     Compute the DICE loss, similar to generalized IOU for masks
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     """

#     if inputs.ndim == 4:
#         inputs = inputs.permute(1, 0, 2, 3).sigmoid()
#         inputs = inputs.flatten(1)
#         targets = targets.permute(1, 0, 2, 3).flatten(1)
#         numerator = 2 * (inputs * targets).sum(-1)
#         denominator = inputs.sum(-1) + targets.sum(-1)
#         loss = 1 - (numerator + eps) / (denominator + eps)
#         print(numerator, denominator, loss)
#         loss = loss.sum() / (num_masks + 1e-8)
#     else:
#         inputs = inputs.sigmoid()
#         inputs = inputs.flatten(1, 2)
#         targets = targets.flatten(1, 2)
#         numerator = 2 * (inputs * targets).sum(-1)
#         denominator = inputs.sum(-1) + targets.sum(-1)
#         loss = 1 - (numerator + eps) / (denominator + eps)
#         loss = loss.sum() / (num_masks + 1e-8)
#     return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    gt_score: torch.Tensor = None,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # 根据inputs的形状调整gt_score的形状
    # inputs: torch.Size([video_len, seg_num, 1200, 1920])
    if gt_score is not None:
        gt_score = gt_score.squeeze(2)

    if inputs.ndim == 4:
        inputs = inputs.permute(1, 0, 2, 3).sigmoid()
        inputs = inputs.flatten(2)  # [obj_num, video_len, 1200*1920]
        targets = targets.permute(1, 0, 2, 3).flatten(2)
        gt_score = gt_score.permute(1, 0) if gt_score is not None else None

        numerator = 2 * (inputs * targets).sum(-1)  # sum over spatial dimensions
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        if gt_score is not None:
            loss = loss * gt_score.float()  # apply frame-wise mask
        loss = loss.mean(1).sum() / (num_masks + 1e-8)  # normalize by the number of masks
    else:
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1, 2)  # [video_len*obj_num, 1200*1920]
        targets = targets.flatten(1, 2)
        gt_score = gt_score.flatten(1, 2) if gt_score is not None else None

        numerator = 2 * (inputs * targets).sum(-1)  # sum over spatial dimensions
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        if gt_score is not None:
            loss = loss * gt_score.float()  # apply frame-wise mask
        loss = loss.sum() / (num_masks + 1e-8)  # normalize by the number of masks
    return loss

def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    gt_score: torch.Tensor = None
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # inputs: torch.Size([video_len, seg_num, 1200, 1920])
    # gt_score: torch.Size([video_len, seg_num, 1])
    # 根据inputs的形状调整gt_score的形状
    if gt_score is not None:
        gt_score = gt_score.unsqueeze(-1).expand_as(inputs)

    if inputs.ndim == 4:
        loss = F.binary_cross_entropy_with_logits(inputs.permute(1, 0, 2, 3), targets.permute(1, 0, 2, 3), reduction="none")
        if gt_score is not None:
            loss = loss * gt_score.permute(1, 0, 2, 3)  # 仅计算存在目标的帧的损失
        loss = loss.flatten(1).mean(1).sum() / (num_masks + 1e-8)
    else:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        if gt_score is not None:
            loss = loss * gt_score  # 仅计算存在目标的帧的损失
        loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def l1_loss(pred_boxes, target_boxes, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # video, pred_boxes:torch.Size([5, 1, 4]) (video_len, num_boxes, 4)
        loss_bbox = F.l1_loss(pred_boxes.permute(1,0,2), target_boxes.permute(1,0,2), reduction='none')
        return loss_bbox.mean(1).sum() / num_boxes
    
def giou_loss(pred_boxes, target_boxes, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # video, pred_boxes:torch.Size([5, 1, 4])

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_boxes.permute(1,0,2).flatten(0,1)),
            box_ops.box_cxcywh_to_xyxy(target_boxes.permute(1,0,2)).flatten(0,1)))
        # torch.Size([1 * 5])
        
        return loss_giou.reshape(pred_boxes.shape[1], pred_boxes.shape[0], -1).mean(1).sum() / num_boxes
    

def cls_ce_loss(pred_score, gt_score, num_masks):
    """
    计算二元交叉熵损失。

    参数:
    - gt_score (torch.Tensor): 形状为 [video_len, seg_num, 1] 的视频掩码GT张量。
    - pred_score (torch.Tensor): 形状为 [video_len, seg_num, 1] 的预测对象得分张量, dtype为torch.bfloat16。

    返回:
    - loss (torch.Tensor): 计算得到的损失值。
    """

    # 计算二元交叉熵损失
    loss = F.binary_cross_entropy_with_logits(pred_score, gt_score, reduction="none")

    return loss.mean(0).sum() / (num_masks + 1e-8)
    
def check_nan_parameters(model):
    nan_parameters = {}
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_parameters[name] = param
    return nan_parameters

def get_gpu_tensors():
    import gc
    tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            tensors.append(obj)
    return tensors

def print_all_gpu_tensors():
    total_memory = 0
    tensors = get_gpu_tensors()
    for tensor in tensors:
        size_in_bytes = tensor.element_size() * tensor.nelement()
        print(f"Tensor of shape {tensor.shape}: {size_in_bytes / 1024 ** 2:.2f} MB")
        total_memory += size_in_bytes
    print(f"Total memory used by tensors: {total_memory / 1024 ** 2:.2f} MB")
class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config, mode=kwargs['mode'] if 'mode' in kwargs else "eval")

    def _get_clones(self, module, N, layer_share=False):
        # import ipdb; ipdb.set_trace()
        if layer_share:
            return nn.ModuleList([module for i in range(N)])
        else:
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def initialize_fusion_modules(self):
        post_fusion_type = getattr(self.config, "post_fusion_type", "none")

        if post_fusion_type in ['gdino_bi', 'gdino_bi_updated_img', 'gdino_uni']:
            feature_fusion_layer = BiAttentionBlock(
                    v_dim=256,
                    l_dim=256,
                    embed_dim=1024,
                    num_heads=4,
                    dropout=0.0,
                    drop_path=0.1,
                    fusion_type=post_fusion_type
            )    
            
            
            self.fusion_layers = self._get_clones(
                feature_fusion_layer, 1, layer_share=False
            )
            for param in self.fusion_layers.parameters():
                param.requires_grad = True

        elif post_fusion_type == 'nn':
            self.fusion_layers = nn.ModuleList([
                nn.Sequential(
                    nn.MultiheadAttention(256, 4, dropout=0.1),
                    # nn.Dropout(0.1),
                    nn.LayerNorm(256)
                ) for _ in range(6)
            ])
            
            for layer in self.fusion_layers:
                for param in layer.parameters():
                    param.requires_grad = True  

            
            # self.fusion_layers = nn.MultiheadAttention(256, 4, dropout=0.0)
            # for param in self.fusion_layers.parameters():
            #     param.requires_grad = True

    def initialize_lisa_modules(self, config, mode = 'eval'):
        # SAM
        if getattr(config, "sam2", False):
            self.visual_model = build_sam2_video_predictor_refine(
                config_file=config.sam2_cfg,
                ckpt_path=config.sam2_checkpoint,
                mode = mode,
                apply_postprocessing=False,
            )
        else:
            self.visual_model = build_sam_vit_h(self.vision_pretrained)

        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder and not getattr(config, "sam2", False):
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        elif config.train_mask_decoder and getattr(config, "sam2", False):
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        post_fusion_type = getattr(self.config, "post_fusion_type", "none")
        if post_fusion_type in ['gdino_bi', 'gdino_bi_updated_img', 'gdino_uni']:
            feature_fusion_layer = BiAttentionBlock(
                    v_dim=256,
                    l_dim=256,
                    embed_dim=1024,
                    num_heads=4,
                    dropout=0.0,
                    drop_path=0.1,
                    fusion_type=post_fusion_type
            )    
            
            
            self.fusion_layers = self._get_clones(
                feature_fusion_layer, 1, layer_share=False
            )
            for param in self.fusion_layers.parameters():
                param.requires_grad = True

        elif post_fusion_type == 'nn':
            self.fusion_layers = nn.ModuleList([
                nn.Sequential(
                    nn.MultiheadAttention(256, 4, dropout=0.1),
                    # nn.Dropout(0.1),
                    nn.LayerNorm(256)
                ) for _ in range(6)
            ])
            
            for layer in self.fusion_layers:
                for param in layer.parameters():
                    param.requires_grad = True  

            
            # self.fusion_layers = nn.MultiheadAttention(256, 4, dropout=0.0)
            # for param in self.fusion_layers.parameters():
            #     param.requires_grad = True        


class LisaModel(LisaMetaModel, LlavaLlamaModel):
# class LisaModel(LisaMetaModel, LlavaAttLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
# class LISAForCausalLM(LlavaLlamaAttForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # if not hasattr(config, "train_mask_decoder"): # TODO: find the reason
        if not hasattr(config, "aa") and "use_mm_start_end" in kwargs:
            # 推理不走这里
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)

            # config.mm_vision_tower = kwargs.get(
            #     "vision_tower", "openai/clip-vit-large-patch14"
            # )
            if hasattr(config, "vision_tower"):
                config.mm_vision_tower = config.vision_tower
            else:
                 config.mm_vision_tower = kwargs.get(
                    "vision_tower", "openai/clip-vit-large-patch14"
                )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.l1_loss_weight = kwargs.pop("l1_loss_weight", None)
            self.giou_loss_weight = kwargs.pop("giou_loss_weight", None)
            self.obj_loss_weight = kwargs.pop("obj_loss_weight", None)
            
            # 下面的参数在第二阶段之后的训练中，如果原config已经包含对应键值, 则from_pretrained函数会自动将他们替换成kwargs的同名参数（如果有），并且kwargs的参数被替换后会从kwargs删掉
            # 因此默认kwargs一定会提供下面的参数（旧的ckpt可能会不适配）
            if "use_multuple_seg_token" in kwargs:
                config.use_multuple_seg_token = kwargs.pop("use_multuple_seg_token", False)
            if "post_fusion_type" in kwargs:
                config.post_fusion_type = kwargs.pop("post_fusion_type", 'none')
            if "sam2" in kwargs:
                config.sam2 = kwargs.pop("sam2", False)
                config.sam2_cfg = kwargs.pop("sam2_cfg", None)
                config.sam2_checkpoint = kwargs.pop("sam2_checkpoint", None)
                config.seg_refine = kwargs.pop("seg_refine", False)
                config.memory_refine = kwargs.pop("memory_refine", False)
            if "llm_sample_mode" in kwargs:
                config.llm_sample_mode = kwargs.pop("llm_sample_mode", 'nosample')
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.sam2 = getattr(config, "sam2", False)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        # torch.Size([2, 3, 1024, 1024]) -> torch.Size([2, 1, 3, 1024, 1024])
        with torch.no_grad():
            image_embeddings_list = []
            fpn_list = []
            pos_list = []
            for i in range(pixel_values.shape[0]):
                # torch.cuda.empty_cache()
                if pixel_values[i].ndim == 4:
                    video_len = pixel_values[i].shape[0]
                    image_embeddings_list2 = []
                    fpn_list_2 = []
                    pos_list_2 = []
                    for j in range(video_len):
                        image_embeddings = self.model.visual_model.image_encoder(
                            pixel_values[i][j].unsqueeze(0)
                        ) 
                        if not self.sam2:
                            image_embeddings_list2.append(image_embeddings)
                        else:
                            image_embeddings_list2.append(image_embeddings['vision_features'].to(torch.bfloat16))
                            p_list = [p_i.to(torch.bfloat16) for p_i in image_embeddings['vision_pos_enc']]
                            pos_list_2.append(p_list)
                            f_list = [f_i.to(torch.bfloat16) for f_i in image_embeddings['backbone_fpn']]
                            fpn_list_2.append(f_list)
                    video_embeddings = torch.stack(image_embeddings_list2)
                    image_embeddings_list.append(video_embeddings)
                    if self.sam2:
                        fpn_list.append(fpn_list_2)
                        pos_list.append(pos_list_2)

                else:  
                    image_embeddings = self.model.visual_model.image_encoder(
                        pixel_values[i].unsqueeze(0)
                    )
                    image_embeddings_list.append(image_embeddings)
            # torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
            if len(fpn_list) > 0:
                fpn_embeddings = fpn_list
                pos_embeddings = pos_list
            else:
                fpn_embeddings = None
                pos_embeddings = None

        return image_embeddings, fpn_embeddings, pos_embeddings

    def forward(self, attention_mask=None, **kwargs):
        # 推理的时候好像一定会进入分支(即不会执行self.model_forward)
        if "past_key_values" in kwargs:
            return super().forward(attention_mask=attention_mask, **kwargs)
        # 训练的时候会进入下面分支
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):  
        # 首先说明，在训练时一个batch里可能有多个文本（数量设为qs）即一张图片/视频要跑多个文本，所以最终是同时跑qs次（即实际bs为qs,而这里的bs仅为视频/图片个数)
        # 这个方法包含了所有步骤(即最终输出mask)，返回的是loss，具体步骤和推理时是类似的，类似于推理时的model.evaluate()
        # input_ids:tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
        #                    21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
        #                      322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
        #                    29889,  3148,  1001, 29901, 32001,  -200, 32002, 22172,   319,  1799,
        #                     9047, 13566, 29901]], device='cuda:0'), torch.Size([qs, 73])
        # images_clip: 原本是torch.Size([bs, 3, 224, 224])，需要修改为torch.Size([bs, video_len, 3, 224, 224]),由于我们是采样训练，可以保证一个bs内所有视频帧数相同
        # images: 原本是torch.Size([bs, 3, 1024, 1024])，需要修改为torch.Size([bs, video_len, 3, 1024, 1024]), 用于SAM, images_clip用于LLM, 这样对我们很方便，因为输给SAM的图片和输给LLM的视频可以分开
        
        # labels: 训练时用,文本GT, torch.Size([qs, 73])
        # offset: 训练时用,tensor([0, 3, 6], device='cuda:0'), 长度为bs+1, 用于确认每个视频对应文本input_ids的索引范围
        # masks_list: 训练时用, mask的GT, [len=bs], 如[torch.Size([3, 330, 440]), torch.Size([3, 427, 640])](lisa原版), 需要修改为[torch.Size([video_len1, 3, 330, 440]), torch.Size([video_len2,3, 427, 640])]
        # kwargs['box_list']: 训练时用, box的GT, [len=bs], 元素为torch.Size([video_len, 4])
        # label_list:训练时用, [len=bs], 如[torch.Size([330, 440]), torch.Size([427, 640])], 意义待研究，但好像只用到了其中的shape信息
        # resize_list: [len=bs], 如[[768, 1024], [683, 1024]]
        # kwargs['prompts']: 暂时只有训练时用, [len=qs], 每个问题对应一个prompt


        # 关于视频采样部分: llm_sample_mode用于控制采样类型，默认为"nosample", 即送入llm的为完整视频; 当采用均匀采样时,llm_sample_mode='uniform',
        # 此时会有1个参数kwargs['sampled_indices']表示采样的帧号(具体为1个list,里面是帧号)
        # 当启用llm_sample_mode='uniform'时, 假设原视频有n帧, 采样了k帧, 那么需要送入LLM的images_clip的shape变为[bs,k,3,224,224], 送入SAM2的images还是[bs, n, 3, 1024, 1024]
        # 前面LLM的部分不需要修改, 当启用均匀采样是LLM返回的
        training_type = kwargs.get("training_type", 1)
        llm_sample_mode = getattr(self.config, "llm_sample_mode", "nosample")
        sampled_indices = kwargs.get("sampled_indices", None)
        post_fusion_type = getattr(self.config, "post_fusion_type", "none")
        use_fusion_module = (post_fusion_type != 'none')
        seg_refine = getattr(self.config, "seg_refine", False)
        memory_refine = getattr(self.config, "memory_refine", False)
        if images.shape[1] == 1:
            training_type = 1
            llm_sample_mode = "nosample"

        bs, t, _, _, _ = images.shape

        image_embeddings, fpn_embeddings, pos_embeddings = self.get_visual_embs(images) # 10 1 256 64 64
        image_embeddings = image_embeddings.squeeze(1)
        image_embeddings = rearrange(image_embeddings, '(b t) c h w -> b t c h w', b=bs, t=t)
        # image_embeddings: torch.Size([bs, 256, 64, 64]) -> torch.Size([bs, video_len, 256, 64, 64])
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        
        if training_type != 2:
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            
            if type(images_clip) is list or images_clip.ndim == 5:
                # 由于我们是采样训练，可以保证一个bs内所有视频帧数相同
                llm_video_len = images_clip[0].shape[0] if type(images_clip) is list else images_clip.shape[1]
                
                token_per_frame = 256
                
                if (input_ids == TRAJECTORY_TOKEN_INDEX).any():
                    seg_token_mask = torch.stack([torch.cat([torch.zeros(llm_video_len * token_per_frame - 1).bool().cuda(),
                                                                seg_token_mask[i]]) if TRAJECTORY_TOKEN_INDEX in input_ids[i] 
                                                            else torch.cat([torch.zeros(llm_video_len * token_per_frame - llm_video_len).bool().cuda(),
                                                                seg_token_mask[i], torch.zeros(llm_video_len - 1).bool().cuda()]) for i in range(len(seg_token_mask))])
                else:
                    seg_token_mask = torch.cat(
                        [
                            torch.zeros((seg_token_mask.shape[0], llm_video_len * token_per_frame - llm_video_len)).bool().cuda(),
                            seg_token_mask,
                        ],
                        dim=1,
                    )
                
            else:
                # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
                seg_token_mask = torch.cat(
                    [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
                    dim=1,
                )
            

            if inference:
                # 推理(这里其实这是validation会跑)一定是一个视频+一个文本
                n_batch = 1
                length = input_ids.shape[0]
                assert images_clip.shape[0] == 1
                images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

                output_hidden_states = []
                for i in range(n_batch):
                    start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                    output_i = super().forward(
                        images=images_clip_extend[: end_i - start_i],
                        attention_mask=attention_masks[start_i:end_i],
                        input_ids=input_ids[start_i:end_i],
                        output_hidden_states=True,
                    )
                    output_hidden_states.append(output_i.hidden_states)
                    torch.cuda.empty_cache()

                output_hidden_states_list = []
                output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
                output_hidden_states_list.append(output_hidden_states_level)
                output_hidden_states = output_hidden_states_list
                output = None

            else:
                images_clip_list = []
                for i in range(len(offset) - 1):
                    start_i, end_i = offset[i], offset[i + 1]
                    if images_clip.ndim == 5:
                        images_clip_i = (
                            images_clip[i]
                            .unsqueeze(0)
                            .expand(end_i - start_i, -1, -1, -1, -1)
                            .contiguous()
                        )
                    else:
                        images_clip_i = (
                            images_clip[i]
                            .unsqueeze(0)
                            .expand(end_i - start_i, -1, -1, -1)
                            .contiguous()
                        )
                    images_clip_list.append(images_clip_i)
                images_clip = torch.cat(images_clip_list, dim=0)

                # 这里替换成LLaMA-VID的LlavaLlamaAttForCausalLM.forward()!!!
                output = super().forward(
                    images=images_clip,
                    attention_mask=attention_masks,
                    input_ids=input_ids,
                    labels=labels,
                    output_hidden_states=True,
                    prompts=kwargs["prompts"],
                    trajectories=torch.stack(kwargs["trajectoies"]) if ("trajectoies" in kwargs and None not in kwargs["trajectoies"]) else None # bs*video_len*4, 这里暂时默认bs=qs了
                )
                # # hidden_states: torch.Size([bs?, 298(input_ids.token_nums + 1), 4096]), 这里不确定，可能包了tuple
                output_hidden_states = output.hidden_states

            
            hidden_states = []
            
            # self.model.text_hidden_fcs[0]:
            # Sequential(
            #   (0): Linear(in_features=4096, out_features=4096, bias=True)
            #   (1): ReLU(inplace=True)
            #   (2): Linear(in_features=4096, out_features=256, bias=True)
            #   (3): Dropout(p=0.0, inplace=False)
            # )
            
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

            # last_hidden_state: torch.Size([bs, 171, 256])
            # seg_token_mask: torch.Size([6, 171])
            # pred_embeddings: torch.Size([6, 256])
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            seg_token_offset = seg_token_offset[offset]
            
            pred2img_indices = torch.searchsorted(seg_token_offset, torch.arange(seg_token_offset[-1]).cuda(), right=True) - 1
            
            if post_fusion_type == 'gdino_uni' and pred_embeddings.shape[0] != 0:
                seg_num = pred_embeddings.shape[0]
                llm_video_len = images_clip.shape[1]
                pred_embeddings = pred_embeddings.repeat_interleave(llm_video_len, dim=0).unsqueeze(1)

                # image_embeddings:torch.Size([bs, video_len, 256, 64 , 64]) -> torch.Size([bs*video_len,N,256]) 
                # pred_embeddings: torch.Size([bs * video_len, 1, 256])
                # key_padding_mask = torch.zeros(_image_embeddings.shape[0], _image_embeddings.shape[1], dtype=torch.bool, device=pred_embeddings.device)
                # text_attention_mask = torch.zeros(pred_embeddings.shape[0], pred_embeddings.shape[1], dtype=torch.bool, device=pred_embeddings.device)
                    
                for i, layer in enumerate(self.model.fusion_layers):
                    # output(src): torch.Size([bs, 20906, 256])
                    # memory_text: torch.Size([bs, 4, 256])
                    # key_padding_mask: torch.Size([bs, 20906])
                    # text_attention_mask: torch.Size([bs, 4])
                    if llm_sample_mode == 'uniform':
                        v=image_embeddings[pred2img_indices][:,sampled_indices].flatten(-2).flatten(0,1).permute(0,2,1)     # sampled-len x 4096 x 256
                    elif llm_sample_mode == 'nosample':
                        v=image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(0,2,1)
                    pred_embeddings = self.model.fusion_layers[i](
                        v=v,
                        l=pred_embeddings,
                        # attention_mask_v=key_padding_mask,
                        # attention_mask_l=text_attention_mask,
                    )
                pred_embeddings = pred_embeddings.reshape(seg_num, llm_video_len, -1)  

            elif post_fusion_type == 'gdino_bi' and pred_embeddings.shape[0] != 0:
                seg_num = pred_embeddings.shape[0]
                llm_video_len = images_clip.shape[1]
                pred_embeddings = pred_embeddings.repeat_interleave(llm_video_len, dim=0).unsqueeze(1)
                
                if llm_sample_mode == 'uniform':
                    _image_embeddings = image_embeddings[pred2img_indices][:,sampled_indices].detach().flatten(-2).flatten(0,1).permute(0,2,1) 
                elif llm_sample_mode == 'nosample':
                    _image_embeddings = image_embeddings[pred2img_indices].detach().flatten(-2).flatten(0,1).permute(0,2,1) 
                
                # _image_embeddings: torch.Size([bs * video_len, 4096, 256])
                
                for i, layer in enumerate(self.model.fusion_layers):    
                    _image_embeddings, pred_embeddings = self.model.fusion_layers[i](
                        v=_image_embeddings,
                        l=pred_embeddings,
                        # attention_mask_v=key_padding_mask,
                        # attention_mask_l=text_attention_mask,
                    )
                pred_embeddings = pred_embeddings.reshape(seg_num, llm_video_len, -1)  
            
            elif post_fusion_type == 'gdino_bi_updated_img' and pred_embeddings.shape[0] != 0:
                #这个模式在QA和qs!=bs时有bug，需要用到再修
                seg_num = pred_embeddings.shape[0]
                llm_video_len = images_clip.shape[1]
                pred_embeddings = pred_embeddings.repeat_interleave(llm_video_len, dim=0).unsqueeze(1)
                if llm_sample_mode == 'uniform':
                    image_embeddings = image_embeddings[pred2img_indices][:,sampled_indices].flatten(-2).flatten(0,1).permute(0,2,1)
                elif llm_sample_mode == 'nosample':
                    image_embeddings = image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(0,2,1)
                
                for i, layer in enumerate(self.model.fusion_layers):
                    # output(src): torch.Size([bs, 20906, 256])
                    # memory_text: torch.Size([bs, 4, 256])
                    # key_padding_mask: torch.Size([bs, 20906])
                    # text_attention_mask: torch.Size([bs, 4])
                    image_embeddings, pred_embeddings = self.model.fusion_layers[i](
                        v=image_embeddings,
                        l=pred_embeddings,
                        # attention_mask_v=key_padding_mask,
                        # attention_mask_l=text_attention_mask,
                    )
                    
                image_embeddings = image_embeddings.permute(0, 2, 1).view(seg_num, llm_video_len, 256, 64, 64) 
                pred_embeddings = pred_embeddings.reshape(seg_num, llm_video_len, -1)  
            
            elif post_fusion_type == 'nn' and pred_embeddings.shape[0] != 0:
                # pred_embeddings: [1, bs (其实应该是seg_num, 视频版本简化为bs) * video_len, 256]
                seg_num = pred_embeddings.shape[0]
                llm_video_len = images_clip.shape[1]
                pred_embeddings = pred_embeddings.repeat_interleave(llm_video_len, dim=0).unsqueeze(0)
                
                if llm_sample_mode == 'uniform':
                    image_embed_kv = image_embeddings[pred2img_indices][:,sampled_indices].flatten(-2).flatten(0,1).permute(2,0,1)
                elif llm_sample_mode == 'nosample':
                    image_embed_kv = image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(2,0,1)
                    
                for layer in self.model.fusion_layers:
                    attn_output = layer[0](  # MultiheadAttention
                        query=pred_embeddings,
                        key=image_embed_kv,
                        value=image_embed_kv, attn_mask=None
                    )[0]
                    
                    # pred_embeddings = layer[2](pred_embeddings + layer[1](attn_output))  # Dropout + LayerNorm 残差连接
                    pred_embeddings = layer[1](pred_embeddings + attn_output)  # 仅残差连接 + LayerNorm
                
                # pred_embeddings = pred_embeddings + self.model.fusion_layers(
                #     query=pred_embeddings,
                #     key=image_embed_kv,
                #     value=image_embed_kv, attn_mask=None,)[0]
                
                pred_embeddings = pred_embeddings.squeeze(0).reshape(seg_num, llm_video_len, -1)
            
            if use_fusion_module and pred_embeddings.shape[0] == 0:
                # 当use_fusion_module，并且一个bs全是QA时, 调整pred_embeddings的shape统一
                pred_embeddings = pred_embeddings.unsqueeze(1).repeat_interleave(images_clip.shape[1], dim=1)
                
            
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

        else:
            pred_embeddings = [None for i in range(batch_size)]
        
        multimask_output = False
        pred_masks = []
        pred_boxes = []
        pred_object_score_logits = []

        # 一个视频所有<SEG>一起算mask, 包括不同问题的<SEG>和一个问题的多个<SEG>, 但事实上好像不考虑后者
        for i in range(len(pred_embeddings)):
            if training_type != 2:
                if use_fusion_module and not self.sam2:    
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].flatten(0,1).unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                elif not use_fusion_module and not self.sam2: 
                    # sparse_embeddings: torch.Size([3 (该视频<SEG>总数), 1, 256])
                    # dense_embeddings: torch.Size([3 (该视频<SEG>总数), 256, 64, 64])
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                elif use_fusion_module and self.sam2:
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].flatten(0,1).unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                elif not use_fusion_module and self.sam2: 
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            
            else: 
                sampled_indices = [0]
            
            # image_embeddings: torch.Size([bs, 256, 64, 64]) -> torch.Size([bs, video_len, 256, 64, 64]), 因为是视频需要循环解码
            # self.model.visual_model.prompt_encoder.get_dense_pe(): torch.Size([1, 256, 64, 64])
            if image_embeddings.ndim == 5 and not getattr(self.config, "use_multuple_seg_token", False):
                video_len = image_embeddings.shape[1]
                video_masks = []
                video_boxes = []
                video_object_score_logits = []
                
                if seg_refine:
                    """
                    优化模式数据准备, 使用已编码特征, 初始化 inference_state
                    """
                    inference_state = self.model.visual_model.init_state(
                        video_height=label_list[i].shape[0],
                        video_width=label_list[i].shape[1],
                        images=images[i],
                        cached_features=image_embeddings[i],
                        fpn_embeddings=fpn_embeddings[i],
                        pos_embeddings=pos_embeddings[i],
                    )
                
                for j in range(video_len):
                    # 区分 sampled & un-sampled frames
                    if sampled_indices is not None:
                        if j in sampled_indices:
                            sampled_indice = sampled_indices.index(j)   # indice in the sampled list (sparse/dense_embeddings only cover the sampled ones, image_embeddings cover the full frames)
                            sampled_frame = True
                        else:
                            sampled_frame = False
                    else:
                        sampled_indice = j
                        sampled_frame = True
                        
                    curr_image_embeddings = image_embeddings[i][j]
                    seg_token_num = pred_embeddings[i].shape[0] if training_type != 2 else None
                    object_score_logits = None
                    # low_res_masks: torch.Size([3 (该视频<SEG>总数), 1, 256, 256])
                    # iou_predictions: torch.Size([3 (该视频<SEG>总数), 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)            
  
                    if use_fusion_module:
                    # low_res_masks: torch.Size([3 (该视频<SEG>总数), 1, 256, 256])
                    # iou_predictions: torch.Size([3 (该视频<SEG>总数), 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)
                    # boxes: torch.Size([3 (该视频<SEG>总数), 1, 4])
                        curr_indices = sampled_indice + torch.arange(seg_token_num) * video_len if training_type != 2 else None
                        if not self.sam2:
                            low_res_masks, boxes, iou_predictions = self.model.visual_model.mask_decoder(
                                image_embeddings=curr_image_embeddings.unsqueeze(0),
                                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings[curr_indices],
                                dense_prompt_embeddings=dense_embeddings[curr_indices],
                                multimask_output=multimask_output,
                            )
                        else:
                            fpn_list = fpn_embeddings[i][j]
                            # fpn_list[0] = self.model.visual_model.sam_mask_decoder.conv_s0(fpn_list[0])
                            # fpn_list[1] = self.model.visual_model.sam_mask_decoder.conv_s1(fpn_list[1])
                            pos_list = pos_embeddings[i][j]
                            if seg_refine: # seg_token_refine_propagation:
                                """
                                0. 给定 video_len 个视频帧, 对每个视频帧, 得到一个 seg_token
                                1. 使用第一帧 seg_token, 得到第一帧 mask
                                2. 使用第一帧 mask, 逐个帧传播, 从 1 - (video_len-1), 第一帧索引为 0, 下文为第 t 帧
                                3. 如果第 t 帧需要 refine, 先用 propagation 得到一个 mask, 然后添加 segtoken 更新 mask
                                4. 更新后, 重置 tracking-state 和 memory, 当前帧 (t) 作为第 0 帧向后传播 (add point 只能影响当前帧预测, 不能影响 memofy, 因此, 在每次更新后, 就重置 tracker 和 memory, 将当前帧作为第一帧向后传播)
                                """
                                if j == 0:
                                    frame_idx = 0
                                    # object_score_logits: torch.Size([1, 1]) e.g. tensor([[4.2188]], device='cuda:0', dtype=torch.bfloat16)
                                    if training_type == 2:
                                        pred_mask = masks_list[i][0].unsqueeze(0)
                                        object_score_logits = torch.zeros((1,1), device = pred_mask.device, dtype = torch.bfloat16)
                                    else:
                                        low_res_masks, iou_predictions, _, object_score_logits = self.model.visual_model.sam_mask_decoder(
                                            image_embeddings = curr_image_embeddings.unsqueeze(0),
                                            image_pe = self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings = sparse_embeddings[curr_indices],
                                            dense_prompt_embeddings = dense_embeddings[curr_indices],
                                            multimask_output = multimask_output,
                                            repeat_image = True,
                                            high_res_features = fpn_list[:-1],
                                        )
                                        
                                        is_obj_appearing = object_score_logits > 0

                                        # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                                        # consistent with the actual mask prediction
                                        low_res_masks = torch.where(
                                            is_obj_appearing[:, None, None],
                                            low_res_masks,
                                            NO_OBJ_SCORE,
                                        )
                                        
                                        pred_mask = self.model.visual_model.postprocess_masks(
                                            low_res_masks,
                                            input_size=resize_list[i],
                                            original_size=label_list[i].shape,
                                        )   # upsample mask

                                # if j == 0 or start_point:
                                # if j == 0:
                                # 第一帧(编号0), 或者是被 seg token 改变过的帧
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 0. initialize inference_state
                                    # if sampled_indice+1 == len(samples_indices) and samples_indices[sampled_indice]<(video_len):
                                    #     inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:video_len], async_loading_frames=False)
                                    # else: #
                                    #     inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:(samples_indices[sampled_indice+1]+1)], async_loading_frames=False)
                                    # inference_state['video_height'] = original_size_list[i][0]
                                    # inference_state['video_width'] = original_size_list[i][1]
                                    # image = images[i][j].float().unsqueeze(0).to(torch.bfloat16)
                                    # backbone_out = {
                                    #     "vision_features": image_embeddings[i, j].unsqueeze(0).to(torch.bfloat16),
                                    #     "backbone_fpn": fpn_list.copy(),
                                    #     "vision_pos_enc": pos_list.copy(),
                                    # }
                                    # # Cache the most recent frame's feature (for repeated interactions with
                                    # # a frame; we can use an LRU cache for more frames in the future).
                                    # inference_state["cached_features"] = {0: (image, backbone_out)}
                                    # self.model.visual_model.reset_state(inference_state)
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 1. add_new_mask
                                    if pred_mask.shape[0] == 0:
                                        object_mask = np.zeros((pred_mask.shape[2], pred_mask.shape[3])) > 0
                                    else:
                                        object_mask = (pred_mask > 0)[0,0].cpu().numpy()    # just propagation
                                    self.model.visual_model.add_new_mask(
                                        inference_state=inference_state,
                                        frame_idx=frame_idx,                                    # frame with masks
                                        obj_id=1,                                       # object id (only one)
                                        mask=object_mask,
                                        input_size=resize_list[i]
                                    )
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 2-1. propagation (preparation stage, for frames with masks)
                                    self.model.visual_model.propagate_in_video_preflight(inference_state)
                                    output_dict = inference_state["output_dict"]
                                    if len(output_dict["cond_frame_outputs"]) == 0:
                                        raise RuntimeError("No points are provided; please add points first")
                                    storage_key = "cond_frame_outputs"
                                    current_out = output_dict[storage_key][0]
                                    pred_masks_prop = current_out["pred_masks"]
                                    self.model.visual_model._add_output_per_object(
                                        inference_state, 0, current_out, storage_key
                                    )
                                    inference_state["frames_already_tracked"][0] = {"reverse": False}
                                    # # Resize the output mask to the original video resolution (we directly use
                                    # # the mask scores on GPU for output to avoid any CPU conversion in between)
                                    # _, video_res_masks = self.model.visual_model._get_orig_video_res_output(
                                    #     inference_state, pred_masks_prop
                                    # )
                                    frame_idx += 1
                                    # start_point = False # 下一帧在 else 中处理

                                    # if j > 0:   # 不然 video_masks 数量比 video_len 多, add_new_mask 过了两次循环
                                    #     video_masks[j-1] = pred_mask[:, 0]
                                else:
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-2. propagation (propagation stage, for frames to segment)
                                        # expanded_image = images[i][j].expand(1, -1, -1, -1)
                                        # expanded_backbone_out = {
                                        #     "backbone_fpn": fpn_list.copy(),
                                        #     "vision_pos_enc": pos_list.copy(),
                                        # }
                                        # for fi, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
                                        #     expanded_backbone_out["backbone_fpn"][fi] = feat.expand(
                                        #         1, -1, -1, -1
                                        #     )
                                        # for fi, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
                                        #     pos = pos.expand(1, -1, -1, -1)
                                        #     expanded_backbone_out["vision_pos_enc"][fi] = pos

                                        # features = self.model.visual_model._prepare_backbone_features(expanded_backbone_out)
                                        # features = (expanded_image,) + features
                                    # 如果是有 seg token 的帧, 就不用在传播时编码 mem 了, 因为后面会 add new seg 后再编码 mem
                                    run_mem_encoder = not sampled_frame
                                    if memory_refine and sampled_frame:     # 如果是采样帧, 还选择使用 seg token 和 memory 同时分割视频帧
                                        seg_token_sparse = sparse_embeddings[curr_indices]
                                        seg_token_dnese = dense_embeddings[curr_indices]
                                        run_mem_encoder = True
                                    else:
                                        seg_token_sparse = None
                                        seg_token_dnese = None

                                    storage_key = "non_cond_frame_outputs" if not sampled_frame else "cond_frame_outputs"
                                    current_out, pred_masks_prop = self.model.visual_model._run_single_frame_inference(
                                        inference_state=inference_state,
                                        output_dict=output_dict,
                                        frame_idx=frame_idx,
                                        batch_size=1,
                                        is_init_cond_frame=sampled_frame,
                                        point_inputs=None,
                                        mask_inputs=None,
                                        reverse=False,
                                        run_mem_encoder=run_mem_encoder,
                                        sparse_embeddings=seg_token_sparse,
                                        dense_embeddings=seg_token_dnese,
                                    )
                                    output_dict[storage_key][frame_idx] = current_out
                                    
                                    # if sampled_frame:
                                    #     self.model.visual_model._clear_non_cond_mem_around_input(inference_state, frame_idx)
                                    # Create slices of per-object outputs for subsequent interaction with each
                                    # individual object after tracking.
                                    self.model.visual_model._add_output_per_object(
                                        inference_state, frame_idx, current_out, storage_key
                                    )
                                    inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                                    
                                    object_score_logits = current_out["object_score_logits"]
                                    
                                    # Resize the output mask to the original video resolution (we directly use
                                    # the mask scores on GPU for output to avoid any CPU conversion in between)
                                    _, pred_mask = self.model.visual_model._get_orig_video_res_output(
                                        inference_state, pred_masks_prop, input_size=resize_list[i]
                                    )
                                    # 两种情况
                                    # 1. 没有 seg token, 什么都不做
                                    # 2. 有 seg token, add new point
                                    if sampled_frame and not memory_refine: # 采样帧, 但是采样帧单独做 refine, 不考虑 memory
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-3. add new point (seg token) (propagation stage, for frames to segment)
                                        _, _, pred_mask, object_score_logits = self.model.visual_model.add_new_seg(
                                            inference_state=inference_state,
                                            frame_idx=frame_idx,                            # frame with masks
                                            obj_id=1,                                       # object id (only one)
                                            sparse_embeddings = sparse_embeddings[curr_indices],
                                            dense_embeddings = dense_embeddings[curr_indices],
                                            # pre_embed=features,
                                            input_size = resize_list[i]
                                        )
                                    frame_idx += 1

                            if not seg_refine and sampled_frame:   # sampled frames. always true if samples_indices is None
                                fpn_list[0] = self.model.visual_model.sam_mask_decoder.conv_s0(fpn_list[0])
                                fpn_list[1] = self.model.visual_model.sam_mask_decoder.conv_s1(fpn_list[1])
                                pos_list = pos_embeddings[i][j]
                                if training_type == 2 and j == 0:
                                    pred_mask = masks_list[i][0].unsqueeze(0)
                                    object_score_logits = torch.zeros((1,1), device = pred_mask.device, dtype = torch.bfloat16)
                                else:  
                                    low_res_masks, iou_predictions, _, object_score_logits = self.model.visual_model.sam_mask_decoder(
                                        image_embeddings = curr_image_embeddings.unsqueeze(0),
                                        image_pe = self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings = sparse_embeddings[curr_indices],
                                        dense_prompt_embeddings = dense_embeddings[curr_indices],
                                        multimask_output = multimask_output,
                                        repeat_image = True,
                                        high_res_features = fpn_list[:-1],
                                    )

                                    # print(43333333333)
                                    is_obj_appearing = object_score_logits > 0

                                    # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                                    # consistent with the actual mask prediction
                                    low_res_masks = torch.where(
                                       is_obj_appearing[:, None, None],
                                       low_res_masks,
                                       NO_OBJ_SCORE,
                                    )

                                    pred_mask = self.model.visual_model.postprocess_masks(
                                        low_res_masks,
                                        input_size=resize_list[i],
                                        original_size=label_list[i].shape,
                                    )   # upsample mask
                                
                                # from PIL import Image
                                # image = Image.open('/hhd2/gaomingqi/ytvos_test/rvos_sam2_video_v2_tracking_sample0_4000/Annotations/0c04834d61/1/00000.png').convert('L')  # 将图像转换为灰度图

                                # image = np.array(image, dtype=np.float32)
                                # # 将图像转换为Tensor，并调整尺寸为 [1, 1, 720, 1280]
                                # tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                                # # 将Tensor移动到cuda设备
                                # pred_mask = tensor.to(device='cuda')

                                # if next frame is not sampled, 1) initialise state, 2) encode mask as memory
                                if sampled_indices is not None:
                                    if (sampled_indice+1 == len(sampled_indices) and sampled_indices[sampled_indice]<(video_len-1)) or \
                                        sampled_indices[min(sampled_indice+1, len(sampled_indices)-1)] - sampled_indices[sampled_indice] > 1:
# ---------------------------------------------------------------------------------------------------------------------------------
# 0. initialize inference_state
                                        if sampled_indice+1 == len(sampled_indices) and sampled_indices[sampled_indice]<(video_len-1):
                                            inference_state = self.model.visual_model.init_state(images[i][sampled_indices[sampled_indice]:video_len], async_loading_frames=False)
                                        else:
                                            inference_state = self.model.visual_model.init_state(images[i][sampled_indices[sampled_indice]:sampled_indices[sampled_indice+1]], async_loading_frames=False)
                                        inference_state['video_height'] = label_list[i].shape[0]
                                        inference_state['video_width'] = label_list[i].shape[1]
                                        image = images[i][j].float().unsqueeze(0).to(torch.bfloat16)
                                        backbone_out = {
                                            "vision_features": image_embeddings[i, j].unsqueeze(0).to(torch.bfloat16),
                                            "backbone_fpn": fpn_list.copy(),
                                            "vision_pos_enc": pos_list.copy(),
                                        }
                                        # Cache the most recent frame's feature (for repeated interactions with
                                        # a frame; we can use an LRU cache for more frames in the future).
                                        inference_state["cached_features"] = {0: (image, backbone_out)}
                                        self.model.visual_model.reset_state(inference_state)
# ---------------------------------------------------------------------------------------------------------------------------------
# 1. add_new_mask
                                        if pred_mask.shape[0] == 0:
                                            object_mask = np.zeros((pred_mask.shape[2], pred_mask.shape[3])) > 0
                                        else:
                                            object_mask = (pred_mask > 0)[0,0].cpu().numpy()    # just propagation
                                        self.model.visual_model.add_new_mask(
                                            inference_state=inference_state,
                                            frame_idx=0,                                    # frame with masks
                                            obj_id=1,                                       # object id (only one)
                                            mask=object_mask,
                                            input_size=resize_list[i]
                                        )
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-1. propagation (preparation stage, for frames with masks)
                                        self.model.visual_model.propagate_in_video_preflight(inference_state)
                                        output_dict = inference_state["output_dict"]
                                        if len(output_dict["cond_frame_outputs"]) == 0:
                                            raise RuntimeError("No points are provided; please add points first")
                                        storage_key = "cond_frame_outputs"
                                        current_out = output_dict[storage_key][0]
                                        pred_masks_prop = current_out["pred_masks"]
                                        self.model.visual_model._add_output_per_object(
                                            inference_state, 0, current_out, storage_key
                                        )
                                        inference_state["frames_already_tracked"][0] = {"reverse": False}
                                        # Resize the output mask to the original video resolution (we directly use
                                        # the mask scores on GPU for output to avoid any CPU conversion in between)
                                        _, video_res_masks = self.model.visual_model._get_orig_video_res_output(
                                            inference_state, pred_masks_prop
                                        )
                                        frame_idx = 1
                            elif not seg_refine:   # other frames
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-2. propagation (propagation stage, for frames to segment)
                                expanded_image = images[i][j].expand(1, -1, -1, -1)
                                expanded_backbone_out = {
                                    "backbone_fpn": fpn_list.copy(),
                                    "vision_pos_enc": pos_list.copy(),
                                }
                                for fi, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
                                    expanded_backbone_out["backbone_fpn"][fi] = feat.expand(
                                        1, -1, -1, -1
                                    )
                                for fi, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
                                    pos = pos.expand(1, -1, -1, -1)
                                    expanded_backbone_out["vision_pos_enc"][fi] = pos

                                features = self.model.visual_model._prepare_backbone_features(expanded_backbone_out)
                                features = (expanded_image,) + features

                                storage_key = "non_cond_frame_outputs"
                                current_out, pred_masks_prop = self.model.visual_model._run_single_frame_inference(
                                    inference_state=inference_state,
                                    output_dict=output_dict,
                                    frame_idx=frame_idx,
                                    batch_size=1,
                                    is_init_cond_frame=False,
                                    point_inputs=None,
                                    mask_inputs=None,
                                    reverse=False,
                                    run_mem_encoder=True,
                                    pre_embed=features,
                                )
                                output_dict[storage_key][frame_idx] = current_out
                                # Create slices of per-object outputs for subsequent interaction with each
                                # individual object after tracking.
                                self.model.visual_model._add_output_per_object(
                                    inference_state, frame_idx, current_out, storage_key
                                )
                                inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                                
                                object_score_logits = current_out["object_score_logits"]

                                # Resize the output mask to the original video resolution (we directly use
                                # the mask scores on GPU for output to avoid any CPU conversion in between)
                                _, pred_mask = self.model.visual_model._get_orig_video_res_output(
                                    inference_state, pred_masks_prop, input_size=resize_list[i]
                                )
                                frame_idx += 1
                            
                            # else:
                            #     # 

                            #     low_res_masks, iou_predictions, _, _ = self.model.visual_model.sam_mask_decoder(
                            #         image_embeddings = curr_image_embeddings.unsqueeze(0),
                            #         image_pe = self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                            #         sparse_prompt_embeddings = sparse_embeddings[curr_indices],
                            #         dense_prompt_embeddings = dense_embeddings[curr_indices],
                            #         multimask_output = multimask_output,
                            #         repeat_image = True,
                            #         high_res_features = fpn_list,
                            #     )
                            boxes = None
                    else:
                        # low_res_masks: torch.Size([3 (该视频<SEG>总数), 1, 256, 256])
                        # iou_predictions: torch.Size([3 (该视频<SEG>总数), 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)
                        # boxes: torch.Size([3 (该视频<SEG>总数), 1, 4])
                        curr_indices = torch.arange(seg_token_num) if training_type != 2 else None
                        if not self.sam2:
                            low_res_masks, boxes, iou_predictions = self.model.visual_model.mask_decoder(
                                image_embeddings=curr_image_embeddings.unsqueeze(0),
                                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings[curr_indices],
                                dense_prompt_embeddings=dense_embeddings[curr_indices],
                                multimask_output=multimask_output,
                            )
                        else:
                            fpn_list = fpn_embeddings[i][j]
                            # fpn_list[0] = self.model.visual_model.sam_mask_decoder.conv_s0(fpn_list[0])
                            # fpn_list[1] = self.model.visual_model.sam_mask_decoder.conv_s1(fpn_list[1])
                            pos_list = pos_embeddings[i][j]
                            
                            if seg_refine: # seg_token_refine_propagation:
                                """
                                0. 给定 video_len 个视频帧, 对每个视频帧, 得到一个 seg_token
                                1. 使用第一帧 seg_token, 得到第一帧 mask
                                2. 使用第一帧 mask, 逐个帧传播, 从 1 - (video_len-1), 第一帧索引为 0, 下文为第 t 帧
                                3. 如果第 t 帧需要 refine, 先用 propagation 得到一个 mask, 然后添加 segtoken 更新 mask
                                4. 更新后, 重置 tracking-state 和 memory, 当前帧 (t) 作为第 0 帧向后传播 (add point 只能影响当前帧预测, 不能影响 memofy, 因此, 在每次更新后, 就重置 tracker 和 memory, 将当前帧作为第一帧向后传播)
                                """
                                if j == 0:
                                    frame_idx = 0
                                    # object_score_logits: torch.Size([1, 1]) e.g. tensor([[4.2188]], device='cuda:0', dtype=torch.bfloat16)
                                    if training_type == 2:
                                        pred_mask = masks_list[i][0].unsqueeze(0)
                                        object_score_logits = torch.zeros((1,1), device = pred_mask.device, dtype = torch.bfloat16)
                                    else:
                                        low_res_masks, iou_predictions, _, object_score_logits = self.model.visual_model.sam_mask_decoder(
                                            image_embeddings = curr_image_embeddings.unsqueeze(0),
                                            image_pe = self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings = sparse_embeddings[curr_indices],
                                            dense_prompt_embeddings = dense_embeddings[curr_indices],
                                            multimask_output = multimask_output,
                                            repeat_image = True,
                                            high_res_features = fpn_list[:-1],
                                        )
                                        
                                        is_obj_appearing = object_score_logits > 0

                                        # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                                        # consistent with the actual mask prediction
                                        low_res_masks = torch.where(
                                            is_obj_appearing[:, None, None],
                                            low_res_masks,
                                            NO_OBJ_SCORE,
                                        )
                                        
                                        pred_mask = self.model.visual_model.postprocess_masks(
                                            low_res_masks,
                                            input_size=resize_list[i],
                                            original_size=label_list[i].shape,
                                        )   # upsample mask

                                # if j == 0 or start_point:
                                # if j == 0:
                                # 第一帧(编号0), 或者是被 seg token 改变过的帧
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 0. initialize inference_state
                                    # if sampled_indice+1 == len(samples_indices) and samples_indices[sampled_indice]<(video_len):
                                    #     inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:video_len], async_loading_frames=False)
                                    # else: #
                                    #     inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:(samples_indices[sampled_indice+1]+1)], async_loading_frames=False)
                                    # inference_state['video_height'] = original_size_list[i][0]
                                    # inference_state['video_width'] = original_size_list[i][1]
                                    # image = images[i][j].float().unsqueeze(0).to(torch.bfloat16)
                                    # backbone_out = {
                                    #     "vision_features": image_embeddings[i, j].unsqueeze(0).to(torch.bfloat16),
                                    #     "backbone_fpn": fpn_list.copy(),
                                    #     "vision_pos_enc": pos_list.copy(),
                                    # }
                                    # # Cache the most recent frame's feature (for repeated interactions with
                                    # # a frame; we can use an LRU cache for more frames in the future).
                                    # inference_state["cached_features"] = {0: (image, backbone_out)}
                                    # self.model.visual_model.reset_state(inference_state)
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 1. add_new_mask
                                    if pred_mask.shape[0] == 0:
                                        object_mask = np.zeros((pred_mask.shape[2], pred_mask.shape[3])) > 0
                                    else:
                                        object_mask = (pred_mask > 0)[0,0].cpu().numpy()    # just propagation
                                    self.model.visual_model.add_new_mask(
                                        inference_state=inference_state,
                                        frame_idx=frame_idx,                                    # frame with masks
                                        obj_id=1,                                       # object id (only one)
                                        mask=object_mask,
                                        input_size=resize_list[i]
                                    )
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 2-1. propagation (preparation stage, for frames with masks)
                                    self.model.visual_model.propagate_in_video_preflight(inference_state)
                                    output_dict = inference_state["output_dict"]
                                    if len(output_dict["cond_frame_outputs"]) == 0:
                                        raise RuntimeError("No points are provided; please add points first")
                                    storage_key = "cond_frame_outputs"
                                    current_out = output_dict[storage_key][0]
                                    pred_masks_prop = current_out["pred_masks"]
                                    self.model.visual_model._add_output_per_object(
                                        inference_state, 0, current_out, storage_key
                                    )
                                    inference_state["frames_already_tracked"][0] = {"reverse": False}
                                    # # Resize the output mask to the original video resolution (we directly use
                                    # # the mask scores on GPU for output to avoid any CPU conversion in between)
                                    # _, video_res_masks = self.model.visual_model._get_orig_video_res_output(
                                    #     inference_state, pred_masks_prop
                                    # )
                                    frame_idx += 1
                                    # start_point = False # 下一帧在 else 中处理

                                    # if j > 0:   # 不然 video_masks 数量比 video_len 多, add_new_mask 过了两次循环
                                    #     video_masks[j-1] = pred_mask[:, 0]
                                else:
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-2. propagation (propagation stage, for frames to segment)
                                        # expanded_image = images[i][j].expand(1, -1, -1, -1)
                                        # expanded_backbone_out = {
                                        #     "backbone_fpn": fpn_list.copy(),
                                        #     "vision_pos_enc": pos_list.copy(),
                                        # }
                                        # for fi, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
                                        #     expanded_backbone_out["backbone_fpn"][fi] = feat.expand(
                                        #         1, -1, -1, -1
                                        #     )
                                        # for fi, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
                                        #     pos = pos.expand(1, -1, -1, -1)
                                        #     expanded_backbone_out["vision_pos_enc"][fi] = pos

                                        # features = self.model.visual_model._prepare_backbone_features(expanded_backbone_out)
                                        # features = (expanded_image,) + features
                                    # 如果是有 seg token 的帧, 就不用在传播时编码 mem 了, 因为后面会 add new seg 后再编码 mem
                                    run_mem_encoder = not sampled_frame
                                    if memory_refine and sampled_frame:     # 如果是采样帧, 还选择使用 seg token 和 memory 同时分割视频帧
                                        seg_token_sparse = sparse_embeddings[curr_indices]
                                        seg_token_dnese = dense_embeddings[curr_indices]
                                        run_mem_encoder = True
                                    else:
                                        seg_token_sparse = None
                                        seg_token_dnese = None

                                    storage_key = "non_cond_frame_outputs" if not sampled_frame else "cond_frame_outputs"
                                    current_out, pred_masks_prop = self.model.visual_model._run_single_frame_inference(
                                        inference_state=inference_state,
                                        output_dict=output_dict,
                                        frame_idx=frame_idx,
                                        batch_size=1,
                                        is_init_cond_frame=sampled_frame,
                                        point_inputs=None,
                                        mask_inputs=None,
                                        reverse=False,
                                        run_mem_encoder=run_mem_encoder,
                                        sparse_embeddings=seg_token_sparse,
                                        dense_embeddings=seg_token_dnese,
                                    )
                                    output_dict[storage_key][frame_idx] = current_out
                                    
                                    # if sampled_frame:
                                    #     self.model.visual_model._clear_non_cond_mem_around_input(inference_state, frame_idx)
                                    # Create slices of per-object outputs for subsequent interaction with each
                                    # individual object after tracking.
                                    self.model.visual_model._add_output_per_object(
                                        inference_state, frame_idx, current_out, storage_key
                                    )
                                    inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                                    
                                    object_score_logits = current_out["object_score_logits"]
                                    
                                    # Resize the output mask to the original video resolution (we directly use
                                    # the mask scores on GPU for output to avoid any CPU conversion in between)
                                    _, pred_mask = self.model.visual_model._get_orig_video_res_output(
                                        inference_state, pred_masks_prop, input_size=resize_list[i]
                                    )
                                    # 两种情况
                                    # 1. 没有 seg token, 什么都不做
                                    # 2. 有 seg token, add new point
                                    if sampled_frame and not memory_refine: # 采样帧, 但是采样帧单独做 refine, 不考虑 memory
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-3. add new point (seg token) (propagation stage, for frames to segment)
                                        _, _, pred_mask, object_score_logits = self.model.visual_model.add_new_seg(
                                            inference_state=inference_state,
                                            frame_idx=frame_idx,                            # frame with masks
                                            obj_id=1,                                       # object id (only one)
                                            sparse_embeddings = sparse_embeddings[curr_indices],
                                            dense_embeddings = dense_embeddings[curr_indices],
                                            # pre_embed=features,
                                            input_size = resize_list[i]
                                        )
                                    frame_idx += 1
                            else:
                                fpn_list[0] = self.model.visual_model.sam_mask_decoder.conv_s0(fpn_list[0])
                                fpn_list[1] = self.model.visual_model.sam_mask_decoder.conv_s1(fpn_list[1])
                                pos_list = pos_embeddings[i][j]
                                
                                low_res_masks, iou_predictions, _, object_score_logits = self.model.visual_model.sam_mask_decoder(
                                    image_embeddings=curr_image_embeddings.unsqueeze(0),
                                    image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse_embeddings[curr_indices],
                                    dense_prompt_embeddings=dense_embeddings[curr_indices],
                                    multimask_output=multimask_output,
                                    repeat_image=True,
                                    high_res_features=fpn_list[:-1],
                                )
                                
                                # print(43333333333)
                                # is_obj_appearing = object_score_logits > 0

                                # # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                                # # consistent with the actual mask prediction
                                # low_res_masks = torch.where(
                                #     is_obj_appearing[:, None, None],
                                #     low_res_masks,
                                #     NO_OBJ_SCORE,
                                # )

                    # pred_mask: torch.Size([3 (该视频<SEG>总数), 1, 1080, 1920])
                    # pred_mask[:, 0]: torch.Size([3 (该视频<SEG>总数), 1080, 1920])
                    if masks_list[i] is not None:
                        # RVOS
                        # if (not use_fusion_module) or (use_fusion_module and not self.sam2):
                        if not self.sam2:
                            pred_mask = self.model.visual_model.postprocess_masks(
                                low_res_masks,
                                input_size=resize_list[i],
                                original_size=label_list[i].shape,
                            )
                    else:
                        # RVOT
                        pred_mask = None

                    if torch.isnan(pred_mask).any():
                        print("NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    
                    # pred_box: torch.Size([1, 1, 4])
                    # pred_box = self.model.visual_model.postprocess_boxes(boxes, input_size=resize_list[i])
                    pred_box = None
                    video_masks.append(pred_mask[:, 0] if masks_list[i] is not None else None)
                    video_object_score_logits.append(object_score_logits)
                    # video_boxes.append(pred_box[:, 0])
                    # video_boxes.append(None)
                # 这样的话pred_masks里是 torch.Size([video_len, 1, 1080, 1920])的tensor
                # pred_boxes[0]: torch.Size([5, 1, 4])
                # pred_boxes.append(torch.stack(video_boxes))
                pred_masks.append(torch.stack(video_masks) if masks_list[i] is not None else None)
                pred_object_score_logits.append(torch.stack(video_object_score_logits) if masks_list[i] is not None else None)
                
            elif image_embeddings.ndim == 5 and getattr(self.config, "use_multuple_seg_token", False):
                # 多SEG
                video_len = image_embeddings.shape[1]
                video_masks = []
                video_boxes = []
                assert video_len == sparse_embeddings.shape[0]
                for j in range(video_len):
                    curr_image_embeddings = image_embeddings[i][j]
                    # low_res_masks: torch.Size([3 (该视频<SEG>总数), 1, 256, 256])
                    # iou_predictions: torch.Size([3 (该视频<SEG>总数), 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)
                    # boxes: torch.Size([3 (该视频<SEG>总数), 1, 4])
                    low_res_masks, boxes, iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=curr_image_embeddings.unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings[j].unsqueeze(0),
                        dense_prompt_embeddings=dense_embeddings[j].unsqueeze(0),
                        multimask_output=multimask_output,
                    )
                    # pred_mask: torch.Size([3 (该视频<SEG>总数), 1, 1080, 1920])
                    # pred_mask[:, 0]: torch.Size([3 (该视频<SEG>总数), 1080, 1920])
                    if masks_list[i] is not None:
                        # RVOS
                        pred_mask = self.model.visual_model.postprocess_masks(
                            low_res_masks,
                            input_size=resize_list[i],
                            original_size=label_list[i].shape,
                        )
                    else:
                        # RVOT
                        pred_mask = None
                    
                    # pred_box: torch.Size([1, 1, 4])
                    pred_box = self.model.visual_model.postprocess_boxes(boxes, input_size=resize_list[i])
                    video_masks.append(pred_mask[:, 0] if masks_list[i] is not None else None)
                    video_boxes.append(pred_box[:, 0])
                # 这样的话pred_masks里是 torch.Size([video_len, 1, 1080, 1920])的tensor
                # pred_boxes[0]: torch.Size([5, 1, 4])
                pred_boxes.append(torch.stack(video_boxes))
                pred_masks.append(torch.stack(video_masks) if masks_list[i] is not None else None)
            
            else:
                # 图像
                low_res_masks, boxes, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_box = self.model.visual_model.postprocess_boxes(boxes, input_size=resize_list[i])
                pred_boxes.append(pred_box[:, 0])
                pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        if training_type != 2:
            ce_loss = output.loss
        else:
            ce_loss = 0
            if use_fusion_module:
                ce_loss += torch.stack([p.sum() for p in self.model.fusion_layers.parameters()]).sum() * 0.0
            ce_loss += torch.stack([p.sum() for p in self.model.text_hidden_fcs.parameters()]).sum() * 0.0
            ce_loss += torch.stack([p.sum() for p in self.model.layers.parameters()]).sum() * 0.0
            ce_loss += torch.stack([p.sum() for p in self.model.embed_tokens.parameters()]).sum() * 0.0
            ce_loss += torch.stack([p.sum() for p in self.lm_head.parameters()]).sum() * 0.0
            ce_loss += torch.stack([p.sum() for p in self.model.spi_module.parameters()]).sum() * 0.0
            
            
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        obj_score_ce_loss = 0
        # box_l1_loss = 0
        # box_giou_loss = 0
        num_masks = 0
        # num_boxes = 0

        for batch_idx in range(len(pred_masks)):
            # gt_mask: torch.Size([3 (该视频<SEG>总数), 500, 674]) -> torch.Size([video_len, 3 (该视频<SEG>总数), 500, 674])
            # gt_box: torch.Size([video_len, 1, 4])
            gt_mask = gt_masks[batch_idx]
            # gt_box = kwargs['box_list'][batch_idx].unsqueeze(1)
            # pred_mask: torch.Size([1, 1080, 1920]) -> torch.Size([video_len, 3 (该视频<SEG>总数), 1080, 1920])
            pred_mask = pred_masks[batch_idx]
            # pred_box = pred_boxes[batch_idx]
            
            pred_score = pred_object_score_logits[batch_idx]
            
            gt_score = gt_mask.any(dim=-1).any(dim=-1).int().unsqueeze(-1).to(device=pred_score.device, dtype = pred_score.dtype)
            
            if training_type == 2:
                gt_mask = gt_mask[1:]
                pred_mask = pred_mask[1:]
                pred_score = pred_score[1:]
                gt_score = gt_score[1:]


            if gt_mask is not None:
                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
            # else:
            #     assert (
            #         gt_box.shape[1] == pred_box.shape[1]
            #     ), "gt_box.shape: {}, pred_box.shape: {}".format(
            #         gt_box.shape, pred_box.shape
            #     )
            
            # 视频需要逐帧算loss
            # 暂时一个视频的loss为所有帧loss的和，没有除以视频帧数
            if image_embeddings.ndim == 5:
                if gt_mask is not None:
                    # 有mask才算mask loss
                    # pred_mask: torch.Size([video_len, seg_num, 1200, 1920])
                    mask_bce_loss += (
                        sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[1], gt_score=gt_score)
                        # sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[1], gt_score=None)
                        * gt_mask.shape[1]
                    )
                    mask_dice_loss += (
                        dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[1], gt_score=gt_score)
                        # dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[1], gt_score=None)
                        * gt_mask.shape[1]
                    )
                    obj_score_ce_loss += (
                        cls_ce_loss(pred_score, gt_score, num_masks=gt_mask.shape[1])
                        * gt_mask.shape[1]
                    )

                    num_masks += gt_mask.shape[1]
                    
                # # box loss
                # box_l1_loss += (
                #     l1_loss(pred_box, gt_box, num_boxes=gt_box.shape[1])
                #     * gt_box.shape[1]
                # )
                # box_giou_loss += (
                #     giou_loss(pred_box, gt_box, num_boxes=gt_box.shape[1])
                #     * gt_box.shape[1]
                # )
                # num_boxes += gt_box.shape[1]
            else:
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]


        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        obj_score_ce_loss = self.obj_loss_weight * obj_score_ce_loss / (num_masks + 1e-8)
        # box_l1_loss = self.l1_loss_weight * box_l1_loss / (num_boxes + 1e-8)
        # box_giou_loss = self.giou_loss_weight * box_giou_loss / (num_boxes + 1e-8)
        
        mask_loss = mask_bce_loss + mask_dice_loss
        # box_loss = box_l1_loss + box_giou_loss
        if num_masks == 0 and use_fusion_module:
            mask_loss += torch.stack([p.sum() for p in self.model.fusion_layers.parameters()]).sum() * 0.0
        
        # loss = ce_loss + mask_loss
        loss = ce_loss + mask_loss + obj_score_ce_loss
        # loss = mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "obj_score_ce_loss": obj_score_ce_loss,
            "mask_loss": mask_loss,
            # "box_l1_loss": box_l1_loss,
            # "box_giou_loss": box_giou_loss,
            # "box_loss": box_loss
        }

    def get_clip_features(self, images, trajectories = None):
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            
            if getattr(self.config, "train_roi_align", False) and trajectories is not None:
                trajectories = trajectories.reshape(trajectories.shape[0]*trajectories.shape[1], 4)
                image_features, trajectory_features, mlvl_features = self.encode_images(concat_images, trajectories, bs=len(images), video_len=images[0].shape[0])
            else:
                if getattr(self.config, "train_roi_align", False) and trajectories is None:
                    fake_traj = torch.zeros((len(images) *images[0].shape[0], 4), device='cuda')
                    # trajectory_features:list, len=bs, torch.Size([video_len, 4096]); low_res_trajectory_features: c = 1024
                    image_features, trajectory_features, mlvl_features = self.encode_images(concat_images, fake_traj, bs=len(images), video_len=images[0].shape[0])
                else:
                    # trajectory_features:list, len=bs, torch.Size([video_len, 4096]); low_res_trajectory_features: c = 1024
                    image_features, trajectory_features, mlvl_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # image_features = [x.flatten(0, 1) for x in image_features]
            image_features = [x for x in image_features]
             # image_features: list[len = bs], torch.Size([video_len, 256, 4096])
        else:
            # torch.Size([1, 256, 4096])
            image_features = self.encode_images(images)
        return image_features
    
    def save_bialign_heatmap(self, pred_embeddings, clip_images, clip_image_features, sam_images, sam_features):
        # pred_embeddings: torch.Size([1, 256]), FCI签的<SEG> token
        # clip_images: torch.Size([video_len, 3, 224, 224]), clip图像(原始图像transform后得到), 和clip特征的heatmap可以直接在这个上面可视化
        # clip_image_features: torch.Size([video_len, 256, 4096]), clip特征
        # sam_images: torch.Size([video_len, 3, 1024, 1024]), sam图像(原始图像transform后得到)
        # sam_features: torch.Size([video_len, 256, 64, 64]), sam特征
        import torch.nn.functional as F
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        import os
        video_len = clip_images.shape[0]
        
        # 确保 pred_embeddings 形状为 (1, 256)
        pred_embeddings = pred_embeddings.squeeze(0) if pred_embeddings.dim() == 2 else pred_embeddings  # [256]
        
        for i in range(video_len):
            # 计算 CLIP attention map
            clip_feat = clip_image_features[i]  # [256, 4096]
            clip_attn = F.cosine_similarity(pred_embeddings.unsqueeze(0), clip_feat.T, dim=1)  # [1, 4096]
            clip_attn = clip_attn.squeeze(0).view(64, 64).cpu().to(torch.float32).numpy()  # 变回 64x64
            clip_attn = cv2.resize(clip_attn, (224, 224))  # 调整到原图大小
            clip_attn = (clip_attn - clip_attn.min()) / (clip_attn.max() - clip_attn.min())  # 归一化
            clip_attn = np.power(clip_attn, 2)  # 增强重点区域，抑制非重点区域
            
            # 计算 SAM attention map
            sam_feat = sam_features[i]  # [256, 64, 64]
            sam_attn = F.cosine_similarity(pred_embeddings[:, None, None], sam_feat, dim=0)  # [64, 64]
            sam_attn = sam_attn.cpu().to(torch.float32).numpy()
            sam_attn = cv2.resize(sam_attn, (1024, 1024))
            sam_attn = (sam_attn - sam_attn.min()) / (sam_attn.max() - sam_attn.min())
            sam_attn = np.power(sam_attn, 2)  # 增强重点区域，抑制非重点区域
            
            # 叠加到原图
            clip_img = clip_images[i].permute(1, 2, 0).cpu().to(torch.float32).numpy()
            clip_img = (clip_img - clip_img.min()) / (clip_img.max() - clip_img.min())
            clip_overlay = (clip_img * 255).astype(np.uint8)
            clip_heatmap = cv2.applyColorMap((clip_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            clip_result = cv2.addWeighted(clip_overlay, 0.6, clip_heatmap, 0.4, 0)
            
            sam_img = sam_images[i].permute(1, 2, 0).to(torch.float32).cpu().numpy()
            sam_img = (sam_img - sam_img.min()) / (sam_img.max() - sam_img.min())
            sam_overlay = (sam_img * 255).astype(np.uint8)
            sam_heatmap = cv2.applyColorMap((sam_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            sam_result = cv2.addWeighted(sam_overlay, 0.6, sam_heatmap, 0.4, 0)
            
            # 保存图片
            cv2.imwrite(os.path.join('heatmap', f"clip_attn_{i}.jpg"), clip_result)
            cv2.imwrite(os.path.join('heatmap', f"sam_attn_{i}.jpg"), sam_result)

    def save_sam_heatmap_with_mask(self, pred_embeddings, sam_images, sam_features, input_size, original_size, threshold=0.6):
        """
        生成 SAM 特征的注意力热力图，并根据高分区域生成 mask 进行保存。
        
        :param pred_embeddings: torch.Size([1, video_len, 256])  预测嵌入向量
        :param sam_images: torch.Size([video_len, 3, 1024, 1024])  SAM 处理的输入图像
        :param sam_features: torch.Size([video_len, 256, 64, 64])  SAM 提取的特征
        :param input_size: tuple, 模型输入的图像尺寸 (H, W)
        :param original_size: tuple, 原始图像的尺寸 (H, W)
        :param threshold: 用于生成 mask 的阈值，默认为 0.6
        """
        import torch.nn.functional as F
        import numpy as np
        import cv2
        import os
        import torch
        
        def postprocess_masks(masks: torch.Tensor, input_size: tuple, original_size: tuple) -> torch.Tensor:
            """
            Remove padding and upscale masks to the original image size.
            """
            dtype = masks.dtype
            masks = masks.float()
            masks = masks[..., :input_size[0], :input_size[1]]  # 去掉 padding
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)  # 放缩到原图尺寸
            return masks.to(dtype)
        
        video_len = sam_images.shape[0]
        if pred_embeddings.ndim == 3:
            save_name = 'post'
        else:
            save_name = 'pre'
            
        pred_embeddings = pred_embeddings.squeeze(0) if pred_embeddings.ndim == 3 else pred_embeddings # [video_len, 256]
        
        os.makedirs('heatmap_sam', exist_ok=True)
        os.makedirs('mask_sam', exist_ok=True)
        
        for i in range(video_len):
            # 计算 SAM attention map
            sam_feat = sam_features[i]  # [256, 64, 64]
            if pred_embeddings.shape[0] == 1:
                sam_attn = F.cosine_similarity(pred_embeddings[0][:, None, None], sam_feat, dim=0)  # [64, 64]
            else:
                sam_attn = F.cosine_similarity(pred_embeddings[i][:, None, None], sam_feat, dim=0)  # [64, 64]
            sam_attn = sam_attn.cpu().to(torch.float32).numpy()
            
            # 归一化
            sam_attn = (sam_attn - sam_attn.min()) / (sam_attn.max() - sam_attn.min())
            sam_attn = np.power(sam_attn, 2)  # 增强重点区域
            
            # 转换为张量并进行尺寸还原
            sam_attn_tensor = torch.tensor(sam_attn).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]
            sam_attn_resized = F.interpolate(sam_attn_tensor, (1024, 1024), mode="bilinear", align_corners=False)
            sam_attn_resized = postprocess_masks(sam_attn_resized, input_size, original_size).squeeze(0).squeeze(0).numpy()
            
            # 生成 mask（最终为 original_size）
            mask = (sam_attn_resized >= threshold).astype(np.uint8) * 255  # 二值化 mask
            
            # 还原 sam_img 到 original_size
            sam_img = sam_images[i].permute(1, 2, 0).cpu().to(torch.float32).numpy()
            sam_img = (sam_img - sam_img.min()) / (sam_img.max() - sam_img.min())  # 归一化
            sam_img = torch.tensor(sam_img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 1024, 1024]
            sam_overlay = postprocess_masks(sam_img, input_size, original_size).squeeze(0).permute(1, 2, 0).numpy()
            sam_overlay = (sam_overlay * 255).astype(np.uint8)  # 恢复到 uint8 颜色格式
            
            # 生成热力图
            sam_heatmap = cv2.applyColorMap((sam_attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            sam_result = cv2.addWeighted(sam_overlay, 0.6, sam_heatmap, 0.4, 0)  # 现在 sam_overlay 和 sam_heatmap 尺寸一致
            
            # 保存热力图和 mask
            cv2.imwrite(os.path.join('heatmap_sam', f"sam_attn_{i}_{save_name}.jpg"), sam_result)
            cv2.imwrite(os.path.join('mask_sam', f"sam_mask_{i}_{save_name}.png"), mask)
    
    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
        trajectory = None,
        use_multuple_seg_token = False,
        llm_sample_mode = 'nosample',
        samples_indices = None,
        seg_refine=True,
        memory_refine=True,
        memory_efficient=False
    ):  
        # images_clip: torch.Size([1, 3, 224, 224]), 用于LLM
        # images: torch.Size([1, 3, 1024, 1024]), 用于SAM分割
        # input_ids: tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
        #                     21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
        #                       322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
        #                     29889,  3148,  1001, 29901, 32001,  -200, 32002, 22172,   319,  1799,
        #                      9047, 13566, 29901]], device='cuda:0') torch.Size([1, 43])
        # resize_list: [(576, 1024)]
        # original_size_list:[(1080, 1920)]
        # tokenizer: <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>
        
        # 如果llm_sample_mode = 'uniform', 则推理时启用均匀采样(默认llm_sample_mode = 'nosample', 即llm输入的是完整视频)
        # 均匀采样时, 假设原视频长度n帧, 采样k帧, 则images_clip[0]: torch.Size([k, 3, 224, 224]); images: torch.Size([1, n, 3, 1024, 1024])
        # samples_indices: list, 记录所有采样帧的索引, 默认采12帧, 如[0, 1, 3, 5, 7, 8, 10, 11, 13, 14, 16, 18] (仅当使用均匀采样时开启)
        
        with torch.no_grad():
            # 这一步调用库函数生成完整的回答(即内部多次调用model.forward())(训练的时候应该不会走这里，因为训练时每次是根据GT生成新token，不会根据模型前面的输出生成新token)
            # LISA和LLaMA-VID生成策略的策略不一样，LISA是greedy_search(直接选最大概率?), LLaMA-VID是sample(按概率采样，即概率小的token也有可能被选择)
            # use_cache必须是False
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                trajectory = trajectory
            )
            # 并不保证LLM一定输出<SEG>token, 所以需要文本监督训练
            # outputs.keys(): odict_keys(['sequences', 'hidden_states'])
            # outputs['sequences']: tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
            #                                21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
            #                                  322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
            #                                29889,  3148,  1001, 29901, 32001,  -200, 32002, 22172,   319,  1799,
            #                                 9047, 13566, 29901, 18585, 29892,   372,   338, 32003,   869,     2]],
            #                                device='cuda:0')   torch.Size([1, 50])
            # outputs['hidden_states']: tuple, 7 * (torch.Size([1, 298, 4096]), torch.Size([1, 299, 4096]), ..., torch.Size([1, 304, 4096]))
            
            # output_hidden_states: torch.Size([1, 304, 4096])
            # llama-vid返回完整的hidden_states,元组第一层为输出的token数量,第二层为layer层数
            output_hidden_states = outputs.hidden_states[-1][-1] if type(outputs.hidden_states[-1]) is tuple else outputs.hidden_states[-1]
            output_ids = outputs.sequences

            # self.seg_token_idx: 32003
            # seg_token_mask: tensor([[False, False, False, False, False, False, False, False, False, False,
            #                          False, False, False, False, False, False, False, False, False, False,
            #                          False, False, False, False, False, False, False, False, False, False,
            #                          False, False, False, False, False, False, False, False, False, False,
            #                          False, False, False, False, False, False,  True, False, False]],
            #                        device='cuda:0')  torch.Size([1, 49])
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)


            if type(images_clip) is list:
                video_len = images_clip[0].shape[0] if type(images_clip) is list else images_clip.shape[1]
                token_per_frame = 256
                
                
                if (input_ids == TRAJECTORY_TOKEN_INDEX).any():
                    seg_token_mask = torch.stack([torch.cat([torch.zeros(video_len * token_per_frame - 1).bool().cuda(),
                                                                seg_token_mask[i]]) if TRAJECTORY_TOKEN_INDEX in input_ids[i] 
                                                            else torch.cat([torch.zeros(video_len * token_per_frame - video_len).bool().cuda(),
                                                                seg_token_mask[i], torch.zeros(video_len - 1).bool().cuda()]) for i in range(len(seg_token_mask))])
                else:
                    seg_token_mask = torch.cat(
                        [
                            torch.zeros((seg_token_mask.shape[0], video_len * token_per_frame - video_len)).bool().cuda(),
                            seg_token_mask,
                        ],
                        dim=1,
                    )
                
            else:
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                        seg_token_mask,
                    ],
                    dim=1,
                )
                # seg_token_mask: torch.Size([1, 304]) 304 = 255 + 49

            
            hidden_states = []

            # self.model.text_hidden_fcs:
            # ModuleList(
            #    (0): Sequential(
            #      (0): Linear(in_features=4096, out_features=4096, bias=True)
            #      (1): ReLU(inplace=True)
            #      (2): Linear(in_features=4096, out_features=256, bias=True)
            #      (3): Dropout(p=0.0, inplace=False)
            #    )
            #  )
            
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
            # hidden_states[0].shape: torch.Size([1, 304, 256])
            
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            # last_hidden_state: torch.Size([1, 304, 256])
            # pred_embeddings: torch.Size([1, 256])
    
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )
            # seg_token_counts: tensor([1], device='cuda:0')
            # seg_token_offset: tensor([0, 1], device='cuda:0')
            
            bs, t, _, _, _ = images.shape
            if memory_efficient:
                # 此时在这里只抽取samples_indices的特征
                assert llm_sample_mode == 'uniform'
                image_embeddings, fpn_embeddings, pos_embeddings = self.get_visual_embs(images[:,samples_indices]) # 10 1 256 64 64    # sam2 image features
                # image_embeddings: torch.Size([bt, 1, 256, 64, 64])
                image_embeddings = image_embeddings.squeeze(1)                  # full video frames, video_len x 256 x 64 x 64
                image_embeddings = rearrange(image_embeddings, '(b t) c h w -> b t c h w', b=bs, t=len(samples_indices))
                # image_embeddings: b t c h w (t=n, 均匀采样时为完整视频长度)
            else:
                image_embeddings, fpn_embeddings, pos_embeddings = self.get_visual_embs(images) # 10 1 256 64 64    # sam2 image features
                # image_embeddings: torch.Size([bt, 1, 256, 64, 64])
                image_embeddings = image_embeddings.squeeze(1)                  # full video frames, video_len x 256 x 64 x 64
                image_embeddings = rearrange(image_embeddings, '(b t) c h w -> b t c h w', b=bs, t=t)
                # image_embeddings: b t c h w (t=n, 均匀采样时为完整视频长度)

            # self.save_sam_heatmap_with_mask(pred_embeddings, images[0][samples_indices], image_embeddings[0][samples_indices], resize_list[0], original_size_list[0], threshold=0.4)

            post_fusion_type = getattr(self.config, "post_fusion_type", "none")
            use_fusion_module = (post_fusion_type != 'none')
            
            pred2img_indices = torch.searchsorted(seg_token_offset, torch.arange(seg_token_offset[-1]).cuda(), right=True) - 1

            if post_fusion_type == 'gdino_uni' and pred_embeddings.shape[0] != 0:
                seg_num = pred_embeddings.shape[0]
                video_len = images_clip[0].shape[0]
                pred_embeddings = pred_embeddings.repeat_interleave(video_len, dim=0).unsqueeze(1)
                # pred_embeddings: torch.Size([bs * video_len, 1, 256])
                # key_padding_mask = torch.zeros(_image_embeddings.shape[0], _image_embeddings.shape[1], dtype=torch.bool, device=pred_embeddings.device)
                # text_attention_mask = torch.zeros(pred_embeddings.shape[0], pred_embeddings.shape[1], dtype=torch.bool, device=pred_embeddings.device)
                for i, layer in enumerate(self.model.fusion_layers):
                    # output(src): torch.Size([bs, 20906, 256])
                    # memory_text: torch.Size([bs, 4, 256])
                    # key_padding_mask: torch.Size([bs, 20906])
                    # text_attention_mask: torch.Size([bs, 4])
                    if llm_sample_mode == 'uniform':
                        if memory_efficient:
                            v=image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(0,2,1)     # sampled-len x 4096 x 256
                        else:
                            v=image_embeddings[pred2img_indices][:,samples_indices].flatten(-2).flatten(0,1).permute(0,2,1)     # sampled-len x 4096 x 256
                    elif llm_sample_mode == 'nosample':
                        v=image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(0,2,1)
                    
                    pred_embeddings = self.model.fusion_layers[i](
                        v=v,
                        l=pred_embeddings,
                        # attention_mask_v=key_padding_mask,
                        # attention_mask_l=text_attention_mask,
                    )
                pred_embeddings = pred_embeddings.reshape(seg_num, video_len, -1)  
            
            elif post_fusion_type == 'gdino_bi' and pred_embeddings.shape[0] != 0:
                seg_num = pred_embeddings.shape[0]
                video_len = images_clip[0].shape[0]
                pred_embeddings = pred_embeddings.repeat_interleave(video_len, dim=0).unsqueeze(1)
                
                if llm_sample_mode == 'uniform':
                    if memory_efficient:
                        _image_embeddings = image_embeddings[pred2img_indices].detach().flatten(-2).flatten(0,1).permute(0,2,1) 
                    else:
                        _image_embeddings = image_embeddings[pred2img_indices][:,samples_indices].detach().flatten(-2).flatten(0,1).permute(0,2,1) 
                elif llm_sample_mode == 'nosample':
                    _image_embeddings = image_embeddings[pred2img_indices].detach().flatten(-2).flatten(0,1).permute(0,2,1) 
                # _image_embeddings: torch.Size([bs * video_len, 4096, 256])
                
                for i, layer in enumerate(self.model.fusion_layers):    
                    _image_embeddings, pred_embeddings = self.model.fusion_layers[i](
                        v=_image_embeddings,
                        l=pred_embeddings,
                        # attention_mask_v=key_padding_mask,
                        # attention_mask_l=text_attention_mask,
                    )
                pred_embeddings = pred_embeddings.reshape(seg_num, video_len, -1)  
            
            elif post_fusion_type == 'gdino_bi_updated_img' and pred_embeddings.shape[0] != 0:
                seg_num = pred_embeddings.shape[0]
                video_len = images_clip[0].shape[0]
                pred_embeddings = pred_embeddings.repeat_interleave(video_len, dim=0).unsqueeze(1)
                
                if llm_sample_mode == 'uniform':
                    if memory_efficient:
                        image_embeddings = image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(0,2,1)
                    else:
                        image_embeddings = image_embeddings[pred2img_indices][:,samples_indices].flatten(-2).flatten(0,1).permute(0,2,1)
                elif llm_sample_mode == 'nosample':
                    image_embeddings = image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(0,2,1)
                
                
                for i, layer in enumerate(self.model.fusion_layers):
                    # output(src): torch.Size([bs, 20906, 256])
                    # memory_text: torch.Size([bs, 4, 256])
                    # key_padding_mask: torch.Size([bs, 20906])
                    # text_attention_mask: torch.Size([bs, 4])
                    image_embeddings, pred_embeddings = self.model.fusion_layers[i](
                        v=image_embeddings,
                        l=pred_embeddings,
                        # attention_mask_v=key_padding_mask,
                        # attention_mask_l=text_attention_mask,
                    )
                    
                image_embeddings = image_embeddings.permute(0, 2, 1).view(seg_num, video_len, 256, 64, 64) 
                pred_embeddings = pred_embeddings.reshape(seg_num, video_len, -1)  
            
            elif post_fusion_type == 'nn' and pred_embeddings.shape[0] != 0:
                # pred_embeddings: [1, bs * video_len, 256]
                seg_num = pred_embeddings.shape[0]
                video_len = images_clip[0].shape[0]
                pred_embeddings = pred_embeddings.repeat_interleave(video_len, dim=0).unsqueeze(0)
                
                if llm_sample_mode == 'uniform':
                    if memory_efficient:
                        image_embed_kv = image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(2,0,1)
                    else:
                        image_embed_kv = image_embeddings[pred2img_indices][:,samples_indices].flatten(-2).flatten(0,1).permute(2,0,1)
                elif llm_sample_mode == 'nosample':
                    image_embed_kv = image_embeddings[pred2img_indices].flatten(-2).flatten(0,1).permute(2,0,1)
                    
                for layer in self.model.fusion_layers:
                    attn_output = layer[0](  # MultiheadAttention
                        query=pred_embeddings,
                        key=image_embed_kv,
                        value=image_embed_kv, attn_mask=None
                    )[0]
                    
                    # pred_embeddings = layer[2](pred_embeddings + layer[1](attn_output))  # Dropout + LayerNorm 残差连接
                    pred_embeddings = layer[1](pred_embeddings + attn_output)  # 仅残差连接 + LayerNorm
                
                # pred_embeddings = pred_embeddings + self.model.fusion_layers(
                #     query=pred_embeddings,
                #     key=image_embed_kv,
                #     value=image_embed_kv, attn_mask=None,)[0]
                
                pred_embeddings = pred_embeddings.squeeze(0).reshape(seg_num, video_len, -1)
            
            if use_fusion_module and pred_embeddings.shape[0] == 0:
            # 当use_fusion_module，并且一个bs全是QA时, 调整pred_embeddings的shape统一
                pred_embeddings = pred_embeddings.unsqueeze(1).repeat_interleave(images_clip[0].shape[0], dim=1)
           
            # self.save_sam_heatmap_with_mask(pred_embeddings, images[0][samples_indices], image_embeddings[0][samples_indices], resize_list[0], original_size_list[0], threshold=0.4)

            # 这里offset用于为batch里每张图片找到对应的token区间
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                # start_i, end_i: 0,1
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            # pred_embeddings[0]: torch.Size([1, 256])
    
            multimask_output = False
            pred_masks = []
            pred_boxes = []
            pred_object_score_logits = []
            
            # 在启用均匀采样后, pred_embeddings[0].shape: torch.Size([1, k, 256])
            # 这个循环是循环图像(bs维度)
            for i in range(len(pred_embeddings)):
                # 推理时可以简单认为sparse_embeddings: torch.Size([video_len, 1, 256]), 启用均匀采样时则为torch.Size([k, 1, 256])
                if use_fusion_module and not self.sam2:    
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].flatten(0,1).unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                elif not use_fusion_module and not self.sam2: 
                    # sparse_embeddings: torch.Size([3 (该视频<SEG>总数), 1, 256])
                    # dense_embeddings: torch.Size([3 (该视频<SEG>总数), 256, 64, 64])
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                elif use_fusion_module and self.sam2:
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].flatten(0,1).unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                elif not use_fusion_module and self.sam2: 
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                
                # 推理时可以简单认为sparse_embeddings: torch.Size([video_len, 1, 256]), 启用均匀采样时则为torch.Size([k, 1, 256])
                # 下面是需要补充的均匀采样传播部分: 默认时为遍历完整视频每一帧j, 分别用sparse_embeddings[j]来解码mask
                # 使用均匀采样时, 大致思路为: 首先使用用sparse_embeddings得到采样帧的mask, 再根据samples_indices(采样帧帧号)将mask传播至完整视频
                if image_embeddings.ndim == 5 and (not use_multuple_seg_token):
                    video_len = images.shape[1]
                    video_masks = []
                    video_boxes = []
                    video_object_score_logits = []
                    start_point = True # 当前帧, 是否为传播起始帧
                    
                    if seg_refine:
                        """
                        优化模式数据准备, 使用已编码特征, 初始化 inference_state
                        """
                        # memory_efficient=True时,不cache, 推理时动态加载
                        inference_state = self.model.visual_model.init_state(
                            video_height=original_size_list[i][0],
                            video_width=original_size_list[i][1],
                            images=images[i],
                            cached_features=image_embeddings[i],
                            fpn_embeddings=fpn_embeddings[i],
                            pos_embeddings=pos_embeddings[i],
                            memory_efficient = memory_efficient
                        )
                    
                    # 下面是消融实验部分，要删掉
                    # def random_sample(lst, m):
                    #     import random
                    #     if m >= len(lst):
                    #         return lst
                    #     # 从第二个元素开始随机选择m-1个元素
                    #     sample = random.sample(lst[1:], m-1)
                    #     # 将第一个元素添加到结果中
                    #     sample.insert(0, lst[0])
                    #     return sorted(sample)
                    # abl_sampled_indice = random_sample(samples_indices, 8)
                    # print(samples_indices)
                    # print(abl_sampled_indice)
                    
                    j = 0
                    while j < video_len:
                        if sparse_embeddings.shape[0] == 0:
                            # 如果没有输出<SEG> token, 直接输出全0 mask (仅限推理)
                            video_masks.append(torch.zeros(1, original_size_list[i][0], original_size_list[i][1]).to(device = 'cuda'))
                            video_object_score_logits.append(torch.zeros(1,1).to(device = 'cuda'))
                            j += 1 
                            continue
                            
                    # for j in range(video_len):
                        # 区分 sampled & un-sampled frames
                        if samples_indices is not None:
                            if j in samples_indices:
                            # 下面这行也是消融, 及得调整回来！！！
                            # if j in abl_sampled_indice:
                                sampled_indice = samples_indices.index(j)   # indice in the sampled list (sparse/dense_embeddings only cover the sampled ones, image_embeddings cover the full frames)
                                sampled_frame = True
                            else:
                                sampled_frame = False
                        else:
                            sampled_indice = j
                            sampled_frame = True
                        
                        if not memory_efficient or j == 0:
                            curr_image_embeddings = image_embeddings[i][j]
                        seg_token_num = pred_embeddings[i].shape[0]
                        object_score_logits = None
                        # low_res_masks: torch.Size([3 (该视频<SEG>总数), 1, 256, 256])
                        # iou_predictions: torch.Size([3 (该视频<SEG>总数), 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)                     
                        if use_fusion_module:
                        # low_res_masks: torch.Size([3 (该视频<SEG>总数), 1, 256, 256])
                        # iou_predictions: torch.Size([3 (该视频<SEG>总数), 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)
                        # boxes: torch.Size([3 (该视频<SEG>总数), 1, 4])
                            curr_indices = sampled_indice + torch.arange(seg_token_num) * video_len
                            if not self.sam2:
                                assert not memory_efficient
                                low_res_masks, boxes, iou_predictions = self.model.visual_model.mask_decoder(
                                    image_embeddings=curr_image_embeddings.unsqueeze(0),
                                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse_embeddings[curr_indices],
                                    dense_prompt_embeddings=dense_embeddings[curr_indices],
                                    multimask_output=multimask_output,
                                )
                            else:
                                if not memory_efficient or j == 0:
                                    fpn_list = fpn_embeddings[i][j]
                                    pos_list = pos_embeddings[i][j]
                                    if memory_efficient and j == 0:
                                        fpn_list[0] = self.model.visual_model.sam_mask_decoder.conv_s0(fpn_list[0])
                                        fpn_list[1] = self.model.visual_model.sam_mask_decoder.conv_s1(fpn_list[1])
                                if seg_refine: # seg_token_refine_propagation:
                                    """
                                    0. 给定 video_len 个视频帧, 对每个视频帧, 得到一个 seg_token
                                    1. 使用第一帧 seg_token, 得到第一帧 mask
                                    2. 使用第一帧 mask, 逐个帧传播, 从 1 - (video_len-1), 第一帧索引为 0, 下文为第 t 帧
                                    3. 如果第 t 帧需要 refine, 先用 propagation 得到一个 mask, 然后添加 segtoken 更新 mask
                                    4. 更新后, 重置 tracking-state 和 memory, 当前帧 (t) 作为第 0 帧向后传播 (add point 只能影响当前帧预测, 不能影响 memofy, 因此, 在每次更新后, 就重置 tracker 和 memory, 将当前帧作为第一帧向后传播)
                                    """
                                    if j == 0:
                                        # 由于一定采样第0帧,即使开启memory_efficient也没问题
                                        frame_idx = 0
                                        # object_score_logits: torch.Size([1, 1]) e.g. tensor([[4.2188]], device='cuda:0', dtype=torch.bfloat16)
                                        low_res_masks, iou_predictions, _, object_score_logits = self.model.visual_model.sam_mask_decoder(
                                            image_embeddings = curr_image_embeddings.unsqueeze(0),
                                            image_pe = self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings = sparse_embeddings[curr_indices],
                                            dense_prompt_embeddings = dense_embeddings[curr_indices],
                                            multimask_output = multimask_output,
                                            repeat_image = True,
                                            high_res_features = fpn_list[:-1],
                                        )
                                        # print(33333)
                                        is_obj_appearing = object_score_logits > 0

                                        # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                                        # consistent with the actual mask prediction
                                        low_res_masks = torch.where(
                                            is_obj_appearing[:, None, None],
                                            low_res_masks,
                                            NO_OBJ_SCORE,
                                        )
                                        
                                        
                                        pred_mask = self.model.visual_model.postprocess_masks(
                                            low_res_masks,
                                            input_size=resize_list[i],
                                            original_size=original_size_list[i],
                                        )   # upsample mask

                                    # if j == 0 or start_point:
                                    # if j == 0:
                                    # 第一帧(编号0), 或者是被 seg token 改变过的帧
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 0. initialize inference_state
                                        # if sampled_indice+1 == len(samples_indices) and samples_indices[sampled_indice]<(video_len):
                                        #     inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:video_len], async_loading_frames=False)
                                        # else: #
                                        #     inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:(samples_indices[sampled_indice+1]+1)], async_loading_frames=False)
                                        # inference_state['video_height'] = original_size_list[i][0]
                                        # inference_state['video_width'] = original_size_list[i][1]
                                        # image = images[i][j].float().unsqueeze(0).to(torch.bfloat16)
                                        # backbone_out = {
                                        #     "vision_features": image_embeddings[i, j].unsqueeze(0).to(torch.bfloat16),
                                        #     "backbone_fpn": fpn_list.copy(),
                                        #     "vision_pos_enc": pos_list.copy(),
                                        # }
                                        # # Cache the most recent frame's feature (for repeated interactions with
                                        # # a frame; we can use an LRU cache for more frames in the future).
                                        # inference_state["cached_features"] = {0: (image, backbone_out)}
                                        # self.model.visual_model.reset_state(inference_state)
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 1. add_new_mask
                                        if pred_mask.shape[0] == 0:
                                            object_mask = np.zeros((pred_mask.shape[2], pred_mask.shape[3])) > 0
                                        else:
                                            object_mask = (pred_mask > 0)[0,0].cpu().numpy()    # just propagation
                                        self.model.visual_model.add_new_mask(
                                            inference_state=inference_state,
                                            frame_idx=frame_idx,                                    # frame with masks
                                            obj_id=1,                                       # object id (only one)
                                            mask=object_mask,
                                            input_size=resize_list[i]
                                        )
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 2-1. propagation (preparation stage, for frames with masks)
                                        self.model.visual_model.propagate_in_video_preflight(inference_state)
                                        output_dict = inference_state["output_dict"]
                                        if len(output_dict["cond_frame_outputs"]) == 0:
                                            raise RuntimeError("No points are provided; please add points first")
                                        storage_key = "cond_frame_outputs"
                                        current_out = output_dict[storage_key][0]
                                        pred_masks_prop = current_out["pred_masks"]
                                        self.model.visual_model._add_output_per_object(
                                            inference_state, 0, current_out, storage_key
                                        )
                                        inference_state["frames_already_tracked"][0] = {"reverse": False}
                                        # # Resize the output mask to the original video resolution (we directly use
                                        # # the mask scores on GPU for output to avoid any CPU conversion in between)
                                        # _, video_res_masks = self.model.visual_model._get_orig_video_res_output(
                                        #     inference_state, pred_masks_prop
                                        # )
                                        frame_idx += 1
                                        # start_point = False # 下一帧在 else 中处理

                                        # if j > 0:   # 不然 video_masks 数量比 video_len 多, add_new_mask 过了两次循环
                                        #     video_masks[j-1] = pred_mask[:, 0]
                                    else:
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-2. propagation (propagation stage, for frames to segment)
                                            # expanded_image = images[i][j].expand(1, -1, -1, -1)
                                            # expanded_backbone_out = {
                                            #     "backbone_fpn": fpn_list.copy(),
                                            #     "vision_pos_enc": pos_list.copy(),
                                            # }
                                            # for fi, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
                                            #     expanded_backbone_out["backbone_fpn"][fi] = feat.expand(
                                            #         1, -1, -1, -1
                                            #     )
                                            # for fi, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
                                            #     pos = pos.expand(1, -1, -1, -1)
                                            #     expanded_backbone_out["vision_pos_enc"][fi] = pos

                                            # features = self.model.visual_model._prepare_backbone_features(expanded_backbone_out)
                                            # features = (expanded_image,) + features
                                        # 如果是有 seg token 的帧, 就不用在传播时编码 mem 了, 因为后面会 add new seg 后再编码 mem
                                        run_mem_encoder = not sampled_frame
                                        if memory_refine and sampled_frame:     # 如果是采样帧, 还选择使用 seg token 和 memory 同时分割视频帧
                                            seg_token_sparse = sparse_embeddings[curr_indices]
                                            seg_token_dnese = dense_embeddings[curr_indices]
                                            run_mem_encoder = True
                                        else:
                                            seg_token_sparse = None
                                            seg_token_dnese = None

                                        storage_key = "non_cond_frame_outputs" if not sampled_frame else "cond_frame_outputs"
                                        current_out, pred_masks_prop = self.model.visual_model._run_single_frame_inference(
                                            inference_state=inference_state,
                                            output_dict=output_dict,
                                            frame_idx=frame_idx,
                                            batch_size=1,
                                            is_init_cond_frame=sampled_frame,
                                            point_inputs=None,
                                            mask_inputs=None,
                                            reverse=False,
                                            run_mem_encoder=run_mem_encoder,
                                            sparse_embeddings=seg_token_sparse,
                                            dense_embeddings=seg_token_dnese,
                                        )
                                        output_dict[storage_key][frame_idx] = current_out
                                        
                                        # if sampled_frame:
                                        #     self.model.visual_model._clear_non_cond_mem_around_input(inference_state, frame_idx)
                                        # Create slices of per-object outputs for subsequent interaction with each
                                        # individual object after tracking.
                                        self.model.visual_model._add_output_per_object(
                                            inference_state, frame_idx, current_out, storage_key
                                        )
                                        inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                                        
                                        object_score_logits = current_out["object_score_logits"]
                                        
                                        # Resize the output mask to the original video resolution (we directly use
                                        # the mask scores on GPU for output to avoid any CPU conversion in between)
                                        _, pred_mask = self.model.visual_model._get_orig_video_res_output(
                                            inference_state, pred_masks_prop, input_size=resize_list[i]
                                        )
                                        # 两种情况
                                        # 1. 没有 seg token, 什么都不做
                                        # 2. 有 seg token, add new point
                                        if sampled_frame and not memory_refine: # 采样帧, 但是采样帧单独做 refine, 不考虑 memory
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-3. add new point (seg token) (propagation stage, for frames to segment)
                                            _, _, pred_mask, object_score_logits = self.model.visual_model.add_new_seg(
                                                inference_state=inference_state,
                                                frame_idx=frame_idx,                            # frame with masks
                                                obj_id=1,                                       # object id (only one)
                                                sparse_embeddings = sparse_embeddings[curr_indices],
                                                dense_embeddings = dense_embeddings[curr_indices],
                                                # pre_embed=features,
                                                input_size = resize_list[i]
                                            )
                                        frame_idx += 1

                                if not seg_refine and sampled_frame:   # sampled frames. always true if samples_indices is None
                                    assert not memory_efficient
                                    fpn_list[0] = self.model.visual_model.sam_mask_decoder.conv_s0(fpn_list[0])
                                    fpn_list[1] = self.model.visual_model.sam_mask_decoder.conv_s1(fpn_list[1])
                                    pos_list = pos_embeddings[i][j]
                                    low_res_masks, iou_predictions, _, object_score_logits = self.model.visual_model.sam_mask_decoder(
                                        image_embeddings = curr_image_embeddings.unsqueeze(0),
                                        image_pe = self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings = sparse_embeddings[curr_indices],
                                        dense_prompt_embeddings = dense_embeddings[curr_indices],
                                        multimask_output = multimask_output,
                                        repeat_image = True,
                                        high_res_features = fpn_list[:-1],
                                    )
                                    # print(3333333333333333)
                                    is_obj_appearing = object_score_logits > 0

                                    # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                                    # consistent with the actual mask prediction
                                    low_res_masks = torch.where(
                                        is_obj_appearing[:, None, None],
                                        low_res_masks,
                                        NO_OBJ_SCORE,
                                    )

                                    pred_mask = self.model.visual_model.postprocess_masks(
                                        low_res_masks,
                                        input_size=resize_list[i],
                                        original_size=original_size_list[i],
                                    )   # upsample mask
                                    
                                    # from PIL import Image
                                    # image = Image.open('/hhd2/gaomingqi/ytvos_test/rvos_sam2_video_v2_tracking_sample0_4000/Annotations/0c04834d61/1/00000.png').convert('L')  # 将图像转换为灰度图

                                    # image = np.array(image, dtype=np.float32)
                                    # # 将图像转换为Tensor，并调整尺寸为 [1, 1, 720, 1280]
                                    # tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                                    # # 将Tensor移动到cuda设备
                                    # pred_mask = tensor.to(device='cuda')

                                    # if next frame is not sampled, 1) initialise state, 2) encode mask as memory
                                    if samples_indices is not None:
                                        if (sampled_indice+1 == len(samples_indices) and samples_indices[sampled_indice]<(video_len-1)) or \
                                            samples_indices[min(sampled_indice+1, len(samples_indices)-1)] - samples_indices[sampled_indice] > 1:
# ---------------------------------------------------------------------------------------------------------------------------------
# 0. initialize inference_state
                                            if sampled_indice+1 == len(samples_indices) and samples_indices[sampled_indice]<(video_len-1):
                                                inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:video_len], async_loading_frames=False)
                                            else:
                                                inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:samples_indices[sampled_indice+1]], async_loading_frames=False)
                                            inference_state['video_height'] = original_size_list[i][0]
                                            inference_state['video_width'] = original_size_list[i][1]
                                            image = images[i][j].float().unsqueeze(0).to(torch.bfloat16)
                                            backbone_out = {
                                                "vision_features": image_embeddings[i, j].unsqueeze(0).to(torch.bfloat16),
                                                "backbone_fpn": fpn_list.copy(),
                                                "vision_pos_enc": pos_list.copy(),
                                            }
                                            # Cache the most recent frame's feature (for repeated interactions with
                                            # a frame; we can use an LRU cache for more frames in the future).
                                            inference_state["cached_features"] = {0: (image, backbone_out)}
                                            self.model.visual_model.reset_state(inference_state)
# ---------------------------------------------------------------------------------------------------------------------------------
# 1. add_new_mask
                                            if pred_mask.shape[0] == 0:
                                                object_mask = np.zeros((pred_mask.shape[2], pred_mask.shape[3])) > 0
                                            else:
                                                object_mask = (pred_mask > 0)[0,0].cpu().numpy()    # just propagation
                                            self.model.visual_model.add_new_mask(
                                                inference_state=inference_state,
                                                frame_idx=0,                                    # frame with masks
                                                obj_id=1,                                       # object id (only one)
                                                mask=object_mask,
                                                input_size=resize_list[i]
                                            )
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-1. propagation (preparation stage, for frames with masks)
                                            self.model.visual_model.propagate_in_video_preflight(inference_state)
                                            output_dict = inference_state["output_dict"]
                                            if len(output_dict["cond_frame_outputs"]) == 0:
                                                raise RuntimeError("No points are provided; please add points first")
                                            storage_key = "cond_frame_outputs"
                                            current_out = output_dict[storage_key][0]
                                            pred_masks_prop = current_out["pred_masks"]
                                            self.model.visual_model._add_output_per_object(
                                                inference_state, 0, current_out, storage_key
                                            )
                                            inference_state["frames_already_tracked"][0] = {"reverse": False}
                                            # Resize the output mask to the original video resolution (we directly use
                                            # the mask scores on GPU for output to avoid any CPU conversion in between)
                                            _, video_res_masks = self.model.visual_model._get_orig_video_res_output(
                                                inference_state, pred_masks_prop
                                            )
                                            frame_idx = 1
                                elif not seg_refine:   # other frames
                                    assert not memory_efficient
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-2. propagation (propagation stage, for frames to segment)
                                    expanded_image = images[i][j].expand(1, -1, -1, -1)
                                    expanded_backbone_out = {
                                        "backbone_fpn": fpn_list.copy(),
                                        "vision_pos_enc": pos_list.copy(),
                                    }
                                    for fi, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
                                        expanded_backbone_out["backbone_fpn"][fi] = feat.expand(
                                            1, -1, -1, -1
                                        )
                                    for fi, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
                                        pos = pos.expand(1, -1, -1, -1)
                                        expanded_backbone_out["vision_pos_enc"][fi] = pos

                                    features = self.model.visual_model._prepare_backbone_features(expanded_backbone_out)
                                    features = (expanded_image,) + features

                                    storage_key = "non_cond_frame_outputs"
                                    current_out, pred_masks_prop = self.model.visual_model._run_single_frame_inference(
                                        inference_state=inference_state,
                                        output_dict=output_dict,
                                        frame_idx=frame_idx,
                                        batch_size=1,
                                        is_init_cond_frame=False,
                                        point_inputs=None,
                                        mask_inputs=None,
                                        reverse=False,
                                        run_mem_encoder=True,
                                        pre_embed=features,
                                    )
                                    output_dict[storage_key][frame_idx] = current_out
                                    # Create slices of per-object outputs for subsequent interaction with each
                                    # individual object after tracking.
                                    self.model.visual_model._add_output_per_object(
                                        inference_state, frame_idx, current_out, storage_key
                                    )
                                    inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                                    
                                    object_score_logits = current_out["object_score_logits"]

                                    # Resize the output mask to the original video resolution (we directly use
                                    # the mask scores on GPU for output to avoid any CPU conversion in between)
                                    _, pred_mask = self.model.visual_model._get_orig_video_res_output(
                                        inference_state, pred_masks_prop, input_size=resize_list[i]
                                    )
                                    frame_idx += 1
                                
                                # else:
                                #     # 

                                #     low_res_masks, iou_predictions, _, _ = self.model.visual_model.sam_mask_decoder(
                                #         image_embeddings = curr_image_embeddings.unsqueeze(0),
                                #         image_pe = self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                #         sparse_prompt_embeddings = sparse_embeddings[curr_indices],
                                #         dense_prompt_embeddings = dense_embeddings[curr_indices],
                                #         multimask_output = multimask_output,
                                #         repeat_image = True,
                                #         high_res_features = fpn_list,
                                #     )
                                boxes = None
                        else:
                            # low_res_masks: torch.Size([3 (该视频<SEG>总数), 1, 256, 256])
                            # iou_predictions: torch.Size([3 (该视频<SEG>总数), 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)
                            # boxes: torch.Size([3 (该视频<SEG>总数), 1, 4])
                            curr_indices = torch.arange(seg_token_num)
                            if not self.sam2:
                                assert not memory_efficient
                                low_res_masks, boxes, iou_predictions = self.model.visual_model.mask_decoder(
                                    image_embeddings=curr_image_embeddings.unsqueeze(0),
                                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse_embeddings[curr_indices],
                                    dense_prompt_embeddings=dense_embeddings[curr_indices],
                                    multimask_output=multimask_output,
                                )
                            else:
                                if not memory_efficient or j == 0:
                                    fpn_list = fpn_embeddings[i][j]
                                    pos_list = pos_embeddings[i][j]
                                    if memory_efficient and j == 0:
                                        fpn_list[0] = self.model.visual_model.sam_mask_decoder.conv_s0(fpn_list[0])
                                        fpn_list[1] = self.model.visual_model.sam_mask_decoder.conv_s1(fpn_list[1])

                                if seg_refine: # seg_token_refine_propagation:
                                    """
                                    0. 给定 video_len 个视频帧, 对每个视频帧, 得到一个 seg_token
                                    1. 使用第一帧 seg_token, 得到第一帧 mask
                                    2. 使用第一帧 mask, 逐个帧传播, 从 1 - (video_len-1), 第一帧索引为 0, 下文为第 t 帧
                                    3. 如果第 t 帧需要 refine, 先用 propagation 得到一个 mask, 然后添加 segtoken 更新 mask
                                    4. 更新后, 重置 tracking-state 和 memory, 当前帧 (t) 作为第 0 帧向后传播 (add point 只能影响当前帧预测, 不能影响 memofy, 因此, 在每次更新后, 就重置 tracker 和 memory, 将当前帧作为第一帧向后传播)
                                    """
                                    if j == 0:
                                        # 由于一定采样第0帧,即使开启memory_efficient也没问题
                                        frame_idx = 0
                                        # object_score_logits: torch.Size([1, 1]) e.g. tensor([[4.2188]], device='cuda:0', dtype=torch.bfloat16)
                                        low_res_masks, iou_predictions, _, object_score_logits = self.model.visual_model.sam_mask_decoder(
                                            image_embeddings = curr_image_embeddings.unsqueeze(0),
                                            image_pe = self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings = sparse_embeddings[curr_indices],
                                            dense_prompt_embeddings = dense_embeddings[curr_indices],
                                            multimask_output = multimask_output,
                                            repeat_image = True,
                                            high_res_features = fpn_list[:-1],
                                        )
                                        # print(43534534534)
                                        is_obj_appearing = object_score_logits > 0
                                        # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                                        # consistent with the actual mask prediction
                                        low_res_masks = torch.where(
                                            is_obj_appearing[:, None, None],
                                            low_res_masks,
                                            NO_OBJ_SCORE,
                                        )
                                        
                                        
                                        pred_mask = self.model.visual_model.postprocess_masks(
                                            low_res_masks,
                                            input_size=resize_list[i],
                                            original_size=original_size_list[i],
                                        )   # upsample mask

                                    # if j == 0 or start_point:
                                    # if j == 0:
                                    # 第一帧(编号0), 或者是被 seg token 改变过的帧
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 0. initialize inference_state
                                        # if sampled_indice+1 == len(samples_indices) and samples_indices[sampled_indice]<(video_len):
                                        #     inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:video_len], async_loading_frames=False)
                                        # else: #
                                        #     inference_state = self.model.visual_model.init_state(images[i][samples_indices[sampled_indice]:(samples_indices[sampled_indice+1]+1)], async_loading_frames=False)
                                        # inference_state['video_height'] = original_size_list[i][0]
                                        # inference_state['video_width'] = original_size_list[i][1]
                                        # image = images[i][j].float().unsqueeze(0).to(torch.bfloat16)
                                        # backbone_out = {
                                        #     "vision_features": image_embeddings[i, j].unsqueeze(0).to(torch.bfloat16),
                                        #     "backbone_fpn": fpn_list.copy(),
                                        #     "vision_pos_enc": pos_list.copy(),
                                        # }
                                        # # Cache the most recent frame's feature (for repeated interactions with
                                        # # a frame; we can use an LRU cache for more frames in the future).
                                        # inference_state["cached_features"] = {0: (image, backbone_out)}
                                        # self.model.visual_model.reset_state(inference_state)
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 1. add_new_mask
                                        if pred_mask.shape[0] == 0:
                                            object_mask = np.zeros((pred_mask.shape[2], pred_mask.shape[3])) > 0
                                        else:
                                            object_mask = (pred_mask > 0)[0,0].cpu().numpy()    # just propagation
                                        self.model.visual_model.add_new_mask(
                                            inference_state=inference_state,
                                            frame_idx=frame_idx,                                    # frame with masks
                                            obj_id=1,                                       # object id (only one)
                                            mask=object_mask,
                                            input_size=resize_list[i]
                                        )
# ---------------------------------------------------------------------------------------------------------------------------------
# [new] 2-1. propagation (preparation stage, for frames with masks)
                                        self.model.visual_model.propagate_in_video_preflight(inference_state)
                                        output_dict = inference_state["output_dict"]
                                        if len(output_dict["cond_frame_outputs"]) == 0:
                                            raise RuntimeError("No points are provided; please add points first")
                                        storage_key = "cond_frame_outputs"
                                        current_out = output_dict[storage_key][0]
                                        pred_masks_prop = current_out["pred_masks"]
                                        self.model.visual_model._add_output_per_object(
                                            inference_state, 0, current_out, storage_key
                                        )
                                        inference_state["frames_already_tracked"][0] = {"reverse": False}
                                        # # Resize the output mask to the original video resolution (we directly use
                                        # # the mask scores on GPU for output to avoid any CPU conversion in between)
                                        # _, video_res_masks = self.model.visual_model._get_orig_video_res_output(
                                        #     inference_state, pred_masks_prop
                                        # )
                                        frame_idx += 1
                                        # start_point = False # 下一帧在 else 中处理

                                        # if j > 0:   # 不然 video_masks 数量比 video_len 多, add_new_mask 过了两次循环
                                        #     video_masks[j-1] = pred_mask[:, 0]
                                    else:
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-2. propagation (propagation stage, for frames to segment)
                                            # expanded_image = images[i][j].expand(1, -1, -1, -1)
                                            # expanded_backbone_out = {
                                            #     "backbone_fpn": fpn_list.copy(),
                                            #     "vision_pos_enc": pos_list.copy(),
                                            # }
                                            # for fi, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
                                            #     expanded_backbone_out["backbone_fpn"][fi] = feat.expand(
                                            #         1, -1, -1, -1
                                            #     )
                                            # for fi, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
                                            #     pos = pos.expand(1, -1, -1, -1)
                                            #     expanded_backbone_out["vision_pos_enc"][fi] = pos

                                            # features = self.model.visual_model._prepare_backbone_features(expanded_backbone_out)
                                            # features = (expanded_image,) + features
                                        # 如果是有 seg token 的帧, 就不用在传播时编码 mem 了, 因为后面会 add new seg 后再编码 mem
                                        run_mem_encoder = not sampled_frame
                                        if memory_refine and sampled_frame:     # 如果是采样帧, 还选择使用 seg token 和 memory 同时分割视频帧
                                            seg_token_sparse = sparse_embeddings[curr_indices]
                                            seg_token_dnese = dense_embeddings[curr_indices]
                                            run_mem_encoder = True
                                        else:
                                            seg_token_sparse = None
                                            seg_token_dnese = None

                                        storage_key = "non_cond_frame_outputs" if not sampled_frame else "cond_frame_outputs"
                                        current_out, pred_masks_prop = self.model.visual_model._run_single_frame_inference(
                                            inference_state=inference_state,
                                            output_dict=output_dict,
                                            frame_idx=frame_idx,
                                            batch_size=1,
                                            is_init_cond_frame=sampled_frame,
                                            point_inputs=None,
                                            mask_inputs=None,
                                            reverse=False,
                                            run_mem_encoder=run_mem_encoder,
                                            sparse_embeddings=seg_token_sparse,
                                            dense_embeddings=seg_token_dnese,
                                        )
                                        output_dict[storage_key][frame_idx] = current_out
                                        
                                        # if sampled_frame:
                                        #     self.model.visual_model._clear_non_cond_mem_around_input(inference_state, frame_idx)
                                        # Create slices of per-object outputs for subsequent interaction with each
                                        # individual object after tracking.
                                        self.model.visual_model._add_output_per_object(
                                            inference_state, frame_idx, current_out, storage_key
                                        )
                                        inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                                        
                                        object_score_logits = current_out["object_score_logits"]
                                        
                                        # Resize the output mask to the original video resolution (we directly use
                                        # the mask scores on GPU for output to avoid any CPU conversion in between)
                                        _, pred_mask = self.model.visual_model._get_orig_video_res_output(
                                            inference_state, pred_masks_prop, input_size=resize_list[i]
                                        )
                                        # 两种情况
                                        # 1. 没有 seg token, 什么都不做
                                        # 2. 有 seg token, add new point
                                        if sampled_frame and not memory_refine: # 采样帧, 但是采样帧单独做 refine, 不考虑 memory
# ---------------------------------------------------------------------------------------------------------------------------------
# 2-3. add new point (seg token) (propagation stage, for frames to segment)
                                            _, _, pred_mask, object_score_logits = self.model.visual_model.add_new_seg(
                                                inference_state=inference_state,
                                                frame_idx=frame_idx,                            # frame with masks
                                                obj_id=1,                                       # object id (only one)
                                                sparse_embeddings = sparse_embeddings[curr_indices],
                                                dense_embeddings = dense_embeddings[curr_indices],
                                                # pre_embed=features,
                                                input_size = resize_list[i]
                                            )
                                        frame_idx += 1
                                
                                # low_res_masks, iou_predictions, _, _ = self.model.visual_model.sam_mask_decoder(
                                #     image_embeddings=curr_image_embeddings.unsqueeze(0),
                                #     image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                                #     sparse_prompt_embeddings=sparse_embeddings[curr_indices],
                                #     dense_prompt_embeddings=dense_embeddings[curr_indices],
                                #     multimask_output=multimask_output,
                                #     repeat_image=True,
                                #     high_res_features=fpn_list[:-1],
                                # )
                        
                        # # pred_mask: torch.Size([3 (该视频<SEG>总数), 1, 1080, 1920])
                        # # pred_mask[:, 0]: torch.Size([3 (该视频<SEG>总数), 1080, 1920])
                        # if (not use_fusion_module) or (use_fusion_module and not self.sam2):
                        if not self.sam2:
                            pred_mask = self.model.visual_model.postprocess_masks(
                                low_res_masks,
                                input_size=resize_list[i],
                                original_size=original_size_list[i],
                            )
                            
                        if torch.isnan(low_res_masks).any() or torch.isnan(pred_mask).any():
                            print("NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        
                        # pred_box = self.model.visual_model.postprocess_boxes(boxes, input_size=resize_list[i])
                        pred_box = None
                        if pred_mask.shape[0] == 0:
                            video_masks.append(torch.zeros(1, pred_mask.shape[2], pred_mask.shape[3]).to(device = 'cuda'))
                            video_object_score_logits.append(0)
                        else:
                            video_masks.append(pred_mask[:, 0])
                            video_object_score_logits.append(object_score_logits)
                        j += 1
                        # video_boxes.append(pred_box[:, 0])
                    # 这样的话pred_masks里是 torch.Size([video_len, 1, 1080, 1920])的tensor
                    # pred_boxes[0]: torch.Size([video_len, 1, 4])
                    pred_masks.append(torch.stack(video_masks))    
                    pred_object_score_logits.append(torch.stack(video_object_score_logits))
                    # pred_boxes.append(torch.stack(video_boxes))
                elif image_embeddings.ndim == 5 and use_multuple_seg_token:
                    video_len = image_embeddings.shape[1]
                    video_masks = []
                    video_boxes = []
                    assert video_len == sparse_embeddings.shape[0], f"Real video len:{video_len}, Output len:{sparse_embeddings.shape[0]}"
                    for j in range(video_len):
                        curr_image_embeddings = image_embeddings[i][j]
                        # low_res_masks: torch.Size([3 (该视频<SEG>总数), 1, 256, 256])
                        # iou_predictions: torch.Size([3 (该视频<SEG>总数), 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)
                        # boxes: torch.Size([3 (该视频<SEG>总数), 1, 4])
                        low_res_masks, boxes, iou_predictions = self.model.visual_model.mask_decoder(
                            image_embeddings=curr_image_embeddings.unsqueeze(0),
                            image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings[j].unsqueeze(0),
                            dense_prompt_embeddings=dense_embeddings[j].unsqueeze(0),
                            multimask_output=multimask_output,
                        )
                        # pred_mask: torch.Size([3 (该视频<SEG>总数), 1, 1080, 1920])
                        # pred_mask[:, 0]: torch.Size([3 (该视频<SEG>总数), 1080, 1920])
                       
                        # RVOS
                        pred_mask = self.model.visual_model.postprocess_masks(
                            low_res_masks,
                            input_size=resize_list[i],
                            original_size=original_size_list[i]
                        )
                        
                        # pred_box: torch.Size([1, 1, 4])
                        pred_box = self.model.visual_model.postprocess_boxes(boxes, input_size=resize_list[i])
                        video_masks.append(pred_mask[:, 0])
                        video_boxes.append(pred_box[:, 0])
                    # 这样的话pred_masks里是 torch.Size([video_len, 1, 1080, 1920])的tensor
                    # pred_boxes[0]: torch.Size([5, 1, 4])
                    pred_boxes.append(torch.stack(video_boxes))
                    pred_masks.append(torch.stack(video_masks))            
                else:
                    # self.model.visual_model.mask_decoder: <class 'model.segment_anything.modeling.mask_decoder.MaskDecoder'>
                    # low_res_masks: torch.Size([1, 1, 256, 256])
                    # iou_predictions: torch.Size([1, 1]), tensor([[0.6289]], device='cuda:0', dtype=torch.bfloat16)
                    low_res_masks, boxes, iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    
                    # pred_mask: torch.Size([1, 1, 1080, 1920])
                    # pred_mask[:, 0]: torch.Size([1, 1080, 1920])
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=original_size_list[i],
                    )
                    pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks, pred_boxes
    
