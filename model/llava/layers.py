# copied from https://github.com/jshilong/GPT4RoI
# build spi query for llava
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmcv.cnn import ConvModule, Linear, normal_init
from .mmdet.models import BaseRoIExtractor


def str2spi(input_str):
    bbox_regex = r'<bbox>\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)\s*</bbox>'
    # only attention inter the instruction
    results = []
    matches = re.findall(bbox_regex, input_str)
    for match in matches:
        results.append([float(match[0]), float(match[1]), float(match[2]),
                        float(match[3])])
    return results


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def coordinate_to_encoding(coord_tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)
    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
            coord_tensor.size(-1)))
    return pos


def align_tensor(inputs, max_len=None):
    if max_len is None:
        max_len = max([len(item) for item in inputs])

    return torch.stack([padding_to(item, max_len) for item in inputs])


def padding_to(inputs, max=300):
    if max is None:
        return inputs
    num_padding = max - len(inputs)
    if inputs.dim() > 1:
        padding = inputs.new_zeros(num_padding,
                                   *inputs.size()[1:],
                                   dtype=inputs.dtype)
    else:
        padding = inputs.new_zeros(num_padding, dtype=inputs.dtype)
    inputs = torch.cat([inputs, padding], dim=0)
    return inputs


class MLVLFuseModule(nn.Module):
    def __init__(self, input_dims=1024, embed_dims=1024, num_levels=3, num_fuse=4):
        super(MLVLFuseModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_fuse = num_fuse
        self.input_dims = input_dims
        self.shuffle_channles = embed_dims // 4

        # contains the tuple of level indices that will do the interaction
        self.fuse_lvl_list = []
        num_levels = self.num_levels
        for lvl in range(num_levels):
            top_lvl = min(lvl + 1, num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            tar_lvl = lvl
            self.fuse_lvl_list.append((tar_lvl, top_lvl, dow_lvl))

        self.remain_chs = self.embed_dims - self.shuffle_channles * 2
        self._init_layers()

    def generate_coordinate(self, featmap_sizes, device='cuda'):

        x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
        y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([featmap_sizes[0], 1, -1, -1])
        x = x.expand([featmap_sizes[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        return coord_feat

    def _init_layers(self):
        self.input_conv = nn.ModuleList([nn.Conv2d(self.input_dims + 2,
                                                   self.embed_dims, 1)
                                         for _ in range(self.num_levels)])
        self.fuse_convs = nn.ModuleList()
        for i in range(self.num_fuse):
            self.fuse_convs.append(
                ConvModule(self.embed_dims,
                           self.embed_dims,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=None,
                           norm_cfg=dict(type='GN',
                                         num_groups=64,
                                         requires_grad=True)
                           ))

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def _single_shuffle(self, inputs, conv_module):
        if not isinstance(conv_module, (nn.ModuleList, list)):
            conv_module = [conv_module]
        for single_conv_m in conv_module:
            fused_inputs = []
            for fuse_lvl_tuple in self.fuse_lvl_list:
                tar_lvl, top_lvl, dow_lvl = fuse_lvl_tuple
                tar_input = inputs[tar_lvl]
                top_input = inputs[top_lvl]
                down_input = inputs[dow_lvl]
                remain = tar_input[:, :self.remain_chs]
                from_top = top_input[:,
                           self.remain_chs:][:,
                           self.shuffle_channles:]
                from_top = F.interpolate(from_top.to(torch.float32),
                                         size=tar_input.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True).to(tar_input.dtype)

                from_down = down_input[:, self.remain_chs:][:, :self.
                shuffle_channles]
                from_down = F.interpolate(from_down.to(torch.float32),
                                          size=tar_input.shape[-2:],
                                          mode='bilinear',
                                          align_corners=True).to(tar_input.dtype)
                fused_inputs.append(
                    torch.cat([remain, from_top, from_down], dim=1))
            fused_inputs = [single_conv_m(item) for item in fused_inputs]
            inputs = fused_inputs
        return inputs

    def forward(self, inputs, ):
        feat_size = [item.shape for item in inputs]
        new_inputs = []
        for feat, single_feat_size in zip(inputs, feat_size):
            coord_feat = self.generate_coordinate(single_feat_size, device=inputs[0].device).to(dtype=feat.dtype)
            feat = torch.cat([feat, coord_feat], dim=1)
            new_inputs.append(feat)
        inputs = new_inputs

        inputs = [self.input_conv[lvl](item) for lvl, item in enumerate(inputs)]

        for conv_m in self.fuse_convs:
            inputs = self._single_shuffle(inputs, [conv_m])
        return inputs


class MLVLROIQueryModule(nn.Module):
    def __init__(self, embed_dims=1024, out_dims=4096,
                 num_levels=3):
        super(MLVLROIQueryModule, self).__init__()
        # 暂时一层特征, 先不定义融合层
        self.mlvl_fuse = MLVLFuseModule(input_dims=embed_dims,
                                        embed_dims=embed_dims,
                                        num_levels=num_levels,
                                        num_fuse=5)
        # 这里可能要改，硬编码了
        strids = [14 / 8, 14 / 4, 14 / 2, 14]
        assert len(strids) == num_levels
        bbox_roi_extractor = dict(roi_layer=dict(type='RoIAlign',
                                                 output_size=14,
                                                 sampling_ratio=2),
                                  out_channels=embed_dims,
                                  embed_dims=embed_dims,
                                  fuse_level=num_levels,
                                  featmap_strides=strids)

        self.roi_align = MlvlRoIExtractor(**bbox_roi_extractor)

    def forward(self, mlvl_feats, bboxes):
        # mlvl_feats: tuple, len=4, 文中提到的倒数2,5,8,11层, [0]: torch.Size([1, 256, 1024])
        # 暂时我们只用单层特征
        # bboxes: [tensor([[0.0320, 0.0266, 0.2848, 0.8895]], device='cuda:0')]; list[len = 1(bs?)], bboxes[0]: torch.Size([1, 4])
        if mlvl_feats[0].dim() == 3:
            # h,w: 16
            h = w = int(math.sqrt(mlvl_feats[0].shape[1]))
            assert h == 16
            assert w == 16
            # b, c: 1, 1024 
            b, c = mlvl_feats[0].shape[0], mlvl_feats[0].shape[-1]
            mlvl_feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2) for item in mlvl_feats]
            # mlvl_feats[0].shape: torch.Size([1, 1024, 16, 16])
                        
        # base_shape: torch.Size([16, 16])
        # num_level: 4 (我们现在是1)
        base_shape = mlvl_feats[0].shape[-2:]
        num_level = len(mlvl_feats)
        to_shape = [(base_shape[0] * 2 ** level, base_shape[1] * 2 ** level) for level in range(num_level)]
        # to_shape: [(16, 16), (32, 32), (64, 64), (128, 128)]
        # 我们目前只是[(16,16)]
        to_shape = to_shape[::-1]
        # to_shape: [(128, 128), (64, 64), (32, 32), (16, 16)]
        for level in range(num_level):
            # torch.Size([1, 1024, 16, 16])
            feat = mlvl_feats[level]
            shape = to_shape[level]
            mlvl_feats[level] = F.interpolate(feat.float(), size=shape, mode='bilinear', align_corners=True).bfloat16()

        # mlvl_feats: [torch.Size([1, 1024, 128, 128]), torch.Size([1, 1024, 64, 64]), torch.Size([1, 1024, 32, 32]), torch.Size([1, 1024, 16, 16])]
        # 先不fuse
        mlvl_feats = self.mlvl_fuse(mlvl_feats)
        # mlvl_feats: 同上，没变
        
        return self.roi_align(mlvl_feats, bboxes)
        # self.roi_align(mlvl_feats, bboxes): torch.Size([1, 4096]), bbox经过编码

class MlvlRoIExtractor(BaseRoIExtractor):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 embed_dims=1024,
                 stride=1,
                 norm_init=True,
                 fuse_level=3,
                 finest_scale=56,
                 init_cfg=None):
        super(MlvlRoIExtractor, self).__init__(roi_layer, out_channels,
                                               featmap_strides, init_cfg)
        self.embed_dims = embed_dims
        self.finest_scale = finest_scale
        self.fuse_level = fuse_level
        self.norm_init = norm_init

        self.pconvs = nn.ModuleList(
            nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=1, padding=1)
            for _ in range(self.fuse_level))
        self.pos_embedd = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.LayerNorm(1024),
        )
        self.updims = nn.Linear(1024, 4096)

        self.flatten_linear = nn.Linear(self.embed_dims * self.roi_layers[0].output_size[0] ** 2, 1024)
        self.norm_init_weights()

    #  self.dtype = torch.float32
    def norm_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)

    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        # rois(bboxes): [tensor([[0.0320, 0.0266, 0.2848, 0.8895]], device='cuda:0')]; list[len = 1(bs?)], bboxes[0]: torch.Size([1, 4])
        # feats: [torch.Size([1, 1024, 128, 128]), torch.Size([1, 1024, 64, 64]), torch.Size([1, 1024, 32, 32]), torch.Size([1, 1024, 16, 16])]
        # num_imgs: 1
        num_imgs = len(rois)
        # feats = [item for item in feats]
        
        # batch_rois: torch.Size([1, 4])
        if type(rois) != torch.Tensor:
            batch_rois = torch.cat(rois, dim=0)
        else:
            batch_rois = rois.reshape(rois.shape[0],4)
        
        
        batch_rois = batch_rois.to(dtype=self.pos_embedd[0].weight.dtype)
        # pos_embedd: torch.Size([1, 1024])
        pos_embedd = self.pos_embedd(batch_rois)
        # out_size: (14, 14)
        out_size = self.roi_layers[0].output_size
        # num_levels: 4
        num_levels = len(feats)
        if feats[0].dim() == 3:
            h = w = int(math.sqrt(feats[0].shape[1]))
            assert h == 16
            assert w == 16
            b, c = feats[0].shape[0], feats[0].shape[-1]
            feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2) for item in feats]
        new_rois = []
        for img_id, single_img_roi in enumerate(rois):
            # rescale to original img scale
            # single_img_roi:tensor([[7.1715, 5.9547, 63.8015, 199.2527]], device='cuda:0'), torch.Size([1, 4])
            # 这里注意224硬编码了, 可能要调整代码
            single_img_roi = single_img_roi * 224

            # roi_img_id: tensor([0.], device='cuda:0')
            roi_img_id = single_img_roi.new_ones(len(single_img_roi)) * img_id
            # single_img_roi: torch.Size([1, 5])， tensor([[0.0000, 7.1715, 5.9547, 63.8015, 199.2527]], device='cuda:0')
            single_img_roi = torch.cat([roi_img_id[:, None], single_img_roi], dim=1)
            new_rois.append(single_img_roi)
        # rois: torch.Size([1, 5])
        rois = torch.cat(new_rois)

        # roi_feats: torch.Size([4 (level_num), 1, 1024, 14, 14])
        roi_feats = feats[0].new_zeros(self.fuse_level,
                                       rois.size(0), self.out_channels, *out_size)

        for i in range(num_levels):
            if len(rois) > 0:
                rois_ = rois
                ori_dtype = feats[i].dtype
                # self.roi_layers: ModuleList(
                #   (0): RoIAlign(output_size=(14, 14), spatial_scale=0.5714285714285714, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
                #   (1): RoIAlign(output_size=(14, 14), spatial_scale=0.2857142857142857, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
                #   (2): RoIAlign(output_size=(14, 14), spatial_scale=0.14285714285714285, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
                #   (3): RoIAlign(output_size=(14, 14), spatial_scale=0.07142857142857142, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
                # )
                # 这里的spatial_scale计算方式举个例子， 第一层ss=4/7, 那么224 * 4/7 = 128, 也是硬编码
                # <class 'mmcv.ops.roi_align.RoIAlign'>
                # roi_feats_t: torch.Size([1, 1024, 14, 14])
                roi_feats_t = self.roi_layers[i](feats[i].to(torch.float32), rois_.to(torch.float32))

                roi_feats[i] = roi_feats_t.to(ori_dtype)

            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        

        fuse_roi_feats = []
        for i in range(self.fuse_level):
            # self.pconvs: ModuleList(
            #   (0-3): 4 x Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # )
            fuse_roi_feats.append(self.pconvs[i](roi_feats[i]))
            # fuse_roi_feats[0]: torch.Size([1, 1024, 14, 14])
        fuse_roi_feats = sum(fuse_roi_feats)
        # fuse_roi_feats: torch.Size([1, 1024, 14, 14])
        fuse_roi_feats = F.relu(fuse_roi_feats)
        fuse_roi_feats = fuse_roi_feats.flatten(1, -1)
        # fuse_roi_feats: torch.Size([1, 200704])
        fuse_roi_feats = self.flatten_linear(fuse_roi_feats)
        # fuse_roi_feats: torch.Size([1, 1024])
        fuse_roi_feats = fuse_roi_feats + pos_embedd
        fuse_roi_feats = self.updims(fuse_roi_feats)
        # fuse_roi_feats: torch.Size([1, 4096])
        query_feats = []
        for i in range(num_imgs):
            mask = rois[:, 0] == i
            query_feats.append(fuse_roi_feats[mask])
        # query_feats: torch.Size([1, 4096])
        return query_feats
