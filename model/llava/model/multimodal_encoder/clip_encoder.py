import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, low_cpu_mem_usage=True
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # self.select_layer: -2
        # self.select_feature: "patch"
        # image_forward_outs.hidden_states[0]: torch.Size([6, 257, 1024])
        if type(self.select_layer) is list:
            image_features = [image_forward_outs.hidden_states[i] for i in self.select_layer]
            if self.select_feature == "patch":
                image_features = [fea[:, 1:] for fea in image_features]
            elif self.select_feature == "cls_patch":
                image_features = image_features
            else:
                raise ValueError(f"Unexpected select feature: {self.select_feature}")
            return image_features[-1], image_features
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.select_feature == "patch":
                image_features = image_features[:, 1:]
            elif self.select_feature == "cls_patch":
                image_features = image_features
            else:
                raise ValueError(f"Unexpected select feature: {self.select_feature}")
            return image_features, None

    @torch.no_grad()
    def forward(self, images):
        # images: torch.Size([bs*video_len, 3, 224, 224])
        if type(images) is list:
            image_features = []
            mlvl_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature, mlvl_feature = self.feature_select(image_forward_out)
                image_feature = image_feature.to(images.dtype)
                if mlvl_feature is not None:
                    mlvl_feature = [fea.to(images.dtype) for fea in mlvl_feature]
                image_features.append(image_feature)
                mlvl_features.append(mlvl_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            # image_forward_outs.hidden_states: tuple[len=25], [0]:torch.Size([bs*video_len, 257, 1024])
            # image_features: torch.Size([6, 256, 1024]), 倒数第2层
            # mlvl_features: tuple, len=4, 倒数2,5,8,11层, [0]: torch.Size([1, 256, 1024])
            image_features, mlvl_features = self.feature_select(image_forward_outs)
            image_features = image_features.to(images.dtype)
            if mlvl_features is not None:
                mlvl_features = [fea.to(images.dtype) for fea in mlvl_features]

        # torch.cuda.empty_cache()
        return image_features, mlvl_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
