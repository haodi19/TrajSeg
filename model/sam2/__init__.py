# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
from .build_sam import build_sam2, build_sam2_video_predictor, build_sam2_video_predictor_refine

initialize_config_module("model.sam2.sam2_configs", version_base="1.2")
