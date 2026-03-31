# TrajSeg

TrajSeg is a training & inference codebase for video referring segmentation built on top of a LISA-style VLM and SAM/SAM2 visual backbones.  
This repository focuses on **two-stage training** (image stage → video stage) and **uniform frame sampling** inference on Ref-Youtube-VOS / MeVIS / ReVOS.

## Installation

This project is typically run on Linux with CUDA GPUs.

Option A (conda):

```bash
conda env create -f environment.yml
conda activate llm
```

Option B (pip):

```bash
pip install -r requirements.txt
```

## Pretrained weights

You need to download two key assets:

1) **LLaVA-Lightning-7B-v1-1** (base VLM weights)  
- Download from the HuggingFace model hub (search for `LLaVA-Lightning-7B-v1-1`).  
- Set `--version /path/to/LLaVA-Lightning-7B-v1-1`.

2) **SAM2 checkpoint**: `sam2_hiera_large.pt`  
- Download the official SAM2 checkpoint, then set `--sam2_checkpoint /path/to/sam2_hiera_large.pt`.  
- The SAM2 config file is already in this repo: `model/sam2/sam2_configs/sam2_hiera_l.yaml` (use it via `--sam2_cfg`).


## Data layout

This repo supports two groups of datasets:

- Stage 1 (image) datasets: semantic segmentation / referring segmentation / VQA / reasoning segmentation
- Stage 2 (video) datasets: RVOS (Ref-Youtube-VOS, MeVIS, ReVOS)

We recommend the following structure (names can be adapted; pass paths via `--dataset_dir` and `--image_dir`):

```text
data/
  stage1_images/                  # for --dataset_dir (stage 1)
    ade20k/
    cocostuff/
    coco/                         # e.g., train2017/
    mapillary/
    vlpart/
    refer_seg/
    llava_dataset/                # e.g., llava_instruct_150k.json
    reason_seg/

  stage2_videos/                  # for --dataset_dir (stage 2)
    ref-youtube-vos/
      meta_expressions/
        valid/meta_expressions.json
        test/meta_expressions.json
      valid/JPEGImages/<video_name>/*.jpg
      test/JPEGImages/<video_name>/*.jpg

    MeVIS/
      valid/JPEGImages/<video_name>/*.jpg
      valid/meta_expressions.json

    ReVOS/
      <video_name>/*.jpg
      meta_expressions_valid_.json
      meta_expressions_test_.json
```

Note: the inference scripts expect dataset-specific metadata filenames as implemented in each script.

## Training

### Stage 1 (image stage)

Example command (adjust paths for your machine):

```bash
deepspeed --master_port=23532 -- train_ds.py \
  --stage 1 \
  --exp_name trajseg_stage1 \
  --version /path/to/LLaVA-Lightning-7B-v1-1 \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --dataset_dir data/stage1_images \
  --dataset sem_seg||refer_seg||vqa||reason_seg \
  --sample_rates 9,3,3,1 \
  --batch_size 2 \
  --train_mode_rates 1,0 \
  --answer_type 2 \
  --num_frames 10 \
  --post_fusion_type nn \
  --sam2 --sam2_checkpoint /path/to/sam2_hiera_large.pt \
  --sam2_cfg model/sam2/sam2_configs/sam2_hiera_l.yaml \
  --no_eval \
  --auto_resume \
  --epochs 20
```

### Stage 2 (video stage)

`--version` should point to the **stage-1 output**. If you saved a HF-style model folder, pass that folder path here.

```bash
deepspeed --master_port=23563 train_ds.py \
  --stage 2 \
  --exp_name trajseg_stage2 \
  --version /path/to/stage1_ckpt_or_hf_folder \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --dataset_dir data/stage2_videos \
  --image_dir data/stage1_images \
  --dataset rvos||refseg \
  --rvos_data ytvos||mevis||revos \
  --sample_rates 3,1 \
  --batch_size 1 \
  --train_mode_rates 1,1 \
  --answer_type 2 \
  --num_frames 10 \
  --post_fusion_type nn \
  --sam2 --sam2_checkpoint /path/to/sam2_hiera_large.pt \
  --sam2_cfg model/sam2/sam2_configs/sam2_hiera_l.yaml \
  --train_roi_align \
  --seg_refine --memory_refine \
  --grad_accumulation_steps 20 \
  --no_eval \
  --auto_resume \
  --epochs 10
```

## Export: merge LoRA weights into a HF model folder

After training, you may want to convert the saved weights into a HF-style folder:

```bash
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version /path/to/base_llava_or_model_folder \
  --weight /path/to/pytorch_model.bin \
  --save_path /path/to/output_hf_model \
  --sam2 --sam2_checkpoint /path/to/sam2_hiera_large.pt \
  --sam2_cfg model/sam2/sam2_configs/sam2_hiera_l.yaml
```

## Inference examples

### Ref-Youtube-VOS

```bash
python inference_ytvos_uniform.py \
  --version /path/to/ckpt_or_hf_model_folder \
  --ytvos_path data/stage2_videos/ref-youtube-vos \
  --output_dir /path/to/output_dir \
  --ngpus 1 \
  --llm_sample_frames 10
```

### MeVIS / ReVOS

The commands are analogous; only the dataset path argument changes:

```bash
python inference_mevis_uniform.py --version /path/to/ckpt --mevis_path data/stage2_videos/MeVIS --output_dir /path/to/output_dir --ngpus 1 --llm_sample_frames 10
python inference_revos_uniform.py --version /path/to/ckpt --revos_path data/stage2_videos/ReVOS --output_dir /path/to/output_dir --ngpus 1 --llm_sample_frames 10
```
