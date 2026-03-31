import argparse
import os
import random
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Sampler

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.dataset_vid import HybridDatasetVid, collate_fn as collate_fn_vid
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_SEG_END_TOKEN, DEFAULT_SEG_START_TOKEN, DEFAULT_TRAJ_END_TOKEN, DEFAULT_TRAJ_START_TOKEN, DEFAULT_VD_END_TOKEN, DEFAULT_VD_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
torch.autograd.set_detect_anomaly(True)

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    # Training mode rates:
    # Like sample rates
    # a,b
    # a: text to mask trajectory (traditional mode), b: trajector to text (tighten vision-language correspondence)
    parser.add_argument("--train_mode_rates", default="1,0", type=str)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    
    parser.add_argument(
        "--rvos_data", default="ytvos||mevis", type=str
    )
    
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--image_dir", default="./lisa", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
 
    parser.add_argument("--bert_base_uncased", default="bert-base-uncased", type=str)

    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--l1_loss_weight", default=0.5, type=float)
    parser.add_argument("--giou_loss_weight", default=2.0, type=float)
    parser.add_argument("--obj_loss_weight", default=0.5, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=True)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--train_projector_layer", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--use_multuple_seg_token", action="store_true", default=False)

    # parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    parser.add_argument(
        "--post_fusion_type",
        default="none",
        type=str,
        choices=["none", "nn", "gdino_uni", "gdino_bi", "gdino_bi_updated_img"],
    )
    
    # RVOS/RVOT数据集部分    
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of clip frames for training")
    # * Segmentation
    parser.add_argument('--masks', action='store_true', default=True,
                        help="Train segmentation head if the flag is provided")
    # dataset parameters
    # ['ytvos', 'davis', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasets for pretraining
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name') 
    parser.add_argument('--coco_path', type=str, default='refering_datasets/coco')
    parser.add_argument('--ytvos_path', type=str, default='refering_datasets/ref-youtube-vos')
    parser.add_argument('--a2d_path', type=str, default='refering_datasets/a2d_sentences')
    parser.add_argument('--mevis_path', type=str, default='refering_datasets/MeVIS')
    parser.add_argument('--davis_path', type=str, default='data/ref-davis')
    parser.add_argument('--tnl2k_path', type=str, default='refering_datasets/TNL2K/TNL2K_train_subset/train_data')
    parser.add_argument('--lasot_path', type=str, default='refering_datasets/LaSOT')
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=640, type=int, help="max size for the frame")
    
    parser.add_argument('--compress_type', default='mean', type=str, help="compress_type")
    parser.add_argument("--train_roi_align", action="store_true", default=False)
    parser.add_argument("--answer_type", default='1', type=str)  # 1：版本 1，2：版本 2，3：版本 3
    parser.add_argument("--stage", default=2, type=int)  # 1：版本 1，2：版本 2，3：版本 3

    # SAM 2
    parser.add_argument('--sam2', action='store_true', default=False, help="Train with sam2")
    parser.add_argument('--sam2_cfg', type=str, default='./model/sam2/sam2_hiera_l.yaml', help="sam2 config files")
    parser.add_argument('--sam2_checkpoint', type=str, default='./checkpoints/sam2_hiera_large.pt', help="sam2 checkpoint")
    
    parser.add_argument('--seg_refine', action='store_true', default=False, help="Train with sam2")
    parser.add_argument('--memory_refine', action='store_true', default=False, help="Train with sam2")
    
    parser.add_argument(
        "--llm_sample_mode",
        default="nosample",
        type=str,
        choices=["nosample", "uniform"],
    )
    
    return parser.parse_args(args)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    seed = 42
    set_seed(seed + args.local_rank)
    
    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # 原始的LLaMA-VID没有这两个特殊token，先不开
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VD_START_TOKEN, DEFAULT_VD_END_TOKEN, DEFAULT_TRAJ_START_TOKEN, DEFAULT_TRAJ_END_TOKEN], special_tokens=True
        )
        
    if args.use_multuple_seg_token:
        tokenizer.add_tokens(
            [DEFAULT_SEG_START_TOKEN, DEFAULT_SEG_END_TOKEN], special_tokens=True
        )

    # 在第二阶段及后续训练时,无论ckpt的config是否自带下面的参数, 都需在args重新指定, config会被args覆盖
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "train_roi_align": args.train_roi_align,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "l1_loss_weight": args.l1_loss_weight,
        "giou_loss_weight": args.giou_loss_weight,
        "obj_loss_weight": args.obj_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "use_multuple_seg_token": args.use_multuple_seg_token,
        "post_fusion_type": args.post_fusion_type,
        "sam2": args.sam2,
        "sam2_cfg": args.sam2_cfg,
        "sam2_checkpoint": args.sam2_checkpoint,
        "llm_sample_mode": args.llm_sample_mode,
        "seg_refine": args.seg_refine,
        "memory_refine": args.memory_refine,
        "mode": "train"
    }
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if args.train_roi_align:
        model.config.mm_vision_select_layer = [-11, -8, -5, -2]

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    if args.train_roi_align:
        model.get_model().initialize_spi_modules()
    
    model.config.model_path = args.version

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    
    if not args.eval_only and args.stage == 1:
        model.get_model().initialize_lisa_modules(model.get_model().config, mode='train')
        
    origin_cfg = AutoConfig.from_pretrained(args.version) if args.stage != 1 else None

    if args.post_fusion_type != 'none':
        if hasattr(origin_cfg, "post_fusion_type"):
            assert origin_cfg.post_fusion_type == args.post_fusion_type
            # 二阶段训练如果使用的预训练ckpt如果已经包含fusion module, 则不初始化
            if args.post_fusion_type in ['gdino_bi', 'gdino_bi_updated_img', 'gdino_uni']:
                print('Loading model fusion_layers.0.gamma_l and model.fusion_layers.0.gamma_v...')
                # 不知道什么原因, LISAForCausalLM.from_pretrained不加载下面2个参数(保存的模型文件是有这些参数的)，手动加载一下
                state_dict = torch.load(f'{args.version}/pytorch_model-00002-of-00002.bin', map_location=torch.device('cpu'))
                model.model.fusion_layers[0].gamma_l.data = state_dict['model.fusion_layers.0.gamma_l'].to(torch_dtype)
                model.model.fusion_layers[0].gamma_v.data = state_dict['model.fusion_layers.0.gamma_v'].to(torch_dtype)
        else:
            model.get_model().initialize_fusion_modules()
    
    if getattr(model.config, "sam2", False) and getattr(origin_cfg, "sam2", False):
        print('Loading model.visual_model.memory_encoder.fuser.layers.0.gamma and model.visual_model.memory_encoder.fuser.layers.1.gamma...')
        # 不知道什么原因, LISAForCausalLM.from_pretrained不加载下面2个参数(保存的模型文件是有这些参数的)，手动加载一下
        # 下面暂时直接加载sam2的ckpt，之后替换为训练ckpt的数据
        sam2_state_dict = torch.load(model.config.sam2_checkpoint, map_location=torch.device('cpu'))
        
        model.model.visual_model.memory_encoder.fuser.layers[0].gamma.data = sam2_state_dict['model']['memory_encoder.fuser.layers.0.gamma'].to(torch_dtype)
        model.model.visual_model.memory_encoder.fuser.layers[1].gamma.data = sam2_state_dict['model']['memory_encoder.fuser.layers.1.gamma'].to(torch_dtype)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "fusion_layers",
                                "spi_module",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    # <class 'peft.peft_model.PeftModelForCausalLM'>

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in (["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "spi_module", "fusion_layers"] if not args.train_projector_layer 
                          else ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "spi_module", "vlm_att_projector", "vlm_att_key_projector", "vlm_att_val_projector", "mm_projector", "fusion_layers"])
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    
    # for n, p in model.named_parameters():
    #     if 'fusion_layers' not in n:
    #         p.requires_grad=False
    
    print('-------------------------------------')            
    for n, p in model.named_parameters():
        if p.requires_grad==True:
            print(n, p.shape)
    
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    if args.stage == 1:
        train_dataset = HybridDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            samples_per_epoch=args.batch_size
            * args.grad_accumulation_steps
            * args.steps_per_epoch
            * world_size,
            precision=args.precision,
            image_size=args.image_size,
            num_classes_per_sample=args.num_classes_per_sample,
            exclude_val=args.exclude_val,
            dataset=args.dataset,
            sample_rate=[float(x) for x in args.sample_rates.split(",")],
            sem_seg_data=args.sem_seg_data,
            refer_seg_data=args.refer_seg_data,
            vqa_data=args.vqa_data,
            reason_seg_data=args.reason_seg_data,
            explanatory=args.explanatory,
        )
    elif args.stage == 2:
        # train_dataset = HybridDatasetVid(
        #     args = args, 
        #     base_video_dir = args.dataset_dir,
        #     tokenizer = tokenizer,
        #     vision_tower = args.vision_tower,
        #     samples_per_epoch = 500 * 8 * 2 * 10,
        #     precision = args.precision,
        #     image_size = args.image_size,
        #     num_classes_per_sample = args.num_classes_per_sample,
        #     exclude_val = False,
        #     dataset = args.dataset,
        #     sample_rate = [float(x) for x in args.sample_rates.split(",")],
        #     train_mode_rate = [float(x) for x in args.train_mode_rates.split(",")],
        #     rvos_data = args.rvos_data,
        #     rvot_data = "lasot||tnl2k",
        #     refseg_data = "refcoco||refcoco+||refcocog",
        #     explanatory = 0,
        #     answer_type = args.answer_type,
        # )
        
        train_dataset = HybridDatasetVid(
            args = args, 
            base_video_dir = args.dataset_dir,
            tokenizer = tokenizer,
            vision_tower = args.vision_tower,
            samples_per_epoch = 500 * 8 * 2 * 10,
            precision = args.precision,
            image_size = args.image_size,
            num_classes_per_sample = args.num_classes_per_sample,
            exclude_val = False,
            dataset = args.dataset,
            sample_rate = [float(x) for x in args.sample_rates.split(",")],
            train_mode_rate = [float(x) for x in args.train_mode_rates.split(",")],
            rvos_data = args.rvos_data,
            rvot_data = "lasot||tnl2k",
            refseg_data = "refcoco||refcoco+||refcocog",
            explanatory = 0,
            answer_type = args.answer_type,
            image_dataset = "sem_seg||refer_seg||vqa||reason_seg",
            base_image_dir = args.image_dir,
        )
    
    print('Initialised Done.')

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    # import pdb
    # pdb.set_trace()
    if args.stage == 1:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
            config=ds_config,
        )
    elif args.stage == 2:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn_vid,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                use_multuple_seg_token = args.use_multuple_seg_token,
                llm_sample_mode = args.llm_sample_mode,
                local_rank=args.local_rank,
            ),
            config=ds_config,
        )
        
    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn_vid,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                use_multuple_seg_token = args.use_multuple_seg_token,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    # save_dir = os.path.join('test_ckpt', "ckpt_model")
    # model_engine.save_checkpoint(save_dir)
    # exit(0)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
            train_dataset
        )

        if args.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
    train_dataset
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    obj_score_ce_losses = AverageMeter("ObjScoreCELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    # box_l1_losses = AverageMeter("BoxL1Loss", ":.4f")
    # box_giou_losses = AverageMeter("BoxGIOULoss", ":.4f")
    # box_losses = AverageMeter("BoxLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            obj_score_ce_losses,
            # box_l1_losses,
            # box_giou_losses,
            # box_losses
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            obj_score_ce_loss = output_dict["obj_score_ce_loss"]
            mask_loss = output_dict["mask_loss"]
            # box_l1_loss = output_dict["box_l1_loss"]
            # box_giou_loss = output_dict["box_giou_loss"]
            # box_loss = output_dict["box_loss"]
            
            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            obj_score_ce_losses.update(obj_score_ce_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            # box_l1_losses.update(box_l1_loss.item(), input_dict["images"].size(0))
            # box_giou_losses.update(box_giou_loss.item(), input_dict["images"].size(0))
            # box_losses.update(box_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                obj_score_ce_losses.all_reduce()
                mask_losses.all_reduce()
                # box_l1_losses.all_reduce()
                # box_giou_losses.all_reduce()
                # box_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/obj_score_ce_loss", obj_score_ce_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                # writer.add_scalar(
                #     "train/box_l1_loss", box_l1_losses.avg, global_step
                # )
                # writer.add_scalar(
                #     "train/box_giou_loss", box_giou_losses.avg, global_step
                # )
                # writer.add_scalar(
                #     "train/box_loss", box_losses.avg, global_step
                # )
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            obj_score_ce_losses.reset()
            mask_losses.reset()
            # box_l1_losses.reset()
            # box_giou_losses.reset()
            # box_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])