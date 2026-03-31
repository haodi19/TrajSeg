import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, image_transforms

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
						 DEFAULT_IMAGE_TOKEN, DEFAULT_VD_END_TOKEN, DEFAULT_VD_START_TOKEN, IMAGE_TOKEN_INDEX)
import requests
import json, time
from PIL import Image, ImageDraw
from io import BytesIO
from pathlib import Path

from decord import VideoReader, cpu

import multiprocessing as mp

from transformers import AutoConfig

from tqdm import tqdm


from utils.painter import colormap, mask_painter


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def parse_args(args):
	parser = argparse.ArgumentParser(description="LISA chat")
	parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
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
	parser.add_argument("--local-rank", default=0, type=int, help="node rank")
	parser.add_argument("--load_in_8bit", action="store_true", default=False)
	parser.add_argument("--load_in_4bit", action="store_true", default=False)
	parser.add_argument("--use_mm_start_end", action="store_true", default=True)
	parser.add_argument("--output_dir", type=str)
	parser.add_argument("--ngpus", type=int, default=1)
	parser.add_argument("--visualize", action="store_true", default=False)
	# YTVOS
	parser.add_argument("--ytvos_path", type=str, required=True)
	parser.add_argument("--split", type=str, default="valid", required=False)
 
	parser.add_argument(
        "--image_resize_type",
        default="clip",
        type=str,
        choices=["clip", "simple"],
    )

	parser.add_argument(
		"--conv_type",
		default="llava_v1",
		type=str,
		choices=["llava_v1", "llava_llama_2"],
	)
 
	parser.add_argument("--llm_sample_frames", type=int, default=12)
	return parser.parse_args(args)

def uniform_sample_frames(n, sample_frames=12):
    if n <= sample_frames:
        return [i for i in range(n)]
    
    target_frames = sample_frames

    # 计算理想采样间隔，并加入随机波动
    interval = n / target_frames
    sampled_indices = [0]  # 确保第一帧被采样到
    
    # 使用随机偏差进行采样，确保随机性
    for i in range(1, target_frames):  # 从第二帧开始采样
        index = int(i * interval + random.uniform(0, interval))  # 在每个间隔内随机选择
        index = min(index, n - 1)  # 确保索引不超过n
        sampled_indices.append(index)

    # 去重并排序
    sampled_indices = sorted(set(sampled_indices))

    # 如果采样后帧数不足或过多，进行调整
    while len(sampled_indices) < sample_frames:
        sampled_indices.append(random.choice(range(n)))  # 随机补充
        sampled_indices = sorted(set(sampled_indices))   # 去重排序

    while len(sampled_indices) > sample_frames:
        sampled_indices.pop(random.randint(1, len(sampled_indices) - 1))  # 随机移除

    return sampled_indices

def preprocess(
	x,
	pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
	pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
	img_size=1024,
) -> torch.Tensor:
	"""Normalize pixel values and pad to a square input."""
	# Normalize colors
	x = (x - pixel_mean) / pixel_std
	# Pad
	h, w = x.shape[-2:]
	padh = img_size - h
	padw = img_size - w
	x = F.pad(x, (0, padw, 0, padh))
	return x


def load_image(image_file):
	if image_file.startswith('http') or image_file.startswith('https'):
		response = requests.get(image_file)
		image = Image.open(BytesIO(response.content)).convert('RGB')
	else:
		image = Image.open(image_file).convert('RGB')
	return image

def load_video(video_path, fps=1):
	vr = VideoReader(video_path, ctx=cpu(0))
	fps = round(vr.get_avg_fps()/fps)
	frame_idx = [i for i in range(0, len(vr), fps)]
	spare_frames = vr.get_batch(frame_idx).asnumpy()
	return spare_frames

def load_images_from_dir(directory, fps=1, use_fps_sampling=False):
	# 列出目录下所有的png文件
	image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
	# 对文件名进行排序以保证图片的顺序
	image_files.sort()
	
	# 如果使用fps采样，按fps间隔读取图片
	if use_fps_sampling and fps > 1:
		image_files = image_files[::fps]
	
	images = []
	for filename in image_files:
		# 使用cv2读取每张图片
		img = cv2.imread(filename)
		# 将BGR转换为RGB（cv2默认以BGR格式加载图片）
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		images.append(img)
	
	# 将图片列表转换为Numpy数组
	images_array = np.array(images)
	return images_array

def load_first_frame(video_path):
	# 使用OpenCV打开视频
	cap = cv2.VideoCapture(video_path)
	
	# 检查视频是否打开成功
	if not cap.isOpened():
		print("Error: Could not open video.")
		return None
	
	# 读取第一帧
	ret, frame = cap.read()
	
	# 关闭视频文件
	cap.release()
	
	if ret:
		# 返回第一帧的图像数据
		return frame
	else:
		print("Error: Could not read frame.")
		return None

def load_all_frames(video_path):
	# 使用OpenCV打开视频
	cap = cv2.VideoCapture(video_path)
	
	# 检查视频是否打开成功
	if not cap.isOpened():
		print("Error: Could not open video.")
		return None
	
	frames = []  # 用于存储所有帧的列表
	
	while True:
		# 读取帧
		ret, frame = cap.read()
		
		# 检查是否成功读取帧
		if not ret:
			break  # 如果没有帧了，退出循环
		
		frames.append(frame)  # 将帧添加到列表中
	
	# 关闭视频文件
	cap.release()
	
	if frames:
		return frames  # 返回所有帧的列表
	else:
		print("Error: No frames read.")
		return None

def load_images_as_frames(directory):
	# 列出目录下所有的png文件
	image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg') ]
	# 对文件名进行排序以保证图片的顺序
	image_files.sort()
	
	frames = []
	for filename in image_files:
		# 使用cv2读取每张图片
		frame = cv2.imread(filename)
		if frame is not None:
			frames.append(frame)
	
	if frames:
		return frames  # 返回所有帧的列表
	else:
		print("Error: No frames read.")
		return None


# visuaize functions
def box_cxcywh_to_xyxy(x):
	x_c, y_c, w, h = x.unbind(1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
		 (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
	img_w, img_h = size
	b = box_cxcywh_to_xyxy(out_bbox)
	b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
	return b

def split_video_frames(total_frames, frames_per_group):
    # 初始化最佳步长和其对应的最大分组长度差值
    best_step = 1
    min_diff = float('inf')
    
    # 从1开始尝试每个步长，直到 total_frames
    for step in range(1, total_frames + 1):
        # 计算这个步长下最大的分组长度
        max_group_size = (total_frames - 1) // step + 1
        
        # 如果最大分组长度大于 frames_per_group，则不考虑这个步长
        if max_group_size > frames_per_group:
            continue
        
        # 计算与 frames_per_group 的差异，尽量使分组长度接近 frames_per_group
        diff = frames_per_group - max_group_size
        
        # 更新最佳步长
        if diff < min_diff:
            min_diff = diff
            best_step = step
    
    # 使用最佳步长进行分组
    groups = []
    for start in range(best_step):
        group = list(range(start, total_frames, best_step))
        if len(group) > frames_per_group:
            group = group[:frames_per_group]
        if group:
            groups.append(group)
    
    return groups

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
	os.makedirs(args.vis_save_path, exist_ok=True)
 
	seed = 42
	set_seed(seed + args.local_rank)
    
	torch.multiprocessing.set_start_method('spawn')

	"""
	Inference on Ref-YouTube-VOS
	1. load val/test videos
	2. for each video, infer all frames and save masks
	"""
	# 0. set save_paths
	output_dir = args.output_dir
	split = args.split
	save_path_prefix = os.path.join(output_dir, 'Annotations')
	if not os.path.exists(save_path_prefix):
		os.makedirs(save_path_prefix)

	save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
	if args.visualize:
		if not os.path.exists(save_visualize_path_prefix):
			os.makedirs(save_visualize_path_prefix)

	# 1. load data
	root = Path(args.ytvos_path) # data/ref-youtube-vos
	img_folder = os.path.join(root, split, "JPEGImages")
	meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
	with open(meta_file, "r") as f:
		data = json.load(f)["videos"]
	valid_test_videos = set(data.keys())
	# for some reasons the competition's validation expressions dict contains both the validation (202) & 
	# test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
	# the validation expressions dict:
	test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
	with open(test_meta_file, 'r') as f:
		test_data = json.load(f)['videos']
	test_videos = set(test_data.keys())
	valid_videos = valid_test_videos - test_videos
	video_list = sorted([video for video in valid_videos])
 
	# done_videos = os.listdir('/hhd2/ljn/tmp_lisa/final_lisa/Lisa2/rvos_sam2_video_v8_fusion_500_revos/Annotations')
	# video_list = [v for v in video_list if v not in done_videos]

	# print(len(video_list))
 
 
	# assert len(video_list) == 202, 'error: incorrect number of validation videos'

	# 2. inference, in subprocesses
	thread_num = args.ngpus
	# global result_dict				# not defined in spawned processor
	# result_dict = mp.Manager().dict()

	processes = []
	# lock = threading.Lock()

	video_num = len(video_list)
	per_thread_video_num = video_num // thread_num

	start_time = time.time()
	print('Start inference')
	for i in range(thread_num):
		if i == thread_num - 1:
			sub_video_list = video_list[i * per_thread_video_num:]
		else:
			sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
		p = mp.Process(target=sub_processor, args=(i, args, data, 
												   save_path_prefix, save_visualize_path_prefix, 
												   img_folder, sub_video_list))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

	end_time = time.time()
	total_time = end_time - start_time

	# result_dict = dict(result_dict)
	# num_all_frames_gpus = 0
	# for pid, num_all_frames in result_dict.items():
	# 	num_all_frames_gpus += num_all_frames

	print("Total inference time: %.4f s" %(total_time))


def sub_processor(pid, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list):
	text = 'processor %d' % pid
	# with lock:	# will cause: cannot pickle '_thread.lock' object (when using lock)
	# 	progress = tqdm(
	# 		total=len(video_list),
	# 		position=pid,
	# 		desc=text,
	# 		ncols=0
	# 	)
	progress = tqdm(
		total=len(video_list),
		position=pid,
		desc=text,
		ncols=0
	)
	torch.cuda.set_device(pid)

	key_frame_save_file =  os.path.join(args.output_dir, f'key_frames{pid}.txt')

	# Create model
	# <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>
	tokenizer = AutoTokenizer.from_pretrained(
		args.version,
		cache_dir=None,
		model_max_length=args.model_max_length,
		padding_side="right",
		use_fast=False,
	)
	# tokenizer.pad_token: '<unk>'
	# args.seg_token_idx: 32003
	tokenizer.pad_token = tokenizer.unk_token
	# 这一步是添加[SEG] token的操作
	args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
	torch_dtype = torch.float32
	if args.precision == "bf16":
		torch_dtype = torch.bfloat16
	elif args.precision == "fp16":
		torch_dtype = torch.half

	kwargs = {"torch_dtype": torch_dtype}
	if args.load_in_4bit:
		kwargs.update(
			{
				"torch_dtype": torch.half,
				"load_in_4bit": True,
				"quantization_config": BitsAndBytesConfig(
					load_in_4bit=True,
					bnb_4bit_compute_dtype=torch.float16,
					bnb_4bit_use_double_quant=True,
					bnb_4bit_quant_type="nf4",
					llm_int8_skip_modules=["visual_model"],
				),
			}
		)
	elif args.load_in_8bit:
		kwargs.update(
			{
				"torch_dtype": torch.half,
				"quantization_config": BitsAndBytesConfig(
					llm_int8_skip_modules=["visual_model"],
					load_in_8bit=True,
				),
			}
		)


	model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, seg_token_idx=args.seg_token_idx, **kwargs
    )
 
	if getattr(model.config, "post_fusion_type", "none") in ["gdino_uni", "gdino_bi", "gdino_bi_updated_img"]:
		# 不知道什么原因, LISAForCausalLM.from_pretrained不加载下面2个参数(保存的模型文件是有这些参数的)，手动加载一下
		print('Loading model.fusion_layers.0.gamma_l and model.fusion_layers.0.gamma_v...')
		state_dict = torch.load(f'{args.version}/pytorch_model-00002-of-00002.bin', map_location=torch.device('cpu'))
		
		model.model.fusion_layers[0].gamma_l.data = state_dict['model.fusion_layers.0.gamma_l'].to(torch_dtype)
		model.model.fusion_layers[0].gamma_v.data = state_dict['model.fusion_layers.0.gamma_v'].to(torch_dtype)
 
	if getattr(model.config, "sam2", False):
		print('Loading model.visual_model.memory_encoder.fuser.layers.0.gamma and model.visual_model.memory_encoder.fuser.layers.1.gamma...')
        # 不知道什么原因, LISAForCausalLM.from_pretrained不加载下面2个参数(保存的模型文件是有这些参数的)，手动加载一下
		sam2_state_dict = torch.load(model.config.sam2_checkpoint, map_location=torch.device('cpu'))
        
		model.model.visual_model.memory_encoder.fuser.layers[0].gamma.data = sam2_state_dict['model']['memory_encoder.fuser.layers.0.gamma'].to(torch_dtype)
		model.model.visual_model.memory_encoder.fuser.layers[1].gamma.data = sam2_state_dict['model']['memory_encoder.fuser.layers.1.gamma'].to(torch_dtype)
 
	model.config.eos_token_id = tokenizer.eos_token_id
	model.config.bos_token_id = tokenizer.bos_token_id
	model.config.pad_token_id = tokenizer.pad_token_id
	model.config.model_path = args.version
	
	model.get_model().initialize_vision_modules(model.get_model().config)
	
	vision_tower = model.get_model().get_vision_tower()
	vision_tower.to(dtype=torch_dtype)

	if args.precision == "bf16":
		model = model.bfloat16().cuda(pid)
	elif (
		args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
	):
		vision_tower = model.get_model().get_vision_tower()
		model.model.vision_tower = None
		import deepspeed

		model_engine = deepspeed.init_inference(
			model=model,
			dtype=torch.half,
			replace_with_kernel_inject=True,
			replace_method="auto",
		)
		model = model_engine.module
		model.model.vision_tower = vision_tower.half().cuda(pid)
	elif args.precision == "fp32":
		model = model.float().cuda(pid)

	vision_tower = model.get_model().get_vision_tower()
	vision_tower.to(device=f'cuda:{pid}')

	num_all_frames = 0
	clip_image_processor = vision_tower.image_processor
	transform = ResizeLongestSide(args.image_size)
	model.eval()
	# end of initialisation
	# TODO: move initialisation into an individual module

	print(f'Inference on device: {pid}')

	key_frame_str = ''	
 
	# 1. For each video
	for video in video_list:
		metas = [] # list[dict], length is number of expressions

		expressions = data[video]["expressions"]   
		expression_list = list(expressions.keys()) 
		num_expressions = len(expression_list)
		video_len = len(data[video]["frames"])

		# read all the anno meta
		for i in range(num_expressions):
			meta = {}
			meta["video"] = video
			meta["exp"] = expressions[expression_list[i]]["exp"]
			meta["exp_id"] = expression_list[i]
			meta["frames"] = data[video]["frames"]
			metas.append(meta)
		meta = metas

		# 2. For each expression
		for i in range(num_expressions):
			video_name = meta[i]["video"]
			exp = meta[i]["exp"]
			exp_id = meta[i]["exp_id"]
			frame_names = meta[i]["frames"]

			video_len = len(frame_names)
   
   			# Convert expressions to conversations
			conv = conversation_lib.conv_templates[args.conv_type].copy()
			conv.messages = []

			prompt = "Please segment the " + exp + " in this video."
			
			prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

			# load frames
			frame_path = os.path.join(img_folder, video_name)
			frames = load_images_from_dir(frame_path, use_fps_sampling=False)
			_, origin_h, origin_w, _ = frames.shape
   
			if args.image_resize_type == 'clip':
				image_clip = clip_image_processor.preprocess(frames, return_tensors='pt')['pixel_values'].half().cuda()
			elif args.image_resize_type == 'simple':
				final_size = clip_image_processor.size["shortest_edge"]
				image_clip = np.array([image_transforms.resize(img, size=(final_size, final_size), resample = clip_image_processor.resample) for img in frames])
				image_clip = clip_image_processor.preprocess(image_clip, do_resize=False, do_center_crop=False, return_tensors='pt')['pixel_values'].half().cuda()
			
			image_clip = [image_clip]
			# image_np: [(1080, 1920, 3), (1080, 1920, 3)...]
			images_np = load_images_as_frames(frame_path)
			images_np = [cv2.cvtColor(inp, cv2.COLOR_BGR2RGB) for inp in images_np]
			original_size_list = [images_np[0].shape[:2]]
			
			if args.precision == "bf16":
				image_clip[0] = image_clip[0].bfloat16()
			elif args.precision == "fp16":
				image_clip[0] = image_clip[0].half()
			else:
				image_clip[0] = image_clip[0].float()

			video = [transform.apply_image(inp) for inp in images_np]
			# video: [(576, 1024, 3), (576, 1024, 3)...]
			resize_list = [video[0].shape[:2]]

			video = [(
				preprocess(torch.from_numpy(img).permute(2, 0, 1).contiguous())
				.unsqueeze(0)
				.cuda(pid)
			) for img in video]
			
			video = torch.stack(video).permute(1, 0, 2, 3, 4).cuda(pid)
			
			# video: torch.Size([1, video_len, 3, 1024, 1024])
			if args.precision == "bf16":
				video = video.bfloat16()
			elif args.precision == "fp16":
				video = video.half()
			else:
				video = video.float()

			total_video_len = video.shape[1]
			sampled_frames_cnt = args.llm_sample_frames
			# samples_indices: list, 记录所有采样帧的索引, 默认采12帧, 如[0, 1, 3, 5, 7, 8, 10, 11, 13, 14, 16, 18]
			samples_indices = uniform_sample_frames(total_video_len, sampled_frames_cnt)		

			key_frame_str += f"{video_name},{exp_id}:{','.join(map(str, samples_indices))}\n"

			sub_video_len = len(samples_indices)
			sub_video = video[:,samples_indices]
			sub_image_clip = [image_clip[0][samples_indices]]

			if args.use_mm_start_end:
				replace_token = DEFAULT_IMAGE_TOKEN
				replace_token_vid = (
						DEFAULT_VD_START_TOKEN + replace_token + DEFAULT_VD_END_TOKEN
					)
				replace_token_img = (
						DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
					)
				
				prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token_vid)
				prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * sub_video_len)
				prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token_img)

			conv.append_message(conv.roles[0], prompt)
			conv.append_message(conv.roles[1], "")

			# prompt: "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nhello ASSISTANT:"
			prompt = conv.get_prompt()
			# input_ids: tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
			#                     21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
			#                       322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
			#                     29889,  3148,  1001, 29901, 32001,  -200, 32002, 22172,   319,  1799,
			#                      9047, 13566, 29901]], device='cuda:0')
			input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
			input_ids = input_ids.unsqueeze(0).cuda(pid)

			# 并不保证LLM一定输出<SEG>token, 所以需要文本监督训练
			output_ids, pred_masks, pred_boxes = model.evaluate(
				sub_image_clip,
				video,
				input_ids,
				resize_list,
				original_size_list,
				max_new_tokens=512,
				tokenizer=tokenizer,
                llm_sample_mode = 'uniform',
            	samples_indices = samples_indices,
			)
			# output_ids: tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
			#                      21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
			#                        322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
			#                      29889,  3148,  1001, 29901, 32001,  -200, 32002, 22172,   319,  1799,
			#                       9047, 13566, 29901, 18585, 29892,   372,   338, 32003,   869,     2]],
			#                    device='cuda:0')
			# <s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start> <im_end> Don't generate a [SEG] token ASSISTANT: Sure, the segmentation result is [SEG] .</s>           
			# pred_masks: list[torch.Size([video_len, 1, 1080, 1920])], 可能生成多个masks

			all_pred_masks = pred_masks[0].detach().cpu().numpy()
			output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

			text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
			text_output = text_output.replace("\n", "").replace("  ", " ")


			if 'SEG' in text_output:
				if args.visualize:
					for t, frame_name in enumerate(frame_names):
						# original
						img_path = os.path.join(img_folder, video_name, frame_name + '.jpg')
						source_img = Image.open(img_path).convert('RGBA') 	# PIL image
						draw = ImageDraw.Draw(source_img)

						# draw mask
						pred_mask = all_pred_masks[t] > 0
						mask = pred_mask[0].astype(np.uint8) * 255
						source_img = mask_painter(np.array(source_img.convert("RGB")), mask, i+3)

						# save
						save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video_name, str(i))
						if not os.path.exists(save_visualize_path_dir):
							os.makedirs(save_visualize_path_dir)
						save_visualize_path = os.path.join(save_visualize_path_dir, frame_name + '.png')
						Image.fromarray(source_img).save(save_visualize_path)

				# save binary image
				save_path = os.path.join(save_path_prefix, video_name, exp_id)
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				for j in range(video_len):
					frame_name = frame_names[j]
					pred_mask = all_pred_masks[j] > 0
					mask = pred_mask[0].astype(np.float32)
					mask = Image.fromarray(mask * 255).convert('L')
					save_file = os.path.join(save_path, frame_name + ".png")
					mask.save(save_file)
			else:
				# No [SEG], save empty masks
				print(f'empty predictions! {os.path.join(img_folder, video_name)}')
				img_path = os.path.join(img_folder, video_name, frame_names[0] + '.jpg')
				source_img = Image.open(img_path).convert('RGBA') # PIL image
				width, height = source_img.size
				empty_mask = np.zeros((height, width))
	
				# save binary image
				save_path = os.path.join(save_path_prefix, video_name, exp_id)
				if not os.path.exists(save_path):
					os.makedirs(save_path)
	
				for j in range(video_len):
					frame_name = frame_names[j]
					mask = Image.fromarray(empty_mask * 255).convert('L')
					save_file = os.path.join(save_path, frame_name + ".png")
					mask.save(save_file)

		# with lock:	# pickle problem in spawned process
		progress.update(1)
	
	with open(f"{key_frame_save_file}", "w", encoding="utf-8") as file:
		file.write(key_frame_str)
	
 	# result_dict[str(pid)] = num_all_frames	# not defined in spawned process
	# with lock:		# pickle problem in spawned process
	progress.close()


if __name__ == "__main__":
	main(sys.argv[1:])
