import base64
from io import BytesIO

import torch
from PIL import Image
from transformers import StoppingCriteria

import re

from .constants import IMAGE_TOKEN_INDEX, TRAJECTORY_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors="pt")["pixel_values"]


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):  
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


# def tokenizer_trajectory_token(
#     prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, trajectory_token_index=TRAJECTORY_TOKEN_INDEX, return_tensors=None
# ):  
#     prompt_chunks = [tokenizer(chunk).input_ids for chunk in re.split("<trajectory>|<image>", prompt)]
#     # test test <image> test test <trajectory> test -> [1, tokenizer(test)]
#     def insert_separator(X, sep):   # Input: X: [x1, x2], sep [-200, -200] -> Output: [x1, [-200, -200], x2]
#         return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]
#     # X: list, len = 2, 每个元素是一段话
#     # split chunks into two groups
#     # just_a_break = 1
#     prompt_chunks_0 = prompt_chunks[0:2]
#     # prompt_chunks_0[1] = prompt_chunks_0[1][:just_a_break]
#     prompt_chunks_1 = prompt_chunks[1:]
#     # prompt_chunks_1[0] = prompt_chunks_1[0][just_a_break:]

#     # <image>
#     input_ids_0 = []
#     offset = 0
#     if (
#         len(prompt_chunks_0) > 0
#         and len(prompt_chunks_0[0]) > 0                         # <image> token 前的序列
#         and prompt_chunks_0[0][0] == tokenizer.bos_token_id     # 该序列是否 starts with bos 
#     ):
#         offset = 1
#         input_ids_0.append(prompt_chunks_0[0][0])
#     # <image> 的插入结果，作为 <trajectory> 的起始段
#     for x in insert_separator(prompt_chunks_0, [image_token_index] * (offset + 1)):
#         input_ids_0.extend(x[offset:])

#     # <trajectory>
#     prompt_chunks_1[0] = input_ids_0
#     input_ids_1 = []
#     offset = 0
#     if (
#         len(prompt_chunks_1) > 0
#         and len(prompt_chunks_1[0]) > 0
#         and prompt_chunks_1[0][0] == tokenizer.bos_token_id
#     ):
#         offset = 1
#         input_ids_1.append(prompt_chunks_1[0][0])

#     for x in insert_separator(prompt_chunks_1, [trajectory_token_index] * (offset + 1)):
#         input_ids_1.extend(x[offset:])

#     input_ids = input_ids_1

#     # # <trajectory only>
#     # input_ids_traj_only = []
#     # offset = 0
#     # if (
#     #     len(prompt_chunks) > 0
#     #     and len(prompt_chunks[0]) > 0
#     #     and prompt_chunks[0][0] == tokenizer.bos_token_id
#     # ):
#     #     offset = 1
#     #     input_ids_traj_only.append(prompt_chunks[0][0])

#     # for x in insert_separator(prompt_chunks, [trajectory_token_index] * (offset + 1)):
#     #     input_ids_traj_only.extend(x[offset:])

#     if return_tensors is not None:
#         if return_tensors == "pt":
#             return torch.tensor(input_ids, dtype=torch.long)
#         raise ValueError(f"Unsupported tensor type: {return_tensors}")
#     return input_ids

def tokenizer_trajectory_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, trajectory_token_index=TRAJECTORY_TOKEN_INDEX, return_tensors=None
):
    # 分割prompt为不同块，处理图像和轨迹标记
    prompt_chunks = prompt.replace("<trajectory>", "<image>").split("<image>")
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks if chunk != ""]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    # 初始化input_ids列表和offset
    input_ids = []
    offset = 0
    if (len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # 根据分割符生成最终的input_ids
    prompt_separators = prompt.split("<image>")
    separators = []
    for chunk in prompt_separators:
        sub_chunks = chunk.split("<trajectory>")
        for _ in sub_chunks[:-1]:
            separators.append([trajectory_token_index])
        separators.append([image_token_index])
    separators.pop()  # 移除最后一个多余的image_token_index

    # 构建input_ids列表
    for x, sep in zip(prompt_chunks, separators + [[]] * (len(prompt_chunks) - len(separators))):
        input_ids.extend(x[offset:])
        input_ids.extend(sep)

    # 处理return_tensors参数
    if return_tensors is not None:
        if return_tensors == "pt":
            import torch
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    
    return input_ids



def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
