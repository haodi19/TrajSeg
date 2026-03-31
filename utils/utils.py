from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
TRAJECTORY_TOKEN_INDEX = -300
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VD_START_TOKEN = "<vd_start>"
DEFAULT_VD_END_TOKEN = "<vd_end>"

DEFAULT_TRAJ_START_TOKEN = "<traj_start>"
DEFAULT_TRAJ_END_TOKEN = "<traj_end>"

DEFAULT_SEG_START_TOKEN = "<seg_start>"
DEFAULT_SEG_END_TOKEN = "<seg_end>"

DEFAULT_TRAJECTORY_TOKEN = "<trajectory>"

DEFAULT_P_START_TOKEN = "<p>"
DEFAULT_P_END_TOKEN = "</p>"

# SEG_IDX_TOKENS = ['<seg{}>'.format(i) for i in range(0, 35)]

PREFIX = "This is a video of {video_len} frames in length. "

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this video?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this video.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this video? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this video? Please output segmentation mask.",
]

SHORT_QUESTION_MULTI_SEG_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + PREFIX + "Can you segment the {class_name} in this video? Please respond by outputting {video_len} masks.",
    DEFAULT_IMAGE_TOKEN + "\n" + PREFIX + "Please segment the {class_name} in this video by outputting {video_len} masks.",
    DEFAULT_IMAGE_TOKEN
    + "\n" + PREFIX
    + "What is {class_name} in this video? Please respond with {video_len} segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n" + PREFIX
    + "What is {class_name} in this video? Please output {video_len} segmentation mask.",
]


LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

NEW_MODE_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you describe " + DEFAULT_TRAJECTORY_TOKEN + " in this video?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please describe " + DEFAULT_TRAJECTORY_TOKEN + " in this video.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is " + DEFAULT_TRAJECTORY_TOKEN + " in this video? Please respond with detailed description.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is " + DEFAULT_TRAJECTORY_TOKEN + " in this video? Please provide a thorough description.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please provide a detailed description of " + DEFAULT_TRAJECTORY_TOKEN + " in the video."
]


NEW_MODE_ANSWER_LIST = [
    "Sure, {class_name} [SEG].",
    "{class_name} [SEG]."
]


ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

MR_SINGLE_ANSWER_LIST = [
    "{class_name} is [SEG].",
]

MR_MULTI_ANSWER_LIST = [
    "{class_name} are {seg}, separately.",
    "{class_name} are {seg}.",
    "Sure, {class_name} are {seg}, separately.",
    "Sure, {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}, separately.",
    "Sure, the segmentation result of {class_name} are {seg}.",
    "Sure, the segmentation result of {class_name} are {seg}, separately.",
]


EXPAND_LONG_QUESTION_LIST = [

    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Provide the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please show the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} I'd appreciate segmentation masks.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please highlight the segmentation mask.",

]

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            # and isinstance(input_dict[k][0], torch.Tensor)
            and any(isinstance(item, torch.Tensor) for item in input_dict[k])
        ):  
            input_dict[k] = [ele.cuda(non_blocking=True) if ele is not None else None for ele in v]
    return input_dict
