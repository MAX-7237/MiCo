from functools import partial
from .vlm import Qwen2_5VLChat, InternVL3Chat

supported_VLM = {
    "Qwen2.5-VL-7B-Instruct": partial(
        Qwen2_5VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=1003520,
        max_pixels=12845056,
    ),
    "InternVL3-8B": partial(InternVL3Chat, model_path="OpenGVLab/InternVL3-8B"),
}
