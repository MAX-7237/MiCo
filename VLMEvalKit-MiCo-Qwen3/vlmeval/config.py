from functools import partial
from .vlm import Qwen3VLChat

supported_VLM = {
    "Qwen3-VL-8B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        min_pixels=1310720,
        max_pixels=16777216,
        max_new_tokens=2048,
    ),
}
