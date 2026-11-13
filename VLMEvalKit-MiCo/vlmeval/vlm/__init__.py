import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .qwen2_5_vl import Qwen2_5VLChat
from .internvl3 import InternVL3Chat
