import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    name = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    if name and (
        os.path.exists(name)
        or name.startswith("openai")
        or name.startswith("laion")
        or "clip" in name.lower()
        or "ShareGPT4V" in name
    ):
        return CLIPVisionTower(name, args=vision_tower_cfg, **kwargs)
    raise ValueError(f"Unknown vision tower: {name}")
