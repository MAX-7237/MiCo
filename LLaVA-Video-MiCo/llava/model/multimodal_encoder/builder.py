from .siglip_encoder import SiglipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    name = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    if not name:
        raise ValueError("The video checkpoint must configure its SigLIP vision tower")
    return SiglipVisionTower(name, args=vision_tower_cfg, **kwargs)
