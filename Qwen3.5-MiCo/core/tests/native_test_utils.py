"""Random-weight native32 CPU qualification of the new controller interface."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from unified_core import install_pruning
from linear_beta import linear_beta_quality


def model():
    torch.manual_seed(3519)
    cfg = Qwen3_5Config(image_token_id=61, video_token_id=63, vision_start_token_id=60, vision_end_token_id=62,
        text_config=dict(vocab_size=80, hidden_size=32, intermediate_size=48, num_hidden_layers=32,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            layer_types=(['linear_attention']*3+['full_attention'])*8,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4, pad_token_id=0, eos_token_id=None,
            rope_parameters=dict(rope_type='default', rope_theta=10000, partial_rotary_factor=.25,
                                 mrope_section=[1,1,2])),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=48, num_heads=4,
            patch_size=2, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32,
            num_position_embeddings=64))
    cfg._attn_implementation = cfg.text_config._attn_implementation = cfg.vision_config._attn_implementation = 'sdpa'
    return Qwen3_5ForConditionalGeneration(cfg).eval()


def inputs(grid=8):
    n = grid*grid//4
    ids = torch.tensor([[1,60]+[61]*n+[62]+list(range(2,15))])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                image_grid_thw=torch.tensor([[1,grid,grid]]),
                pixel_values=torch.randn(grid*grid,24,generator=torch.Generator().manual_seed(392+grid)))


def mask(data, scope):
    out = torch.zeros_like(data['input_ids'], dtype=torch.bool)
    out[:, -4 if scope == 'task_content' else -13:] = True
    return out


GEN = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0,
           output_logits=True, return_dict_in_generate=True)


