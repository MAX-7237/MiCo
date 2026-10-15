#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random



@torch.jit.script
def _greedy_select_jit(relevance: torch.Tensor, similarity: torch.Tensor,
                       tokens_to_select: int, rel_w: float, div_w: float) -> torch.Tensor:
    """JIT-compiled greedy diversity-aware token selection."""
    T = relevance.shape[0]
    N = relevance.shape[1]
    device = relevance.device
    available = torch.ones(T, N, dtype=torch.bool, device=device)
    selected = torch.zeros(T, tokens_to_select, dtype=torch.long, device=device)
    max_sim = torch.full((T, N), float("-inf"), device=device, dtype=relevance.dtype)
    weighted_rel = rel_w * relevance
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=relevance.dtype)
    arange_T = torch.arange(T, device=device)

    for step in range(tokens_to_select):
        if step == 0:
            scores = relevance.clone()
        else:
            scores = weighted_rel + div_w * (1.0 - max_sim)
        scores = torch.where(available, scores, neg_inf)
        best = scores.argmax(dim=1)
        selected[:, step] = best
        available[arange_T, best] = False
        best_exp = best.unsqueeze(1).unsqueeze(2).expand(T, N, 1)
        new_sim = torch.gather(similarity, 2, best_exp).squeeze(2)
        max_sim = torch.max(max_sim, new_sim)
    return selected

class LlavaMetaModel:

    def __init__(self, config, **kwargs):
        super(LlavaMetaModel, self).__init__(config, **kwargs)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        if self.pruning_method in ('fastervlm', 'cdpruner', 'tops', 'attn_cov_proj_llm_12', 'attn_H_cov_proj_llm_12'):
            image_features, image_attentions, image_keys = self.get_model().get_vision_tower()(images, output_attentions=True)
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_attentions, image_keys = None, None
        # MMTOK_VIDEO_PORT: retain the current encoder features for Q.
        if self.pruning_method == 'mmtok':
            self._mmtok_preprojector = image_features
        image_features = self.get_model().mm_projector(image_features)
        return image_features, image_attentions, image_keys
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)  # (T, 1, H, W, D)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()  # (D, T, H, 1, W)
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)  # (D, T*H, W)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            return image_feature
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, modalities=["image"], image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, 0

        if isinstance(modalities, str):
            modalities = [modalities]

        # Debug: Show current values at the start of prepare_inputs
        rank0_print("="*80)
        rank0_print(f"[llava_arch.prepare_inputs] 函数开始执行")
        rank0_print(f"[llava_arch] self.pruning_method = {getattr(self, 'pruning_method', 'NOT_FOUND')}")
        rank0_print(f"[llava_arch] self.visual_token_num = {getattr(self, 'visual_token_num', 'NOT_FOUND')}")
        rank0_print("="*80)

        # Set default visual_token_num if not specified
        if self.visual_token_num is None and self.pruning_method != 'vanilla':
            self.visual_token_num = 196  # Default value for pruning
            rank0_print(f"[警告] visual_token_num未设置，使用默认值: {self.visual_token_num}")

        # IMPORTANT: visual_token_num should be set via command line, but if it's 169 (default vision token count),
        # it likely means the parameter wasn't passed correctly. For now, keep it as is to allow testing.
        # Users should pass visual_token_num explicitly in model_args.

        rank0_print(f"[prune-debug] method={self.pruning_method}, visual_token_num={self.visual_token_num}")

        if self.pruning_method in ('attn_cov_proj_llm_12', 'attn_H_cov_proj_llm_12'):
            if input_ids.shape[0] != 1 or modalities != ['video'] or len(images) != 1:
                raise ValueError('attn_cov Video requires one video per batch and modalities=[video]')
            self._attn_cov_pending = None

        # Force flat mode if using pruning
        if self.pruning_method != 'vanilla':
            original_merge_type = getattr(self.config, 'mm_patch_merge_type', 'unknown')
            if original_merge_type != 'flat':
                rank0_print(f"[强制修改] mm_patch_merge_type 从 '{original_merge_type}' 改为 'flat' (剪枝需要)")
                self.config.mm_patch_merge_type = 'flat'

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features, encoded_image_attentions, encoded_image_keys = self.encode_images(concat_images) # (T, N, D), (T, N, N), (T, N, C)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            if encoded_image_attentions is not None:
                encoded_image_attentions = torch.split(encoded_image_attentions, split_sizes)
            if encoded_image_keys is not None:
                encoded_image_keys = torch.split(encoded_image_keys, split_sizes)
            image_features, image_attentions, image_keys = [], [], []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                    if encoded_image_attentions is not None:
                        image_attentions.append(self.get_2dPool(self.get_2dPool(encoded_image_attentions[idx]).transpose(1, 2)).transpose(1, 2))
                    else:
                        image_attentions.append(None)
                    if encoded_image_keys is not None:
                        image_keys.append(self.get_2dPool(encoded_image_keys[idx]))
                    else:
                        image_keys.append(None)
                else:
                    image_features.append(image_feat)
                    image_attentions.append(None)
                    image_keys.append(None)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                rank0_print(f"[Visual Token Pruning] 剪枝方法: {self.pruning_method}")
                new_image_features = []
                # Visual Token Pruning
                for image_feature, image_attention, image_key in zip(image_features, image_attentions, image_keys):
                    T, N, D = image_feature.shape
                    device = image_feature.device

                    if self.pruning_method == 'fastervlm':
                        rank0_print("="*60)
                        rank0_print(f"✓ [FasteRVLM] 开始执行剪枝")
                        rank0_print(f"  剪枝前: 每帧token数 = {N}, 总帧数 = {T}, 总tokens = {T*N}")
                        rank0_print(f"  目标: 每帧保留 {self.visual_token_num} 个tokens")

                        cls_attn = image_attention.mean(dim=1)

                        topk_idx = cls_attn.topk(self.visual_token_num, dim=-1).indices
                        topk_idx = torch.sort(topk_idx).values

                        image_feature = image_feature[torch.arange(T).unsqueeze(-1), topk_idx]

                        rank0_print(f"  剪枝后: 每帧token数 = {image_feature.shape[1]}, 总tokens = {T*image_feature.shape[1]}")
                        rank0_print(f"  压缩率: {(1 - image_feature.shape[1]/N)*100:.1f}% (减少了 {N - image_feature.shape[1]} tokens/frame)")
                        rank0_print("="*60)
                    
                    elif self.pruning_method == 'trim':
                        rank0_print("="*60)
                        rank0_print(f"✓ [TRIM] 开始执行剪枝")
                        rank0_print(f"  剪枝前: 每帧token数 = {N}, 总帧数 = {T}, 总tokens = {T*N}")
                        rank0_print(f"  目标: 每帧保留 {self.visual_token_num} 个tokens")

                        image_embed = image_feature
                        text_embed = self.get_model().embed_tokens(input_ids[:, 15:]).mean(dim=1)

                        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
                        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                        similarity = torch.matmul(image_embed, text_embed.t()).squeeze(-1)

                        # select topk tokens by similarity scores
                        topk_idx = similarity.topk(self.visual_token_num - 1, dim=-1).indices
                        topk_idx = torch.sort(topk_idx).values
                        selected_tokens = image_feature[torch.arange(T).unsqueeze(-1), topk_idx]

                        # aggregate the remaining tokens
                        remaining_mask = torch.ones_like(similarity, dtype=torch.bool).scatter_(1, topk_idx, False)
                        remaining_tokens = image_feature.masked_select(remaining_mask.unsqueeze(-1)).view(T, N - self.visual_token_num + 1, D)
                        aggregated_token = remaining_tokens.mean(dim=1, keepdim=True)

                        image_feature = torch.cat([selected_tokens, aggregated_token], dim=1)

                        rank0_print(f"  剪枝后: 每帧token数 = {image_feature.shape[1]}, 总tokens = {T*image_feature.shape[1]}")
                        rank0_print(f"  压缩率: {(1 - image_feature.shape[1]/N)*100:.1f}% (减少了 {N - image_feature.shape[1]} tokens/frame)")
                        rank0_print("="*60)
                    
                    elif self.pruning_method == 'dart':
                        rank0_print("="*60)
                        rank0_print(f"✓ [DART] 开始执行剪枝")
                        rank0_print(f"  剪枝前: 每帧token数 = {N}, 总帧数 = {T}, 总tokens = {T*N}")
                        rank0_print(f"  目标: 每帧保留 {self.visual_token_num} 个tokens")

                        image_normalized = image_feature / image_feature.norm(dim=-1, keepdim=True)
                        visual_token_num = (self.visual_token_num - 4) // 4
                        step = N // (4 + 1) + 1

                        retained_idx = []
                        for t in range(T):
                            retained_set = set(range(step, N, step))
                            valid_set = set(range(N)) - retained_set
                            for pivot_idx in list(retained_set):
                                valid_list = list(valid_set)
                                cos_sim = torch.matmul(image_normalized[t, pivot_idx], image_normalized[t, valid_list].t())
                                topk_idx = torch.topk(cos_sim, k=visual_token_num, largest=False).indices

                                topk_real_idx = [valid_list[idx] for idx in topk_idx]
                                retained_set.update(topk_real_idx)
                                valid_set.difference_update(topk_real_idx)
                            retained_idx.append(list(retained_set))

                        retained_idx = torch.tensor(retained_idx, dtype=torch.long, device=device)
                        retained_idx = torch.sort(retained_idx).values
                        image_feature = image_feature[torch.arange(T).unsqueeze(-1), retained_idx]

                        rank0_print(f"  剪枝后: 每帧token数 = {image_feature.shape[1]}, 总tokens = {T*image_feature.shape[1]}")
                        rank0_print(f"  压缩率: {(1 - image_feature.shape[1]/N)*100:.1f}% (减少了 {N - image_feature.shape[1]} tokens/frame)")
                        rank0_print("="*60)
                    
                    elif self.pruning_method == 'divprune':
                        rank0_print("="*60)
                        rank0_print(f"✓ [DivPrune] 开始执行剪枝")
                        rank0_print(f"  剪枝前: 每帧token数 = {N}, 总帧数 = {T}, 总tokens = {T*N}")
                        rank0_print(f"  目标: 每帧保留 {self.visual_token_num} 个tokens")

                        image_normalized = image_feature / image_feature.norm(dim=-1, keepdim=True)
                        cosine_matrix = 1.0 - torch.matmul(image_normalized, image_normalized.transpose(1, 2))

                        select_idx = torch.empty((T, self.visual_token_num), dtype=torch.long, device=device)
                        for i in range(self.visual_token_num):
                            m2 = cosine_matrix if i==0 \
                                else cosine_matrix[torch.arange(T).unsqueeze(1).expand(-1, i), select_idx[:, :i]]
                            scores = torch.topk(m2, 2, dim=1, largest=False).values[:, 1] \
                                if i==0 else torch.min(m2, dim=1).values
                            select_idx[:, i] = torch.argmax(scores, dim=-1)

                        select_idx = torch.sort(select_idx).values
                        image_feature = image_feature[torch.arange(T).unsqueeze(-1), select_idx]

                        rank0_print(f"  剪枝后: 每帧token数 = {image_feature.shape[1]}, 总tokens = {T*image_feature.shape[1]}")
                        rank0_print(f"  压缩率: {(1 - image_feature.shape[1]/N)*100:.1f}% (减少了 {N - image_feature.shape[1]} tokens/frame)")
                        rank0_print("="*60)
                    
                    elif self.pruning_method == 'dppruner':
                        rank0_print("="*60)
                        rank0_print(f"✓ [DPPruner] 开始执行剪枝")
                        rank0_print(f"  剪枝前: 每帧token数 = {N}, 总帧数 = {T}, 总tokens = {T*N}")
                        rank0_print(f"  目标: 每帧保留 {self.visual_token_num} 个tokens")

                        image_normalized = image_feature / image_feature.norm(dim=-1, keepdim=True) # (T, N, D)
                        similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (T, N, N)

                        kernel = similarity

                        cis = torch.zeros((self.visual_token_num, T, N), device=device) # (M, T, N)
                        di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (T, N)
                        select_idx = torch.empty((self.visual_token_num, T), dtype=torch.long, device=device) # (N, T)
                        for i in range(self.visual_token_num):
                            j = torch.argmax(di2s, dim=-1)
                            select_idx[i] = j

                            eis = (kernel[torch.arange(T), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(T), j], cis[:i])) \
                                / torch.sqrt(di2s[torch.arange(T), j]).unsqueeze(-1)
                            cis[i, :, :] = eis
                            di2s -= torch.square(eis)
                            di2s[torch.arange(T), j] = -float('inf')

                        select_idx = torch.sort(select_idx.t()).values # (T, M)
                        image_feature = image_feature[torch.arange(T).unsqueeze(-1), select_idx]

                        rank0_print(f"  剪枝后: 每帧token数 = {image_feature.shape[1]}, 总tokens = {T*image_feature.shape[1]}")
                        rank0_print(f"  压缩率: {(1 - image_feature.shape[1]/N)*100:.1f}% (减少了 {N - image_feature.shape[1]} tokens/frame)")
                        rank0_print("="*60)
                    
                    elif self.pruning_method == 'mmtok':
                        if len(image_features) != 1 or video_idx_in_batch != [0]:
                            raise ValueError('MMTok video port supports exactly one video per request')
                        from llava.model.mmtok_video import prune_video
                        image_feature = prune_video(self, image_feature)

                    elif self.pruning_method == 'cdpruner':
                        rank0_print("="*60)
                        rank0_print(f"✓ [CDPruner] 开始执行剪枝")
                        rank0_print(f"  剪枝前: 每帧token数 = {N}, 总帧数 = {T}, 总tokens = {T*N}")
                        rank0_print(f"  目标: 每帧保留 {self.visual_token_num} 个tokens")

                        image_normalized = image_feature / image_feature.norm(dim=-1, keepdim=True) # (T, N, D)
                        similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (T, N, N)

                        # text_embed = self.get_model().embed_tokens(input_ids[:, 15:]).mean(dim=1)
                        # text_normalized = text_embed / text_embed.norm(dim=-1, keepdim=True)
                        relevance = image_attention.mean(dim=1)
                        # relevance = torch.matmul(image_normalized, text_normalized.t()).squeeze(-1)
                        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min()) # (T, N)

                        kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (T, N, N)

                        cis = torch.zeros((self.visual_token_num, T, N), device=device) # (M, T, N)
                        di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (T, N)
                        select_idx = torch.empty((self.visual_token_num, T), dtype=torch.long, device=device) # (N, T)
                        for i in range(self.visual_token_num):
                            j = torch.argmax(di2s, dim=-1)
                            select_idx[i] = j

                            eis = (kernel[torch.arange(T), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(T), j], cis[:i])) \
                                / torch.sqrt(di2s[torch.arange(T), j]).unsqueeze(-1)
                            cis[i, :, :] = eis
                            di2s -= torch.square(eis)
                            di2s[torch.arange(T), j] = -float('inf')

                        select_idx = torch.sort(select_idx.t()).values # (T, M)
                        image_feature = image_feature[torch.arange(T).unsqueeze(-1), select_idx]

                        rank0_print(f"  剪枝后: 每帧token数 = {image_feature.shape[1]}, 总tokens = {T*image_feature.shape[1]}")
                        rank0_print(f"  压缩率: {(1 - image_feature.shape[1]/N)*100:.1f}% (减少了 {N - image_feature.shape[1]} tokens/frame)")
                        rank0_print("="*60)

                    elif self.pruning_method == 'star_v3':
                        rank0_print("="*60)
                        rank0_print(f"✓ [STAR-V3] 开始执行剪枝 (Adaptive THCP)")
                        rank0_print(f"  剪枝前: 每帧token数 = {N}, 总帧数 = {T}, 总tokens = {T*N}")

                        # 1. Adaptive target (key innovation)
                        stage1_keep_num = self.visual_token_num * 2
                        tokens_to_select = min(stage1_keep_num, N)
                        rank0_print(f"  目标(final): 每帧保留 {self.visual_token_num} tokens")
                        rank0_print(f"  Stage 1 adaptive: 保留 {tokens_to_select} tokens (目标 × 2)")

                        # 2. Extract text embeds
                        text_embed = self.get_model().embed_tokens(input_ids[:, 15:])  # (B, M, D)
                        text_normalized = text_embed / (text_embed.norm(dim=-1, keepdim=True) + 1e-8)
                        rank0_print(f"  Text tokens: {text_embed.shape[1]}")

                        # 3. Normalize image features
                        image_normalized = image_feature / (image_feature.norm(dim=-1, keepdim=True) + 1e-8)  # (T, N, D)

                        # 4. Compute relevance (text-visual similarity, mean across text tokens)
                        relevance = torch.matmul(image_normalized, text_normalized.squeeze(0).t()).mean(dim=-1)  # (T, N)
                        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-6)

                        # 5. Compute visual similarity
                        similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (T, N, N)

                        # 6. Greedy selection with diversity (replace DPP)
                        select_idx = torch.empty((tokens_to_select, T), dtype=torch.long, device=device)

                        relevance_weight = 0.5
                        diversity_weight = 0.5

                        # JIT-compiled greedy selection (same logic, GPU-optimized)
                        selected = _greedy_select_jit(
                            relevance, similarity, tokens_to_select,
                            relevance_weight, diversity_weight)

                        # 7. Apply selection
                        select_idx = torch.sort(selected, dim=1).values  # (T, tokens_to_select)
                        image_feature = image_feature[torch.arange(T).unsqueeze(-1), select_idx]  # (T, tokens_to_select, D)

                        rank0_print(f"  剪枝后: 每帧token数 = {image_feature.shape[1]}, 总tokens = {T*image_feature.shape[1]}")
                        rank0_print(f"  压缩率: {(1 - image_feature.shape[1]/N)*100:.1f}% (减少了 {N - image_feature.shape[1]} tokens/frame)")
                        rank0_print("="*60)
                        
                    elif self.pruning_method in ('attn_cov_proj_llm_12', 'attn_H_cov_proj_llm_12'):
                        from llava.model.language_model.attn_cov_video import weighted_coverage_select, pack_video, hidden_l2_weights
                        if image_attention is None:
                            raise ValueError('attn_cov S1 requires actual SigLIP attention')
                        raw_weights = image_attention.mean(dim=1) if image_attention.ndim == 3 else image_attention
                        if raw_weights.shape != (T, N):
                            raise ValueError(f'wrong SigLIP attention shape: {raw_weights.shape}, expected {(T, N)}')
                        n_keep = min(2 * int(self.visual_token_num), N)
                        masks = torch.zeros((T, N), dtype=torch.bool, device=device)
                        for frame in range(T):
                            weights = raw_weights[frame]
                            if self.pruning_method == 'attn_H_cov_proj_llm_12':
                                # H is the raw L2 norm at the pooled projector prune site.
                                weights = hidden_l2_weights(image_feature[frame], weights)
                            indices = weighted_coverage_select(image_feature[frame], weights, n_keep)
                            masks[frame, indices] = True
                        packed, metadata = pack_video(image_feature, masks, 'flat')
                        self._attn_cov_pending = metadata
                        new_image_features.append(packed)
                        continue

                    elif self.pruning_method == 'tops':
                        # TOPS Stage 1: Vision-Side RDC Greedy Selection
                        # Relevance (R): CLS attention from vision encoder
                        # Diversity (D): cosine dissimilarity to selected set
                        # Coverage (C): avg cosine similarity to unselected set / SCOPE marginal gain
                        import os
                        a1 = float(os.environ.get("A1", "1.0"))
                        alpha_1 = float(os.environ.get("ALPHA_1", "1.0"))
                        lambda_1 = float(os.environ.get("LAMBDA_1", "1.0"))
                        coverage_method = os.environ.get("COVERAGE", "MEAN")

                        rank0_print("="*60)
                        rank0_print(f"✓ [TOPS] 开始执行剪枝 (Vision-Side RDC)")
                        rank0_print(f"  剪枝前: 每帧token数 = {N}, 总帧数 = {T}, 总tokens = {T*N}")
                        self.model.video_frame_num = T

                        # Adaptive Stage 1: keep target × 2 tokens (per frame)
                        stage1_keep_num = min(max(1, self.visual_token_num * 2), N)
                        rank0_print(f"  目标(final): 每帧保留 {self.visual_token_num} tokens")
                        rank0_print(f"  Stage 1 adaptive: 保留 {stage1_keep_num} tokens (目标 × 2)")
                        rank0_print(f"  a1={a1}, alpha_1={alpha_1}, lambda_1={lambda_1}, coverage={coverage_method}")

                        # Relevance: CLS attention from vision encoder
                        if image_attention is not None:
                            if image_attention.dim() == 3:
                                cls_attn = image_attention.mean(dim=1)  # (T, N)
                            elif image_attention.dim() == 2:
                                cls_attn = image_attention  # (T, N)
                            else:
                                cls_attn = image_attention.view(T, -1)
                        else:
                            rank0_print("[TOPS S1] Warning: image_attns is None, using L2-norm fallback")
                            cls_attn = image_feature.norm(dim=-1)  # (T, N)

                        selected_masks = torch.zeros(T, N, dtype=torch.bool, device=device)

                        for t in range(T):
                            attn_t = cls_attn[t]
                            feat_t = image_feature[t]
                            keep_t = min(stage1_keep_num, N)

                            # Normalize relevance to [0, 1]
                            attn_t = (attn_t - attn_t.min()) / (attn_t.max() - attn_t.min() + 1e-8)

                            # Cosine similarity matrix
                            feat_n = feat_t / (feat_t.norm(dim=-1, keepdim=True) + 1e-8)
                            sim = torch.matmul(feat_n, feat_n.t())  # (N, N)

                            avail = torch.ones(N, dtype=torch.bool, device=device)

                            # Seed: highest relevance token
                            i0 = torch.argmax(attn_t).item()
                            selected = [i0]
                            avail[i0] = False
                            max_sim_to_S = sim[i0].clone()
                            max_sim_to_S[i0] = 1.0

                            for _ in range(1, keep_t):
                                if not avail.any():
                                    break

                                diversity = 1.0 - max_sim_to_S

                                if coverage_method == "SCOPE":
                                    # SCOPE: marginal coverage gain
                                    coverage = torch.clamp(sim - max_sim_to_S.unsqueeze(0), min=0.0).sum(dim=1)
                                    coverage[~avail] = 0.0
                                else:
                                    # MEAN: avg similarity to remaining unselected tokens
                                    sim_to_avail = sim[:, avail].sum(dim=1)
                                    n_avail = avail.sum().item()
                                    if n_avail > 1:
                                        coverage = (sim_to_avail - 1.0) / (n_avail - 1)
                                    else:
                                        coverage = torch.zeros(N, device=device)

                                a_mean = attn_t.mean().clamp(min=1e-8)
                                div_mean = diversity.mean().clamp(min=1e-8)
                                cov_mean = coverage.mean().clamp(min=1e-8)

                                scores = a1 * (attn_t / a_mean) + \
                                         alpha_1 * (diversity / div_mean) + \
                                         lambda_1 * (coverage / cov_mean)
                                scores[~avail] = -float("inf")
                                idx = torch.argmax(scores).item()

                                selected.append(idx)
                                avail[idx] = False
                                max_sim_to_S = torch.maximum(max_sim_to_S, sim[idx])
                                max_sim_to_S[idx] = 1.0

                            selected_t = torch.tensor(selected, device=device, dtype=torch.long)
                            selected_masks[t, selected_t] = True

                        select_idx = torch.stack([
                            torch.where(selected_masks[t])[0] for t in range(T)
                        ], dim=0)
                        select_idx = torch.sort(select_idx, dim=-1).values
                        image_feature = image_feature[torch.arange(T).unsqueeze(-1), select_idx]

                        rank0_print(f"  剪枝后: 每帧token数 = {image_feature.shape[1]}, 总tokens = {T*image_feature.shape[1]}")
                        rank0_print(f"  压缩率: {(1 - image_feature.shape[1]/N)*100:.1f}% (减少了 {N - image_feature.shape[1]} tokens/frame)")
                        rank0_print("="*60)

                    else:
                        # FastV/SparseVLM/PDrop are implemented in model forward, not here
                        if self.pruning_method in ['fastv', 'sparsevlm', 'pdrop']:
                            rank0_print(f"[{self.pruning_method.upper()}] 剪枝在模型内部实现，跳过prepare_inputs剪枝")
                        elif self.pruning_method != 'vanilla':
                            rank0_print(f"⚠ [警告] 未知的剪枝方法: {self.pruning_method}, 将不进行剪枝")
                        rank0_print(f"[No Pruning in prepare_inputs] 保持原始token数: 每帧={N}, 总帧={T}, 总tokens={T*N}")

                    new_image_features.append(image_feature.flatten(0, 1))

                image_features = new_image_features

            elif mm_patch_merge_type.startswith("spatial"):
                if self.pruning_method != 'vanilla':
                    rank0_print(f"⚠ [警告] mm_patch_merge_type='{mm_patch_merge_type}' 不支持剪枝，剪枝方法 '{self.pruning_method}' 将被忽略")
                    rank0_print(f"  提示: 如需使用剪枝，请设置 mm_patch_merge_type='flat'")
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    if image_idx in video_idx_in_batch:  # video operations
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, image_attentions, image_keys = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    if self.pruning_method in ('attn_cov_proj_llm_12', 'attn_H_cov_proj_llm_12'):
                        if num_images != 1 or i != 0 or self._attn_cov_pending is None:
                            raise ValueError('attn_cov requires exactly one visual sentinel and one packed video')
                        metadata = self._attn_cov_pending
                        visual_start = sum(part.shape[0] for part in cur_new_input_embeds)
                        self.model._tops_request_id = getattr(self, '_tops_request_id', None)
                        self.model.set_visual_layout(visual_start, len(cur_image_features),
                            metadata['patch_positions'], metadata['frame_ids'], metadata['frames'], metadata['original_patch_ids'])
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        if self.pruning_method in ('attn_cov_proj_llm_12', 'attn_H_cov_proj_llm_12'):
            if tokenizer_model_max_length is not None and any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                raise ValueError('attn_cov video prompt would be truncated; increase explicit context limit')
        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features[0].shape[0] // images[0].shape[0] if type(images) is list else image_features[0].shape[0]

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
