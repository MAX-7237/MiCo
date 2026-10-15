# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LlamaForCausalLM, Qwen2ForCausalLM
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_qwen2_fastv import Qwen2ForCausalLMPruning
from ..sparsevlm_core import SPARSEVLM_PRUNING_LAYERS_28

logger = logging.get_logger(__name__)


def _is_attn_h_cov_vit_proj_llm(method: str) -> bool:
    prefix = 'attn_H_cov_vit_proj_llm_'
    if not isinstance(method, str) or not method.startswith(prefix):
        return False
    body = method[len(prefix):]
    if "_s1m" in body:  # `<K>_s1m<M>` ablation arms
        body = body.rsplit("_s1m", 1)[0]
    return body.isdigit()


def _attn_h_s1_mult(method: str) -> int:
    """S1 budget multiple of T, resolved method-name-first.

    `<...>_llm_<K>_s1m<M>` pins it per arm; otherwise fall back to the
    H_VIT_S1_MULT env knob. Default 2 preserves the published
    `attn_H_cov_vit_proj_llm_K` behaviour exactly.
    """
    from vlmeval.vlm.attn_cov_select import attn_h_s1_mult as _env_mult
    if isinstance(method, str):
        head, _, tail = method.rpartition("_s1m")
        if head and tail.isdigit() and int(tail) >= 1:
            return int(tail)
    return _env_mult()


def _is_attn_h_cov_vit_proj(method: str) -> bool:
    return method == 'attn_H_cov_vit_proj' or _is_attn_h_cov_vit_proj_llm(method)

# Variable names used to hold the cache at generation time
ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModelPruning(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True,
                 pruning_method='vanilla', visual_token_num=-1):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                # Use custom pruning model for fastv method
                if pruning_method == 'fastv':
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for fastv pruning, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] 使用 FastV 剪枝方法初始化 Language Model")
                    print("="*80)
                    print(f"  Pruning method: fastv")
                    print(f"  Target AVERAGE visual tokens: {visual_token_num} (across all 28 layers)")
                    print(f"  Per-patch token count: {self.num_image_token}")
                    print(f"  Note: FastV will prune at layer 2:")
                    print(f"        - Layers 0-1: Keep all visual tokens (original count)")
                    print(f"        - Layers 2-27: Keep R tokens (calculated from target average)")
                    print(f"        - Formula: avg = (original*2 + R*26) / 28 = {visual_token_num}")
                    print("="*80 + "\n")

                    self.language_model = Qwen2ForCausalLMPruning(
                        config.llm_config,
                        pruning_method='fastv',
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method == 'apet':
                    from .modeling_qwen2_apet import Qwen2ForCausalLMApET
                    self.language_model = Qwen2ForCausalLMApET(config.llm_config, visual_token_num)
                    self.apet_schedule = self.language_model.model.apet_schedule
                    print(f'[InternVL3 ApET] whole-image budget={visual_token_num}, '
                          f'S1={self.apet_schedule["stage1"]}, S2={self.apet_schedule["stage2"]}, '
                          f'after {self.apet_schedule["prune_after"]} completed decoder layers')
                elif pruning_method in ('dart', 'dart_coverage'):
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for {pruning_method}, got {visual_token_num}")
                    print("\n" + "="*80)
                    print(f"[InternVL3] 使用 {pruning_method.upper()} in-LLM 剪枝 (Qwen2 backbone, layer K=2)")
                    print(f"  Target visual tokens: {visual_token_num}  per-patch: {self.num_image_token}")
                    print("="*80 + "\n")
                    from .modeling_qwen2_dart import Qwen2ForCausalLMDART
                    self.language_model = Qwen2ForCausalLMDART(
                        config.llm_config,
                        pruning_method=pruning_method,
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method == 'star_v3':
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for star_v3, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] 使用 STAR-V3 两阶段剪枝方法初始化 Language Model")
                    print("="*80)
                    print(f"  Pruning method: star_v3")
                    print(f"  Stage 1 (THCP): Vision encoder → target×2 tokens")
                    print(f"  Stage 2 (Progressive): Decoder progressive pruning")
                    print(f"  Target FINAL visual tokens: {visual_token_num} (average across layers)")

                    from .modeling_qwen2_star_v3 import StarV3Qwen2ForCausalLM, STAR_V3_SCHEDULE
                    schedule = STAR_V3_SCHEDULE.get("internvl3_8b", {}).get(visual_token_num, [])
                    print(f"  Stage 2 schedule: {schedule}")
                    print("="*80 + "\n")

                    # Initialize Stage 1 parameters
                    self.stage1_keep_tokens = max(1, visual_token_num * 2)

                    self.language_model = StarV3Qwen2ForCausalLM(
                        config.llm_config,
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method == 'star_qr':
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for star_qr, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] 使用 STAR-Pro-QR 两阶段剪枝方法初始化 Language Model")
                    print("="*80)
                    print(f"  Pruning method: star_qr")
                    print(f"  Stage 1 (QR): query-agnostic global residual QR → target×2 tokens")
                    print(f"  Stage 2 (Progressive): decoder pruning, text_aggregation=average_all")
                    print(f"  Target FINAL visual tokens: {visual_token_num} (average across layers)")

                    from .modeling_qwen2_star_qr import StarQRQwen2ForCausalLM, STAR_QR_SCHEDULE
                    schedule = STAR_QR_SCHEDULE.get("internvl3_8b", {}).get(visual_token_num, [])
                    print(f"  Stage 2 schedule: {schedule}")
                    print("="*80 + "\n")

                    # Initialize Stage 1 parameters
                    self.stage1_keep_tokens = max(1, visual_token_num * 2)

                    self.language_model = StarQRQwen2ForCausalLM(
                        config.llm_config,
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method == 'tops':
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for tops, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] Using TOPS Two-Stage Pruning Method")
                    print("="*80)
                    print(f"  Stage 1 (Vision Encoder): Split RDC — local patches + thumbnail")
                    print(f"  Stage 2 (Decoder): Progressive RDC pruning")
                    print(f"  Target FINAL visual tokens: {visual_token_num} (average across layers)")

                    from .modeling_qwen2_tops import TopsQwen2ForCausalLM, TOPS_SCHEDULE
                    schedule = TOPS_SCHEDULE.get("internvl3_8b", {}).get(visual_token_num, [])
                    print(f"  Stage 2 schedule: {schedule}")
                    print("="*80 + "\n")

                    self.stage1_keep_tokens = max(1, visual_token_num * 2)
                    self.language_model = TopsQwen2ForCausalLM(
                        config.llm_config,
                        tops_config={"T": visual_token_num}
                    )
                elif pruning_method == 'tops_2':
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for tops_2, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] Using TOPS_2 Single-Stage Pruning Method")
                    print("="*80)
                    print(f"  Stage 1 (Vision Encoder): Split RDC — local patches + thumbnail (prune directly to target)")
                    print(f"  Stage 2 (Decoder): DISABLED (single-stage)")
                    print(f"  Target FINAL visual tokens: {visual_token_num}")

                    from .modeling_qwen2_tops import TopsQwen2ForCausalLM
                    print("="*80 + "\n")

                    # Single-stage: Stage-1 prunes directly to the target (NOT target×2).
                    self.stage1_keep_tokens = max(1, visual_token_num)
                    self.language_model = TopsQwen2ForCausalLM(
                        config.llm_config,
                        tops_config={"T": visual_token_num, "disable_stage2": True}
                    )
                elif pruning_method == 'tops_plus':
                    # tops+: vision-side merge (merge-first/CRM/merge-back) + single-stage RDC (S2 off, like tops_2)
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for tops_plus, got {visual_token_num}")
                    print("\n[InternVL3] Using TOPS+ (vision-side merge + single-stage RDC)")
                    from .modeling_qwen2_tops import TopsQwen2ForCausalLM
                    self.stage1_keep_tokens = max(1, visual_token_num)
                    self.language_model = TopsQwen2ForCausalLM(
                        config.llm_config,
                        tops_config={"T": visual_token_num, "disable_stage2": True}
                    )
                elif pruning_method == 'vscan':
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for vscan, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] Using VScan Two-Stage Pruning Method")
                    print("="*80)
                    print(f"  Stage 1 (Vision Encoder): complementary local+global attention scan + token merging")
                    print(f"  Stage 2 (Decoder): last-text-token guided progressive pruning (FastV-style)")
                    print(f"  Target FINAL visual tokens: {visual_token_num} (average across layers)")

                    from .modeling_qwen2_vscan import VScanQwen2ForCausalLM, VSCAN_SCHEDULE
                    schedule = VSCAN_SCHEDULE.get("internvl3_8b", {}).get(visual_token_num, [])
                    print(f"  Stage 2 schedule: {schedule}")
                    print("="*80 + "\n")

                    self.stage1_keep_tokens = max(1, visual_token_num * 2)
                    self.language_model = VScanQwen2ForCausalLM(
                        config.llm_config,
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method in ('duet', 'duet_official'):
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for {pruning_method}, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] Using DUET Two-Stage Pruning Method")
                    print("="*80)
                    print(f"  Stage 1 (V2V / VisionZip): dominant attention top-k + contextual cluster merge")
                    print(f"  Stage 2 (T2V / PyramidDrop): text->image attention progressive pruning")
                    print(f"  Target FINAL visual tokens: {visual_token_num} (average across layers)")

                    from .modeling_qwen2_duet import DuetQwen2ForCausalLM, DUET_SCHEDULE
                    schedule = DUET_SCHEDULE.get("internvl3_8b", {}).get(visual_token_num, [])
                    print(f"  Stage 2 schedule: {schedule}")
                    print("="*80 + "\n")

                    self.stage1_keep_tokens = max(1, visual_token_num * 2)
                    self.language_model = DuetQwen2ForCausalLM(
                        config.llm_config,
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method == 'sparsevlm':
                    # SparseVLM (arXiv 2406.11717): in-decoder pruning at 3 layers, text-guided
                    # relevance top-k plus DPC-kmeans recycling of the dropped tokens.
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for sparsevlm, got {visual_token_num}")

                    print("\n" + "="*80)
                    print("[InternVL3] Using SparseVLM In-Decoder Pruning")
                    print("="*80)
                    print(f"  Pruning layers: {SPARSEVLM_PRUNING_LAYERS_28} (text-guided top-k + DPC-kmeans recycle)")
                    print(f"  Target layer-average visual tokens: {visual_token_num}")
                    print("="*80 + "\n")

                    from .modeling_qwen2_sparsevlm import Qwen2ForCausalLMSparseVLM
                    self.language_model = Qwen2ForCausalLMSparseVLM(
                        config.llm_config,
                        pruning_method='sparsevlm',
                        visual_token_num=visual_token_num,
                        num_image_token=self.num_image_token
                    )
                elif pruning_method == 'cover_prune_2':
                    # COVER_Prune_2 双阶段 (InternVL port): stage-1(projector, cls_attn内加权coverage,
                    #   保留2×target) + stage-2(LLM第12/16层, all_text_attn内加权coverage, 复用TOPS schedule).
                    #   LLM 侧复用 TopsQwen2ForCausalLM, 通过 env COVER_S2_INSIDE=1 切换 stage-2 打分为内加权.
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for cover_prune_2, got {visual_token_num}")
                    print("\n[InternVL3] Using COVER_Prune_2 Two-Stage (cls/text attn × coverage, inside-weighted)")
                    from .modeling_qwen2_tops import TopsQwen2ForCausalLM, TOPS_SCHEDULE
                    schedule = TOPS_SCHEDULE.get("internvl3_8b", {}).get(visual_token_num, [])
                    print(f"  Stage 2 schedule (shared with TOPS): {schedule}")
                    self.stage1_keep_tokens = max(1, visual_token_num * 2)
                    self.language_model = TopsQwen2ForCausalLM(
                        config.llm_config,
                        tops_config={"T": visual_token_num}
                    )
                elif pruning_method in ('attn_cov_vit_proj_llm_12', 'attn_cov_proj_llm_12') or _is_attn_h_cov_vit_proj_llm(pruning_method):
                    # attn_cov_vit_proj_llm_12: ViT-space S1 keep 2T
                    # attn_cov_proj_llm_12: projector-space S1 keep 2T
                    # Both: one LLM prune at TOPS first layer (12), FastV-style avg T.
                    if visual_token_num <= 0:
                        raise ValueError(
                            f"visual_token_num must be positive for {pruning_method}, got {visual_token_num}"
                        )
                    from .modeling_qwen2_tops import TopsQwen2ForCausalLM, TOPS_SCHEDULE
                    from vlmeval.vlm.attn_cov_select import (
                        attn_h_s1_mult,
                        attn_h_s2_keep,
                        fastv_keep_after_k,
                        llm12_pruning_layers,
                    )
                    n_layers = config.llm_config.num_hidden_layers
                    t = int(visual_token_num)
                    attn_h = _is_attn_h_cov_vit_proj_llm(pruning_method)
                    if attn_h:
                        # Strip an optional `_s1m<M>` ablation suffix before reading K,
                        # so `..._llm_4_s1m5` still resolves to K=4.
                        k_layer = int(pruning_method.rsplit('_s1m', 1)[0].rsplit('_', 1)[1])
                        if not 1 <= k_layer < n_layers:
                            raise ValueError(f"H-ViT S2 K must be in [1, {n_layers - 1}], got {k_layer}")
                        # S1 keeps M×T, so the per-layer average identity is
                        # T*L = K*(M*T) + (L-K)*R.
                        s1_m = _attn_h_s1_mult(pruning_method)
                        s2 = {k_layer: attn_h_s2_keep(t, n_layers, k_layer, s1_m)}
                    else:
                        s1_m = 2
                        s2 = llm12_pruning_layers(TOPS_SCHEDULE.get("internvl3_8b", {}), t, n_layers)
                    space = "ViT" if "vit_proj" in pruning_method else "projector"
                    print(
                        f"\n[InternVL3] {pruning_method}: S1 {space} raw_cls×coverage {s1_m}T, "
                        f"S2={s2} (TOPS first layer, R=(T*L-K*{s1_m}T)/(L-K), L={n_layers})"
                    )
                    self.stage1_keep_tokens = max(1, t * s1_m)
                    self._s1_mult = s1_m
                    self.language_model = TopsQwen2ForCausalLM(
                        config.llm_config,
                        tops_config={
                            "T": t,
                            "pruning_layers": s2,
                            "cover_s2_inside": True,
                            "attn_h": attn_h,
                            "dynamic_r": attn_h,
                        },
                    )
                elif pruning_method in ['divprune', 'divprune_coverage', 'divprune_coverage_to_others_only', 'cover_qe', 'cover_prune', 'cover_e', 'cover_q', 'cover_u', 'cover_prune_1_proj', 'cover_prune_2_proj', 'mmtok', 'cdpruner', 'dppruner', 'visionzip', 'visionzip+', 'holov', 'attn_cov_vit_proj', 'attn_cov_proj', 'attn_H_cov_vit_proj']:
                    # These methods prune in extract_feature(), use vanilla LLM
                    if visual_token_num <= 0:
                        raise ValueError(f"visual_token_num must be positive for {pruning_method}, got {visual_token_num}")

                    print("\n" + "="*80)
                    print(f"[InternVL3] Using {pruning_method.upper()} Pruning Method")
                    print("="*80)
                    print(f"  Pruning location: Vision encoder output (extract_feature)")
                    print(f"  Target visual tokens: {visual_token_num}")
                    print(f"  Method description:")
                    if pruning_method == 'divprune':
                        print(f"    - Diversity-based pruning using cosine distance")
                        print(f"    - Selects most diverse visual tokens greedily")
                    elif pruning_method == 'cdpruner':
                        print(f"    - Content-Diversity Pruner (DPP-based)")
                        print(f"    - Combines text relevance + visual diversity")
                        print(f"    - Kernel: relevance * similarity * relevance")
                    elif pruning_method == 'dppruner':
                        print(f"    - Pure DPP-based pruning")
                        print(f"    - Maximizes determinantal point process over visual tokens")
                    elif pruning_method == 'visionzip':
                        print(f"    - VisionZip: Dominant + Contextual token compression")
                        print(f"    - Total target tokens: {visual_token_num}")
                        print(f"    - Ratio per patch: Dominant 27/32, Contextual 5/32")
                    elif pruning_method == 'visionzip+':
                        print(f"    - VisionZip+: Enhanced with fixed 5% contextual tokens")
                        print(f"    - Total target tokens: {visual_token_num}")
                        print(f"    - Contextual: 5% of original per patch")
                    elif pruning_method == 'holov':
                        print(f"    - HoloV: holistic visual context retention")
                        print(f"    - Region-partitioned budget allocation + per-region attention top-k")
                        print(f"    - Total target tokens: {visual_token_num}")
                    print("="*80 + "\n")

                    self.language_model = Qwen2ForCausalLM(config.llm_config)
                else:
                    self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        self.pruning_method = pruning_method
        self.visual_token_num = visual_token_num

        # Process position_ids with pruning
        def _update_model_kwargs_for_generation_pruning(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
        ) -> Dict[str, Any]:
            # update past_key_values keeping its naming used in model code
            for possible_cache_name in ALL_CACHE_NAMES:
                if possible_cache_name in outputs:
                    # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                    if possible_cache_name in ("past_buckets_states", "mems"):
                        cache_name = "past_key_values"
                    else:
                        cache_name = possible_cache_name
                    model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                    break

            # update token_type_ids with last value
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

            if not is_encoder_decoder:
                # update attention mask
                if "attention_mask" in model_kwargs:
                    attention_mask = model_kwargs["attention_mask"]
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
            if "position_ids" in model_kwargs:
                position_ids = model_kwargs["position_ids"]
                if position_ids is not None:
                    model_kwargs["position_ids"] = torch.cat(
                        [position_ids, position_ids[:, -1].unsqueeze(-1) + 1], dim=-1
                    )
                else:
                    model_kwargs.pop("position_ids")
            else:
                # update decoder attention mask
                if "decoder_attention_mask" in model_kwargs:
                    decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                    model_kwargs["decoder_attention_mask"] = torch.cat(
                        [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                        dim=-1,
                    )

            if model_kwargs.get("use_cache", True):
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
            else:
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                ).to(past_positions.device)
                model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
            return model_kwargs

        self.language_model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation_pruning, self.language_model)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # Extract visual features with pruning
        vit_embeds, selected_masks = self.extract_feature(pixel_values)

        # DEBUG: Log before/after pruning
        original_tokens = vit_embeds.shape[0] * vit_embeds.shape[1] if len(vit_embeds.shape) > 2 else vit_embeds.shape[0]

        # Apply pruning mask if any tokens were pruned
        if selected_masks.sum() < selected_masks.numel():
            # Filter vit_embeds using selected_masks: keep only selected tokens
            # selected_masks shape: [B, N] where B=num_patches, N=tokens_per_patch
            # We need to flatten and filter
            vit_embeds = vit_embeds[selected_masks]  # [total_selected_tokens, D]

            print(f"[DEBUG] Stage 1 pruning applied: {original_tokens} → {vit_embeds.shape[0]} tokens")
        else:
            print(f"[DEBUG] No Stage 1 pruning: {original_tokens} tokens kept")

        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        # STAR-V3 / STAR-Pro-QR: Set system prompt length for pruning
        if self.pruning_method in ('star_v3', 'star_qr') and N > 1:  # Prefill mode
            system_prompt_length = int(selected.max(dim=-1).indices) if selected.sum() > 0 else 41
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)
                
        # TOPS: Set system prompt length for pruning
        if self.pruning_method in ('tops', 'tops_2', 'tops_plus', 'cover_prune_2', 'attn_cov_vit_proj_llm_12', 'attn_cov_proj_llm_12') and N > 1:  # Prefill mode
            system_prompt_length = int(selected.max(dim=-1).indices) if selected.sum() > 0 else 41
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)

        # VScan / DUET: Set system prompt length for Stage 2 pruning
        if self.pruning_method in ['vscan', 'duet', 'duet_official'] and N > 1:  # Prefill mode
            system_prompt_length = int(selected.max(dim=-1).indices) if selected.sum() > 0 else 41
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)

        # FastV: Set system prompt length for pruning
        if self.pruning_method == 'fastv' and N > 1:  # Prefill mode
            # Calculate where visual tokens start (right after the last image context token in the first occurrence)
            # selected is a boolean mask where image context tokens are located
            if selected.sum() > 0:
                # Find the first image context token position
                first_img_token = int(selected.nonzero()[0].item())
                # System prompt is everything before the first image token
                system_prompt_length = first_img_token

                # Typical InternVL3 system prompt is ~41 tokens
                # Warn if significantly different (might indicate tokenization issues)
                if system_prompt_length < 30 or system_prompt_length > 100:
                    logger.warning(f"[FastV] Unexpected system_prompt_length: {system_prompt_length} "
                                 f"(expected ~41). This might affect pruning performance.")
            else:
                system_prompt_length = 41  # Default for InternVL3

            # Set system_prompt_length on the language model
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"[FastV] System prompt length: {system_prompt_length}")
                    print(f"[FastV] Visual tokens: {self.num_image_token}")
                    print(f"[FastV] Target tokens: {self.visual_token_num}")

        # DART / DART_coverage / SparseVLM: Set visual segment (system prompt length + actual visual token count)
        if self.pruning_method in ('dart', 'dart_coverage', 'sparsevlm') and N > 1:  # Prefill mode
            if selected.sum() > 0:
                sp_len = int(selected.nonzero()[0].item())      # 首个视觉token位置
                vis_len = int(selected.sum().item())            # 实际视觉token总数(=保留前)
            else:
                sp_len, vis_len = 41, self.num_image_token
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(sp_len)
            if hasattr(self.language_model, 'set_visual_token_length'):
                self.language_model.set_visual_token_length(vis_len)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values, image_num_patches=None, text_embeds=None):
        vit_attns = None
        if self.pruning_method in ['fastervlm', 'prumerge', 'prumerge+', 'visionzip', 'visionzip+', 'vispruner', 'tops', 'tops_2', 'tops_plus', 'holov', 'vscan', 'duet', 'duet_official', 'cover_prune_1_proj', 'cover_prune_2_proj', 'cover_prune_2', 'attn_cov_vit_proj', 'attn_cov_vit_proj_llm_12', 'attn_cov_proj', 'attn_cov_proj_llm_12'] or _is_attn_h_cov_vit_proj(self.pruning_method):
            if self.select_layer == -1:
                vit_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                    select_layer=self.select_layer)
                vit_embeds = vit_outputs.last_hidden_state
                vit_attns = vit_outputs.attentions[self.select_layer]
            else:
                vit_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                    select_layer=self.select_layer)
                vit_embeds = vit_outputs.hidden_states[self.select_layer]
                vit_attns = vit_outputs.attentions[self.select_layer]
        else:
            if self.select_layer == -1:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True).last_hidden_state
            else:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]
        if vit_attns is not None:
            vit_attns = vit_attns[:, :, 0, 1:]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        # Pre-mlp1 tokens are still ViT space (after pixel_shuffle). S1 for
        # attn_cov_vit_proj* uses this, matching LLaVA CLIP-before-projector.
        vit_pre_proj = vit_embeds
        if self.pruning_method == 'apet':
            from ..apet_select import apet_stage1
            if image_num_patches is not None and len(image_num_patches) != 1:
                raise ValueError('ApET benchmark adapter requires one image per prompt')
            crops, crop_tokens, channels = vit_embeds.shape
            total_tokens = crops * crop_tokens
            keep = min(self.apet_schedule['stage1'], total_tokens)
            # One budget covers all dynamic-resolution crops and the thumbnail;
            # pixel shuffle only concatenates patches, before mlp1 projects them.
            merged, indices = apet_stage1(
                vit_pre_proj.reshape(1, total_tokens, channels), keep=keep,
                basis=self.language_model.model.apet_basis,
                seed=self.language_model.model.apet_seed)
            if not torch.isfinite(merged).all():
                raise ValueError('ApET S1 produced non-finite ViT features')
            projected = self.mlp1(merged.reshape(crops, crop_tokens, channels).to(vit_embeds.dtype))
            selected_masks = torch.zeros(total_tokens, dtype=torch.bool, device=vit_embeds.device)
            selected_masks[indices[0]] = True
            self.apet_stage1_record = dict(crops=crops, crop_tokens=crop_tokens,
                                          original=total_tokens, selected=keep)
            print(f'[ApET S1] N={total_tokens} -> keep {keep} '
                  f'(3T/2={self.apet_schedule["stage1"]}, basis=10, space=pre-projector-ViT)', flush=True)
            return projected, selected_masks.reshape(crops, crop_tokens)
        vit_embeds = self.mlp1(vit_embeds)

        if vit_attns is not None:
            vit_attns = vit_attns.reshape(-1, h, w).unsqueeze(-1)
            vit_attns = self.pixel_shuffle(vit_attns, scale_factor=self.downsample_ratio).mean(dim=-1)
            vit_attns = vit_attns.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[1])

        # Generate selection mask
        B, N, D = vit_embeds.shape
        device = vit_embeds.device
        selected_masks = torch.ones(B, N, dtype=torch.bool, device=device)

        if self.pruning_method == 'thumbnail':
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
        elif self.pruning_method in ('random', 'rand'):
            # 每patch保留 visual_token_num//B 个 (与 divprune 一致), 否则 topk(256/patch) 等于不剪
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1
            random_scores = torch.rand(B, N).to(device)
            selected_indices = random_scores.topk(min(tokens_per_patch, N), dim=-1).indices
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, selected_indices, True)
        elif self.pruning_method == "fastervlm":
            image_attention = vit_attns.mean(dim=1)
            patch_attention, thumb_attention = image_attention[:-1], image_attention[-1:]
            patch_token_num = self.visual_token_num // 2 // patch_attention.shape[0]
            patch_indices = patch_attention.topk(patch_token_num, dim=-1).indices
            thumb_token_num = self.visual_token_num // 2
            thumb_indices = thumb_attention.topk(thumb_token_num, dim=-1).indices
            # selected_indices = image_attention.topk(self.visual_token_num, dim=-1).indices

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks[:-1].scatter_(1, patch_indices, True)
            selected_masks[-1:].scatter_(1, thumb_indices, True)
            # selected_masks.scatter_(1, selected_indices, True)
        elif self.pruning_method == "trim":
            _text_embeds_list = [text_embed[None].expand(num_patch, -1, -1) for text_embed, num_patch in zip(text_embeds, image_num_patches)]
            text_embeds = torch.cat(_text_embeds_list, dim=0)

            image_normalized = (vit_embeds / vit_embeds.norm(dim=-1, keepdim=True))
            text_normalized = (text_embeds / text_embeds.norm(dim=-1, keepdim=True))
            query_relevance = torch.matmul(image_normalized, text_normalized.transpose(1, 2)).max(dim=-1).values
            selected_indices = query_relevance.topk(self.visual_token_num, dim=-1).indices

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, selected_indices, True)
        elif self.pruning_method == 'visionzip' or self.pruning_method == 'visionzip+':
            # VisionZip: Select dominant tokens + merge contextual tokens
            # B = num_patches (e.g., 5 patches), N = tokens_per_patch (e.g., 256)
            # visual_token_num = target TOTAL tokens across ALL patches (e.g., 256)
            if vit_attns is None:
                raise ValueError("VisionZip requires attention weights, but they are not available")

            # Calculate tokens per patch: total_target / num_patches
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1  # At least 1 token per patch

            if self.pruning_method == 'visionzip':
                # For each patch: dominant = tokens_per_patch * 27/32
                dominant_per_patch = int(tokens_per_patch * 27 / 32)
                contextual_per_patch = tokens_per_patch - dominant_per_patch
            elif self.pruning_method == 'visionzip+':
                # For each patch: contextual = 5% of original N
                contextual_per_patch = int(N * 5 / 100)
                dominant_per_patch = tokens_per_patch - contextual_per_patch

            # Select dominant tokens based on attention
            image_attention = vit_attns.mean(dim=1)  # [B, N]
            dominant_indices = torch.empty((B, dominant_per_patch), dtype=torch.long, device=device)
            for b in range(B):
                dominant_indices[b] = image_attention[b].topk(dominant_per_patch, dim=-1).indices

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, dominant_indices, True)

            # For each image in batch, process contextual tokens
            for b in range(B):
                # Get normalized embeddings for non-dominant tokens
                filtered_embeds = vit_embeds[b, ~selected_masks[b]]  # [remaining, D]
                if filtered_embeds.shape[0] == 0:
                    continue

                normalized_embeds = filtered_embeds / filtered_embeds.norm(dim=-1, keepdim=True)

                # Select contextual tokens evenly spaced
                target_step = max(1, normalized_embeds.shape[0] // contextual_per_patch)
                target_indices = torch.arange(0, normalized_embeds.shape[0], target_step, device=device)[:contextual_per_patch]
                target_embeds = normalized_embeds[target_indices, :]

                # Compute similarity for source tokens to targets
                source_mask = ~torch.isin(torch.arange(normalized_embeds.shape[0], device=device), target_indices)
                source_embeds = normalized_embeds[source_mask]

                if source_embeds.shape[0] > 0 and target_embeds.shape[0] > 0:
                    similarity = torch.matmul(source_embeds, target_embeds.t())
                    # Use vit_embeds dtype to avoid dtype mismatch (BFloat16)
                    assign_one_hot = vit_embeds.new_zeros(source_embeds.shape[0], contextual_per_patch)
                    assign_one_hot.scatter_(1, similarity.argmax(dim=-1).unsqueeze(-1), 1)
                    counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)

                    # Get actual indices in the filtered space
                    contextual_positions = (~selected_masks[b]).nonzero(as_tuple=True)[0]
                    target_positions = contextual_positions[target_indices]
                    source_positions = contextual_positions[source_mask]

                    # Mark target tokens as selected
                    selected_masks[b, target_positions] = True

                    # Aggregate source embeddings into targets
                    aggregated_embeds = torch.matmul(assign_one_hot.t(), source_embeds) / counts
                    vit_embeds[b, target_positions] += aggregated_embeds
        elif self.pruning_method == 'vispruner':
            # VisPruner: keep important_ratio of the per-patch budget by CLS-attn
            # rank, then fill the rest by an iterative cosine-redundancy cull.
            # Selector runs on ViT-space features (vit_pre_proj), matching the
            # LLaVA path where it runs before the projector.
            if vit_attns is None:
                raise ValueError("VisPruner requires attention weights, but they are not available")

            from vlmeval.vlm.vispruner_select import vispruner_select

            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1
            tokens_per_patch = min(tokens_per_patch, N)

            selected_indices = vispruner_select(
                vit_pre_proj, vit_attns, tokens_per_patch, important_ratio=0.5)

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, selected_indices, True)
            print(f"[VisPruner] {B} patch(es), tokens/patch {N} -> {tokens_per_patch} "
                  f"(important {int(tokens_per_patch * 0.5)} + diverse "
                  f"{tokens_per_patch - int(tokens_per_patch * 0.5)})")
        elif self.pruning_method == 'divprune':
            # Calculate tokens per patch: total_target / num_patches (same as VisionZip)
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1  # At least 1 token per patch

            image_normalized = (vit_embeds / vit_embeds.norm(dim=-1, keepdim=True))
            cosine_matrix = 1.0 - torch.matmul(image_normalized, image_normalized.transpose(1, 2))

            select_idx = torch.empty((B, tokens_per_patch), dtype=torch.long, device=device)

            # Process each patch independently to avoid indexing issues
            for b in range(B):
                for i in range(tokens_per_patch):
                    if i == 0:
                        m2 = cosine_matrix[b]  # [N, N]
                        scores = torch.topk(m2, 2, dim=0, largest=False).values[1]  # [N]
                    else:
                        # Select rows corresponding to already selected tokens
                        m2 = cosine_matrix[b, select_idx[b, :i], :]  # [i, N]
                        scores = torch.min(m2, dim=0).values  # [N]

                    select_idx[b, i] = torch.argmax(scores)

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx, True)
        elif self.pruning_method in ('divprune_coverage', 'divprune_coverage_to_others_only'):
            # DivPrune 选择准则换成 SCOPE coverage 边际增益贪心 (与 Qwen path 一致)
            # others_only: 去掉自覆盖项(对角), 仅用交叉覆盖
            others_only = (self.pruning_method == 'divprune_coverage_to_others_only')
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1
            image_normalized = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)
            sim_all = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # [B,N,N] 余弦
            select_idx = torch.empty((B, tokens_per_patch), dtype=torch.long, device=device)
            for b in range(B):
                sim = sim_all[b]                              # [N,N]
                sg = sim.clone()
                if others_only:
                    sg = sim.clone(); sg.fill_diagonal_(0.0)
                Nb = sim.shape[0]; Kt = min(tokens_per_patch, Nb)
                avail = torch.ones(Nb, dtype=torch.bool, device=device)
                g = torch.zeros(Nb, device=device)
                i0 = int(torch.argmax(torch.clamp(sg, min=0.0).sum(dim=1)))
                select_idx[b, 0] = i0; avail[i0] = False; g = torch.maximum(g, sim[i0])
                for i in range(1, Kt):
                    gain = torch.clamp(sg - g.unsqueeze(0), min=0.0).sum(dim=1)  # 边际覆盖增益
                    gain[~avail] = -float('inf')
                    j = int(torch.argmax(gain))
                    select_idx[b, i] = j; avail[j] = False; g = torch.maximum(g, sim[j])
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx, True)
        elif self.pruning_method in ('cover_qe', 'cover_prune', 'cover_e', 'cover_q', 'cover_u'):
            # COVER-Prune 加权 facility-location coverage (逐patch). 主方法 cover_qe:
            #   w_v = energy_v · task_v; energy=||x_v||²(归一), task=(1-λ)+λ·Np·softmax(r_v/τ)
            #   r_v(Q)=max_t cos(视觉v, 文本query t); 目标 max_S Σ_v w_v·max_{s∈S}κ(v,s)
            import os as _os
            lam = float(_os.environ.get('COVER_LAMBDA', '0.5'))
            tau = float(_os.environ.get('COVER_TAU', '0.1'))
            use_e = self.pruning_method in ('cover_qe', 'cover_prune', 'cover_e')
            use_q = self.pruning_method in ('cover_qe', 'cover_prune', 'cover_q')
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1
            image_normalized = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)
            sim_all = torch.clamp(torch.matmul(image_normalized, image_normalized.transpose(1, 2)), min=0.0)  # [B,N,N]
            # energy [B,N]
            if use_e:
                energy_all = vit_embeds.float().pow(2).sum(dim=-1)              # [B,N]
                energy_all = energy_all / (energy_all.mean(dim=1, keepdim=True) + 1e-6)
            else:
                energy_all = torch.ones(B, N, device=device)
            # task relevance [B,N]: 用 extract_feature 传入的 text_embeds (generate路径), 每patch广播
            task_all = torch.ones(B, N, device=device)
            if use_q and text_embeds is not None:
                try:
                    if image_num_patches is not None:
                        _tl = [te[None].expand(np, -1, -1) for te, np in zip(text_embeds, image_num_patches)]
                        _te = torch.cat(_tl, dim=0)                                # [B,T,C]
                    else:
                        _te = text_embeds if text_embeds.dim() == 3 else text_embeds[None].expand(B, -1, -1)
                    tn = _te.float() / (_te.float().norm(dim=-1, keepdim=True) + 1e-8)   # [B,T,C]
                    r_all = torch.matmul(image_normalized.float(), tn.transpose(1, 2)).max(dim=-1).values  # [B,N]
                    pi_all = torch.softmax(r_all / tau, dim=1)
                    task_all = (1.0 - lam) + lam * N * pi_all
                except Exception as _e:
                    print(f"[cover_qe] task relevance fallback (w=energy only): {_e}")
            w_all = (energy_all * task_all)                                     # [B,N]
            w_all = w_all / (w_all.mean(dim=1, keepdim=True) + 1e-6)
            select_idx = torch.empty((B, tokens_per_patch), dtype=torch.long, device=device)
            for b in range(B):
                sim = sim_all[b]; w = w_all[b].to(sim.dtype)
                Kt = min(tokens_per_patch, N)
                avail = torch.ones(N, dtype=torch.bool, device=device)
                g = torch.zeros(N, device=device, dtype=sim.dtype)
                i0 = int(torch.argmax((w.unsqueeze(0) * sim).sum(dim=1)))
                select_idx[b, 0] = i0; avail[i0] = False; g = torch.maximum(g, sim[i0])
                for i in range(1, Kt):
                    gain = (w.unsqueeze(0) * torch.clamp(sim - g.unsqueeze(0), min=0.0)).sum(dim=1)
                    gain[~avail] = -float('inf')
                    j = int(torch.argmax(gain)); select_idx[b, i] = j; avail[j] = False; g = torch.maximum(g, sim[j])
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx, True)
        elif self.pruning_method == 'mmtok':
            # MMTok (arXiv 2508.18264, 竞品, InternVL port): 多模态最大覆盖, 单阶段 projector 后, 逐patch.
            #   P = softmax(z@x.T / tv)/M 文本→视觉[M,N]; Q = softmax(x@x.T / vv)/N 视觉→视觉[N,N]
            #   Combined=[P; α·Q], 贪心最大覆盖. 注: 原文 Q 用 ViT/CLIP 空间, 此处用 projector 输出.
            import os as _os
            tv = float(_os.environ.get('MMTOK_TV_TEMP', '0.02'))
            vv = float(_os.environ.get('MMTOK_VV_TEMP', '0.2'))
            alpha = float(_os.environ.get('MMTOK_ALPHA', '0.5'))
            tokens_per_patch = max(1, self.visual_token_num // B)
            Kt = min(tokens_per_patch, N)
            image_normalized = vit_embeds / (vit_embeds.norm(dim=-1, keepdim=True) + 1e-8)  # [B,N,D]
            Q_all = torch.softmax(torch.matmul(image_normalized, image_normalized.transpose(1, 2)).float() / vv, dim=2) / N  # [B,N,N]
            P_all = None
            if text_embeds is not None:
                try:
                    if image_num_patches is not None:
                        _tl = [te[None].expand(np, -1, -1) for te, np in zip(text_embeds, image_num_patches)]
                        _te = torch.cat(_tl, dim=0)                                    # [B,T,C]
                    else:
                        _te = text_embeds if text_embeds.dim() == 3 else text_embeds[None].expand(B, -1, -1)
                    zn = _te.float() / (_te.float().norm(dim=-1, keepdim=True) + 1e-8)  # [B,T,C]
                    M = zn.shape[1]
                    P_all = torch.softmax(torch.matmul(zn, image_normalized.transpose(1, 2).float()) / tv, dim=2) / M  # [B,T,N]
                except Exception as _e:
                    print(f"[mmtok] text P fallback (Q-only): {_e}")
            select_idx = torch.empty((B, Kt), dtype=torch.long, device=device)
            for b in range(B):
                Q = Q_all[b]                                                            # [N,N]
                Combined = torch.cat([P_all[b], alpha * Q], dim=0) if P_all is not None else (alpha * Q)
                best = torch.zeros(Combined.shape[0], device=device, dtype=Combined.dtype)
                smask = torch.zeros(N, device=device, dtype=Combined.dtype)
                for i in range(Kt):
                    delta = (Combined - best.unsqueeze(1)).clamp_min(0.0).sum(dim=0) + smask
                    j = int(torch.argmax(delta).item())
                    select_idx[b, i] = j
                    best = torch.maximum(best, Combined[:, j])
                    smask[j] = -float('inf')
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx, True)
            print(f"[mmtok] Kt={Kt} tv={tv} vv={vv} alpha={alpha} text={'yes' if P_all is not None else 'no'}")
        elif self.pruning_method in ('cover_prune_1_proj', 'cover_prune_2_proj', 'cover_prune_2'):
            # COVER-Prune (InternVL port): cls_attn × coverage 选择, projector后 (逐patch).
            #   cover_prune_1_proj (SCOPE, 外加权): score_i = cls_attn_i · Σ_v max(0, sim(v,i)-g_v)
            #   cover_prune_2_proj (内加权):        gain_i = Σ_v cls_attn_v · max(0, sim(v,i)-g_v)
            #   cover_prune_2 (双阶段 stage-1):     同 cp2_proj 内加权, 但保留 2×target (stage-2 在 LLM)
            is_inside = self.pruning_method in ('cover_prune_2_proj', 'cover_prune_2')
            keep_total = self.visual_token_num * 2 if self.pruning_method == 'cover_prune_2' else self.visual_token_num
            tokens_per_patch = max(1, keep_total // B)
            Kt = min(tokens_per_patch, N)
            # cls_attn [B,N] (vit_attns 已在 552 白名单里算出并 shuffle 对齐)
            if vit_attns is not None:
                cls_attn_all = vit_attns.mean(dim=1) if vit_attns.dim() == 3 else vit_attns
            else:
                print(f"[{self.pruning_method}] vit_attns is None, fallback to norm")
                cls_attn_all = vit_embeds.norm(dim=-1)
            image_normalized = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)
            sim_all = torch.clamp(torch.matmul(image_normalized, image_normalized.transpose(1, 2)), min=0.0)  # [B,N,N]
            # 每patch归一化 cls_attn 到 [0,1]
            a_all = cls_attn_all.float()
            a_all = (a_all - a_all.min(dim=1, keepdim=True).values) / (
                a_all.max(dim=1, keepdim=True).values - a_all.min(dim=1, keepdim=True).values + 1e-8)
            select_idx = torch.empty((B, Kt), dtype=torch.long, device=device)
            for b in range(B):
                sim = sim_all[b]; a = a_all[b].to(sim.dtype)
                avail = torch.ones(N, dtype=torch.bool, device=device)
                g = torch.zeros(N, device=device, dtype=sim.dtype)
                # seed = 各自打分公式在 g=0 时的 argmax
                if is_inside:
                    i0 = int(torch.argmax((a.unsqueeze(0) * sim).sum(dim=1)))       # Σ_v a_v·sim(v,i)
                else:
                    i0 = int(torch.argmax(a * sim.sum(dim=1)))                       # a_i·Σ_v sim(v,i)
                select_idx[b, 0] = i0; avail[i0] = False; g = torch.maximum(g, sim[i0])
                for i in range(1, Kt):
                    if is_inside:
                        score = (a.unsqueeze(0) * torch.clamp(sim - g.unsqueeze(0), min=0.0)).sum(dim=1)  # 内加权
                    else:
                        coverage = torch.clamp(sim - g.unsqueeze(0), min=0.0).sum(dim=1)
                        score = a * coverage                                          # 外加权 SCOPE
                    score[~avail] = -float('inf')
                    j = int(torch.argmax(score)); select_idx[b, i] = j; avail[j] = False; g = torch.maximum(g, sim[j])
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx, True)
        elif self.pruning_method in ('attn_cov_vit_proj_llm_12', 'attn_cov_vit_proj', 'attn_cov_proj_llm_12', 'attn_cov_proj') or _is_attn_h_cov_vit_proj(self.pruning_method):
            # Same inner raw-CLS×coverage as LLaVA. vit_proj*: cosine on
            # pre-mlp1 ViT; attn_cov_proj*: cosine on mlp1 (projector) output.
            from vlmeval.vlm.attn_cov_select import inner_weighted_raw_cls_select
            keep_2t = '_llm_' in self.pruning_method
            use_vit = 'vit_proj' in self.pruning_method
            # S1 budget = m*T (m reads H_VIT_S1_MULT / the `_s1m<M>` suffix).
            _s1m = int(getattr(self, '_s1_mult', 2) or 2) if keep_2t else 1
            keep_total = min(self.visual_token_num * _s1m, B * N)
            # Split the budget per patch, handing the remainder to the first
            # patches instead of dropping it: keep_total//B alone would silently
            # shrink the S1 count (e.g. 768 over B=5 -> 765), which breaks the
            # average-T identity the Stage-2 cut assumes.
            base_per_patch, _rem = divmod(int(keep_total), B)
            per_patch_keep = [
                min(base_per_patch + (1 if b < _rem else 0), N) for b in range(B)
            ]
            kt = min(max(per_patch_keep) if per_patch_keep else 1, N)
            cov = vit_pre_proj if use_vit else vit_embeds
            if vit_attns is not None:
                cls_attn_all = vit_attns.mean(dim=1) if vit_attns.dim() == 3 else vit_attns
            else:
                print(f"[{self.pruning_method}] vit_attns is None, fallback to {'pre-proj' if use_vit else 'proj'} L2")
                cls_attn_all = cov.norm(dim=-1)
            masks = []
            for b in range(B):
                masks.append(inner_weighted_raw_cls_select(
                    cov[b], cls_attn_all[b], per_patch_keep[b],
                    attn_h=_is_attn_h_cov_vit_proj(self.pruning_method)
                ))
            selected_masks = torch.stack(masks, dim=0)
            s2 = "S2=TOPS first LLM layer avg T" if keep_2t else "no LLM prune"
            _per = f"{min(per_patch_keep)}-{max(per_patch_keep)}" if per_patch_keep else "0"
            print(
                f"[{self.pruning_method}] S1 {'ViT' if use_vit else 'proj'} raw_cls"
                f"{'*|h|' if _is_attn_h_cov_vit_proj(self.pruning_method) else ''}*coverage: B={B} N={N} "
                f"keep/patch={_per} kept={int(selected_masks.sum().item())} "
                f"({f'{_s1m}T' if keep_2t else 'T'}={keep_total}, {s2})"
            )
        elif self.pruning_method == 'dppruner':
            # Calculate tokens per patch: total_target / num_patches (same as VisionZip)
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1  # At least 1 token per patch

            image_normalized = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)
            kernel_matrix = torch.matmul(image_normalized, image_normalized.transpose(1, 2))

            cis = torch.zeros((tokens_per_patch, B, N), device=device)
            di2s = torch.diagonal(kernel_matrix, dim1=1, dim2=2).clone()
            select_idx = torch.empty((tokens_per_patch, B), dtype=torch.long, device=device)
            for i in range(tokens_per_patch):
                j = torch.argmax(di2s, dim=-1)
                select_idx[i] = j

                eis = (kernel_matrix[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
                    / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
                cis[i, :, :] = eis
                di2s -= torch.square(eis)
                di2s[torch.arange(B), j] = -float('inf')

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx.t(), True)
        elif self.pruning_method == 'cdpruner':
            # Calculate tokens per patch: total_target / num_patches (same as VisionZip)
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1  # At least 1 token per patch

            # CDPruner: Content-Diversity Pruner
            # Combines text relevance and visual diversity using DPP kernel
            image_normalized = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)

            # Compute text relevance
            if text_embeds is not None and len(text_embeds) > 0:
                # Broadcast the SAME text tokens across all image tiles (B). This is
                # robust to multi-image MMMU samples where the tile count B differs
                # from the number of images (the old per-image zip(text_embeds,
                # image_num_patches) assumed 1 image per question and crashed with a
                # batch-dim mismatch on multi-image). Mirrors LLaVA-STAR-Pro2 cdp3:
                # a shared 2D [M, D] text tensor matmul'd against every tile.
                text_all = torch.cat([t.reshape(-1, t.shape[-1]) for t in text_embeds], dim=0)  # [M, D]
                text_normalized = text_all / (text_all.norm(dim=-1, keepdim=True) + 1e-8)
                text_normalized = text_normalized.to(image_normalized.dtype)

                # Relevance: max similarity to any text token, per image tile
                relevance = torch.matmul(image_normalized, text_normalized.t()).max(dim=-1).values  # [B, N]
                # Normalize relevance to [0, 1] per batch
                relevance_min = relevance.min(dim=1, keepdim=True).values
                relevance_max = relevance.max(dim=1, keepdim=True).values
                relevance = (relevance - relevance_min) / (relevance_max - relevance_min + 1e-8)
            else:
                # Fallback: uniform relevance
                relevance = torch.ones(B, N, device=device)

            # Compute visual similarity (for DPP)
            similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # [B, N, N]

            # DPP kernel: relevance * similarity * relevance
            kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # [B, N, N]

            # DPP sampling per batch
            select_idx = torch.empty((B, tokens_per_patch), dtype=torch.long, device=device)
            for b in range(B):
                cis = torch.zeros((tokens_per_patch, N), device=device)
                di2s = torch.diag(kernel[b])  # [N]
                for i in range(tokens_per_patch):
                    j = torch.argmax(di2s)
                    select_idx[b, i] = j

                    # Update for next iteration
                    if i < tokens_per_patch - 1:  # No need to update on last iteration
                        eis = (kernel[b, j, :] - (cis[:i, j] @ cis[:i, :])) / torch.sqrt(di2s[j] + 1e-8)
                        cis[i, :] = eis
                        di2s -= torch.square(eis)
                        di2s[j] = -float('inf')

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks.scatter_(1, select_idx, True)
        elif self.pruning_method == 'star_v3':
            # STAR-V3 Stage 1: THCP (Text-Concept Hierarchical Pruning)
            total_tokens = B * N  # Total visual tokens across all patches
            print(f"\n{'='*80}")
            print(f"[STAR-V3 Stage 1] THCP Text-Concept Coverage")
            print(f"  Visual tokens: {N} per patch ({B} patches) = {total_tokens} total")
            print(f"  Target (final): {self.visual_token_num}")
            print(f"  Stage 1 target: {self.stage1_keep_tokens} (adaptive: target × 2)")

            # Stage 1: Keep target × 2 tokens GLOBALLY (not per-patch)
            stage1_keep_num = min(self.stage1_keep_tokens, total_tokens)

            # Check if text embeddings are available
            if text_embeds is not None and len(text_embeds) > 0:
                # Expand text embeddings for each image patch
                _text_embeds_list = [text_embed[None].expand(num_patch, -1, -1)
                                    for text_embed, num_patch in zip(text_embeds, image_num_patches)]
                text_embeds_expanded = torch.cat(_text_embeds_list, dim=0)  # [B, M, D]
                M = text_embeds_expanded.shape[1]
                print(f"  Text tokens: {M}")
                print(f"  Mode: {'Coverage' if M > 1 else 'Relevance'}")

                # Flatten to treat all patches globally: [B*N, D]
                image_flat = vit_embeds.reshape(-1, D)  # [B*N, D]
                image_normalized = image_flat / (image_flat.norm(dim=-1, keepdim=True) + 1e-8)

                # Use first text embedding for all patches (same question for all)
                text_normalized = text_embeds_expanded[0] / (text_embeds_expanded[0].norm(dim=-1, keepdim=True) + 1e-8)  # [M, D]

                # Compute response matrix globally [B*N, M]
                response_matrix = torch.matmul(image_normalized, text_normalized.t())  # [B*N, M]

                if M > 1:
                    # Coverage mode: average relevance across text tokens
                    text_relevance = response_matrix.mean(dim=-1)  # [B*N]
                else:
                    # Relevance mode: single text token
                    text_relevance = response_matrix.squeeze(-1)  # [B*N]

                # Compute visual similarity globally [B*N, B*N]
                visual_similarity = torch.matmul(image_normalized, image_normalized.t())

                # Global greedy selection
                selected_indices = []
                available_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
                relevance_weight = 0.5
                diversity_weight = 0.5

                for step in range(stage1_keep_num):
                    if not available_mask.any():
                        break

                    scores = text_relevance.clone()
                    if step > 0 and len(selected_indices) > 0:
                        selected_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)
                        max_similarity = visual_similarity[:, selected_tensor].max(dim=1).values
                        diversity_scores = 1 - max_similarity
                        scores = relevance_weight * scores + diversity_weight * diversity_scores

                    scores[~available_mask] = -float('inf')
                    selected_idx = scores.argmax().item()
                    selected_indices.append(selected_idx)
                    available_mask[selected_idx] = False

                # Convert global indices to per-patch mask
                selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
                if len(selected_indices) > 0:
                    selected_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)
                    # Convert flat indices to (batch, position) indices
                    batch_indices = selected_tensor // N
                    pos_indices = selected_tensor % N
                    selected_masks[batch_indices, pos_indices] = True

                print(f"  Selected {len(selected_indices)} tokens total ({len(selected_indices)/B:.1f} per patch average)")
            else:
                # No text embeddings: use attention-based selection as fallback
                print(f"  No text embeddings available, using attention-based fallback")
                if vit_attns is not None:
                    # Use attention from vision encoder - global selection
                    image_attention = vit_attns.mean(dim=1).reshape(-1)  # [B*N]
                    selected_indices_flat = image_attention.topk(stage1_keep_num, dim=-1).indices

                    # Convert to per-patch mask
                    selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
                    batch_indices = selected_indices_flat // N
                    pos_indices = selected_indices_flat % N
                    selected_masks[batch_indices, pos_indices] = True
                else:
                    # Random fallback - global selection
                    random_scores = torch.rand(total_tokens).to(device)
                    selected_indices_flat = random_scores.topk(stage1_keep_num, dim=-1).indices

                    # Convert to per-patch mask
                    selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
                    batch_indices = selected_indices_flat // N
                    pos_indices = selected_indices_flat % N
                    selected_masks[batch_indices, pos_indices] = True

                print(f"  Selected {stage1_keep_num} tokens total ({stage1_keep_num/B:.1f} per patch average)")

            print(f"{'='*80}\n")

        elif self.pruning_method == 'star_qr':
            # STAR-Pro-QR Stage 1 for InternVL3.
            #
            # InternVL emits B dynamic image patches with N visual tokens each.
            # Flatten them in their original patch-major order and run the same
            # global, deterministic residual QR selector used by Qwen STAR-Pro-QR.
            # This remains query agnostic: no text embedding participates.
            #
            # Ported verbatim from the isolated build that produced the
            # STAR-Pro-QR row of the InternVL3-8B main table
            # (/path/to/local-assets, 2026-07-24).
            # Only difference: there, STAGE1_SCORER=qr had to be set explicitly
            # because that build hijacked the star_v3 name; here the method has
            # its own name, so `qr` is the default and the env var only exists
            # to catch a caller asking for a scorer this branch cannot serve.
            import os

            scorer = os.environ.get('STAGE1_SCORER', 'qr').strip().lower() or 'qr'
            if scorer not in ('qr', 'random'):
                raise ValueError(
                    f'star_qr implements qr | random, got STAGE1_SCORER={scorer!r}'
                )
            total_tokens = B * N  # Total visual tokens across all patches
            print(f"\n{'='*80}")
            print(f"[STAR-INTERNVL S1] Global deterministic residual QR")
            print(f"  Visual tokens: {N} per patch ({B} patches) = {total_tokens} total")
            print(f"  Target (final): {self.visual_token_num}")
            print(f"  Stage 1 target: {self.stage1_keep_tokens} (adaptive: target × 2)")

            stage1_mult = float(os.environ.get('STAGE1_MULT', '2'))
            stage1_keep_num = min(
                max(1, int(self.visual_token_num * stage1_mult)),
                total_tokens,
            )
            scope = os.environ.get('STAGE1_QR_SCOPE', 'global').strip().lower()
            if scope != 'global':
                raise ValueError(
                    f'star_qr supports global scope only, got {scope!r}'
                )
            if os.environ.get('STAGE1_QR_REWEIGHT', 'none').strip().lower() != 'none':
                raise ValueError('star_qr requires reweight=none')
            if os.environ.get('STAGE1_MERGE', '0') != '0':
                raise ValueError('star_qr requires merge=0')

            if scorer == 'random':
                # Control arm: keep the same number of tokens, chosen uniformly
                # at random instead of by QR coverage. Everything downstream --
                # budget, schedule, Stage-2 -- is untouched, so a score gap here
                # is attributable to the selector alone. Seeded per call site so
                # the run is reproducible.
                seed = int(os.environ.get('STAGE1_RANDOM_SEED', '0'))
                gen = torch.Generator(device='cpu').manual_seed(
                    seed * 1000003 + total_tokens
                )
                perm = torch.randperm(total_tokens, generator=gen).to(device)
                selected_tensor = perm[:stage1_keep_num]
                selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
                selected_masks[
                    selected_tensor // N,
                    selected_tensor % N,
                ] = True
                print(
                    f"[STAR-INTERNVL S1] scorer=random seed={seed} "
                    f"selected={int(selected_tensor.numel())}/{total_tokens} "
                    f"mult={stage1_mult} target={self.visual_token_num}",
                    flush=True,
                )
                print(f"{'='*80}\n")
                return vit_embeds, selected_masks

            image_flat = vit_embeds.detach().reshape(-1, D).to(torch.float32)
            initial_norms = (image_flat ** 2).sum(dim=1)
            if not torch.isfinite(initial_norms).all():
                raise ValueError('[STAR-INTERNVL S1] non-finite token norms')

            residual = image_flat.clone()
            available = torch.ones(total_tokens, dtype=torch.bool, device=device)
            selected_indices = []
            eps = 1e-8
            while len(selected_indices) < stage1_keep_num:
                residual_norms = (residual ** 2).sum(dim=1)
                residual_norms = torch.where(
                    available,
                    residual_norms,
                    torch.full_like(residual_norms, -1.0),
                )
                selected_idx = int(torch.argmax(residual_norms).item())
                if residual_norms[selected_idx] <= eps:
                    remaining = torch.nonzero(available).flatten()
                    remaining = remaining[
                        torch.argsort(
                            initial_norms[remaining],
                            descending=True,
                            stable=True,
                        )
                    ]
                    selected_indices.extend(
                        int(idx)
                        for idx in remaining[
                            :stage1_keep_num - len(selected_indices)
                        ].tolist()
                    )
                    break
                selected_indices.append(selected_idx)
                available[selected_idx] = False
                basis = residual[selected_idx] / (residual[selected_idx].norm() + eps)
                residual = residual - torch.outer(residual @ basis, basis)

            if (
                len(selected_indices) != stage1_keep_num
                or len(set(selected_indices)) != stage1_keep_num
            ):
                raise ValueError(
                    f'[STAR-INTERNVL S1] bad selection size '
                    f'{len(selected_indices)} (want {stage1_keep_num})'
                )

            selected_tensor = torch.tensor(
                selected_indices,
                dtype=torch.long,
                device=device,
            )
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            selected_masks[
                selected_tensor // N,
                selected_tensor % N,
            ] = True
            print(
                f"[STAR-INTERNVL S1] scorer=qr scope=global "
                f"selected={len(selected_indices)}/{total_tokens} "
                f"mult={stage1_mult} target={self.visual_token_num}",
                flush=True,
            )

            print(f"{'='*80}\n")

        elif self.pruning_method in ('tops', 'tops_2', 'tops_plus'):
            # TOPS Stage 1: Split RDC greedy selection
            # Local patches (first B-1) are pruned together, thumbnail (last patch)
            # is pruned separately. Each group gets half of the total keep budget.
            # For tops_2 (single-stage), stage1_keep_tokens = target (not target×2).
            import os
            a1 = float(os.environ.get("A1", "1.0"))
            alpha_1 = float(os.environ.get("ALPHA_1", "1.0"))
            lambda_1 = float(os.environ.get("LAMBDA_1", "1.0"))
            coverage_method = os.environ.get("COVERAGE", "MEAN")
            enable_debug = os.environ.get("ENABLE_DEBUG", "0") == "1"

            stage1_keep = min(self.stage1_keep_tokens, B * N)
            B_local = B - 1
            if B_local > 0:
                local_keep = (B_local) * stage1_keep // B
                thumb_keep = stage1_keep - local_keep  # handles odd numbers
            else:
                # Only thumbnail exists — give it the full budget
                local_keep = 0
                thumb_keep = stage1_keep

            # ── CLS attention as relevance (R) ──
            if vit_attns is not None:
                if vit_attns.dim() == 3:
                    cls_attn = vit_attns.mean(dim=1)  # (B, N)
                else:
                    cls_attn = vit_attns
            else:
                print("[TOPS S1] error: vit_attns is None")
                cls_attn = vit_embeds.norm(dim=-1)  # (B, N)

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)

            # Helper: RDC greedy selection on a token pool
            def _rdc_select(attn_vec, feat_mat, keep_num):
                """Run greedy RDC selection. Returns list of selected indices (local to the pool)."""
                n_tokens = feat_mat.shape[0]
                keep_num = min(keep_num, n_tokens)
                # Normalize attention to [0, 1]
                attn_vec = (attn_vec - attn_vec.min()) / (attn_vec.max() - attn_vec.min() + 1e-8)
                # Cosine similarity matrix
                feat_n = feat_mat / (feat_mat.norm(dim=-1, keepdim=True) + 1e-8)
                sim = torch.matmul(feat_n, feat_n.t())  # (n, n)

                selected = []
                avail = torch.ones(n_tokens, dtype=torch.bool, device=device)

                # Step 0: pick highest attention token
                i0 = torch.argmax(attn_vec).item()
                selected.append(i0)
                avail[i0] = False
                max_sim_to_S = sim[i0]
                max_sim_to_S[i0] = 1.0

                for _ in range(1, keep_num):
                    if not avail.any():
                        break
                    diversity = 1.0 - max_sim_to_S

                    if coverage_method == "SCOPE":
                        # SCOPE: marginal coverage gain
                        coverage = torch.clamp(sim - max_sim_to_S.unsqueeze(0), min=0.0).sum(dim=1)
                        coverage[~avail] = 0.0
                    else:
                        # MEAN: avg similarity to remaining available tokens
                        sim_to_avail = sim[:, avail].sum(dim=1)
                        n_avail = avail.sum().item()
                        coverage = (sim_to_avail - 1.0) / (n_avail - 1) if n_avail > 1 else torch.zeros(n_tokens, device=device)

                    scores = a1 * (attn_vec / attn_vec.mean().clamp(min=1e-8)) + \
                        alpha_1 * (diversity / diversity.mean().clamp(min=1e-8)) + \
                        lambda_1 * (coverage / coverage.mean().clamp(min=1e-8))
                    scores[~avail] = -float("inf")
                    idx = torch.argmax(scores).item()

                    selected.append(idx)
                    avail[idx] = False
                    max_sim_to_S = torch.maximum(max_sim_to_S, sim[idx])

                return selected

            # ── tops+ merge-first (InternVL): RDC前对每池做bipartite近重复预合并(参数无关,向量化) ──
            # 匹配对共享aff²质心→RDC的diversity自动跳过重复→名额花在不同内容。最终仍RDC选target。
            if self.pruning_method == 'tops_plus' and os.environ.get("TOPS_PLUS_FIRST", "1") == "1":
                vit_embeds = vit_embeds.clone()
                def _premerge_pool(feat):
                    M = feat.shape[0]
                    if M < 2: return feat
                    fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                    sm = torch.matmul(fn, fn.t())
                    ia = torch.arange(0, M, 2, device=feat.device); ib = torch.arange(1, M, 2, device=feat.device)
                    if ia.numel() == 0 or ib.numel() == 0: return feat
                    sab = sm[ia][:, ib]
                    bestb = sab.argmax(dim=1); aff = sab.max(dim=1).values.clamp(min=0.0).float()
                    w = aff * aff
                    nb = feat[ib].clone().float(); ws = torch.ones(ib.shape[0], device=feat.device)
                    nb.index_add_(0, bestb, w.unsqueeze(-1) * feat[ia].float()); ws.index_add_(0, bestb, w)
                    nb = nb / ws.unsqueeze(-1).clamp(min=1e-6)
                    out = feat.clone()
                    out[ib] = nb.to(feat.dtype); out[ia] = nb[bestb].to(feat.dtype)
                    return out
                if B_local > 0:
                    lf = _premerge_pool(vit_embeds[:B_local].reshape(-1, D))
                    vit_embeds[:B_local] = lf.reshape(B_local, N, D)
                vit_embeds[-1] = _premerge_pool(vit_embeds[-1])
                print(f"[tops+FIRST-IV] pool bipartite pre-merge (aff²)")

            # ── Part 1: Local patches (first B-1 patches) — pruned together ──
            if B_local > 0:
                local_keep_clamped = min(local_keep, B_local * N)
                local_attn = cls_attn[:B_local].reshape(-1)       # (B_local*N,)
                local_feat = vit_embeds[:B_local].reshape(-1, D)  # (B_local*N, D)
                local_selected = _rdc_select(local_attn, local_feat, local_keep_clamped)
                local_t = torch.tensor(local_selected, device=device, dtype=torch.long)
                batch_idx = local_t // N
                pos_idx = local_t % N
                selected_masks[batch_idx, pos_idx] = True

            # ── Part 2: Thumbnail patch (last patch) — pruned independently ──
            thumb_keep_clamped = min(thumb_keep, N)
            thumb_attn = cls_attn[-1]             # (N,)
            thumb_feat = vit_embeds[-1]            # (N, D)
            thumb_selected = _rdc_select(thumb_attn, thumb_feat, thumb_keep_clamped)
            thumb_t = torch.tensor(thumb_selected, device=device, dtype=torch.long)
            selected_masks[B - 1, thumb_t] = True

            # ── tops+ merge-back / CRM (InternVL): 选择后把被剪token信息注入保留token(flatten) ──
            # TOPS_PLUS_MERGE=1 开启; TOPS_PLUS_CRM=1 用覆盖残差(正交+范数保持), 否则aff²加权。
            if self.pruning_method == 'tops_plus' and os.environ.get("TOPS_PLUS_MERGE", "0") == "1":
                ve = vit_embeds.reshape(-1, D)
                sm_flat = selected_masks.reshape(-1)
                sel_i = sm_flat.nonzero().squeeze(-1); pru_i = (~sm_flat).nonzero().squeeze(-1)
                if sel_i.numel() > 0 and pru_i.numel() > 0:
                    fn = ve / (ve.norm(dim=-1, keepdim=True) + 1e-8)
                    simps = torch.matmul(fn[pru_i], fn[sel_i].t())
                    assign = simps.argmax(dim=1); aff = simps.max(dim=1).values.clamp(min=0.0).float()
                    ve = ve.clone()
                    if os.environ.get("TOPS_PLUS_CRM", "0") == "1":
                        pe = ve[pru_i].float(); anc = ve[sel_i][assign].float()
                        ah = anc / (anc.norm(dim=-1, keepdim=True) + 1e-8)
                        resid = pe - (pe * ah).sum(-1, keepdim=True) * ah
                        acc = torch.zeros(sel_i.shape[0], D, device=device, dtype=torch.float32)
                        acc.index_add_(0, assign, aff.unsqueeze(-1) * resid)
                        orig = ve[sel_i].float(); v = orig + acc
                        v = v * (orig.norm(dim=-1, keepdim=True) / (v.norm(dim=-1, keepdim=True) + 1e-8))
                        ve[sel_i] = v.to(ve.dtype)
                    else:
                        w = aff * aff
                        mg = ve[sel_i].clone().float(); wsm = torch.ones(sel_i.shape[0], device=device)
                        mg.index_add_(0, assign, w.unsqueeze(-1) * ve[pru_i].float()); wsm.index_add_(0, assign, w)
                        mg = mg / wsm.unsqueeze(-1).clamp(min=1e-6)
                        ve[sel_i] = mg.to(ve.dtype)
                    vit_embeds = ve.reshape(B, N, D)
                    print(f"[tops+MERGE-IV] injected pruned->kept (CRM={os.environ.get('TOPS_PLUS_CRM','0')})")

            if enable_debug:
                per_patch_counts = selected_masks.sum(dim=1).tolist()
                total_selected = selected_masks.sum().item()
                print(
                    f"[TOPS S1] Split: {B}patches ({B_local}local+1thumb), "
                    f"keep={int(total_selected)} (local={int(sum(per_patch_counts[:B_local]))}, "
                    f"thumb={int(per_patch_counts[-1])}), "
                    f"alpha_1={alpha_1}, lambda_1={lambda_1}"
                )
                print(f"[TOPS S1] Per-patch distribution: {[int(c) for c in per_patch_counts]}, stage1_keep={stage1_keep}")
                print(f"[TOPS S1] actual kept={int(total_selected)} (expected ~{stage1_keep})")

        elif self.pruning_method == 'holov':
            # HoloV (single-stage, feature-level): partition each patch's token grid
            # into spatial regions, allocate a keep-budget across regions proportional
            # to region score (attention + feature diversity), then keep the top
            # attention tokens within each region. Selection only (no merge).
            tokens_per_patch = self.visual_token_num // B
            if tokens_per_patch == 0:
                tokens_per_patch = 1
            if vit_attns is None:
                raise ValueError("HoloV requires attention weights, but they are not available")
            cls_attn = vit_attns.mean(dim=1) if vit_attns.dim() == 3 else vit_attns  # [B, N]

            alpha, beta, power, eps = 1.0, 0.09, 1.0, 1e-6
            Hs = int(round(N ** 0.5)); Ws = Hs
            if Hs * Ws != N:
                Hs, Ws = 1, N
            # Region grid (near-square); aim for ~tokens_per_patch regions (spatial coverage)
            R_target = max(1, min(tokens_per_patch, 16))
            gh = max(1, int(R_target ** 0.5))
            gw = max(1, (R_target + gh - 1) // gh)
            row_id = (torch.arange(Hs, device=device) * gh) // Hs        # [Hs]
            col_id = (torch.arange(Ws, device=device) * gw) // Ws        # [Ws]
            region_of_token = (row_id.view(Hs, 1) * gw + col_id.view(1, Ws)).reshape(N)  # [N]
            num_regions = gh * gw

            image_normalized = vit_embeds / (vit_embeds.norm(dim=-1, keepdim=True) + eps)
            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            for b in range(B):
                attn_b = cls_attn[b]  # [N]
                # Per-token feature diversity: variance of cosine sim to other tokens
                S = torch.matmul(image_normalized[b], image_normalized[b].t())  # [N, N]
                eye = 1.0 - torch.eye(N, device=device, dtype=S.dtype)
                S = S * eye
                denom = max(N - 1, 1)
                mean_sim = S.sum(dim=1) / denom
                var_sim = ((S - mean_sim.unsqueeze(1)) ** 2 * eye).sum(dim=1) / denom  # [N]
                attn_scaled = attn_b * 1e3
                var_scale = attn_scaled.abs().mean() / (var_sim.abs().mean() + eps)
                token_scores = alpha * attn_scaled + beta * (var_sim * var_scale)  # [N]

                # Mean token score per region
                region_score = torch.zeros(num_regions, device=device, dtype=token_scores.dtype)
                region_count = torch.zeros(num_regions, device=device, dtype=token_scores.dtype)
                region_score.scatter_add_(0, region_of_token, token_scores)
                region_count.scatter_add_(0, region_of_token, torch.ones(N, device=device, dtype=token_scores.dtype))
                region_score = region_score / region_count.clamp(min=1)

                # Budget allocation proportional to region_score^power
                keep_total = min(tokens_per_patch, N)
                w = region_score.clamp(min=0) ** power
                w = w / (w.sum() + eps)
                alloc = torch.floor(w * keep_total).long()
                remaining = keep_total - int(alloc.sum().item())
                if remaining > 0:
                    extra = torch.topk(w, k=min(remaining, num_regions)).indices
                    alloc[extra] += 1

                # Keep top-attention tokens within each region
                for r in range(num_regions):
                    a = int(alloc[r].item())
                    if a <= 0:
                        continue
                    region_idx = (region_of_token == r).nonzero(as_tuple=True)[0]
                    if region_idx.numel() == 0:
                        continue
                    a = min(a, region_idx.numel())
                    top = attn_b[region_idx].topk(a).indices
                    selected_masks[b, region_idx[top]] = True

                # Fill any shortfall (from region-size capping) by global attention
                cur = int(selected_masks[b].sum().item())
                if cur < keep_total:
                    fill = attn_b.clone()
                    fill[selected_masks[b]] = -float('inf')
                    avail = int((~selected_masks[b]).sum().item())
                    need = min(keep_total - cur, avail)
                    if need > 0:
                        selected_masks[b, fill.topk(need).indices] = True

        elif self.pruning_method == 'vscan':
            # VScan Stage 1 (feature-level): complementary local + global attention
            # scan followed by token MERGING of dropped tokens into their nearest
            # kept token. Keeps stage1_keep_tokens (= target x 2); the LLM Stage 2
            # (modeling_qwen2_vscan) prunes the rest so the per-layer average = target.
            if vit_attns is None:
                raise ValueError("VScan requires attention weights, but they are not available")
            cls_attn = vit_attns.mean(dim=1) if vit_attns.dim() == 3 else vit_attns  # [B, N]

            stage1_keep = getattr(self, 'stage1_keep_tokens', max(1, self.visual_token_num * 2))
            tokens_per_patch = max(1, min(stage1_keep // B if B > 0 else stage1_keep, N))

            Hs = int(round(N ** 0.5)); Ws = Hs
            if Hs * Ws != N:
                Hs, Ws = 1, N

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            for b in range(B):
                attn_b = cls_attn[b]  # [N]
                keep = min(tokens_per_patch, N)
                local_keep = keep // 2
                global_keep = keep - local_keep

                # --- Local scan: spatial coverage via windowed selection ---
                local_sel = torch.zeros(N, dtype=torch.bool, device=device)
                if local_keep > 0:
                    lh = max(1, int(local_keep ** 0.5))
                    lw = max(1, (local_keep + lh - 1) // lh)
                    lrow = (torch.arange(Hs, device=device) * lh) // Hs
                    lcol = (torch.arange(Ws, device=device) * lw) // Ws
                    win_of_token = (lrow.view(Hs, 1) * lw + lcol.view(1, Ws)).reshape(N)
                    num_wins = lh * lw
                    picked = 0
                    for wi in range(num_wins):
                        if picked >= local_keep:
                            break
                        widx = (win_of_token == wi).nonzero(as_tuple=True)[0]
                        if widx.numel() == 0:
                            continue
                        best = widx[attn_b[widx].argmax()]
                        local_sel[best] = True
                        picked += 1
                    # top up if some windows were empty
                    if picked < local_keep:
                        fill = attn_b.clone()
                        fill[local_sel] = -float('inf')
                        avail = int((~local_sel).sum().item())
                        need = min(local_keep - picked, avail)
                        if need > 0:
                            local_sel[fill.topk(need).indices] = True

                # --- Global scan: top attention among the non-local tokens ---
                keep_mask = local_sel.clone()
                if global_keep > 0:
                    gattn = attn_b.clone()
                    gattn[local_sel] = -float('inf')
                    avail = int((~local_sel).sum().item())
                    need = min(global_keep, avail)
                    if need > 0:
                        keep_mask[gattn.topk(need).indices] = True

                selected_masks[b] = keep_mask

                # --- Token merging: fuse dropped tokens into nearest kept token ---
                kept_idx = keep_mask.nonzero(as_tuple=True)[0]
                drop_idx = (~keep_mask).nonzero(as_tuple=True)[0]
                if kept_idx.numel() > 0 and drop_idx.numel() > 0:
                    kept_norm = vit_embeds[b, kept_idx] / (vit_embeds[b, kept_idx].norm(dim=-1, keepdim=True) + 1e-8)
                    drop_norm = vit_embeds[b, drop_idx] / (vit_embeds[b, drop_idx].norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.matmul(drop_norm, kept_norm.t())  # [n_drop, n_keep]
                    assign = sim.argmax(dim=-1)                    # [n_drop]
                    one_hot = vit_embeds.new_zeros(drop_idx.numel(), kept_idx.numel())
                    one_hot.scatter_(1, assign.unsqueeze(-1), 1)
                    counts = one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
                    aggregated = torch.matmul(one_hot.t(), vit_embeds[b, drop_idx]) / counts
                    vit_embeds[b, kept_idx] = vit_embeds[b, kept_idx] + aggregated

        elif self.pruning_method == 'duet':
            # DUET Stage 1 (V2V / VisionZip, feature-level): keep dominant tokens by
            # attention top-k + merge the remaining "contextual" tokens into a small
            # set of anchors. Keeps stage1_keep_tokens (= target x 2); the LLM Stage 2
            # (modeling_qwen2_duet, PyramidDrop) prunes to a per-layer average = target.
            if vit_attns is None:
                raise ValueError("DUET requires attention weights, but they are not available")
            cls_attn = vit_attns.mean(dim=1) if vit_attns.dim() == 3 else vit_attns  # [B, N]

            stage1_keep = getattr(self, 'stage1_keep_tokens', max(1, self.visual_token_num * 2))
            tokens_per_patch = max(1, min(stage1_keep // B if B > 0 else stage1_keep, N))
            dominant_frac = 540.0 / 640.0  # DUET duet_640 preset: 540 dominant / 100 contextual

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            for b in range(B):
                attn_b = cls_attn[b]  # [N]
                keep = min(tokens_per_patch, N)
                dominant_num = min(N, max(1, int(keep * dominant_frac)))
                contextual_num = max(0, keep - dominant_num)

                # --- Dominant: attention top-k selection ---
                dom_idx = attn_b.topk(dominant_num).indices
                dom_mask = torch.zeros(N, dtype=torch.bool, device=device)
                dom_mask[dom_idx] = True

                # --- Contextual: cluster-merge the rest into evenly-spaced anchors ---
                rest_idx = (~dom_mask).nonzero(as_tuple=True)[0]
                if contextual_num > 0 and rest_idx.numel() > 0:
                    contextual_num = min(contextual_num, rest_idx.numel())
                    rest_norm = vit_embeds[b, rest_idx] / (vit_embeds[b, rest_idx].norm(dim=-1, keepdim=True) + 1e-8)
                    step = max(1, rest_idx.numel() // contextual_num)
                    anchor_local = torch.arange(0, rest_idx.numel(), step, device=device)[:contextual_num]
                    anchor_idx = rest_idx[anchor_local]
                    selected_masks[b, dom_idx] = True
                    selected_masks[b, anchor_idx] = True
                    # assign non-anchor rest tokens to nearest anchor and merge
                    src_local_mask = torch.ones(rest_idx.numel(), dtype=torch.bool, device=device)
                    src_local_mask[anchor_local] = False
                    src_local = src_local_mask.nonzero(as_tuple=True)[0]
                    if src_local.numel() > 0:
                        anchor_norm = rest_norm[anchor_local]
                        src_norm = rest_norm[src_local]
                        sim = torch.matmul(src_norm, anchor_norm.t())  # [n_src, n_anchor]
                        assign = sim.argmax(dim=-1)
                        one_hot = vit_embeds.new_zeros(src_local.numel(), anchor_idx.numel())
                        one_hot.scatter_(1, assign.unsqueeze(-1), 1)
                        counts = one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
                        aggregated = torch.matmul(one_hot.t(), vit_embeds[b, rest_idx[src_local]]) / counts
                        vit_embeds[b, anchor_idx] = vit_embeds[b, anchor_idx] + aggregated
                else:
                    selected_masks[b, dom_idx] = True

        elif self.pruning_method == 'duet_official':
            # DUET Stage 1, faithful to the official VisionZip implementation
            # (AMD-AGI/DUET-VLM, visionzip/clip_encoder_cw.py).
            #
            # The ONLY difference from the `duet` branch above is the merge
            # scope. Official preselects `cluster_width * contextual` tokens by
            # CLS attention and clusters ONLY those into the anchors; every
            # token outside dominant + preselect is discarded outright. The
            # `duet` branch instead merges the entire non-dominant remainder,
            # which hands the baseline strictly more information than its paper
            # describes. Dominant fraction, anchor spacing and the Stage-2
            # schedule are held identical so the gap isolates this one factor.
            import os

            if vit_attns is None:
                raise ValueError("DUET requires attention weights, but they are not available")
            cls_attn = vit_attns.mean(dim=1) if vit_attns.dim() == 3 else vit_attns  # [B, N]

            stage1_keep = getattr(self, 'stage1_keep_tokens', max(1, self.visual_token_num * 2))
            tokens_per_patch = max(1, min(stage1_keep // B if B > 0 else stage1_keep, N))
            # Official preset: --dominant 300 --contextual 7 --cluster_width 4
            # => contextual is 7/307 of the kept budget, dominant is the rest.
            # (The `duet` branch next door uses 540/640 = 15.6% contextual, i.e.
            # ~7x more contextual budget than DUET's own scripts prescribe.)
            contextual_frac = float(os.environ.get('DUET_CONTEXTUAL_FRAC', str(7.0 / 307.0)))
            dominant_frac = 1.0 - contextual_frac
            cluster_width = int(os.environ.get('DUET_CLUSTER_WIDTH', '4'))  # official default

            selected_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            merged_total = 0
            discarded_total = 0
            for b in range(B):
                attn_b = cls_attn[b]  # [N]
                keep = min(tokens_per_patch, N)
                dominant_num = min(N, max(1, int(keep * dominant_frac)))
                contextual_num = max(0, keep - dominant_num)

                # --- Dominant: attention top-k selection ---
                dom_idx = attn_b.topk(dominant_num).indices
                dom_mask = torch.zeros(N, dtype=torch.bool, device=device)
                dom_mask[dom_idx] = True
                selected_masks[b, dom_idx] = True

                rest_idx = (~dom_mask).nonzero(as_tuple=True)[0]
                if contextual_num <= 0 or rest_idx.numel() == 0:
                    continue

                contextual_num = min(contextual_num, rest_idx.numel())

                # --- Official preselect: top (cluster_width x contextual) by attention ---
                preselect_num = min(max(1, cluster_width * contextual_num), rest_idx.numel())
                rest_attn = attn_b[rest_idx]
                preselect_local = rest_attn.topk(preselect_num).indices          # into rest_idx
                preselect_idx = rest_idx[preselect_local]

                # --- Anchors: evenly spaced WITHIN the preselected pool ---
                step = max(1, preselect_num // max(1, contextual_num))
                anchor_pos = torch.arange(0, preselect_num, step, device=device)[:contextual_num]
                if anchor_pos.numel() == 0:
                    anchor_pos = torch.tensor([0], device=device)
                anchor_idx = preselect_idx[anchor_pos]
                selected_masks[b, anchor_idx] = True

                # --- Merge: pool-minus-anchors into nearest anchor; rest DISCARDED ---
                keep_mask = torch.ones(preselect_num, dtype=torch.bool, device=device)
                keep_mask[anchor_pos] = False
                merge_idx = preselect_idx[keep_mask]
                merged_total += int(merge_idx.numel())
                discarded_total += int(rest_idx.numel() - preselect_num)
                if merge_idx.numel() > 0:
                    denom = vit_embeds[b, merge_idx].norm(dim=-1, keepdim=True) + 1e-8
                    src_norm = vit_embeds[b, merge_idx] / denom
                    anchor_norm = vit_embeds[b, anchor_idx] / (
                        vit_embeds[b, anchor_idx].norm(dim=-1, keepdim=True) + 1e-8)
                    sim = torch.matmul(src_norm, anchor_norm.t())
                    assign = sim.argmax(dim=-1)
                    one_hot = vit_embeds.new_zeros(merge_idx.numel(), anchor_idx.numel())
                    one_hot.scatter_(1, assign.unsqueeze(-1), 1)
                    counts = one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
                    aggregated = torch.matmul(one_hot.t(), vit_embeds[b, merge_idx]) / counts
                    vit_embeds[b, anchor_idx] = vit_embeds[b, anchor_idx] + aggregated

            print(
                f"[DUET-OFFICIAL S1] contextual_frac={contextual_frac:.5f} "
                f"cluster_width={cluster_width} "
                f"kept={int(selected_masks.sum().item())} "
                f"merged={merged_total} discarded={discarded_total} "
                f"(the `duet` branch would merge {merged_total + discarded_total} "
                f"and discard 0)",
                flush=True,
            )

        return vit_embeds, selected_masks

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            num_patches_list: Optional[List[int]] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if self.pruning_method == 'apet' and (pixel_values is None or visual_features is not None):
            raise ValueError('ApET requires pixel_values so that S1 runs before mlp1')
        if pixel_values is not None:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

            reverse_input_ids = input_ids.flip(-1)
            reverse_mask = (reverse_input_ids == self.img_context_token_id)
            text_instruction_length = int(reverse_mask.max(dim=-1).indices[0])

            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, selected_masks = self.extract_feature(
                    pixel_values,
                    image_num_patches=num_patches_list,
                    text_embeds=input_embeds[:, -text_instruction_length:, :],
                )
            # For FastV or STAR-V3 pruning, use average visual tokens from language model
            if self.pruning_method in ['fastv', 'star_v3', 'tops', 'vscan', 'duet', 'duet_official'] and hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'visual_token_num'):
                lm_model = self.language_model.model
                self.visual_token_len = lm_model.visual_token_num
            else:
                # For vision encoder pruning methods, use total selected tokens
                # selected_masks shape: [num_patches, tokens_per_patch]
                # We want to display TOTAL tokens across all patches, not per-patch average
                self.visual_token_len = selected_masks.sum().item()

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            if self.pruning_method == 'apet':
                visual_positions = selected.nonzero().flatten()
                if B != 1 or not torch.equal(
                        visual_positions,
                        torch.arange(visual_positions[0], visual_positions[-1] + 1,
                                     device=visual_positions.device)):
                    raise ValueError('ApET requires a single contiguous visual span')
                self.language_model.set_system_prompt_length(int(visual_positions[0]))
                self.language_model.set_visual_token_length(int(selected_masks.sum()))
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)

            # Prepare default position ids
            position_ids = torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0)

            # Select visual tokens using `selected_masks`
            if selected_masks.sum() < selected_masks.numel():
                system_prompt_length = int(selected.max(dim=-1).indices)
                selected_masks = selected_masks.reshape(-1)
                visual_token_length = selected_masks.shape[0]
                visual_token_indices = selected_masks.nonzero().squeeze(-1)
                retain_token_indices = torch.cat((
                    torch.arange(system_prompt_length, device=input_embeds.device),
                    visual_token_indices + system_prompt_length,
                    torch.arange(system_prompt_length+visual_token_length, input_embeds.shape[1], device=input_embeds.device),
                ))

                input_embeds = input_embeds[:, retain_token_indices, :]
                attention_mask = attention_mask[:, retain_token_indices]
                position_ids = position_ids[:, retain_token_indices]

                # Update system_prompt_length for STAR-V3 / TOPS / VScan / DUET Stage 2
                if self.pruning_method in ['star_v3', 'tops', 'vscan', 'duet', 'duet_official']:
                    self.system_prompt_length = system_prompt_length
            else:
                position_ids = None
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            position_ids = None

        if self.pruning_method == 'apet':
            # The reference ApET starts S1-compressed prompts at position zero;
            # its midpoint merge independently compacts the later-layer positions.
            position_ids = torch.arange(input_embeds.shape[1], device=input_embeds.device)[None]

        # FastV: Set system prompt length for pruning in generate mode
        if self.pruning_method == 'fastv' and pixel_values is not None and input_embeds.shape[1] > 1:
            # Calculate where visual tokens start
            if 'selected' in locals() and selected.sum() > 0:
                # Find the first image context token position
                first_img_token = int(selected.nonzero()[0].item())
                system_prompt_length = first_img_token

                # Typical InternVL3 system prompt is ~41 tokens
                if system_prompt_length < 30 or system_prompt_length > 100:
                    logger.warning(f"[FastV Generate] Unexpected system_prompt_length: {system_prompt_length} "
                                 f"(expected ~41).")
            else:
                system_prompt_length = 41  # Default for InternVL3

            # Set system_prompt_length on the language model
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)

        # TOPS: Set system prompt length and visual token length for Stage 2
        if (self.pruning_method in ('tops', 'tops_2', 'tops_plus', 'cover_prune_2', 'attn_cov_vit_proj_llm_12', 'attn_cov_proj_llm_12') or _is_attn_h_cov_vit_proj_llm(self.pruning_method)) and pixel_values is not None and input_embeds.shape[1] > 1:
            if 'selected' in locals() and selected.sum() > 0:
                first_img_token = int(selected.nonzero()[0].item())
                system_prompt_length = first_img_token
            else:
                system_prompt_length = 41
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)
            # Tell TOPS decoder how many visual tokens survived Stage 1
            if hasattr(self.language_model, 'model'):
                self.language_model.model.visual_token_length = int(selected_masks.sum().item())

        # STAR-V3 / STAR-Pro-QR: Set system prompt length
        if self.pruning_method in ('star_v3', 'star_qr') and input_embeds.shape[1] > 1:
            if 'selected' in locals() and selected.sum() > 0:
                first_img_token = int(selected.nonzero()[0].item())
                system_prompt_length = first_img_token
            else:
                system_prompt_length = 41
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)

        # VScan / DUET: Set system prompt length for Stage 2 (decoder self-detects
        # the Stage-1 survivor count from the already-sliced sequence, like STAR-V3).
        if self.pruning_method in ['vscan', 'duet', 'duet_official'] and pixel_values is not None and input_embeds.shape[1] > 1:
            if 'selected' in locals() and selected.sum() > 0:
                first_img_token = int(selected.nonzero()[0].item())
                system_prompt_length = first_img_token
            else:
                system_prompt_length = 41
            if hasattr(self.language_model, 'set_system_prompt_length'):
                self.language_model.set_system_prompt_length(system_prompt_length)

        generate_inputs = dict(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        if position_ids is not None:
            generate_inputs["position_ids"] = position_ids

        outputs = self.language_model.generate(**generate_inputs)

        if self.pruning_method == 'apet':
            self.visual_token_len = self.language_model.model.visual_token_num

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
