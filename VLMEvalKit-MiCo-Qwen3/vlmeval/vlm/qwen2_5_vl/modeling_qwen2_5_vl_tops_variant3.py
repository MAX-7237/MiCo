# coding=utf-8

import math
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel


logger = logging.get_logger(__name__)


# TOPS_VARIANT3 剪枝配置: Qwen2.5-VL-7B (28 layers)
#
# 算法: initial = target × 2，三阶段剪枝
#   Stage 1 (L0-11, 12层): 保持 initial tokens
#   Stage 2 (L12-15, 4层): 剪枝到 a tokens
#   Stage 3 (L16-27, 12层): 剪枝到 b tokens
#
# 约束: (initial×12 + a×4 + b×12) / 28 = target
#      即: a + 3b = target
TOPS_VARIANT3_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "7b": {
        512: [(12, 281), (16, 77)],   # initial=1024, (1024×12+281×4+77×12)/28=512
        256: [(12, 139), (16, 39)],   # initial=512,  (512×12+139×4+39×12)/28=256
        128: [(12, 71), (16, 19)],    # initial=256,  (256×12+71×4+19×12)/28=128.0 (a+3b=71+57=128 ✓)
    },
}

class TOPSVariant3Qwen2_5_VLModel(Qwen2_5_VLModel):
    """
    TOPS-Variant3 stage-2 implementation for Qwen2.5-VL.

    Progressive three-step pruning with multi-token text guidance, following
    TOPS-Variant3 technical specification. Assumes Stage-1 THCP has already reduced
    visual tokens close to `2 × target_visual_tokens`.
    """

    def __init__(self, config: Qwen2_5_VLConfig, visual_token_num: int):
        super().__init__(config)
        # transformers 4.52+ moved transformer layers into self.language_model
        # Create aliases so the forward() method (written for old structure) still works
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._update_causal_mask = lm._update_causal_mask

        self.system_prompt_length = 15
        self.visual_token_length = 1296
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

        try:
            visual_token_num = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")

        # num_hidden_layers lives on text_config in transformers 4.52+
        n_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers
        if n_layers == 28:
            self.scale = "7b"
        else:
            self.scale = "unknown"

        self.target_visual_tokens = visual_token_num
        self.stage1_keep_tokens = (
            max(1, visual_token_num) * 2 if visual_token_num > 0 else self.visual_token_length
        )
        self.stage0_visual_tokens = self.visual_token_length
        self.stage1_expected_tokens = min(self.visual_token_length, self.stage1_keep_tokens)

        schedule = TOPS_VARIANT3_SCHEDULE.get(self.scale, {}).get(self.target_visual_tokens)
        if schedule is None:
            self.pruning_layers: Dict[int, int] = {}
            if visual_token_num > 0:
                logger.warning_once(
                    "TOPS-Variant3 schedule missing for target %s on scale %s. Progressive pruning disabled.",
                    visual_token_num,
                    self.scale,
                )
        else:
            self.pruning_layers = {layer_idx: target for layer_idx, target in schedule}

        self.reset_state()

    def reset_state(self):
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.visual_token_sum = 0.0
        self.stage1_actual_tokens: Optional[int] = None
        self.stage2_events: List[Tuple[int, int, int]] = []
        self.stage2_skip_events: List[Tuple[int, int, int]] = []
        self.stage2_reported_layers: Set[int] = set()
        self._stage_summary_emitted = False
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _compute_schedule_average(
        self,
        stage1_tokens: Optional[int],
        total_layers: int,
    ) -> Optional[float]:
        if stage1_tokens is None or total_layers <= 0:
            return None

        sorted_layers = sorted(self.pruning_layers.items())
        total_tokens = 0.0
        current_tokens = float(stage1_tokens)
        current_layer = 1

        for layer_idx, target in sorted_layers:
            if layer_idx < current_layer or layer_idx > total_layers:
                continue

            if layer_idx > current_layer:
                span = layer_idx - current_layer
                total_tokens += span * current_tokens

            new_tokens = float(min(current_tokens, target))
            total_tokens += new_tokens  # pruning layer uses pruned tokens
            current_tokens = new_tokens
            current_layer = layer_idx + 1

        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens

        return total_tokens / total_layers

    def _compute_observed_average(self, total_layers: int) -> Optional[float]:
        if (
            not self.prefill_done
            or self.stage1_actual_tokens is None
            or total_layers <= 0
        ):
            return None

        total_tokens = 0.0
        current_tokens = float(self.stage1_actual_tokens)
        current_layer = 1

        for layer_idx, before_tokens, after_tokens in sorted(self.stage2_events):
            if layer_idx < current_layer or layer_idx > total_layers:
                continue

            if before_tokens is not None and abs(before_tokens - current_tokens) > 1e-5:
                logger.debug(
                    "TOPS-Variant3 average mismatch detected at layer %s: expected %.2f, recorded %.2f",
                    layer_idx,
                    current_tokens,
                    before_tokens,
                )

            if layer_idx > current_layer:
                span = layer_idx - current_layer
                total_tokens += span * current_tokens

            new_tokens = float(after_tokens)
            total_tokens += new_tokens
            current_tokens = new_tokens
            current_layer = layer_idx + 1

        if current_layer <= total_layers:
            total_tokens += (total_layers - current_layer + 1) * current_tokens

        return total_tokens / total_layers

    def _select_text_raters(
        self,
        hidden_states: torch.Tensor,
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        # hidden_states: [batch, seq, hidden]
        text_hidden = hidden_states[:, visual_end:, :]
        if text_hidden.size(1) == 0:
            return hidden_states.new_tensor([hidden_states.shape[1] - 1], dtype=torch.long)

        visual_hidden = hidden_states[:, visual_start:visual_end, :]
        text_norm = F.normalize(text_hidden, dim=-1)
        visual_norm = F.normalize(visual_hidden, dim=-1)

        # [batch, text_len, visual_len]
        text_visual_sim = torch.matmul(text_norm, visual_norm.transpose(1, 2))
        text_importance = text_visual_sim.mean(dim=-1)  # average over visual tokens
        if text_importance.dim() == 2:
            text_importance = text_importance.mean(dim=0)  # aggregate across batch

        mean_importance = text_importance.mean()
        selected = torch.nonzero(text_importance > mean_importance, as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            keep = max(1, text_importance.numel() // 2)
            selected = text_importance.topk(keep).indices

        return selected

    def _compute_visual_token_importance(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        self_attn = decoder_layer.self_attn

        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        num_heads = self_attn.num_heads
        num_kv_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim

        print(f"    [_compute_visual_token_importance]")
        print(f"      Input: bsz={bsz}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
        print(f"      Visual range: [{visual_start}:{visual_end}]")

        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)

        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)

        cos, sin = position_embeddings
        print(f"      Position embeddings: cos.shape={cos.shape}, sin.shape={sin.shape}")
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            print(f"      After unsqueeze: cos.shape={cos.shape}, sin.shape={sin.shape}")

        query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        text_rater_indices = self._select_text_raters(hidden_states, visual_start, visual_end)
        text_positions = text_rater_indices + visual_end
        text_positions = text_positions.clamp(max=seq_len - 1)

        query_raters = torch.index_select(query_states, 2, text_positions)
        key_visual = key_states[:, :, visual_start:visual_end, :]

        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(head_dim)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights.mean(dim=2).mean(dim=1)  # average over raters and heads
        visual_importance = attn_weights.mean(dim=0)  # aggregate across batch

        return visual_importance

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        def _is_empty_cache(cache: Optional[Union[DynamicCache, List[torch.FloatTensor]]]) -> bool:
            if cache is None:
                return True
            if isinstance(cache, DynamicCache):
                return cache.get_seq_length() == 0
            return False

        if _is_empty_cache(past_key_values):
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.is_gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        seq_length = inputs_embeds.shape[1]

        if seq_length > 1 and not self.prefill_done:
            print("\n" + "="*80)
            print("[TOPS-Variant3 Qwen2.5-VL DEBUG] Prefill Phase Start")
            print("="*80)
            print(f"  Sequence length: {seq_length}")
            print(f"  System prompt length: {self.system_prompt_length}")
            print(f"  Inputs_embeds shape: {inputs_embeds.shape}")
            print(f"  Position_ids shape: {position_ids.shape}")
            print(f"  Position_embeddings type: {type(position_embeddings)}")
            if isinstance(position_embeddings, (list, tuple)):
                print(f"  Position_embeddings[0] shape: {position_embeddings[0].shape}")
                print(f"  Position_embeddings[1] shape: {position_embeddings[1].shape}")
            detected_visual_len = max(seq_length - self.system_prompt_length - 10, 0)
            if self.target_visual_tokens > 0:
                expected = min(self.visual_token_length, self.stage1_keep_tokens)
                detected_visual_len = min(detected_visual_len, expected)
            print(f"  Calculated visual length: {detected_visual_len}")

            if detected_visual_len <= 0:
                detected_visual_len = max(
                    min(self.visual_token_length, seq_length - self.system_prompt_length - 1), 0
                )

            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self.prefill_done = True
            self.visual_token_sum = 0.0
            print(
                f"[TOPS-Variant3 DEBUG] Detected visual_len={detected_visual_len}, stage1_keep_tokens={self.stage1_keep_tokens}",
                flush=True,
            )
            print("\n[TOPS-Variant3 Stage 1] Input analysis:", flush=True)
            print(f"  - Stage-0 (THCP input) tokens: {self.stage0_visual_tokens}", flush=True)
            print(f"  - Stage-1 expected tokens (target×2 cap): {self.stage1_expected_tokens}", flush=True)
            print(f"  - Stage-1 detected tokens: {detected_visual_len}", flush=True)
            if self.stage0_visual_tokens > 0:
                reduction = self.stage0_visual_tokens - detected_visual_len
                reduction_pct = 100.0 * reduction / self.stage0_visual_tokens
                print(f"  - Stage-1 reduction: {reduction} tokens ({reduction_pct:.2f}%)", flush=True)

        star_enabled = (
            self.prefill_done
            and self.current_visual_length is not None
            and self.current_visual_length > 0
            and len(self.pruning_layers) > 0
        )
        visual_token_accumulator = 0.0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1

            # Accumulate visual tokens BEFORE processing (to record what this layer uses)
            if seq_length > 1 and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            # Debug: Print layer info before processing
            if seq_length > 1 and layer_idx <= 3:
                print(f"\n[Layer {layer_idx}] Before layer processing:")
                print(f"  Hidden states: {hidden_states.shape}")
                print(f"  Current visual length: {self.current_visual_length}")

            should_prune = (
                star_enabled
                and layer_idx in self.pruning_layers
                and self.current_visual_length is not None
                and self.current_visual_length > self.pruning_layers[layer_idx]
            )

            if should_prune:
                target = self.pruning_layers[layer_idx]
                print(f"\n[TOPS-Variant3 PRUNING] Layer {layer_idx}")
                print(f"  Before: {self.current_visual_length} visual tokens → Target: {target}")
                print(f"  Hidden states shape: {hidden_states.shape}")
                print(f"  Position_ids shape: {position_ids.shape}")

                visual_start = self.system_prompt_length
                visual_end = visual_start + self.current_visual_length
                print(f"  Visual range: [{visual_start}:{visual_end}]")

                visual_importance = self._compute_visual_token_importance(
                    hidden_states,
                    decoder_layer,
                    position_embeddings,
                    visual_start,
                    visual_end,
                )

                print(f"  Visual importance shape: {visual_importance.shape}")
                print(f"  Visual importance stats: min={visual_importance.min():.6f}, "
                      f"max={visual_importance.max():.6f}, mean={visual_importance.mean():.6f}")

                keep = min(target, visual_importance.shape[0])
                keep = max(keep, 0)

                if keep == 0:
                    visual_token_indices = visual_importance.new_empty((0,), dtype=torch.long)
                else:
                    visual_token_indices = torch.sort(visual_importance.topk(keep).indices).values
                print(f"[TOPS-Variant3 Stage 2] Keeping {visual_token_indices.numel()} tokens.", flush=True)
                print(
                    f"[TOPS-Variant3 Stage 2] visual_importance stats: mean={visual_importance.mean():.4f}, "
                    f"max={visual_importance.max():.4f}, min={visual_importance.min():.4f}",
                    flush=True,
                )
            else:
                visual_importance = None
                visual_token_indices = None
                if (
                    star_enabled
                    and self.current_visual_length is not None
                    and layer_idx in self.pruning_layers
                    and layer_idx not in self.stage2_reported_layers
                ):
                    target = self.pruning_layers[layer_idx]
                    print(
                        f"[TOPS-Variant3 Stage 2] Layer {layer_idx}: skip pruning (current {self.current_visual_length} ≤ target {target})",
                        flush=True,
                    )
                    self.stage2_skip_events.append((layer_idx, self.current_visual_length, target))
                    self.stage2_reported_layers.add(layer_idx)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if should_prune:
                device = hidden_states.device
                prev_length = self.current_visual_length
                visual_start = self.system_prompt_length
                visual_end = visual_start + prev_length

                retain_token_indices = torch.cat(
                    (
                        torch.arange(self.system_prompt_length, device=device),
                        visual_token_indices + self.system_prompt_length,
                        torch.arange(visual_end, hidden_states.shape[1], device=device),
                    )
                )

                hidden_states = hidden_states[:, retain_token_indices].contiguous()
                position_ids = position_ids[:, :, retain_token_indices]
                cache_position = cache_position[: retain_token_indices.shape[0]]
                position_embeddings = tuple(
                    pos_emb[:, :, retain_token_indices] for pos_emb in position_embeddings
                )
                causal_mask = self._update_causal_mask(
                    None, hidden_states, cache_position, past_key_values, output_attentions
                )

                self.current_visual_length = visual_token_indices.numel()
                star_enabled = self.current_visual_length > 0
                self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                self.stage2_reported_layers.add(layer_idx)
                print(f"  After pruning:")
                print(f"    Hidden states shape: {hidden_states.shape}")
                print(f"    Position_ids shape: {position_ids.shape}")
                print(f"    Visual tokens: {self.current_visual_length} (target {target})")
                print(f"    System prompt: {self.system_prompt_length}")
                print(f"    Text tokens: {hidden_states.shape[1] - self.system_prompt_length - self.current_visual_length}")
                print(f"    Total sequence: {hidden_states.shape[1]}")
                print(f"    Breakdown: {self.system_prompt_length} (sys) + {self.current_visual_length} (vis) + {hidden_states.shape[1] - self.system_prompt_length - self.current_visual_length} (text) = {hidden_states.shape[1]}")

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Debug: Print layer info after processing
            if seq_length > 1 and (layer_idx <= 3 or layer_idx in [12, 13, 16, 17, 27, 28]):
                print(f"[Layer {layer_idx}] After layer processing:")
                print(f"  Hidden states: {hidden_states.shape}")
                print(f"  Current visual length: {self.current_visual_length}")
                if self.current_visual_length:
                    print(f"  Visual range: [{self.system_prompt_length}:{self.system_prompt_length + self.current_visual_length}]")

        hidden_states = self.norm(hidden_states)

        if seq_length > 1 and self.current_visual_length is not None:
            average_tokens = (
                visual_token_accumulator / len(self.layers) if self.layers else 0.0
            )
            self.visual_token_num = float(average_tokens)
            self.visual_token_len = float(average_tokens)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if (
            seq_length > 1
            and self.prefill_done
            and not self._stage_summary_emitted
        ):
            total_layers = len(self.layers)
            observed_average = self._compute_observed_average(total_layers)
            schedule_average_expected = self._compute_schedule_average(
                self.stage1_expected_tokens,
                total_layers,
            )

            print("\n[TOPS-Variant3 DEBUG] Stage summary:", flush=True)
            if self.stage1_actual_tokens is not None:
                print(
                    f"  - Stage 1: {self.stage0_visual_tokens} → {self.stage1_actual_tokens} "
                    f"(expected cap {self.stage1_expected_tokens})",
                    flush=True,
                )
            else:
                print("  - Stage 1: not detected", flush=True)

            if self.stage2_events:
                print("  - Stage 2 reductions:", flush=True)
                for layer_idx, before, after in self.stage2_events:
                    print(f"      Layer {layer_idx}: {before} → {after}", flush=True)
                print(f"  - Stage 2 final tokens: {self.current_visual_length}", flush=True)
            else:
                print("  - Stage 2: no pruning applied (schedule missing or already at target)", flush=True)

            if self.stage2_skip_events:
                print("  - Stage 2 skipped layers:", flush=True)
                for layer_idx, current_len, target in self.stage2_skip_events:
                    print(
                        f"      Layer {layer_idx}: current {current_len} ≤ target {target} (no pruning performed)",
                        flush=True,
                    )

            if observed_average is not None:
                print(
                    f"  - Layer-average tokens (observed): {observed_average:.2f} "
                    f"(target {self.target_visual_tokens})",
                    flush=True,
                )
            if schedule_average_expected is not None:
                print(
                    f"  - Layer-average tokens (schedule, stage1={self.stage1_expected_tokens}): "
                    f"{schedule_average_expected:.2f}",
                    flush=True,
                )

            self._stage_summary_emitted = True

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
