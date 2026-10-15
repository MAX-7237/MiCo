# coding=utf-8
"""
STAR-V3 Stage 2 implementation for LLaVA-Video with Qwen2.

Progressive two-layer pruning with multi-token text guidance.
Adapted from Qwen2.5-VL implementation for Qwen2 + LLaVA-Video.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Cache,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa
)


# Schedule: two-layer pruning (layer 12, 21) for 28-layer model
# Values are per-frame tokens
# Layer distribution: Layer 1-12 (12 layers), Layer 13-21 (9 layers), Layer 22-28 (7 layers)
# Average calculation: (stage1_per_frame × 12 + layer12_target × 9 + layer21_target × 7) / 28
STAR_V3_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "7b": {
        64: [(12, 23), (21, 7)],   # per-frame: 128→23→7, avg=64.00
        32: [(12, 11), (21, 4)],   # per-frame: 64→11→4, avg=31.96
        16: [(12, 5), (21, 3)],    # per-frame: 32→5→3, avg=16.07
    }
}


class STARV3Qwen2Model(Qwen2Model):
    """
    STAR-V3 Stage 2 for LLaVA-Video with Qwen2.

    Two-layer progressive pruning with multi-token text guidance.
    Assumes Stage-1 THCP has already reduced visual tokens to ~2×target per frame.
    """

    def __init__(self, config: Qwen2Config, starv3_config: Dict):
        super().__init__(config)

        # LLaVA-Video configuration
        self.system_prompt_length = 35
        self.visual_token_length = 169  # 13×13 per frame
        self.video_frame_num = 0
        self.visual_token_num = 0.0

        # STAR-V3 configuration
        try:
            self.target_visual_tokens = int(starv3_config["T"])
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"starv3_config must contain 'T' (int), got {starv3_config}")

        self.stage1_keep_tokens = max(1, self.target_visual_tokens) * 2
        self.stage0_visual_tokens = self.visual_token_length

        # Load schedule
        if config.num_hidden_layers == 28:
            self.scale = "7b"
        else:
            self.scale = "unknown"

        schedule = STAR_V3_SCHEDULE.get(self.scale, {}).get(self.target_visual_tokens)
        if schedule is None:
            self.pruning_layers: Dict[int, int] = {}
            print(f"[STAR-V3 Warning] No schedule for T={self.target_visual_tokens}, "
                  f"scale={self.scale}. Stage 2 pruning disabled.")
        else:
            self.pruning_layers = {layer_idx: target for layer_idx, target in schedule}

        self.reset_state()

        print(f"[STAR-V3 Stage2 Init] T={self.target_visual_tokens}, "
              f"Stage1_per_frame={self.stage1_keep_tokens}, "
              f"Pruning layers: {list(self.pruning_layers.keys())}")

    def reset_state(self):
        """Reset state for new generation"""
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.actual_video_frames: Optional[int] = None  # Fixed frame count
        self.visual_token_sum = 0.0
        self.stage1_actual_tokens: Optional[int] = None
        self.stage2_events: List[Tuple[int, int, int]] = []
        self.stage2_skip_events: List[Tuple[int, int, int]] = []
        self.stage2_reported_layers: Set[int] = set()
        self._stage_summary_emitted = False
        self.visual_token_num = 0.0

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input (for RoPE)"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _select_text_raters(
        self,
        hidden_states: torch.Tensor,
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        """Select important text tokens as raters (SparseVLM-style)"""
        text_hidden = hidden_states[:, visual_end:, :]
        if text_hidden.size(1) == 0:
            # Fallback: use last token
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
            # Fallback: keep top 50%
            keep = max(1, text_importance.numel() // 2)
            selected = text_importance.topk(keep).indices

        return selected

    def _compute_visual_token_importance(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        """Compute visual token importance using multi-token raters"""
        self_attn = decoder_layer.self_attn

        # Apply input layernorm
        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        num_heads = self_attn.num_heads
        num_kv_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim

        # Compute Q, K
        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)

        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # GQA: repeat key_states if needed
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)

        # Note: Skipping RoPE for simplicity (attention scores still meaningful)
        # In production, should apply rotary_emb properly

        # Select text raters
        text_rater_indices = self._select_text_raters(hidden_states, visual_start, visual_end)
        text_positions = text_rater_indices + visual_end
        text_positions = text_positions.clamp(max=seq_len - 1)

        # Extract queries from raters and keys from visual tokens
        query_raters = torch.index_select(query_states, 2, text_positions)
        key_visual = key_states[:, :, visual_start:visual_end, :]

        # Compute attention scores
        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(head_dim)

        # Softmax and aggregate
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights.mean(dim=2).mean(dim=1)  # average over raters and heads
        visual_importance = attn_weights.mean(dim=0)  # aggregate batch

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
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # Reset state on new generation
        if past_key_values is None or (
            isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() == 0
        ):
            self.reset_state()

        # Standard Qwen2 initialization
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print("[STAR-V3] Warning: use_cache=True incompatible with gradient checkpointing. Setting False.")
            use_cache = False

        # Initialize cache
        past_seen_tokens = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_usable_length(seq_length) if past_key_values else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare position_ids
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, inputs_embeds.shape[1])
        else:
            position_ids = position_ids.view(-1, inputs_embeds.shape[1]).long()

        # Prepare attention_mask
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                inputs_embeds,
                past_seen_tokens,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds
        seq_length = inputs_embeds.shape[1]

        # Prefill initialization
        if seq_length > 1 and not self.prefill_done:
            # Detect visual tokens (rough estimate)
            detected_visual_len = max(seq_length - self.system_prompt_length - 10, 0)

            # Clamp to expected Stage1 output
            expected_stage1 = min(self.stage0_visual_tokens * 10, self.stage1_keep_tokens * 10)  # allow multi-frame
            detected_visual_len = min(detected_visual_len, expected_stage1)

            # Calculate and fix frame count (will not change during forward pass)
            self.actual_video_frames = max(1, detected_visual_len // self.stage1_keep_tokens)

            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self.prefill_done = True
            self.visual_token_sum = 0.0

            print(f"\n[STAR-V3 Stage 1 Output] Detected {detected_visual_len} visual tokens")
            print(f"  Expected per-frame Stage1: {self.stage1_keep_tokens}")
            print(f"  Fixed frame count: {self.actual_video_frames}")

        # Layer-wise processing
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        star_enabled = (
            seq_length > 1  # Only prune during prefill, not decode
            and self.prefill_done
            and self.current_visual_length is not None
            and self.current_visual_length > 0
            and len(self.pruning_layers) > 0
        )
        visual_token_accumulator = 0.0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1

            # Check if pruning should be applied at this layer
            if star_enabled and layer_idx in self.pruning_layers and self.actual_video_frames is not None:
                # Use fixed frame count instead of dynamic inference
                target_per_frame = self.pruning_layers[layer_idx]
                target = target_per_frame * self.actual_video_frames
                should_prune = self.current_visual_length > target
            else:
                should_prune = False
                target_per_frame = 0
                target = 0

            # Accumulate tokens BEFORE layer forward (record what this layer actually used)
            if star_enabled and self.current_visual_length is not None and self.actual_video_frames is not None:
                per_frame_tokens = self.current_visual_length / self.actual_video_frames
                visual_token_accumulator += float(per_frame_tokens)
            # Execute layer forward (output_attentions at pruning layers for importance)
            if should_prune:
                attn_mask = torch.ones((batch_size, hidden_states.shape[1]), device=hidden_states.device)
                attn_mask = _prepare_4d_causal_attention_mask(attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0)
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=attn_mask, position_ids=position_ids,
                    past_key_value=past_key_values, output_attentions=True, use_cache=use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if should_prune:
                print(f"\n{'='*80}")
                print(f"[STAR-V3 Stage 2] Layer {layer_idx} Progressive Pruning")
                print(f"  Current visual tokens: {self.current_visual_length}")
                print(f"  Fixed frames: {self.actual_video_frames}")
                print(f"  Target per-frame: {target_per_frame}")
                print(f"  Target total: {target}")

                visual_start = self.system_prompt_length
                visual_end = visual_start + self.current_visual_length

                # Extract importance from attention weights (like SparseVLM)
                # Extract importance from attention weights
                import os
                attn_weights = layer_outputs[1].mean(dim=1)  # (bsz, seq, seq) avg over heads
                use_text_rater = os.environ.get("USE_TEXT_RATER", "1") != "0"
                if use_text_rater:
                    # Multi-token text rater importance
                    text_rater_indices = self._select_text_raters(hidden_states, visual_start, visual_end)
                    text_positions = text_rater_indices + visual_end
                    text_positions = text_positions.clamp(max=hidden_states.shape[1] - 1)
                    rater_to_visual = attn_weights[0, text_positions, visual_start:visual_end]
                    visual_importance = rater_to_visual.mean(dim=0)
                else:
                    # Last-token attention (no text rater overhead)
                    visual_importance = attn_weights[0, -1, visual_start:visual_end]

                keep = min(target, visual_importance.shape[0])
                visual_token_indices = torch.sort(visual_importance.topk(keep).indices).values

                print(f"  Selected {visual_token_indices.numel()} tokens via attention-based raters")
                # Apply pruning after layer forward
                device = hidden_states.device
                prev_length = self.current_visual_length
                visual_start = self.system_prompt_length
                visual_end = visual_start + prev_length

                retain_token_indices = torch.cat([
                    torch.arange(self.system_prompt_length, device=device),
                    visual_token_indices + self.system_prompt_length,
                    torch.arange(visual_end, hidden_states.shape[1], device=device),
                ])

                hidden_states = hidden_states[:, retain_token_indices].contiguous()
                position_ids = position_ids[:, retain_token_indices]

                # Update attention_mask for next layer
                if self._attn_implementation == "flash_attention_2":
                    attention_mask = None
                elif self._attn_implementation == "sdpa":
                    attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                        None,
                        (hidden_states.shape[0], hidden_states.shape[1]),
                        hidden_states,
                        0,
                    )
                else:
                    attention_mask = _prepare_4d_causal_attention_mask(
                        None,
                        (hidden_states.shape[0], hidden_states.shape[1]),
                        hidden_states,
                        0,
                        sliding_window=self.config.sliding_window,
                    )

                self.current_visual_length = visual_token_indices.numel()
                star_enabled = self.current_visual_length > 0
                self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                self.stage2_reported_layers.add(layer_idx)

                print(f"  Pruned: {prev_length} → {self.current_visual_length}")
                print(f"{'='*80}\n")

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Compute average tokens per layer
        if seq_length > 1 and star_enabled:
            average_tokens = visual_token_accumulator / len(self.layers) if self.layers else 0.0
            self.visual_token_num = float(average_tokens)
        elif seq_length > 1:
            self.visual_token_num = float(self.current_visual_length or 0.0)

        # Emit summary once
        if seq_length > 1 and self.prefill_done and not self._stage_summary_emitted:
            print(f"\n{'='*80}")
            print(f"[STAR-V3 Summary]")
            print(f"  Stage 0 (input): {self.stage0_visual_tokens} tokens/frame")
            print(f"  Stage 1 (THCP): {self.stage1_actual_tokens} total tokens detected")
            if self.stage2_events:
                print(f"  Stage 2 progressive pruning:")
                for layer_idx, before, after in self.stage2_events:
                    print(f"    Layer {layer_idx}: {before} → {after} tokens")
            print(f"  Final visual tokens: {self.current_visual_length}")
            print(f"  Average tokens per layer: {self.visual_token_num:.2f}")
            print(f"  Target per-frame: {self.target_visual_tokens}")
            print(f"{'='*80}\n")
            self._stage_summary_emitted = True

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
