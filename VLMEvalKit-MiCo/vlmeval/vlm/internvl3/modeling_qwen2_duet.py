# coding=utf-8
"""
DUET Stage 2 (T2V / PyramidDrop): LLM-level visual token pruning for InternVL3.

Faithful port of DUET-VLM's Stage-2 PyramidDrop path (the Qwen `pdrop_rank_drop`
routine) adapted to the InternVL3 Qwen2 LLM. At a schedule of decoder layers it
scores the surviving visual tokens by the mean attention that ALL text-token
queries pay to each visual key, keeps the top-k, and drops the rest from the
hidden states / position ids while leaving the KV cache intact.

Stage 1 (DUET's V2V VisionZip: dominant attention top-k + contextual cluster
merge) is performed feature-level in InternVLChatModelPruning.extract_feature
(pruning_method=='duet').

The (layer, keep) schedule is chosen so the average visual-token count across the
28 decoder layers equals the requested target, keeping DUET directly comparable
to fastv / tops_variant3 / tops at the same token budget.
"""

from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Set
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)

# DUET 剪枝配置: InternVL3-8B (28 layers)
#
# 算法: initial = target × 2，三阶段剪枝
#   Stage 1 (L0-11, 12层): 保持 initial tokens
#   Stage 2 (L12-15, 4层): 剪枝到 a tokens
#   Stage 3 (L16-27, 12层): 剪枝到 b tokens
#
# 约束: (initial×12 + a×4 + b×12) / 28 = target
#      即: a + 3b = target
DUET_SCHEDULE: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
    "internvl3_8b": {
        512: [(12, 281), (16, 77)],   # initial=1024, (1024×12+281×4+77×12)/28=512
        256: [(12, 139), (16, 39)],   # initial=512,  (512×12+139×4+39×12)/28=256
        128: [(12, 71), (16, 19)],    # initial=256,  (256×12+71×4+19×12)/28=128.0 (a+3b=71+57=128 ✓)
    },
}

class DuetQwen2Model(Qwen2Model):
    """
    DUET Stage 2 implementation for InternVL3.
    Progressive pruning at specific decoder layers with multi-token text guidance.
    """

    def __init__(self, config: Qwen2Config, visual_token_num: int, num_image_token: int = 256):
        super().__init__(config)

        # InternVL3 parameters
        self.system_prompt_length = 41  # Will be updated dynamically
        self.num_image_token = num_image_token  # 256 tokens per patch
        self.target_visual_tokens = visual_token_num
        self.stage1_keep_tokens = max(1, visual_token_num * 2)  # Stage 1 output
        self.stage0_visual_tokens = 1280  # Original (5 patches × 256)

        # Model scale detection
        if config.num_hidden_layers == 28:
            self.scale = "internvl3_8b"
        else:
            self.scale = "internvl3_8b"
            logger.warning(f"Unknown model scale with {config.num_hidden_layers} layers, defaulting to internvl3_8b")

        # Load pruning schedule
        schedule = DUET_SCHEDULE.get(self.scale, {}).get(self.target_visual_tokens)
        if schedule is None:
            self.pruning_layers: Dict[int, int] = {}
            logger.warning(f"No DUET schedule for target={visual_token_num}, progressive pruning disabled")
        else:
            self.pruning_layers = {layer_idx: target for layer_idx, target in schedule}

        self.reset_state()

        print(f"[DUET DuetQwen2Model Init]")
        print(f"  Model scale: {self.scale} ({config.num_hidden_layers} layers)")
        print(f"  Target visual tokens (final): {self.target_visual_tokens}")
        print(f"  Stage 1 expected output: {self.stage1_keep_tokens} (target × 2)")
        print(f"  Stage 2 schedule: {self.pruning_layers}")

    def reset_state(self):
        """Reset state tracking for new generation."""
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.stage1_actual_tokens: Optional[int] = None
        self.stage2_events: List[Tuple[int, int, int]] = []  # (layer, before, after)
        self.stage2_skip_events: List[Tuple[int, int, int]] = []
        self.stage2_reported_layers: Set[int] = set()
        self._stage_summary_emitted = False
        self.visual_token_num = float(self.target_visual_tokens)  # Target visual tokens for progress display

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half for RoPE application."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _prune_position_embeddings(
        self,
        pos_emb: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Prune position embeddings adaptively based on their dimensions.
        Matches FastV's approach for 2D indexing but supports all dimensions.

        Args:
            pos_emb: Position embedding tensor (2D, 3D, or 4D)
            indices: Token indices to keep

        Returns:
            Pruned position embedding tensor
        """
        if pos_emb.dim() == 2:
            # [seq_len, head_dim] - match FastV approach
            return pos_emb[indices]
        elif pos_emb.dim() == 3:
            # [batch, seq_len, head_dim]
            return pos_emb[:, indices]
        elif pos_emb.dim() == 4:
            # [batch, num_heads, seq_len, head_dim]
            return pos_emb[:, :, indices]
        else:
            raise ValueError(f"Unexpected position embedding dimension: {pos_emb.dim()}")

    def _select_text_raters(
        self,
        hidden_states: torch.Tensor,
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        """
        DUET Stage-2 (PyramidDrop) rater selection: use ALL text tokens after the
        visual span as query raters. DUET's `pdrop_rank_drop` computes
        text->image attention with `text_query_states = cur_query_states[:, text_start:, :]`
        and averages over all text queries (and heads) to score visual keys.

        Returns indices RELATIVE to the text region [visual_end:], so the caller's
        `text_positions = raters + visual_end` covers the whole trailing text span.
        """
        seq_len = hidden_states.shape[1]
        text_len = seq_len - visual_end
        if text_len <= 0:
            # No text after the visual span; fall back to the last position.
            fallback = max(0, seq_len - 1 - visual_end)
            return hidden_states.new_tensor([fallback], dtype=torch.long)
        # All trailing text tokens as raters (PyramidDrop text->image attention).
        return torch.arange(text_len, device=hidden_states.device, dtype=torch.long)

    def _compute_visual_token_importance(
        self,
        hidden_states: torch.Tensor,
        decoder_layer: nn.Module,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        """
        Compute visual token importance using multi-token text guidance.
        Uses attention probe with selected text rater tokens.
        """
        self_attn = decoder_layer.self_attn

        # Get pre-attention normalized input
        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        num_heads = self_attn.num_heads if hasattr(self_attn, 'num_heads') else self_attn.config.num_attention_heads
        num_kv_heads = self_attn.num_key_value_heads if hasattr(self_attn, 'num_key_value_heads') else self_attn.config.num_key_value_heads
        head_dim = self_attn.head_dim if hasattr(self_attn, 'head_dim') else (self_attn.config.hidden_size // num_heads)

        # Project Q and K
        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)

        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Handle GQA
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)

        # Apply RoPE
        cos, sin = position_embeddings
        # Handle position embedding dimensions - match FastV approach
        # query_states and key_states are 4D: [batch, num_heads, seq_len, head_dim]
        # cos/sin can be 2D [seq_len, head_dim], 3D [batch, seq_len, head_dim], or 4D
        if cos.dim() == 2:
            # [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            # [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        # If already 4D, use as-is

        query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        # Select text raters
        text_rater_indices = self._select_text_raters(hidden_states, visual_start, visual_end)
        text_positions = text_rater_indices + visual_end
        text_positions = text_positions.clamp(max=seq_len - 1)

        query_raters = torch.index_select(query_states, 2, text_positions)
        key_visual = key_states[:, :, visual_start:visual_end, :]

        # Compute attention scores
        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(head_dim)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights.mean(dim=2).mean(dim=1)  # Avg over raters and heads
        visual_importance = attn_weights.mean(dim=0)  # Aggregate across batch

        return visual_importance

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        """Forward with DUET Stage 2 progressive pruning."""

        # Detect if this is a new generation (reset state)
        def _is_empty_cache(cache):
            if cache is None:
                return True
            if isinstance(cache, DynamicCache):
                return cache.get_seq_length() == 0
            return False

        if _is_empty_cache(past_key_values):
            self.reset_state()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        seq_length = inputs_embeds.shape[1]

        # DEBUG: Log input state (only during prefill to reduce noise)
        if seq_length > 1:
            print(f"[DEBUG] Forward start: shape={hidden_states.shape}, seq_len={seq_length}, prefill_done={self.prefill_done}")

        # Detect visual token length in prefill phase
        if seq_length > 1 and not self.prefill_done:
            detected_visual_len = max(seq_length - self.system_prompt_length - 10, 0)
            # Cap to Stage 1 expected output
            if self.target_visual_tokens > 0:
                expected = min(self.stage0_visual_tokens, self.stage1_keep_tokens)
                detected_visual_len = min(detected_visual_len, expected)

            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self.prefill_done = True

            print(f"\n[DUET Stage 2 Init]")
            print(f"  Detected visual tokens: {detected_visual_len}")
            print(f"  Expected from Stage 1: {self.stage1_keep_tokens}")
            print(f"  System prompt length: {self.system_prompt_length}")

        # Only enable pruning during prefill phase (seq_length > 1)
        # Generation phase (seq_length == 1) should skip all pruning logic
        star_enabled = (
            seq_length > 1 and
            self.prefill_done and
            self.current_visual_length is not None and
            self.current_visual_length > 0 and
            len(self.pruning_layers) > 0
        )

        # Flag to track if pruning has been performed (disable flash attention after)
        pruning_performed = False

        # Track visual token count for each layer (for average calculation)
        visual_token_accumulator = 0.0

        for layer_idx, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__),
                    hidden_states, causal_mask, position_ids, past_key_values,
                    output_attentions, use_cache, cache_position, position_embeddings,
                )
            else:
                # Check if we should perform DUET progressive pruning at this layer
                should_prune = (
                    star_enabled and
                    layer_idx in self.pruning_layers and
                    self.current_visual_length is not None and
                    self.current_visual_length > self.pruning_layers[layer_idx]
                )

                if should_prune:
                    # DUET Progressive Pruning at this layer
                    target = self.pruning_layers[layer_idx]
                    device = hidden_states.device

                    print(f"\n[DUET Stage 2] Layer {layer_idx}: {self.current_visual_length} → {target}")
                    print(f"[DEBUG] BEFORE pruning: shape={hidden_states.shape}")

                    visual_start = self.system_prompt_length
                    visual_end = visual_start + self.current_visual_length

                    # Validate visual token range
                    if visual_end > hidden_states.shape[1]:
                        logger.warning(
                            f"Visual range [{visual_start}:{visual_end}] exceeds sequence length {hidden_states.shape[1]}. "
                            f"Adjusting to fit."
                        )
                        visual_end = hidden_states.shape[1]
                        self.current_visual_length = visual_end - visual_start

                    # Compute importance
                    visual_importance = self._compute_visual_token_importance(
                        hidden_states, decoder_layer, position_embeddings, visual_start, visual_end
                    )

                    # Select top-k tokens
                    keep = min(target, visual_importance.shape[0])
                    keep = max(keep, 0)

                    if keep == 0:
                        visual_token_indices = visual_importance.new_empty((0,), dtype=torch.long)
                    else:
                        visual_token_indices = torch.sort(visual_importance.topk(keep).indices).values

                    # Forward through the layer FIRST
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

                    # Now apply pruning to the OUTPUT
                    prev_length = self.current_visual_length

                    # Construct retention indices: [system_prompt | selected_visual | text_after]
                    retain_token_indices = torch.cat((
                        torch.arange(visual_start, device=device),
                        visual_token_indices + visual_start,
                        torch.arange(visual_end, hidden_states.shape[1], device=device),
                    ))

                    # Apply pruning to all tensors
                    hidden_states = hidden_states[:, retain_token_indices].contiguous()
                    position_ids = position_ids[:, retain_token_indices]
                    cache_position = cache_position[:retain_token_indices.shape[0]]

                    # Regenerate position embeddings instead of indexing - ensures perfect alignment
                    # This matches FastV's approach but regenerates for consistency after pruning
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

                    # NOTE: Following Qwen2.5-VL's approach - DO NOT prune KV cache
                    # Qwen2.5-VL's working implementation shows that KV cache should remain unpruned
                    # Only hidden_states, position_ids, and position_embeddings are pruned
                    # This avoids complex cache state management issues in generation phase

                    # Recalculate causal mask for the pruned sequence
                    causal_mask = self._update_causal_mask(
                        None, hidden_states, cache_position, past_key_values, output_attentions
                    )

                    # Update state
                    self.current_visual_length = visual_token_indices.numel()
                    star_enabled = self.current_visual_length > 0
                    self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                    self.stage2_reported_layers.add(layer_idx)
                    pruning_performed = True

                    print(f"[DEBUG] AFTER pruning: shape={hidden_states.shape}, kept {visual_token_indices.numel()} visual tokens")
                    print(f"  ✓ Pruned: {prev_length} → {self.current_visual_length}")

                    # Override layer_outputs to ensure correct hidden_states propagation
                    layer_outputs = (hidden_states, layer_outputs[-1])

                else:
                    # Normal forward pass (not a pruning layer, or after all pruning)
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs if not pruning_performed else {},
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Track visual token count for each layer (for average calculation)
            if seq_length > 1 and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

        hidden_states = self.norm(hidden_states)

        # Calculate actual average visual tokens across all layers (for progress display)
        if seq_length > 1 and self.current_visual_length is not None:
            average_tokens = visual_token_accumulator / len(self.layers) if self.layers else 0.0
            self.visual_token_num = float(average_tokens)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Emit summary once
        if seq_length > 1 and self.prefill_done and not self._stage_summary_emitted:
            print(f"\n{'='*80}")
            print(f"[DUET Summary]")
            print(f"  Stage 0: {self.stage0_visual_tokens} tokens (original)")
            if self.stage1_actual_tokens is not None:
                print(f"  Stage 1: {self.stage1_actual_tokens} tokens (THCP output)")
            if self.stage2_events:
                print(f"  Stage 2 pruning events:")
                for layer, before, after in self.stage2_events:
                    print(f"    Layer {layer}: {before} → {after}")
                print(f"  Final visual tokens: {self.current_visual_length}")
            else:
                print(f"  Stage 2: No pruning (schedule missing or already at target)")
            print(f"{'='*80}\n")
            self._stage_summary_emitted = True

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DuetQwen2ForCausalLM(Qwen2ForCausalLM):
    """Wrapper for Qwen2ForCausalLM with DUET Stage 2 decoder."""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, visual_token_num: int, num_image_token: int = 256):
        super().__init__(config)
        self.model = DuetQwen2Model(config, visual_token_num, num_image_token)

    def set_system_prompt_length(self, length: int):
        """Set system prompt length for pruning calculations."""
        self.model.system_prompt_length = length
