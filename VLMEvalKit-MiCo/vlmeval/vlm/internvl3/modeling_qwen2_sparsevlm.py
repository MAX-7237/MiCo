"""SparseVLM for InternVL3-8B's Qwen2 decoder.

InternVL3 lays its inputs out as [system_prompt | visual | text], with the
visual span contiguous and every sample carrying the same 5x256 = 1280 tokens,
so the span can be located by index. Pruning happens inside the decoder at the
layers in `SPARSEVLM_PRUNING_LAYERS_28`; the algorithm itself lives in
`vlmeval/vlm/sparsevlm_core.py`.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

from ..sparsevlm_core import (
    SPARSEVLM_PRUNING_LAYERS_28,
    build_schedule,
    merge_and_retain,
    select_text_raters,
    solve_keep_for_span,
)

logger = logging.get_logger(__name__)


class SparseVLMQwen2Model(Qwen2Model):
    """Qwen2Model with SparseVLM's text-guided in-decoder visual pruning."""

    def __init__(self, config: Qwen2Config, visual_token_num: int, num_image_token: int = 256):
        super().__init__(config)
        self.visual_token_num = visual_token_num
        self.target_visual_token_num = visual_token_num
        self.num_image_token = num_image_token
        self.system_prompt_length = 41
        # InternVL3-8B feeds 5 patches x 256 tokens.
        self.visual_token_length = 1280
        self.pruning_method = 'sparsevlm'
        self.sparsevlm_debug = False
        self.schedule = {}

        num_layers = config.num_hidden_layers
        if num_layers >= max(SPARSEVLM_PRUNING_LAYERS_28):
            self.pruning_locs = set(SPARSEVLM_PRUNING_LAYERS_28)
        else:
            self.pruning_locs = {l for l in SPARSEVLM_PRUNING_LAYERS_28 if l < num_layers}
        # Layer-average bookkeeping, reported at the end of prefill.
        self._visual_token_accum = 0
        self._layer_counter = 0
        self._visual_span = None
        self._active_span = None

        print("[SparseVLM InternVL3] pruning layers: "
              f"{sorted(self.pruning_locs)} (of {num_layers}), target avg={visual_token_num}")

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _visual_cross_attention(
        self,
        hidden_states: torch.Tensor,
        decoder_layer,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        text_rater_index: torch.Tensor,
        visual_start: int,
        visual_end: int,
    ) -> torch.Tensor:
        """SparseVLM's text->visual cross attention, averaged over raters and heads.

        Recomputing Q/K here (rather than reading `output_attentions`) keeps the
        global attention implementation untouched -- forcing `output_attentions`
        would drop the entire stack to eager, which the rest of this repo's
        in-LLM methods deliberately avoid.
        """
        self_attn = decoder_layer.self_attn
        attn_input = hidden_states
        input_layernorm = getattr(decoder_layer, "input_layernorm", None)
        if input_layernorm is not None:
            attn_input = input_layernorm(attn_input)

        bsz, seq_len, _ = attn_input.shape
        num_heads = getattr(self_attn, 'num_heads', None) or self_attn.config.num_attention_heads
        num_kv_heads = getattr(self_attn, 'num_key_value_heads', None) or self_attn.config.num_key_value_heads
        head_dim = getattr(self_attn, 'head_dim', None) or (self_attn.config.hidden_size // num_heads)

        query_states = self_attn.q_proj(attn_input)
        key_states = self_attn.k_proj(attn_input)
        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        if num_kv_heads != num_heads:
            key_states = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)

        cos, sin = position_embeddings
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        text_positions = (text_rater_index + visual_end).clamp(max=seq_len - 1)
        query_raters = torch.index_select(query_states, 2, text_positions)   # [B, H, M, d]
        key_visual = key_states[:, :, visual_start:visual_end, :]            # [B, H, V, d]

        attn_scores = torch.matmul(query_raters, key_visual.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # [B, H, M, V] -> avg over raters and heads -> [B, V]
        return attn_weights.mean(dim=2).mean(dim=1)

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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

        # Prefill only: decoding steps carry a single token and must not re-prune.
        prefill = inputs_embeds.shape[1] > 1
        text_rater_index = None
        visual_start = visual_end = 0
        # Track the span actually visible to each layer so the reported
        # layer-average matches the repo's convention.
        self._active_span = None
        if prefill:
            self._visual_token_accum = 0
            self._layer_counter = 0
            visual_start = min(self.system_prompt_length, hidden_states.shape[1])
            visual_end = min(visual_start + self.visual_token_length, hidden_states.shape[1])
            if visual_end > visual_start + 1:
                text_rater_index = select_text_raters(hidden_states, visual_start, visual_end)
                # Iso-budget schedule: solve target spans so the layer-average is
                # exactly `target_visual_token_num` (matches every other method here).
                self.schedule = build_schedule(
                    visual_end - visual_start,
                    self.target_visual_token_num,
                    num_layers=len(self.layers),
                    pruning_locs=self.pruning_locs,
                )
                if self.sparsevlm_debug:
                    print(f"[SparseVLM] target spans by layer: {self.schedule}")
                self._active_span = (visual_start, visual_end)
            else:
                # Nothing prunable: behave as a plain decoder, and do not let a
                # stale span pollute the layer-average bookkeeping.
                visual_end = visual_start
        elif self._visual_span is not None:
            visual_start, visual_end = self._visual_span

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            prune_here = (
                prefill
                and text_rater_index is not None
                and layer_idx in self.pruning_locs
                and visual_end > visual_start + 1
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states, causal_mask, position_ids, past_key_values,
                    output_attentions, use_cache, cache_position, position_embeddings,
                )
                hidden_states = layer_outputs[0]
            else:
                if prune_here:
                    # Score before the layer consumes hidden_states, matching the official
                    # ordering where attention is read off this layer's own forward.
                    step_visual_weights = self._visual_cross_attention(
                        hidden_states, decoder_layer, position_embeddings,
                        text_rater_index, visual_start, visual_end,
                    )

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
                hidden_states = layer_outputs[0]

                if prune_here:
                    # `schedule` holds the target span for this layer; invert the
                    # recycling rule to learn how many tokens to keep outright.
                    target_span = self.schedule.get(layer_idx, visual_end - visual_start)
                    keep = solve_keep_for_span(visual_end - visual_start, target_span)

                    visual_tokens = hidden_states[:, visual_start:visual_end, :]
                    kept, merged, _ = merge_and_retain(visual_tokens, step_visual_weights, keep)
                    hidden_states = torch.cat((
                        hidden_states[:, :visual_start],
                        kept,
                        merged,
                        hidden_states[:, visual_end:],
                    ), dim=1)
                    new_visual_len = kept.shape[1] + merged.shape[1]

                    # Drop the pruned positions entirely; every surviving token keeps the
                    # position and RoPE table it already had. Merged centroids inherit the
                    # first visual position so the block stays contiguous and ordered.
                    mid = position_ids[:, visual_start:visual_start + 1].expand(-1, new_visual_len)
                    position_ids = torch.cat((
                        position_ids[:, :visual_start],
                        mid,
                        position_ids[:, visual_end:],
                    ), dim=1)
                    cache_position = cache_position[:hidden_states.shape[1]]
                    if isinstance(position_embeddings, (tuple, list)):
                        position_embeddings = tuple(
                            torch.cat((
                                pe[:, :visual_start],
                                pe[:, visual_start:visual_start + 1].expand(-1, new_visual_len, *pe.shape[2:]),
                                pe[:, visual_end:],
                            ), dim=1)
                            for pe in position_embeddings
                        )
                    causal_mask = self._update_causal_mask(
                        None, hidden_states, cache_position, past_key_values, output_attentions
                    )

                    visual_end = visual_start + new_visual_len
                    self._visual_span = (visual_start, visual_end)
                    self._active_span = (visual_start, visual_end)
                    if self.sparsevlm_debug:
                        print(f"[SparseVLM] layer {layer_idx}: keep={keep} -> kept "
                              f"{kept.shape[1]} + merged {merged.shape[1]} = {new_visual_len}")
                    layer_outputs = (hidden_states,) + tuple(layer_outputs[1:])

            hidden_states = layer_outputs[0]

            # Each layer sees the span as it stands *after* that layer's own prune,
            # matching the FastV/DUET convention this repo compares against.
            if prefill and self._active_span is not None:
                self._visual_token_accum += self._active_span[1] - self._active_span[0]
                self._layer_counter += 1

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if prefill and self._layer_counter > 0:
            self.visual_token_num = self._visual_token_accum / self._layer_counter
            print(f"[SparseVLM] actual layer-average visual tokens = {self.visual_token_num:.2f} "
                  f"(target {self.target_visual_token_num})")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForCausalLMSparseVLM(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, pruning_method='sparsevlm', visual_token_num=None, num_image_token=256):
        super().__init__(config)
        self.model = SparseVLMQwen2Model(config, visual_token_num, num_image_token)
        self.pruning_method = pruning_method

    def set_system_prompt_length(self, length: int):
        self.model.system_prompt_length = length

    def set_visual_token_length(self, length: int):
        self.model.visual_token_length = length

    def enable_sparsevlm_debug(self, enable: bool = True):
        self.model.sparsevlm_debug = enable
