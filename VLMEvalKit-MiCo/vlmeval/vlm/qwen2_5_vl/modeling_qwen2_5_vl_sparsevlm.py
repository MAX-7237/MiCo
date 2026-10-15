"""SparseVLM (arXiv 2406.11717) for the Qwen2.5-VL LLM decoder.

Structure follows this repo's DART adapter (`modeling_qwen2_5_vl_dart.py`):
subclass `Qwen2_5_VLModel`, borrow the LLM sublayers, keep the 3D (t/h/w)
`position_ids`, and prune inside the decoder loop. The numeric algorithm is the
backbone-agnostic core in `vlmeval/vlm/sparsevlm_core.py`.

Unlike DART, SparseVLM does not just gather kept tokens -- it also inserts
DPC-kmeans centroids built from the dropped tokens, so the visual span grows
back by a few tokens at each pruning layer. The RoPE tables for the surviving
tokens are sliced (never recomputed), and the merged centroids inherit the
first visual position so the block stays contiguous.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
from transformers.utils import logging

from ..sparsevlm_core import (
    SPARSEVLM_PRUNING_LAYERS_28,
    build_schedule,
    merge_and_retain,
    select_text_raters,
    solve_keep_for_span,
)

logger = logging.get_logger(__name__)


class SparseVLMQwen2_5_VLModel(Qwen2_5_VLModel):
    """Qwen2.5-VL LLM decoder with SparseVLM's text-guided in-decoder pruning."""

    def __init__(self, config, visual_token_num: int):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._update_causal_mask = lm._update_causal_mask

        self.system_prompt_length = 15          # matches this repo's FastV/DART default
        self.visual_token_length = 1296         # single-image default; updated at runtime
        self.visual_token_num = 0
        self.visual_token_len = 0
        self.target_visual_token_num = max(1, visual_token_num)
        self.pruning_method = 'sparsevlm'
        self.sparsevlm_debug = False
        self.schedule = {}

        n_layers = config.text_config.num_hidden_layers if hasattr(config, 'text_config') else config.num_hidden_layers
        if n_layers >= max(SPARSEVLM_PRUNING_LAYERS_28):
            self.pruning_locs = set(SPARSEVLM_PRUNING_LAYERS_28)
        else:
            self.pruning_locs = {l for l in SPARSEVLM_PRUNING_LAYERS_28 if l < n_layers}
        print(f"[SparseVLM Qwen2.5-VL] pruning layers: {sorted(self.pruning_locs)} "
              f"(of {n_layers}), target avg={visual_token_num}")

    def set_system_prompt_length(self, length: int):
        self.system_prompt_length = length

    def set_visual_token_length(self, length: int):
        self.visual_token_length = length

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _visual_cross_attention(
        self, hidden_states, decoder_layer, position_embeddings,
        text_rater_index, visual_start, visual_end,
    ) -> torch.Tensor:
        """Text->visual attention, averaged over rater tokens and heads -> [B, V].

        Q/K are recomputed here so the global attention implementation stays sdpa;
        forcing `output_attentions=True` would drop the whole stack to eager, which
        the other in-LLM methods in this repo deliberately avoid.
        """
        self_attn = decoder_layer.self_attn
        attn_input = hidden_states
        ln = getattr(decoder_layer, "input_layernorm", None)
        if ln is not None:
            attn_input = ln(attn_input)

        bsz, seq_len, _ = attn_input.shape
        num_heads = getattr(self_attn, 'num_heads', None) or self_attn.config.num_attention_heads
        num_kv_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim

        q = self_attn.q_proj(attn_input).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = self_attn.k_proj(attn_input).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        if num_kv_heads != num_heads:
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)

        cos, sin = position_embeddings
        # Qwen2.5-VL RoPE tables are [3, B, seq, head_dim] (t/h/w); collapse the
        # mrope sections the same way apply_multimodal_rotary_pos_emb does by
        # averaging the three axes into one cos/sin per position.
        if cos.dim() == 4:
            cos = cos.mean(0).unsqueeze(1)     # [B, 1, seq, hd]
            sin = sin.mean(0).unsqueeze(1)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        elif cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)

        text_positions = (text_rater_index + visual_end).clamp(max=seq_len - 1)
        q_raters = torch.index_select(q, 2, text_positions)     # [B, H, M, d]
        k_visual = k[:, :, visual_start:visual_end, :]          # [B, H, V, d]
        scores = torch.matmul(q_raters, k_visual.transpose(-2, -1)) / math.sqrt(head_dim)
        w = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return w.mean(dim=2).mean(dim=1)                        # [B, V]

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        prefill = inputs_embeds.shape[1] > 1
        text_rater_index = None
        visual_start = visual_end = 0
        accum = 0
        n_counted = 0
        self._active_span = None
        if prefill:
            seq_now = hidden_states.shape[1]
            visual_start = min(self.system_prompt_length, seq_now)
            visual_end = min(visual_start + self.visual_token_length, seq_now)
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
                self._active_span = (visual_start, visual_end)
                if self.sparsevlm_debug:
                    print(f"[SparseVLM] target spans by layer: {self.schedule}")
            else:
                # Nothing prunable: behave as a plain decoder and keep the
                # layer-average bookkeeping empty.
                visual_end = visual_start
            if self.sparsevlm_debug:
                print(f"[SparseVLM] prefill seq={seq_now} visual=[{visual_start},{visual_end}] "
                      f"target={self.target_visual_token_num}")

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx
            prune_here = (
                prefill
                and text_rater_index is not None
                and layer_idx in self.pruning_locs
                and visual_end > visual_start + 1
            )

            if prune_here:
                step_weights = self._visual_cross_attention(
                    hidden_states, decoder_layer, position_embeddings,
                    text_rater_index, visual_start, visual_end,
                )

            layer_outputs = decoder_layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=past_key_values, output_attentions=output_attentions,
                use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if prune_here:
                device = hidden_states.device
                # `schedule` holds the target span for this layer; invert the
                # recycling rule to learn how many tokens to keep outright.
                target_span = self.schedule.get(layer_idx, visual_end - visual_start)
                keep = solve_keep_for_span(visual_end - visual_start, target_span)

                visual_tokens = hidden_states[:, visual_start:visual_end, :]
                kept, merged, _ = merge_and_retain(visual_tokens, step_weights, keep)
                new_visual_len = kept.shape[1] + merged.shape[1]
                hidden_states = torch.cat((
                    hidden_states[:, :visual_start],
                    kept, merged,
                    hidden_states[:, visual_end:],
                ), dim=1)

                # Kept tokens keep their own (t/h/w) positions and RoPE tables; the merged
                # centroids inherit the first visual position so the span stays ordered.
                mid_pos = position_ids[:, :, visual_start:visual_start + 1].expand(-1, -1, new_visual_len)
                position_ids = torch.cat((
                    position_ids[:, :, :visual_start],
                    mid_pos,
                    position_ids[:, :, visual_end:],
                ), dim=2)
                cache_position = cache_position[:hidden_states.shape[1]]
                position_embeddings = [
                    torch.cat((
                        pe[:, :, :visual_start],
                        pe[:, :, visual_start:visual_start + 1].expand(-1, -1, new_visual_len, *pe.shape[3:]),
                        pe[:, :, visual_end:],
                    ), dim=2)
                    for pe in position_embeddings
                ]
                causal_mask = self._update_causal_mask(None, hidden_states, cache_position, past_key_values, output_attentions)

                visual_end = visual_start + new_visual_len
                self.visual_token_len = new_visual_len
                self._active_span = (visual_start, visual_end)
                if self.sparsevlm_debug:
                    print(f"[SparseVLM] layer {layer_idx}: keep={keep} -> kept {kept.shape[1]} "
                          f"+ merged {merged.shape[1]} = {new_visual_len}")
                layer_outputs = (hidden_states,) + tuple(layer_outputs[1:])

            hidden_states = layer_outputs[0]
            # Each layer sees the span as it stands *after* that layer's own prune,
            # matching the FastV/DUET convention this repo compares against.
            if prefill and self._active_span is not None:
                accum += self._active_span[1] - self._active_span[0]
                n_counted += 1
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if prefill and n_counted > 0:
            self.visual_token_num = accum / n_counted
            print(f"[SparseVLM] actual layer-average visual tokens = {self.visual_token_num:.2f} "
                  f"(target {self.target_visual_token_num})")
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
