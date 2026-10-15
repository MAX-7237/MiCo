# coding=utf-8
"""ApET in-LLM Stage-2 for Qwen2.5-VL.

ApET (Adaptive Patch-pruning with Efficient Token merging) is a two-stage method:
  S1: farthest-point-sampled seeds + linear-reconstruction ranking + cosine merge,
      on ViT-space features before the merger (done in the outer Qwen2_5_VL forward).
  S2: the same merge applied once, mid-decoder, to the visual span of the hidden
      states (this class).

As in the reference ApET hooks, layers after the merge use compact positions
and their own shorter KV caches. Qwen's three RoPE axes share that compact
position in the upper half; the lower half retains the original spatial RoPE.

Schedule: with nominal budget T, S1 keeps floor(3T/2); S2 cuts to floor(T/2)
after layer L//2. Layer-average visual tokens therefore equal T, matching the
single-stage baselines' measurement point.
"""

from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

from vlmeval.vlm.apet_select import apet_stage1, apet_stage2, apet_schedule

logger = logging.get_logger(__name__)


@torch.no_grad()
def apet_merge_vit_groups(merger_input, window_index, merger, spatial_merge_unit,
                         keep, basis, seed=None):
    """Apply S1 in ViT space, retaining the merger's native patch grouping.

    Qwen maps each 2x2 patch group to one LLM token. Concatenating those four
    ViT vectors keeps all channels for ApET's distance/reconstruction/merge;
    after merging, the same learned patch merger projects them to LLM space.
    """
    count = window_index.numel()
    if merger_input.ndim != 2 or merger_input.shape[0] != count * spatial_merge_unit:
        raise ValueError('ApET merger input must contain one complete patch group per output token')
    reverse = window_index.argsort().to(merger_input.device)
    grouped = merger_input.reshape(count, spatial_merge_unit * merger_input.shape[-1])[reverse]
    # Real BF16 Qwen ViT values reach ~46k: FP16 merged sums overflow before
    # averaging. ApET† retains the formula in FP32 and casts back only afterward.
    merged, indices = apet_stage1(grouped[None], keep=keep, basis=basis, seed=seed,
                                 fp32_merge=True)
    # The patch merger acts independently on groups; image order can stay restored.
    projected = merger(merged[0].reshape_as(merger_input).to(merger_input.dtype))
    if projected.shape[0] != count or not bool(torch.isfinite(projected).all()):
        stats = {}
        for label, value in (('input', merger_input), ('merged', merged), ('projected', projected)):
            stats[label] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                finite=int(torch.isfinite(value).sum().item()), total=value.numel(),
                                absmax=float(value.float().abs().max().item()))
        raise ValueError(f'ApET S1 produced invalid projected image features: {stats}, expected_groups={count}')
    return projected, indices[0]


class ApETQwen2_5_VLModel(Qwen2_5_VLModel):
    """Qwen2.5-VL LLM with ApET in-LLM Stage-2 merge at layer L//2."""

    def __init__(self, config: Qwen2_5_VLConfig, visual_token_num: int, basis: int = 10):
        super().__init__(config)
        # transformers 4.52+: transformer layers live inside self.language_model
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._update_causal_mask = lm._update_causal_mask

        try:
            visual_token_num = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")

        n_layers = len(self.layers)
        sched = apet_schedule(visual_token_num, n_layers)
        self.target_visual_tokens = visual_token_num
        self.stage1_keep_tokens = sched['stage1']
        self.stage2_keep_tokens = sched['stage2']
        self.prune_after = sched['prune_after']          # 1-indexed completed-layer count
        self.basis = int(basis)
        self.schedule_layers: Dict[int, int] = {self.prune_after: self.stage2_keep_tokens}

        self.system_prompt_length = 15
        self.visual_token_length = 1296
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

        print("[ApET Init] Qwen2.5-VL")
        print(f"  Layers: {n_layers}  budget T={visual_token_num}")
        print(f"  S1 keep: {self.stage1_keep_tokens}  S2 keep: {self.stage2_keep_tokens} "
              f"after layer {self.prune_after}  layer-avg: {sched['average']:.1f}")
        self.reset_state()

    def reset_state(self):
        self.prefill_done = False
        self.current_visual_length: Optional[int] = None
        self.original_visual_length: Optional[int] = None
        self.visual_token_sum = 0.0
        self.stage1_actual_tokens: Optional[int] = None
        self.stage2_events: List[Tuple[int, int, int]] = []
        self.stage2_selected_indices = None
        self._stage_summary_emitted = False
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        stage1_visual_span: Optional[Tuple[int, int]] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # A fresh cache marks a new sample even when the caller reuses this model.
        is_prefill = past_key_values is None or past_key_values.get_seq_length() == 0
        if is_prefill:
            self.reset_state()
        if inputs_embeds.shape[0] != 1:
            raise ValueError('ApET evaluation requires batch size one per worker')
        if attention_mask is not None and (attention_mask.ndim != 2 or not bool(attention_mask.all())):
            raise ValueError('ApET evaluation requires an unpadded 2D attention mask')

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        hidden_states = inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        seq_length = inputs_embeds.shape[1]

        if is_prefill:
            if stage1_visual_span is not None:
                visual_start, detected_visual_len = stage1_visual_span
                if (inputs_embeds.shape[0] != 1 or visual_start < 0 or detected_visual_len <= 0
                        or detected_visual_len > self.stage1_keep_tokens
                        or visual_start + detected_visual_len >= seq_length):
                    raise ValueError(f'Invalid post-S1 visual span {stage1_visual_span} for seq len {seq_length}')
                self.system_prompt_length = visual_start
            else:
                # Text-only requests have no visual span; never infer it from text length.
                detected_visual_len = 0
            self.current_visual_length = detected_visual_len
            self.original_visual_length = detected_visual_len
            self.stage1_actual_tokens = detected_visual_len
            self.prefill_done = True
            self.visual_token_sum = 0.0
            print(f"[ApET S2] Prefill: seq={seq_length}, visual={detected_visual_len}, "
                  f"S1_keep={self.stage1_keep_tokens}, S2_keep={self.stage2_keep_tokens}")

        apet_enabled = (
            self.prefill_done
            and self.current_visual_length is not None
            and self.current_visual_length > 0
            and len(self.schedule_layers) > 0
        )
        visual_token_accumulator = 0.0
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx + 1
            if is_prefill and self.current_visual_length is not None:
                visual_token_accumulator += float(self.current_visual_length)

            # The lower and upper halves cache different sequence lengths after S2.
            layer_past = past_key_values.get_seq_length(layer_idx - 1) if past_key_values is not None else 0
            layer_cache_position = torch.arange(
                layer_past, layer_past + hidden_states.shape[1], device=hidden_states.device)
            layer_position_ids = position_ids
            if layer_idx > self.prune_after and self.stage2_events:
                layer_position_ids = layer_cache_position[None, None].expand(3, 1, -1)
            position_embeddings = self.rotary_emb(hidden_states, layer_position_ids)
            # Passing an explicit per-layer mask avoids the HF helper's layer-0 cache length.
            layer_mask = torch.ones(1, layer_past + hidden_states.shape[1],
                                    dtype=torch.bool, device=hidden_states.device)
            causal_mask = self._update_causal_mask(
                layer_mask, hidden_states, layer_cache_position, None, output_attentions)

            should_prune = (
                apet_enabled
                and is_prefill
                and layer_idx in self.schedule_layers
                and self.current_visual_length is not None
                and self.current_visual_length > self.schedule_layers[layer_idx]
            )

            if should_prune:
                target = self.schedule_layers[layer_idx]
                visual_start = self.system_prompt_length
                visual_end = visual_start + self.current_visual_length
                if visual_end >= hidden_states.shape[1]:
                    raise ValueError('ApET visual span must precede the answer prompt')

                print(f"\n[ApET S2] Layer {layer_idx}: {self.current_visual_length} → {target} tokens")

                # Run the layer normally, then merge the visual span in place.
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=layer_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=layer_cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]

                merged, selected = apet_stage2(hidden_states, visual_start, self.current_visual_length,
                                               keep=target, basis=self.basis, seed=None)
                # `merged` spans exactly `target` visual tokens.
                hidden_states = merged
                self.stage2_selected_indices = selected

                prev_length = self.current_visual_length
                self.current_visual_length = int(target)
                apet_enabled = self.current_visual_length > 0
                self.stage2_events.append((layer_idx, prev_length, self.current_visual_length))
                print(f"[ApET S2] Layer {layer_idx} done: {prev_length} → {self.current_visual_length}, "
                      f"total_seq={hidden_states.shape[1]}")
                layer_outputs = (hidden_states,) + layer_outputs[1:]
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=layer_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=layer_cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if is_prefill and self.current_visual_length is not None and not self._stage_summary_emitted:
            n_layers = len(self.layers)
            avg = visual_token_accumulator / n_layers if n_layers else 0.0
            self.visual_token_num = float(avg)
            self._stage_summary_emitted = True
            print(f"[ApET S2] summary: stage2_events={self.stage2_events} layer_avg={avg:.1f}")

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values if use_cache else None,
                                    all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
