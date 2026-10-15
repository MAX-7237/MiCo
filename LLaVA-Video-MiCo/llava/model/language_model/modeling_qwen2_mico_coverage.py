"""Single configurable post-layer attn×coverage prune for LLaVA-Video (HF 4.41)."""
import hashlib
import json
import os
from pathlib import Path
import torch
from transformers import Qwen2Model
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from .mico_coverage_video import solve_budget, weighted_coverage_select, post_text_attention, hidden_l2_weights


def _position_safe_rope(module, args, kwargs):
    required = int(getattr(module, '_mico_coverage_required_length', 0))
    if len(args) > 1:
        args = (args[0], max(int(args[1]), required)) + args[2:]
    elif kwargs.get('seq_len') is not None:
        kwargs = dict(kwargs, seq_len=max(int(kwargs['seq_len']), required))
    return args, kwargs


class MiCoCoverageQwen2Model(Qwen2Model):
    def __init__(self, config, mico_coverage_config):
        super().__init__(config)
        self.target_visual_tokens = int(mico_coverage_config['T'])
        self.prune_layer = int(mico_coverage_config.get('K', 12))
        self.mico = bool(mico_coverage_config.get('mico', False))
        if self.target_visual_tokens <= 0 or not 0 < self.prune_layer < len(self.layers):
            raise ValueError('invalid mico_coverage T or 1-based K')
        if getattr(config, 'use_sliding_window', False):
            raise ValueError('mico_coverage Video has not validated sliding-window attention')
        # HF 4.41 FlashAttention's decode cache slicing checks this field even
        # when use_sliding_window=False. Enforce the requested full-attention mode.
        config.sliding_window = None
        self._layout = None
        self.last_debug = None
        self.visual_token_num = 0.0
        for layer in self.layers:
            layer.self_attn.rotary_emb.register_forward_pre_hook(_position_safe_rope, with_kwargs=True)

    def set_visual_layout(self, start, span_length, patch_positions, frame_ids, frames, original_patch_ids=None):
        patch_positions = patch_positions.detach().long()
        frame_ids = frame_ids.detach().long()
        if not 0 <= int(start) or int(span_length) <= 0 or int(frames) <= 0:
            raise ValueError('invalid multimodal span')
        if patch_positions.ndim != 1 or len(patch_positions) != len(frame_ids) or not len(patch_positions):
            raise ValueError('invalid patch/frame metadata')
        if (patch_positions < 0).any() or (patch_positions >= span_length).any():
            raise ValueError('visual patch position outside multimodal span')
        if len(torch.unique(patch_positions)) != len(patch_positions) or not torch.equal(patch_positions, patch_positions.sort().values):
            raise ValueError('visual patch positions must be unique and sorted')
        if (frame_ids < 0).any() or (frame_ids >= frames).any():
            raise ValueError('invalid frame IDs')
        original_patch_ids = patch_positions if original_patch_ids is None else original_patch_ids.detach().long()
        if original_patch_ids.shape != patch_positions.shape or (original_patch_ids < 0).any():
            raise ValueError('invalid original frame patch IDs')
        budget = solve_budget(self.target_visual_tokens * int(frames), len(patch_positions), self.prune_layer, len(self.layers))
        self._layout = dict(start=int(start), span_length=int(span_length), patches=patch_positions,
                            frame_ids=frame_ids, frames=int(frames), budget=budget, original_patch_ids=original_patch_ids)
        self.last_debug = None

    def _mask(self, hidden, past_length, output_attentions):
        if self.config._attn_implementation == 'flash_attention_2':
            return None
        prepare = (_prepare_4d_causal_attention_mask_for_sdpa
                   if self.config._attn_implementation == 'sdpa' and not output_attentions
                   else _prepare_4d_causal_attention_mask)
        return prepare(None, hidden.shape[:2], hidden, past_length)

    def _emit_trace(self):
        debug = self.last_debug
        serial = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in debug.items()}
        selected = serial.pop('retained_sequence_indices')
        serial['retained_sequence_indices_sha256'] = hashlib.sha256(json.dumps(selected, separators=(',', ':')).encode()).hexdigest()
        serial['request_id'] = getattr(self, '_tops_request_id', None)
        print('[MICO_COVERAGE_VIDEO] ' + json.dumps(serial, sort_keys=True), flush=True)
        directory = os.environ.get('VIDEO_TRACE_DIR')
        if directory:
            root = Path(directory); root.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get('RANK', '0'))
            with (root / f'rank-{rank}.jsonl').open('a') as f:
                f.write(json.dumps(serial, sort_keys=True) + '\n')

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        if self._layout is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        if self.training:
            raise ValueError('mico_coverage Video supports evaluation only')
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('specify exactly one of input_ids and inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError('mico_coverage Video currently requires batch_size=1')
        if attention_mask is not None and (attention_mask.ndim != 2 or not attention_mask.bool().all()):
            raise ValueError('mico_coverage Video currently requires unpadded batch_size=1')
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        legacy = not isinstance(past_key_values, Cache)
        cache = DynamicCache.from_legacy_cache(past_key_values) if legacy else past_key_values
        if cache is None:
            cache = DynamicCache()
        past_length = cache.get_seq_length(0)
        q_length = inputs_embeds.shape[1]
        prefill = past_length == 0 and q_length > 1
        if not prefill and self.last_debug is None:
            raise ValueError('decode received without this sample\'s completed prefill')
        if not prefill and q_length != 1:
            raise ValueError('mico_coverage Video requires unchunked prefill and one-token decode')
        if not use_cache and past_length:
            raise ValueError('past_key_values require use_cache=True')
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + q_length, device=inputs_embeds.device)[None]
        else:
            position_ids = position_ids.reshape(1, q_length).long()
        required_length = int(position_ids.max()) + 1
        for layer in self.layers:
            layer.self_attn.rotary_emb._mico_coverage_required_length = required_length
        hidden = inputs_embeds
        layout = self._layout
        visual_positions = layout['patches'].to(hidden.device) + layout['start']
        frame_ids = layout['frame_ids'].to(hidden.device)
        if prefill and layout['start'] + layout['span_length'] > q_length:
            raise ValueError('multimodal sequence was truncated; visual span exceeds prefill')
        current_visual = len(visual_positions)
        visual_counts, sequence_lengths = [], []
        all_hidden = () if output_hidden_states else None
        all_attn = () if output_attentions else None
        for i, layer in enumerate(self.layers, 1):
            if output_hidden_states:
                all_hidden += (hidden,)
            if prefill:
                visual_counts.append(current_visual)
                sequence_lengths.append(hidden.shape[1])
            prune = prefill and i == self.prune_layer
            if prune:
                weights = post_text_attention(layer, hidden, position_ids, visual_positions,
                                              layout['start'] + layout['span_length'])
            layer_past = cache.get_seq_length(i - 1) if use_cache else 0
            result = layer(hidden, attention_mask=self._mask(hidden, layer_past, output_attentions),
                           position_ids=position_ids, past_key_value=cache if use_cache else None,
                           output_attentions=output_attentions, use_cache=use_cache)
            hidden = result[0]
            if output_attentions:
                all_attn += (result[1],)
            if prune:
                if self.mico:
                    # Match image mico_weighted: weight covered tokens by their
                    # post-layer hidden L2 norm before cosine normalization.
                    weights = hidden_l2_weights(hidden[0, visual_positions], weights, normalize=True)
                selected = weighted_coverage_select(hidden[0, visual_positions], weights, layout['budget']['keep'])
                keep_mask = torch.ones(hidden.shape[1], device=hidden.device, dtype=torch.bool)
                keep_mask[visual_positions] = False
                keep_mask[visual_positions[selected]] = True
                retained = keep_mask.nonzero(as_tuple=True)[0]
                hidden = hidden[:, retained].contiguous()
                position_ids = position_ids[:, retained]
                current_visual = len(selected)
                self.last_debug = dict(method='mico_projected_llm_12' if self.mico else 'mico_coverage_proj_llm_12',
                    H='hidden_l2' if self.mico else 'disabled', K=self.prune_layer,
                    frames=layout['frames'], N1=len(visual_positions), R=current_visual,
                    visual_start=layout['start'], multimodal_span_length=layout['span_length'],
                    newline_count=layout['span_length'] - len(visual_positions),
                    target_per_frame=self.target_visual_tokens,
                    selected_per_frame=torch.bincount(frame_ids[selected], minlength=layout['frames']).tolist(),
                    stage1_per_frame=torch.bincount(frame_ids, minlength=layout['frames']).tolist(),
                    retained_sequence_indices=retained.detach().cpu(),
                    selected_visual_indices_sha256=hashlib.sha256(json.dumps(selected.tolist(), separators=(',', ':')).encode()).hexdigest(),
                    stage1_coordinates_sha256=hashlib.sha256(json.dumps(torch.stack([frame_ids, layout['original_patch_ids'].to(hidden.device)], dim=1).tolist(), separators=(',', ':')).encode()).hexdigest(),
                    selected_patch_coordinates_sha256=hashlib.sha256(json.dumps(torch.stack([frame_ids[selected], layout['original_patch_ids'].to(hidden.device)[selected]], dim=1).tolist(), separators=(',', ':')).encode()).hexdigest(),
                    attention_weights_sum=float(weights.sum()),
                    vision_weight_source='SigLIP loaded truncated-tower final-layer mean-head mean-query attention; no CLS',
                    **layout['budget'])
        hidden = self.norm(hidden)
        if output_hidden_states:
            all_hidden += (hidden,)
        if prefill:
            self.visual_token_num = sum(visual_counts) / len(self.layers)
            self.last_debug.update(layers_visual_tokens=visual_counts, layers_sequence_lengths=sequence_lengths,
                actual_average_total=self.visual_token_num,
                actual_average_per_frame=self.visual_token_num / layout['frames'],
                original_prefill_length=q_length, final_prefill_length=hidden.shape[1],
                cache_layer_lengths=[cache.get_seq_length(i) for i in range(len(self.layers))] if use_cache else None)
            self._emit_trace()
        next_cache = (cache.to_legacy_cache() if legacy else cache) if use_cache else None
        if not return_dict:
            return tuple(x for x in (hidden, next_cache, all_hidden, all_attn) if x is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=next_cache,
                                       hidden_states=all_hidden, attentions=all_attn)
