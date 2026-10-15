"""ApET for InternVL3's Qwen2 decoder, with layer-local KV lengths.

The vendored LLaVA ApET hook merges after ``L // 2`` completed layers and
renumbers positions in the remaining layers. Early-layer caches retain the
longer S1 sequence; late-layer caches hold the merged S2 sequence.
"""

import torch
from transformers import Qwen2ForCausalLM, Qwen2Model
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..apet_select import apet_schedule, apet_stage2


class ApETQwen2Model(Qwen2Model):
    def __init__(self, config, visual_token_num):
        super().__init__(config)
        self.apet_schedule = apet_schedule(visual_token_num, config.num_hidden_layers)
        if self.apet_schedule['stage2'] <= 10:
            raise ValueError('ApET budget must leave more than its 10 basis tokens')
        self.system_prompt_length = 0
        self.visual_token_length = 0
        self.visual_token_num = visual_token_num
        self.apet_seed = None  # Official FPS draws from the worker's seeded global RNG.
        self.apet_basis = 10  # Official ApET basis count, shared with S1.
        self.apet_record = None
        self._next_position_ids = None

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None,
                cache_position=None, **flash_attn_kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Specify exactly one of input_ids and inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError('ApET evaluation requires batch size one per worker')
        if attention_mask is not None and (attention_mask.ndim != 2 or not attention_mask.bool().all()):
            raise ValueError('ApET evaluation supports unpadded prompts only')
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError('ApET requires DynamicCache, with a separate length for every layer')
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        prefill = past_seen == 0
        hidden_states = inputs_embeds
        input_length = hidden_states.shape[1]
        if prefill:
            if self.visual_token_length <= 0:
                raise ValueError('Register the current visual span before ApET prefill')
            if position_ids is None:
                position_ids = torch.arange(input_length, device=hidden_states.device)[None]
            self.apet_record = {'stage1': self.visual_token_length, 'layer_visual_tokens': [],
                                'prune_after': self.apet_schedule['prune_after']}
        elif self._next_position_ids is not None:
            # GenerationMixin tracks the outer S1 sequence, while S1 may retain
            # nonconsecutive original positions. Continue those positions below S2.
            position_ids = self._next_position_ids + torch.arange(input_length, device=hidden_states.device)[None]
        else:
            raise ValueError('ApET decode requires this request\'s prefill')
        self._next_position_ids = position_ids[:, -1:] + 1
        lower_positions = position_ids
        visual_length = self.visual_token_length
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if prefill and layer_idx == self.apet_schedule['prune_after']:
                keep = min(self.apet_schedule['stage2'], visual_length)
                if keep < visual_length:
                    hidden_states, indices = apet_stage2(
                        hidden_states, self.system_prompt_length, visual_length,
                        keep=keep, basis=self.apet_basis, seed=self.apet_seed)
                    if not torch.isfinite(hidden_states).all():
                        raise ValueError('ApET S2 produced non-finite hidden states')
                else:
                    indices = torch.arange(visual_length, device=hidden_states.device)[None]
                print(f'[ApET S2] Layer {layer_idx}: {visual_length} → {keep} tokens', flush=True)
                visual_length = keep
                self.apet_record['stage2'] = keep
                self.apet_record['selected_visual_indices'] = indices.detach()
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            query_length = hidden_states.shape[1]
            layer_past = past_key_values.get_seq_length(layer_idx) if past_key_values is not None else 0
            layer_cache_position = torch.arange(layer_past, layer_past + query_length, device=hidden_states.device)
            # Faithful to the reference hook: compact RoPE positions start at
            # the midpoint; earlier caches and their original positions survive.
            layer_positions = (lower_positions if layer_idx < self.apet_schedule['prune_after']
                               else layer_cache_position[None])
            position_embeddings = self.rotary_emb(hidden_states, layer_positions)
            if self.config._attn_implementation == 'flash_attention_2':
                causal_mask = None
            else:
                key_positions = torch.arange(layer_past + query_length, device=hidden_states.device)
                future = key_positions[None] > layer_cache_position[:, None]
                causal_mask = torch.zeros(query_length, layer_past + query_length,
                                          dtype=hidden_states.dtype, device=hidden_states.device)
                causal_mask.masked_fill_(future, torch.finfo(hidden_states.dtype).min)
                causal_mask = causal_mask[None, None]
            layer_outputs = decoder_layer(
                hidden_states, attention_mask=causal_mask, position_ids=layer_positions,
                past_key_value=past_key_values, output_attentions=output_attentions,
                use_cache=use_cache, cache_position=layer_cache_position,
                position_embeddings=position_embeddings, **flash_attn_kwargs)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
            if prefill:
                self.apet_record['layer_visual_tokens'].append(visual_length)

        hidden_states = self.norm(hidden_states)
        if prefill:
            self.visual_token_num = sum(self.apet_record['layer_visual_tokens']) / len(self.layers)
            self.apet_record['average_visual_tokens'] = self.visual_token_num
            print(f'[ApET Summary] layer_avg={self.visual_token_num:.4f}, '
                  f'S1={self.apet_record["stage1"]}, S2={self.apet_record["stage2"]}', flush=True)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states, attentions=all_attentions)


class Qwen2ForCausalLMApET(Qwen2ForCausalLM):
    def __init__(self, config, visual_token_num):
        super().__init__(config)
        self.model = ApETQwen2Model(config, visual_token_num)

    def set_system_prompt_length(self, length):
        self.model.system_prompt_length = int(length)

    def set_visual_token_length(self, length):
        self.model.visual_token_length = int(length)
