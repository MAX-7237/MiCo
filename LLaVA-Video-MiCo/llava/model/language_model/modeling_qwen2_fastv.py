import torch
from typing import List, Dict, Optional, Tuple, Union

from transformers import Qwen2Model
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast


# R_dict = {
#     "7b": {
#         2: {
#             256: 219,
#             128: 81,
#             64: 12,
#         },
#     },
# } # llava-ov-img
# R_dict = {
#     "7b": {
#         2: {
#             64: 54,
#             32: 19,
#             16: 2,
#         },
#     },
# } # llava-ov-vid
R_dict = {
    "7b": {
        2: {
            64: 56,
            32: 21,
            16: 4,
        },
    },
} # llava-video


class FastVQwen2Model(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config, fastv_config: Dict):
        super().__init__(config)
        self.system_prompt_length = 15
        # self.visual_token_length = 729 # llava-ov-img
        # self.visual_token_length = 196 # llava-ov-vid
        self.visual_token_length = 169 # llava-video (13x13)
        self.visual_token_num = 0
        self.video_frame_num = 0

        if config.num_hidden_layers == 28:
            self.scale = "7b"

        # FastV config
        self.K = fastv_config["K"]
        self.T = fastv_config["T"]  # Target tokens per frame
        self.R = R_dict[self.scale][self.K][self.T]  # Actual tokens to keep per frame

        print(f"[FastV Config] K={self.K}, T={self.T}, R={self.R}, visual_token_length={self.visual_token_length}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if seq_length > 1:
            print(f"[FastV Forward] seq_length={seq_length} > 1, 准备执行剪枝")
            print(f"[FastV Forward] video_frame_num={self.video_frame_num}, K={self.K}, R={self.R}")
            visual_token_length = self.visual_token_length
            visual_token_num = 0
        else:
            print(f"[FastV Forward] seq_length={seq_length} <= 1, 跳过剪枝（incremental generation）")

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                # FastV
                if seq_length > 1:
                    visual_token_num += visual_token_length
                    
                    if (decoder_layer.self_attn.layer_idx + 1) == self.K:
                        print("="*80)
                        print(f"[FastV] 在第 {self.K} 层执行剪枝")
                        print(f"[FastV] 剪枝前: seq_length={seq_length}, hidden_states.shape={hidden_states.shape}")
                        print(f"[FastV] video_frame_num={self.video_frame_num}, visual_token_length={self.visual_token_length}")
                        print(f"[FastV] 总视觉tokens={self.visual_token_length*self.video_frame_num}")

                        attn_mask = torch.ones((batch_size, hidden_states.shape[1]), device=hidden_states.device)
                        attn_mask = _prepare_4d_causal_attention_mask(attn_mask, (batch_size, hidden_states.shape[1]), hidden_states, 0)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )
                        hidden_states = layer_outputs[0]
                        last_attention = layer_outputs[1]

                        device = hidden_states.device
                        image_attention = last_attention.mean(dim=1)[0, -1,
                            self.system_prompt_length:self.system_prompt_length+self.visual_token_length*self.video_frame_num] # (N)

                        visual_token_length = self.R
                        print(f"[FastV] 每帧保留 R={self.R} 个tokens (总共 {self.R*self.video_frame_num} 个)")
                        visual_token_index = torch.topk(image_attention, k=self.R*self.video_frame_num).indices # (R)
                        visual_token_index = torch.sort(visual_token_index).values # (R)
                        full_token_index = torch.cat((
                            torch.arange(self.system_prompt_length, device=device),
                            visual_token_index + self.system_prompt_length,
                            torch.arange(self.system_prompt_length+self.visual_token_length*self.video_frame_num, seq_length, device=device),
                        )) # (T)

                        # get tokens by index
                        hidden_states = hidden_states[:,full_token_index]
                        position_ids = full_token_index.unsqueeze(0)

                        print(f"[FastV] 剪枝后: hidden_states.shape={hidden_states.shape}")
                        print(f"[FastV] 压缩率: {(1 - self.R*self.video_frame_num/(self.visual_token_length*self.video_frame_num))*100:.1f}%")
                        print("="*80)

                        layer_outputs = (hidden_states, layer_outputs[2])
                    
                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if seq_length > 1:
            self.visual_token_num = visual_token_num / len(self.layers)

        # add hidden states from the last decoder layer
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