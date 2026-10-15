import torch
import einops as ein
from typing import List, Dict, Optional, Tuple, Union

from transformers import Qwen2Model
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast


# R_dict = {
#     "7b": {
#         2: {
#             256: 350,
#             128: 200,
#             64: 20,
#         },
#         6: {
#             256: 260,
#             128: 80,
#             64: 10,
#         },
#         14: {
#             256: 150,
#             128: 40,
#             64: 5,
#         },
#     },
# } # llava-ov-img
# R_dict = {
#     "7b": {
#         2: {
#             64: 120,
#             32: 40,
#             16: 4,
#         },
#         6: {
#             64: 60,
#             32: 20,
#             16: 2,
#         },
#         14: {
#             64: 30,
#             32: 10,
#             16: 1,
#         },
#     },
# } # llava-ov-vid
R_dict = {
    "7b": {
        2: {
            64: 110,
            32: 50,
            16: 10,
        },
        6: {
            64: 70,
            32: 25,
            16: 5,
        },
        14: {
            64: 30,
            32: 10,
            16: 2,
        },
    },
} # llava-video


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_and_merge(x, cluster_num):
    B, N, C = x.shape

    x1 = ein.rearrange(x, "b l r -> b l () r")
    x2 = ein.rearrange(x, "b l r -> b () l r")
    distance = (x1 - x2).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)

    # get local density
    dist_nearest = torch.topk(dist_matrix, k=cluster_num, dim=-1, largest=False).values
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()

    # # add a little noise to ensure no tokens have the same density.
    # density += torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

    # get distance indicator
    mask = density[:, None, :] > density[:, :, None]
    mask = mask.type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1).values

    # select clustering center according to score
    score = dist * density
    index_down = torch.topk(score, k=cluster_num, dim=-1).indices

    # assign tokens to the nearest center
    dist_matrix = index_points(dist_matrix, index_down)
    idx_cluster = dist_matrix.argmin(dim=1)

    # make sure cluster center merge to itself
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    # merge tokens
    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    # token_weight = x.new_ones(B, N, 1).float()
    token_weight = torch.ones(B, N, 1)
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N).cpu(), source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6

    # average token features
    norm_weight = token_weight / all_weight[idx.cpu()]
    x_source = x.cpu() * norm_weight

    # x_merged = x.new_zeros(B * cluster_num, C).float()
    x_merged = torch.zeros(B * cluster_num, C)
    x_merged.index_add_(dim=0, index=idx.reshape(B * N).cpu(), source=x_source.reshape(B * N, C))
    x_merged = x_merged.reshape(B, cluster_num, C)
    
    return x_merged.to(device=x.device, dtype=x.dtype)


class SparseQwen2Model(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config, sparsevlm_config: Dict):
        super().__init__(config)
        self.system_prompt_length = 15
        # self.visual_token_length = 729 # llava-ov-img
        # self.visual_token_length = 196 # llava-ov-vid
        self.visual_token_length = 169 # llava-video (13x13)
        self.visual_token_num = 0
        self.video_frame_num = 0
        
        if config.num_hidden_layers == 28:
            self.scale = "7b"

        # SparseVLM config
        if config.num_hidden_layers == 28:
            self.pruning_loc = [2, 6, 14]
        self.retained_num = sparsevlm_config["T"]

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

        # SparseVLM
        if seq_length > 1:
            visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length+self.visual_token_length*self.video_frame_num]
            text_tokens = hidden_states[:, self.system_prompt_length+self.visual_token_length*self.video_frame_num:]
            matrix = text_tokens @ visual_tokens.transpose(1, 2)
            matrix = matrix.squeeze(0).softmax(0).mean(1)
            text_rater_index = torch.where(matrix > matrix.mean())[0]

            visual_token_length = self.visual_token_length
            visual_token_num = 0

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
                # SparseVLM
                if seq_length > 1:
                    visual_token_num += visual_token_length

                    if (decoder_layer.self_attn.layer_idx + 1) in self.pruning_loc:
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

                        self_attn_weights = layer_outputs[1].mean(1)
                        cross_attn_weights = self_attn_weights[:, text_rater_index+self.system_prompt_length+visual_token_length*self.video_frame_num, 
                                                               self.system_prompt_length:self.system_prompt_length+visual_token_length*self.video_frame_num]
                        cross_attn_weights = cross_attn_weights.mean(1)

                        retained_visual_tokens = torch.zeros_like(cross_attn_weights, dtype=bool)
                        retained_visual_index = torch.topk(
                            cross_attn_weights, 
                            min(R_dict[self.scale][decoder_layer.self_attn.layer_idx+1][self.retained_num], visual_token_length-1)*self.video_frame_num, 
                            dim=1).indices
                        retained_visual_tokens[0][retained_visual_index] = 1

                        total_visual_tokens = hidden_states[:, self.system_prompt_length:self.system_prompt_length+visual_token_length*self.video_frame_num]
                        sparse_visual_tokens = total_visual_tokens[:, torch.where(retained_visual_tokens == 0)[1]]
                        sparse_attn_weights = cross_attn_weights[:, torch.where(retained_visual_tokens == 0)[1]]
                        merge_num = max(int(sparse_attn_weights.shape[1] * 0.3), 1)
                        merge_token_index = sparse_attn_weights.topk(merge_num).indices

                        merge_visual_tokens = sparse_visual_tokens[:, merge_token_index.squeeze(0)]
                        cluster_num = max(int(merge_visual_tokens.shape[1] * 0.1), 1)
                        if cluster_num == 0:
                            cluster_num = merge_visual_tokens.shape[1]
                        merge_sparse_tokens = cluster_and_merge(merge_visual_tokens, cluster_num)

                        retained_visual_index = torch.sort(retained_visual_index).values
                        select_visual_tokens = total_visual_tokens[:, retained_visual_index.squeeze(0)]
                        hidden_states = torch.cat((
                            hidden_states[:, :self.system_prompt_length],
                            select_visual_tokens,
                            merge_sparse_tokens,
                            hidden_states[:, self.system_prompt_length+visual_token_length*self.video_frame_num:]
                        ), dim=1)
                        position_ids = position_ids[:,:hidden_states.shape[1]]

                        layer_outputs = (hidden_states, layer_outputs[2])
                        # visual_token_length = select_visual_tokens.shape[1] // self.video_frame_num
                        visual_token_length = (select_visual_tokens.shape[1] + merge_sparse_tokens.shape[1]) // self.video_frame_num

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
                        position_ids=torch.tensor([[
                            past_key_values.get_usable_length(1, decoder_layer.self_attn.layer_idx)]], 
                            dtype=torch.int64).cuda(),
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