"""
DART (EMNLP'25) + DART_coverage 移植到 InternVL3 的 Qwen2 LLM 路径。
基于 modeling_qwen2_fastv.py 的 in-LLM 剪枝骨架 (已适配 InternVL3 的 Qwen2 backbone),
仅把"选哪些视觉 token 保留"的准则换成 DART 的:
  pivot(K states L1 范数 topk) + 每个 pivot 吸收 cosine 最不相似(diversity)的邻居。
DART_coverage (use_coverage=True): 把 diversity 换成 SCOPE coverage 边际增益贪心。
与 Qwen2.5-VL path 的 modeling_qwen2_5_vl_dart.py 逻辑一致, 只是位置编码是 2D(无 MROPE)。
"""
from functools import partial
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DARTQwen2Model(Qwen2Model):
    """Qwen2Model (InternVL3 LLM) with DART in-LLM 剪枝. use_coverage=True 时为 DART_coverage。"""

    def __init__(self, config: Qwen2Config, visual_token_num: int, num_image_token: int = 256,
                 use_coverage: bool = False):
        super().__init__(config)
        self.system_prompt_length = 41              # 运行时按 img_context 动态更新
        self.visual_token_length = 1280             # 运行时按实际视觉 token 数更新
        self.visual_token_num = 0
        self.visual_token_len = 0
        self.target = max(1, visual_token_num)      # 目标保留视觉 token 数 (对齐主表 256/128)
        self.use_coverage = use_coverage
        self.pruning_method = 'dart_coverage' if use_coverage else 'dart'

        self.K = 2                                  # 剪枝所在 decoder 层 (同 FastV)
        self.pivot_image_token = 4                  # 图像 pivot 数 (DART 原默认)
        self.dart_debug = False

    def set_system_prompt_length(self, length: int):
        self.system_prompt_length = length

    def _dart_select(self, hidden_states, decoder_layer, current_visual_length, vs):
        """返回要保留的视觉 token 局部索引 (相对 visual 段, 0..current_visual_length-1)。
        vs = 视觉段起始位置。DART: pivot(K的L1范数topk) + 吸收(diversity/coverage) 凑够 target。
        """
        self_attn = decoder_layer.self_attn
        attn_input = hidden_states
        ln = getattr(decoder_layer, "input_layernorm", None)
        if ln is not None:
            attn_input = ln(attn_input)

        # K states (RoPE 前), 用作 pivot 的 L1 范数依据 (DART 用 key states)
        key_states = self_attn.k_proj(attn_input)                    # [b, seq, nkv*hd]
        ve = vs + current_visual_length
        k_img = key_states[0, vs:ve, :]                              # [Nv, nkv*hd]
        # 视觉 token 的 LLM 隐状态 (norm 后) 用于相似度/覆盖
        feat = self.norm(hidden_states)[0, vs:ve, :]                 # [Nv, D]
        Nv = feat.shape[0]
        device = feat.device
        K_target = min(self.target, Nv)

        # ── pivot: K states L1 范数最大的 pivot_image_token 个 ──
        k_l1 = torch.norm(k_img, p=1, dim=-1)                        # [Nv]
        npv = min(self.pivot_image_token, Nv)
        pivots = k_l1.topk(npv).indices
        chosen = set(pivots.tolist())

        featn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(featn, featn.t())                        # [Nv,Nv] 余弦

        if not self.use_coverage:
            # ── DART 原版: 每个已选 token 吸收 cosine 最不相似(最不重复)的邻居 ──
            per = max(1, (K_target - len(chosen)) // max(1, len(chosen)))
            valid = set(range(Nv)) - chosen
            for it in list(chosen):
                if len(chosen) >= K_target or not valid:
                    break
                vlist = list(valid)
                cs = -sim[it, torch.tensor(vlist, device=device)]   # 反相似度=最不重复优先
                take = min(per, len(vlist))
                topk = cs.topk(take).indices
                add = [vlist[i] for i in topk.tolist()]
                chosen.update(add); valid.difference_update(add)
        else:
            # ── DART_coverage: SCOPE coverage 边际增益贪心补齐 ──
            avail = torch.ones(Nv, dtype=torch.bool, device=device)
            sel = list(chosen)
            for i in sel:
                avail[i] = False
            if sel:
                g = sim[torch.tensor(sel, device=device)].max(dim=0).values
            else:
                g = torch.zeros(Nv, device=device)
            while len(chosen) < K_target and avail.any():
                gain = torch.clamp(sim - g.unsqueeze(0), min=0.0).sum(dim=1)
                gain[~avail] = -float('inf')
                j = int(torch.argmax(gain).item())
                chosen.add(j); avail[j] = False
                g = torch.maximum(g, sim[j])

        idx = torch.tensor(sorted(chosen)[:K_target], device=device, dtype=torch.long)
        return idx

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
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        dart_enabled = inputs_embeds.shape[1] > 1
        if dart_enabled and self.dart_debug:
            tag = "DART_coverage" if self.use_coverage else "DART"
            print(f"\n[{tag}] prefill seq={inputs_embeds.shape[1]} vs={self.system_prompt_length} "
                  f"vis_len={self.visual_token_length} K={self.K} target={self.target}", flush=True)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if dart_enabled and layer_idx == self.K:
                device = hidden_states.device
                seq_len_now = hidden_states.shape[1]
                vs = min(self.system_prompt_length, seq_len_now)
                prev_len = min(self.visual_token_length, max(0, seq_len_now - vs))
                if prev_len <= self.target:
                    layer_outputs = decoder_layer(
                        hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                        past_key_value=past_key_values, output_attentions=output_attentions,
                        use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_outputs[0]
                    continue
                # 先跑完该层 (拿该层输出 hidden 用于选择, 与 FastV 一致)
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                    past_key_value=past_key_values, output_attentions=output_attentions,
                    use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]

                keep_local = self._dart_select(hidden_states, decoder_layer, prev_len, vs)
                keep = keep_local.shape[0]
                retain_token_indices = torch.cat((
                    torch.arange(vs, device=device),
                    keep_local + vs,
                    torch.arange(vs + prev_len, seq_len_now, device=device),
                ))
                hidden_states = hidden_states[:, retain_token_indices].contiguous()
                position_ids = position_ids[:, retain_token_indices]
                cache_position = cache_position[:retain_token_indices.shape[0]]
                position_embeddings = [pe[:, retain_token_indices] for pe in position_embeddings]
                causal_mask = self._update_causal_mask(None, hidden_states, cache_position, past_key_values, output_attentions)
                self.visual_token_num = keep
                self.visual_token_len = keep
                if self.dart_debug:
                    print(f"  [{'DART_cov' if self.use_coverage else 'DART'}] pruned {prev_len}->{keep} "
                          f"seq_now={hidden_states.shape[1]}", flush=True)
            else:
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                    past_key_value=past_key_values, output_attentions=output_attentions,
                    use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForCausalLMDART(Qwen2ForCausalLM):
    """InternVL3 LLM (Qwen2) + DART / DART_coverage in-LLM 剪枝。"""

    def __init__(self, config, pruning_method=None, visual_token_num=None, num_image_token=256):
        super().__init__(config)
        use_coverage = (pruning_method == 'dart_coverage')
        self.model = DARTQwen2Model(config, visual_token_num, num_image_token, use_coverage=use_coverage)

    def set_system_prompt_length(self, length: int):
        self.model.set_system_prompt_length(length)

    def set_visual_token_length(self, length: int):
        self.model.visual_token_length = length

    def enable_dart_debug(self, enable: bool = True):
        self.model.dart_debug = enable
