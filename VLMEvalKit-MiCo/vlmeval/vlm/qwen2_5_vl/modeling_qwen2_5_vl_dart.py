# DART (Duplication-Aware / EMNLP'25) + DART_coverage 移植到 VLMEvalKit-TOPS 的 Qwen2.5-VL 路径。
# 参考: /path/to/local-assets
# 因原 DART modeling 依赖 transformers 4.53+ 新 API(masking_utils/TransformersKwargs),
# 与本仓库 4.52.3 不兼容, 故不照搬其 modeling; 而是复用本仓库 FastV 的 in-LLM 剪枝骨架
# (已适配 4.52.3), 仅把"选哪些 token 保留"的准则换成 DART 的:
#   pivot(K states L1 范数 topk) + 每个 pivot 选与其 cosine 最不相似(=最不重复/diversity)的若干 token。
# DART_coverage: 把上面的 diversity(反cosine) 换成 SCOPE coverage 边际增益 (取自 TOPS)。
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DARTQwen2_5_VLModel(Qwen2_5_VLModel):
    """DART in-LLM 剪枝 (第 K 层). use_coverage=True 时为 DART_coverage 变体。"""

    def __init__(self, config, visual_token_num: int, use_coverage: bool = False, pure_cover: bool = False):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._update_causal_mask = lm._update_causal_mask

        self.system_prompt_length = 15          # 与 FastV 一致 (system prompt token 数)
        self.visual_token_length = 1296         # Qwen2.5-VL 单图 token 数 (运行时会按实际更新)
        self.visual_token_num = 0
        self.visual_token_len = 0
        self.target = max(1, visual_token_num)  # 目标保留视觉 token 数 (对齐主表 256/128)
        self.use_coverage = use_coverage
        self.pure_cover = pure_cover            # 纯 coverage 贪心(无DART pivots), 层由 env 控制

        import os as _os
        # DART 超参 (原论文默认). forward里 layer_idx=self_attn.layer_idx+1, 匹配 ==self.K,
        # 即在 decoder 层 (self.K-1) 之后剪. COVER_LLM_LAYER=L 语义="第L层之后剪" → self.K=L+1.
        self.K = int(_os.environ.get('COVER_LLM_LAYER', '2')) + 1 if pure_cover else 2  # 剪枝所在 decoder 层
        self.pivot_image_token = 4              # 图像 pivot 数
        self.dart_debug = False


    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _dart_select(self, hidden_states, decoder_layer, position_embeddings, current_visual_length, vs):
        """返回要保留的视觉 token 局部索引 (相对 visual 段, 0..current_visual_length-1)。
        vs = 视觉段起始位置 (动态传入)。
        DART: pivot(K的L1范数topk) + 每个pivot吸收与其最相似[或按coverage]的邻居, 直到凑够 target。
        """
        self_attn = decoder_layer.self_attn
        attn_input = hidden_states
        ln = getattr(decoder_layer, "input_layernorm", None)
        if ln is not None:
            attn_input = ln(attn_input)
        bsz, seq_len, _ = attn_input.shape
        nkv = self_attn.num_key_value_heads
        hd = self_attn.head_dim

        # K states (含 RoPE 前), 用作 pivot 的 L1 范数依据 (DART 用 key states)
        key_states = self_attn.k_proj(attn_input)                          # [b, seq, nkv*hd]
        ve = vs + current_visual_length
        k_img = key_states[0, vs:ve, :]                                    # [Nv, nkv*hd]  视觉段 K
        # 视觉 token 的 LLM 隐状态 (norm 后) 用于相似度/覆盖
        feat = self.norm(hidden_states)[0, vs:ve, :]                        # [Nv, D]
        Nv = feat.shape[0]
        device = feat.device
        K_target = min(self.target, Nv)

        # ── pivot: K states L1 范数最大的 pivot_image_token 个 (DART 原逻辑) ──
        k_l1 = torch.norm(k_img, p=1, dim=-1)                              # [Nv]
        npv = min(self.pivot_image_token, Nv)
        pivots = k_l1.topk(npv).indices                                    # 局部索引
        chosen = set(pivots.tolist())

        featn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)            # [Nv,D] 归一
        sim = torch.matmul(featn, featn.t())                              # [Nv,Nv] 余弦

        if self.pure_cover:
            # ── 纯 coverage 贪心 (与 divprune_coverage 一致, 无 DART pivots) ──
            simc = torch.clamp(sim, min=0.0)
            avail = torch.ones(Nv, dtype=torch.bool, device=device)
            g = torch.zeros(Nv, device=device); selc = []
            i0 = int(torch.argmax(simc.sum(dim=1)))
            selc.append(i0); avail[i0] = False; g = torch.maximum(g, simc[i0])
            while len(selc) < K_target and avail.any():
                gain = torch.clamp(simc - g.unsqueeze(0), min=0.0).sum(dim=1)
                gain[~avail] = -float('inf')
                j = int(torch.argmax(gain)); selc.append(j); avail[j] = False
                g = torch.maximum(g, simc[j])
            return torch.tensor(sorted(selc)[:K_target], device=device, dtype=torch.long)

        if not self.use_coverage:
            # ── DART 原版: 每个已选 token 吸收与其 cosine 最不相似(最不重复)的邻居 ──
            # TOKEN_TOPK = 每个 pivot 拉进多少个 (总数≈K_target)
            per = max(1, (K_target - len(chosen)) // max(1, len(chosen)))
            valid = set(range(Nv)) - chosen
            for it in list(chosen):
                if len(chosen) >= K_target or not valid:
                    break
                vlist = list(valid)
                cs = -sim[it, torch.tensor(vlist, device=device)]         # 反相似度 = 最不重复优先
                take = min(per, len(vlist))
                topk = cs.topk(take).indices
                add = [vlist[i] for i in topk.tolist()]
                chosen.update(add); valid.difference_update(add)
        else:
            # ── DART_coverage: 用 SCOPE coverage 边际增益贪心补齐, 替代反cosine diversity ──
            # g_k = 已选集对 token k 的最佳覆盖度; 每步选能新增覆盖最多的 token。
            avail = torch.ones(Nv, dtype=torch.bool, device=device)
            sel = list(chosen)
            for i in sel:
                avail[i] = False
            # 初始化 g: 已有 pivot 的覆盖
            if sel:
                g = sim[torch.tensor(sel, device=device)].max(dim=0).values  # [Nv]
            else:
                g = torch.zeros(Nv, device=device)
            while len(chosen) < K_target and avail.any():
                gain = torch.clamp(sim - g.unsqueeze(0), min=0.0).sum(dim=1)  # [Nv] 边际覆盖增益
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

        dart_enabled = inputs_embeds.shape[1] > 1
        if dart_enabled:
            current_visual_token_length = self.visual_token_length
            if self.dart_debug:
                tag = "DART_coverage" if self.use_coverage else "DART"
                print(f"\n[{tag}] prefill seq={inputs_embeds.shape[1]} visual={current_visual_token_length} "
                      f"K={self.K} target={self.target}")

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if dart_enabled:
                layer_idx = decoder_layer.self_attn.layer_idx + 1
                if layer_idx == self.K and current_visual_token_length > self.target:
                    device = hidden_states.device
                    seq_len_now = hidden_states.shape[1]
                    # 边界保护: 视觉段不能超过实际序列长度 (防 Qwen 动态分辨率下越界)
                    vs = min(self.system_prompt_length, seq_len_now)
                    prev_len = min(current_visual_token_length, max(0, seq_len_now - vs))
                    if prev_len <= self.target:
                        # 实际视觉 token 已不多于目标, 跳过剪枝
                        layer_outputs = decoder_layer(
                            hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                            past_key_value=past_key_values, output_attentions=output_attentions,
                            use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
                        )
                        hidden_states = layer_outputs[0]
                        if use_cache:
                            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                        if output_attentions:
                            all_self_attns += (layer_outputs[1],)
                        continue
                    # 先跑完该层 (拿到该层输出的 hidden 用于选择, 与 FastV 一致)
                    layer_outputs = decoder_layer(
                        hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                        past_key_value=past_key_values, output_attentions=output_attentions,
                        use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_outputs[0]

                    keep_local = self._dart_select(hidden_states, decoder_layer, position_embeddings, prev_len, vs)
                    keep = keep_local.shape[0]
                    # 诊断: 检查选择是否合法
                    if self.dart_debug:
                        kmin = int(keep_local.min()) if keep>0 else -1
                        kmax = int(keep_local.max()) if keep>0 else -1
                        ndup = keep - int(torch.unique(keep_local).shape[0])
                        print(f"  [DART-CHK] keep={keep} target={self.target} kmin={kmin} kmax={kmax}(prev_len={prev_len}) dup={ndup}")
                    retain_token_indices = torch.cat((
                        torch.arange(vs, device=device),
                        keep_local + vs,
                        torch.arange(vs + prev_len, seq_len_now, device=device),
                    ))
                    if self.dart_debug:
                        rmax = int(retain_token_indices.max())
                        rdup = retain_token_indices.shape[0] - int(torch.unique(retain_token_indices).shape[0])
                        ntext = int((torch.arange(vs + prev_len, seq_len_now, device=device)).shape[0])
                        print(f"  [DART-CHK] retain_n={retain_token_indices.shape[0]} rmax={rmax}(seq={seq_len_now}) rdup={rdup} ntext={ntext}")
                    hidden_states = hidden_states[:, retain_token_indices].contiguous()
                    if self.dart_debug:
                        nnan = int(torch.isnan(hidden_states).sum())
                        print(f"  [DART-CHK] after-slice hidden nan={nnan} shape={tuple(hidden_states.shape)}")
                    position_ids = position_ids[:, :, retain_token_indices]
                    cache_position = cache_position[:retain_token_indices.shape[0]]
                    position_embeddings = [pe[:, :, retain_token_indices] for pe in position_embeddings]
                    causal_mask = self._update_causal_mask(None, hidden_states, cache_position, past_key_values, output_attentions)
                    current_visual_token_length = keep
                    self.visual_token_num = keep
                    self.visual_token_len = keep
                    if self.dart_debug:
                        print(f"  [{'DART_cov' if self.use_coverage else 'DART'}] pruned {prev_len}->{keep} "
                              f"seq_now={hidden_states.shape[1]}")
                    layer_outputs = (hidden_states, layer_outputs[-1])
                else:
                    layer_outputs = decoder_layer(
                        hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                        past_key_value=past_key_values, output_attentions=output_attentions,
                        use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
                    )
            else:
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                    past_key_value=past_key_values, output_attentions=output_attentions,
                    use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
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
