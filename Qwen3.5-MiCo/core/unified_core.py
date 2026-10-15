"""Fixed layer-average visual budgets on native Transformers 5.2 Qwen3.5.

One unpadded image, batch one, SDPA, cached greedy inference. Changes are scoped
to a model instance; original text/mRoPE anchors and each layer's physical cache
are preserved. Method mathematics live in separately source-parity-tested modules.
"""
import math
import hashlib
import json
from numbers import Integral
from types import MethodType

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache, Qwen3_5ModelOutputWithPast,
    apply_rotary_pos_emb, repeat_kv,
)

METHODS = ('baseline', 'mico_s1', 'mico')
HYBRID = 'native_hybrid_task_v1'
FULL_REDUCTION = 'native_rmsnorm_rope_qk_visual_softmax_native_dtype_mean_query_heads'


class PruningController:
    def __init__(self, model, method, target_tokens=None, record_callback=None, *, cut_after_layer=4, quality_mode='qk',
                 alpha_num=2, alpha_den=1, score_beta_mode=None, score_query_scope=None,
                 record_s2_hidden_digest=False):
        from budget_schedule import validate_alpha
        validate_alpha(alpha_num, alpha_den)
        hybrid = quality_mode == HYBRID
        if score_beta_mode is None: score_beta_mode = 'native_beta' if hybrid else 'none'
        if score_query_scope is None: score_query_scope = 'task_content' if hybrid else 'all_nonvisual'
        expected = ('native_beta', 'task_content') if hybrid else ('none', 'all_nonvisual')
        if (score_beta_mode, score_query_scope) != expected:
            raise ValueError('score controls do not match the fixed layer-sweep recipe')
        if type(record_s2_hidden_digest) is not bool:
            raise ValueError('record_s2_hidden_digest must be boolean')
        if method not in METHODS:
            raise ValueError(f'unknown pruning method: {method}')
        if (isinstance(cut_after_layer, bool) or not isinstance(cut_after_layer, Integral)
                or cut_after_layer not in range(1, 16)
                or (method != 'mico' and cut_after_layer != 4)):
            raise ValueError('MiCo requires K1..15; single-stage controls use interface K4')
        if quality_mode not in ('qk', HYBRID) or (hybrid and method != 'mico'):
            raise ValueError('only original qk and same-layer native_hybrid_task_v1 are supported')
        if quality_mode == 'qk' and cut_after_layer not in (4, 8, 12):
            raise ValueError('original qk requires full-attention K4/K8/K12')
        if method == 'baseline':
            if target_tokens is not None:
                raise ValueError('baseline has no target budget')
        elif isinstance(target_tokens, bool) or not isinstance(target_tokens, Integral) or target_tokens <= 0:
            raise ValueError('target_tokens must be a positive integer')
        if hasattr(model, '_qwen35_pruning_controller'):
            raise ValueError('a pruning controller is already installed')
        self.model, self.method = model, method
        self.quality_mode = quality_mode
        self._requires_query_binding = hybrid
        self._score_evidence = self._score_source_positions = None
        self.alpha_num, self.alpha_den = alpha_num, alpha_den
        self.score_beta_mode, self.score_query_scope = score_beta_mode, score_query_scope
        self.record_s2_hidden_digest = record_s2_hidden_digest
        self._score_query_evidence = self._s2_pre_hidden = None
        self._gate_evidence = None
        self.target_tokens = None if target_tokens is None else int(target_tokens)
        self.record_callback = record_callback
        self.text = model.model.language_model
        if self.text.config._attn_implementation != 'sdpa':
            raise ValueError('this audited adapter requires SDPA')
        self.num_layers = len(self.text.layers)
        full_layers = [i + 1 for i, layer in enumerate(self.text.layers) if layer.layer_type == 'full_attention']
        if (self.num_layers != 32 or [x.layer_type for x in self.text.layers] !=
                (['linear_attention'] * 3 + ['full_attention']) * 8):
            raise ValueError('requires the native 32-layer Qwen3.5 hybrid architecture')
        self.cut = int(cut_after_layer)
        self.score_source_layer = self.cut if method == 'mico' else None
        self.drop_after_layer = self.score_source_layer
        use_linear_source = hybrid and self.cut not in full_layers
        self.score_source_type = (('linear_postconv_qk' if use_linear_source else 'full_attention_qk')
                                  if method == 'mico' else None)
        self.effective_beta_mode = 'native_beta' if use_linear_source else 'none'
        self._linear_module = self._linear_original_chunk = None
        self._linear_capture, self._pending_question = {}, None
        self._beta_evidence = self._question_evidence = None
        self._beta_source_positions = None
        if use_linear_source:
            layer = self.text.layers[self.score_source_layer - 1]
            linear = getattr(layer, 'linear_attn', None)
            if (layer.layer_type != 'linear_attention' or not isinstance(linear, torch.nn.Module)
                    or not callable(getattr(linear, 'chunk_gated_delta_rule', None))
                    or not callable(getattr(linear, 'recurrent_gated_delta_rule', None))
                    or not isinstance(getattr(linear, 'in_proj_b', None), torch.nn.Linear)
                    or linear.num_v_heads % linear.num_k_heads != 0):
                raise ValueError('linear source requires native DeltaNet at the selected cut')
            self._linear_module = linear
            self._linear_original_chunk = linear.chunk_gated_delta_rule
        self.apet_cut = self.num_layers // 2
        self.sparse_cuts = (4, 8, 16)
        if method == 'sparsevlm' and (not set(self.sparse_cuts).issubset(full_layers) or self.num_layers <= 16):
            raise ValueError('SparseVLM requires native full attention at layers 4, 8, 16 and later layers')
        probe_cuts = (self.cut,) if method == 'mico' and not use_linear_source else ()
        for cut in probe_cuts:
            attention = getattr(self.text.layers[cut - 1], 'self_attn', None)
            if (not isinstance(attention, torch.nn.Module)
                    or not isinstance(getattr(attention, 'q_norm', None), torch.nn.Module)
                    or not isinstance(getattr(attention, 'k_norm', None), torch.nn.Module)):
                raise ValueError('selected full-attention block lacks native Q/K hook targets')
            if quality_mode == 'receiver_gate_mean' and not isinstance(getattr(attention, 'q_proj', None), torch.nn.Linear):
                raise ValueError('receiver gate requires the native q_proj hook target')
        self.last_record = self.context = self._cache = None
        self._captured, self._raw_features = {}, None
        self._original_forward = self.text.forward
        self._hooks = [model.register_forward_pre_hook(self._root_pre, with_kwargs=True)]
        for cut in probe_cuts:
            attention = self.text.layers[cut - 1].self_attn
            self._hooks.extend([
                attention.register_forward_pre_hook(self._attention_pre, with_kwargs=True),
                attention.q_norm.register_forward_hook(self._capture_q),
                attention.k_norm.register_forward_hook(self._capture_k),
            ])
            if quality_mode == 'receiver_gate_mean':
                self._hooks.append(attention.q_proj.register_forward_hook(self._capture_gate))
        self.vision_tap = None
        if method in ('mico', 'mico_s1', 'visionzip', 'vispruner', 'holov'):
            from mico_selector import VisionTap
            self.vision_tap = VisionTap(model.model.visual)
        elif method == 'mmtok':
            self._hooks.append(model.model.visual.register_forward_hook(self._capture_raw_vision))
        if self._linear_module is not None:
            # Instance-local callable observation, not a model/global forward patch.
            self._linear_module.chunk_gated_delta_rule = self._observe_linear_chunk
        self.text.forward = MethodType(self._text_forward, self.text)
        model._qwen35_pruning_controller = self

    def remove(self):
        self.text.forward = self._original_forward
        for handle in self._hooks:
            handle.remove()
        if self.vision_tap is not None:
            self.vision_tap.close()
        self._cache = self.context = self._raw_features = None
        self._captured = {}
        self._gate_evidence = None
        if self._linear_module is not None:
            self._linear_module.chunk_gated_delta_rule = self._linear_original_chunk
        self._linear_capture, self._pending_question = {}, None
        self._beta_evidence = self._question_evidence = self._beta_source_positions = None
        self._score_query_evidence = self._s2_pre_hidden = None
        self._score_evidence = self._score_source_positions = None
        del self.model._qwen35_pruning_controller

    def bind_question_tokens(self, input_ids, question_token_mask):
        """Bind one exact prompt to its externally prepared question-only mask.

        The caller must derive this mask from the actual native template/tokenizer
        with question span provenance. This core deliberately does not guess it.
        Every fresh request consumes the binding, including keep-all requests.
        """
        if self.score_query_scope != 'task_content':
            raise ValueError('legacy question binding is only for task_content')
        self.bind_score_query_tokens(input_ids, question_token_mask, score_query_scope='task_content')

    def bind_score_query_tokens(self, input_ids, score_query_mask, *, score_query_scope):
        """Bind exact native IDs and an external mask; never render/tokenize here."""
        question_token_mask = score_query_mask
        if not self._requires_query_binding:
            raise ValueError('question binding is only accepted for native_hybrid_task_v1')
        if score_query_scope != self.score_query_scope:
            raise ValueError('score query scope differs from the configured scope')
        if (not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long
                or input_ids.ndim != 2 or input_ids.shape[0] != 1
                or not isinstance(question_token_mask, torch.Tensor)
                or question_token_mask.dtype != torch.bool
                or question_token_mask.shape != input_ids.shape
                or not bool(question_token_mask.any())):
            raise ValueError('requires exact prompt IDs and nonempty boolean question mask')
        ids, mask = input_ids.detach().cpu().clone(), question_token_mask.detach().cpu().clone()
        visual = (ids[0] == self.model.config.image_token_id).nonzero().flatten()
        question = mask[0].nonzero().flatten()
        if not visual.numel() or int(question[0]) <= int(visual[-1]):
            raise ValueError('question tokens must all follow the image')
        forbidden = set()
        for config in (self.model.config, self.text.config):
            for name in ('image_token_id', 'video_token_id', 'vision_start_token_id',
                         'vision_end_token_id', 'pad_token_id', 'eos_token_id', 'bos_token_id'):
                values = getattr(config, name, None)
                if values is not None:
                    forbidden.update(values if isinstance(values, (list, tuple)) else [values])
        if any(int(x) in forbidden for x in ids[mask]):
            raise ValueError('question mask includes a special/image/padding token')
        if score_query_scope == 'post_image_text':
            expected = torch.arange(ids.shape[1]) > int(visual[-1])
            for token in forbidden: expected &= ids[0] != token
            if not torch.equal(mask[0], expected):
                raise ValueError('post_image_text must contain every eligible post-image token')
        if self._pending_question is not None:
            raise ValueError('unconsumed question binding already exists')
        self._pending_question = (ids, mask)

    def _observe_linear_chunk(self, query, key, value, **kwargs):
        if (self.context is not None and self.context['prefill']
                and self.context['target'] < self.context['n0']):
            if self._linear_capture:
                raise ValueError('linear beta capture was already populated for this request')
            beta = kwargs.get('beta')
            length = self.context['original_length'] - self.context['n0'] + self.context['n1']
            shape = (1, length, self._linear_module.num_v_heads, self._linear_module.head_k_dim)
            if (not isinstance(query, torch.Tensor) or tuple(query.shape) != shape
                    or not isinstance(key, torch.Tensor) or tuple(key.shape) != shape
                    or not isinstance(beta, torch.Tensor) or tuple(beta.shape) != shape[:-1]
                    or kwargs.get('use_qk_l2norm_in_kernel') is not True
                    or not all(bool(torch.isfinite(x).all()) for x in (query, key, beta))
                    or not bool(((beta >= 0) & (beta <= 1)).all())):
                raise ValueError('invalid native postconv Q/K/beta capture')
            self._linear_capture = dict(q=query.detach().clone(), k=key.detach().clone(),
                                        beta=beta.detach().clone(), native_qk_l2norm=True)
        return self._linear_original_chunk(query, key, value, **kwargs)

    def _linear_quality(self, visual_local):
        from linear_beta import linear_beta_quality, REDUCTION, UNIT_REDUCTION
        capture, self._linear_capture = self._linear_capture, {}
        if (set(capture) != {'q', 'k', 'beta', 'native_qk_l2norm'}
                or capture['native_qk_l2norm'] is not True
                or self._beta_source_positions is None or self._question_evidence is None):
            raise ValueError('missing current native linear-beta evidence')
        q, k, beta = capture['q'], capture['k'], capture['beta']
        sources = self._beta_source_positions
        original_queries = self.context['question_positions']
        queries = torch.isin(sources, original_queries).nonzero().flatten()
        if (queries.numel() != original_queries.numel()
                or not torch.equal(sources[queries], original_queries)
                or q.shape[1] != sources.numel()):
            raise ValueError('question-to-current-position projection is not exact')
        score = linear_beta_quality(q, k, beta, queries, visual_local, score_beta_mode=self.score_beta_mode)
        self._beta_evidence = dict(source='native_postconv_chunk_gated_delta_rule_inputs',
            source_layer=self.score_source_layer, drop_after_layer=self.drop_after_layer,
            qk_shape=list(q.shape), beta_shape=list(beta.shape),
            native_key_heads=self._linear_module.num_k_heads,
            native_value_heads=self._linear_module.num_v_heads,
            native_qk_l2norm=True, question_original_positions=original_queries.tolist(),
            question_local_positions=queries.tolist(), query_count=int(queries.numel()),
            finite=True, reduction=REDUCTION if self.score_beta_mode == 'native_beta' else UNIT_REDUCTION,
            all_zero=bool((score == 0).all()), question_binding=dict(self._question_evidence),
            score_beta_mode=self.score_beta_mode, score_query_scope=self.score_query_scope,
            score_query_original_positions=original_queries.tolist(), score_query_local_positions=queries.tolist(),
            score_query_binding=dict(self._score_query_evidence))
        self._score_evidence = dict(self._beta_evidence, source_type=self.score_source_type,
                                    effective_beta_mode=self.effective_beta_mode)
        return score

    def _root_pre(self, module, args, kwargs):
        if module.training:
            raise ValueError('pruning is inference-only')
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('requires input_ids with batch size one')
        if kwargs.get('labels') is not None or kwargs.get('use_cache') is False:
            raise ValueError('requires cached inference without labels')
        mask = kwargs.get('attention_mask')
        if mask is not None and (mask.ndim != 2 or not bool(torch.all(mask == 1))):
            raise ValueError('padding and custom attention masks are unsupported')
        cache = kwargs.get('past_key_values')
        fresh = cache is None or cache.get_seq_length() == 0
        if not fresh:
            if ids.shape[1] != 1 or cache is not self._cache or self.last_record is None:
                raise ValueError('only single-token continuation of this prompt is supported')
            if mask is None and kwargs.get('position_ids') is None:
                raise ValueError('logical decode positions require the original attention mask or explicit position_ids')
            self.context['prefill'] = False
            return
        binding = None
        if self._requires_query_binding:
            binding, self._pending_question = self._pending_question, None
            self._linear_capture = {}
            self._beta_evidence = self._question_evidence = self._beta_source_positions = None
            if binding is None or not torch.equal(ids.detach().cpu(), binding[0]):
                raise ValueError('fresh prompt lacks its exact single-use question binding')
        self._s2_pre_hidden = self._score_query_evidence = None
        self._score_evidence = self._score_source_positions = None
        grid = kwargs.get('image_grid_thw')
        if (grid is None or tuple(grid.shape) != (1, 3) or int(grid[0, 0]) != 1
                or kwargs.get('pixel_values') is None or kwargs.get('pixel_values_videos') is not None
                or bool((ids == module.config.video_token_id).any())):
            raise ValueError('requires exactly one image and no video')
        visual = (ids[0] == module.config.image_token_id).nonzero().flatten()
        if visual.numel() == 0 or not torch.equal(visual, torch.arange(visual[0], visual[-1] + 1, device=visual.device)):
            raise ValueError('requires a nonempty contiguous image-token span')
        if int(visual[-1]) >= ids.shape[1] - 1:
            raise ValueError('a text query must follow the image')
        n0 = visual.numel()
        if int(grid.prod()) // int(module.model.visual.config.spatial_merge_size) ** 2 != n0:
            raise ValueError('native image grid and visual token count disagree')
        target = n0 if self.target_tokens is None else min(self.target_tokens, n0)
        n1 = n2 = n0
        schedule = {}
        if target < n0:
            if self.method in ('fastv', 'mico'):
                remaining = self.num_layers - self.cut
                if remaining <= 0:
                    raise ValueError('no layers remain after pruning')
                if self.method == 'mico':
                    from budget_schedule import mico_budget
                    budget = mico_budget(target, n0, alpha_num=self.alpha_num, alpha_den=self.alpha_den,
                                         cut_after_layer=self.cut)
                    n1, n2 = budget['n1'], budget['n2']
                else:
                    n1 = n0
                    n2 = math.floor((target * self.num_layers - self.cut * n1) / remaining)
                if not 1 <= n2 <= n1:
                    raise ValueError(f'unreachable layer-average budget: N0={n0}, N1={n1}, N2={n2}')
            elif self.method == 'sparsevlm':
                from sparse_selector import build_keep_schedule
                schedule, _, _ = build_keep_schedule(target, self.num_layers, [n0], self.sparse_cuts)
            elif self.method == 'apet':
                n1 = min(round(1.5 * target), n0)
                n2 = round((self.num_layers * target - self.apet_cut * n1) / (self.num_layers - self.apet_cut))
                if not 1 <= n2 <= n1:
                    raise ValueError('unreachable ApET layer-average budget')
            else:
                n1 = n2 = target
        self.context = dict(prefill=True, visual=visual, original_length=ids.shape[1],
                            n0=n0, n1=n1, n2=n2, target=target, schedule=schedule)
        if binding is not None:
            self.context['question_positions'] = binding[1][0].nonzero().flatten().to(ids.device)
            self._question_evidence = {
                'protocol': 'explicit_native_prompt_question_mask_v1',
                'input_ids_sha256': hashlib.sha256(json.dumps(binding[0].tolist(), separators=(',', ':')).encode()).hexdigest(),
                'question_mask_sha256': hashlib.sha256(json.dumps(binding[1].tolist(), separators=(',', ':')).encode()).hexdigest(),
            }
            self._score_query_evidence = dict(protocol='explicit_native_prompt_score_query_mask_v1',
                score_query_scope=self.score_query_scope,
                input_ids_sha256=self._question_evidence['input_ids_sha256'],
                score_query_mask_sha256=self._question_evidence['question_mask_sha256'],
                original_positions=self.context['question_positions'].tolist(),
                query_count=int(self.context['question_positions'].numel()))
        self.last_record = None
        self._gate_evidence = None
        self._captured, self._raw_features = {}, None

    def _capture_raw_vision(self, module, args, output):
        raw = output.last_hidden_state
        n0 = self.context['n0']
        if raw.ndim != 2 or raw.shape[0] != 4 * n0:
            raise ValueError('MMTok vision features do not match native 2x2 image groups')
        self._raw_features = raw.detach().float().reshape(n0, 4, raw.shape[-1]).mean(1)

    def _attention_pre(self, module, args, kwargs):
        self._captured = {}
        if self.context is not None and self.context['prefill'] and self.context['target'] < self.context['n0']:
            self._captured = {'position_embeddings': kwargs['position_embeddings']}

    def _capture_gate(self, module, args, output):
        if not self._captured or self.context is None or not self.context['prefill']:
            return
        attention = self.text.layers[self.cut - 1].self_attn
        heads, dim = self.text.config.num_attention_heads, attention.head_dim
        length = self.context['original_length'] - self.context['n0'] + self.context['n1']
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, length, 2 * heads * dim):
            raise ValueError('native q_proj output does not match current prefill gate shape')
        gate = output.detach().reshape(1, length, heads, 2 * dim).chunk(2, dim=-1)[1]
        if not bool(torch.isfinite(gate).all()):
            raise ValueError('native receiver gate is nonfinite')
        # One declared proxy: sigmoid before the channel mean, both in FP32.
        self._captured['gate_weights'] = gate.float().sigmoid().mean(-1).transpose(1, 2)
        self._captured['gate_shape'] = list(gate.shape)

    def _capture_q(self, module, args, output):
        if self._captured and self.context['prefill']:
            self._captured['q'] = output.transpose(1, 2)

    def _capture_k(self, module, args, output):
        if self._captured and self.context['prefill']:
            self._captured['k'] = output.transpose(1, 2)

    def _rotated_probe(self):
        captured, self._captured = self._captured, {}
        return apply_rotary_pos_emb(captured['q'], captured['k'], *captured['position_embeddings'])

    def _quality(self, visual_local):
        if self._linear_module is not None:
            return self._linear_quality(visual_local)
        attention = self.text.layers[self.cut - 1].self_attn
        gate_weights = self._captured.get('gate_weights')
        gate_shape = self._captured.get('gate_shape')
        q, k = self._rotated_probe()
        k = repeat_kv(k, attention.num_key_value_groups)
        if self._requires_query_binding:
            sources = self._score_source_positions
            original_queries = self.context['question_positions']
            queries = torch.isin(sources, original_queries).nonzero().flatten()
            if (sources.numel() != q.shape[-2] or queries.numel() != original_queries.numel()
                    or not torch.equal(sources[queries], original_queries)):
                raise ValueError('hybrid full-attention query projection is not exact')
        elif self.method == 'fastv':
            queries = torch.tensor([q.shape[-2] - 1], device=q.device)
        else:
            non_visual = torch.ones(q.shape[-2], dtype=torch.bool, device=q.device)
            non_visual[visual_local] = False
            queries = non_visual.nonzero().flatten()
        if not queries.numel():
            raise ValueError('no text queries remain')
        logits = torch.matmul(q.index_select(2, queries), k.index_select(2, visual_local).transpose(-1, -2)) * attention.scaling
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        if self.method == 'mico':
            probabilities = probabilities.to(q.dtype)
            if self.quality_mode == 'receiver_gate_mean':
                if (not isinstance(gate_weights, torch.Tensor) or tuple(gate_weights.shape) != tuple(q.shape[:-1])
                        or gate_shape != [q.shape[0], q.shape[2], q.shape[1], q.shape[3]]
                        or not bool(torch.isfinite(gate_weights).all())
                        or not bool(((gate_weights >= 0) & (gate_weights <= 1)).all())):
                    raise ValueError('missing, nonfinite, or mismatched current receiver gate')
                probabilities = probabilities * gate_weights.index_select(2, queries).to(probabilities.dtype).unsqueeze(-1)
                self._gate_evidence = dict(source='native_q_proj', cut_after_layer=self.cut,
                    gate_shape=gate_shape, weight_shape=list(gate_weights.shape), query_count=int(queries.numel()),
                    finite=True, reduction='sigmoid_fp32_mean_channel_fp32_cast_probability_dtype')
            score = probabilities.mean(2).mean(1).mean(0)
            self._score_evidence = dict(source='native_q_norm_k_norm_rope',
                source_layer=self.score_source_layer, drop_after_layer=self.drop_after_layer,
                source_type=self.score_source_type, effective_beta_mode=self.effective_beta_mode,
                reduction=FULL_REDUCTION, qk_shape=list(q.shape),
                score_query_scope=self.score_query_scope,
                score_query_original_positions=self._score_source_positions[queries].tolist(),
                score_query_local_positions=queries.tolist(), query_count=int(queries.numel()),
                finite=bool(torch.isfinite(score).all()), all_zero=bool((score == 0).all()))
            if not self._score_evidence['finite']:
                raise ValueError('nonfinite full-attention relevance')
            return score
        return probabilities.mean(1).squeeze(1).mean(0)

    @staticmethod
    def _select(hidden, position_ids, positions, sources, visual, selected, values=None):
        """Replace merged values at sorted visual anchors, then retain all text."""
        if selected.ndim != 1 or not selected.numel() or not bool(torch.all(selected[1:] > selected[:-1])):
            raise ValueError('visual selection must contain sorted unique anchors')
        if int(selected[0]) < 0 or int(selected[-1]) >= visual.numel():
            raise ValueError('visual selection outside the current span')
        chosen = visual.index_select(0, selected)
        retain = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        retain[visual] = False
        retain[chosen] = True
        keep = retain.nonzero().flatten()
        if values is not None:
            if values.shape != (chosen.numel(), hidden.shape[-1]):
                raise ValueError('merged embedding shape does not match its anchors')
            hidden = hidden.clone()
            hidden[0, chosen] = values
        is_visual = torch.zeros_like(retain)
        is_visual[chosen] = True
        return (hidden[:, keep], position_ids[..., keep], positions[keep], sources[keep],
                is_visual[keep].nonzero().flatten())

    def _stage1(self, hidden, visual, n1):
        projected = hidden[0].index_select(0, visual)
        capture = self.vision_tap.pop() if self.vision_tap is not None else None
        if capture is not None and (capture[0].shape[0] != len(visual) or capture[1].numel() != len(visual)):
            raise ValueError('vision selector and image-token counts differ')
        if n1 >= len(visual):
            self._raw_features = None
            return None, None
        if self.method in ('mico', 'mico_s1'):
            from mico_selector import coverage_select
            return coverage_select(*capture, n1), None
        if self.method in ('divprune', 'visionzip', 'vispruner', 'holov'):
            from visual_methods import select_visual
            values, indices = select_visual(self.method, projected, capture, n1)
            return indices, values
        if self.method == 'mmtok':
            from mm_methods import mmtok_select
            if self._raw_features is None:
                raise ValueError('MMTok requires the current image raw features')
            indices = mmtok_select(projected, self._raw_features, hidden[0], visual, n1)
            self._raw_features = None
            return indices, None
        if self.method == 'apet':
            from mm_methods import apet_select_merge
            values, indices = apet_select_merge(projected, n1, seed=0, stage='s1')
            return indices, values
        raise RuntimeError('unexpected stage-one method')

    def _text_forward(self, text_model, input_ids=None, attention_mask=None,
                      position_ids=None, past_key_values=None, inputs_embeds=None,
                      use_cache=None, cache_position=None, **kwargs):
        if self.context is None:
            raise ValueError('call the installed multimodal model, not its text submodule')
        if input_ids is not None or inputs_embeds is None:
            raise ValueError('multimodal embeddings are required')
        if use_cache is False or kwargs.get('output_attentions') or kwargs.get('output_hidden_states'):
            raise ValueError('requires cache without full attention/hidden-state histories')
        if past_key_values is None:
            past_key_values = Qwen3_5DynamicCache(text_model.config)
        self._cache = past_key_values
        if cache_position is None:
            start = past_key_values.get_seq_length()
            cache_position = torch.arange(start, start + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            raise ValueError('original multimodal positions must be prepared before pruning')
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        if position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        hidden, positions = inputs_embeds, cache_position
        prefill = self.context['prefill']
        if prefill:
            n0, n1, n2 = (self.context[k] for k in ('n0', 'n1', 'n2'))
            visual = self.context['visual']
            sources = torch.arange(hidden.shape[1], device=hidden.device)
            counts, events = [], []
            raters = None
            if self.context['schedule']:
                from sparse_selector import compute_text_rater
                raters = compute_text_rater(hidden, visual)
                if not raters.numel():
                    raise ValueError('SparseVLM requires at least one text rater')
            selected, values = self._stage1(hidden, visual, n1)
            if selected is not None:
                before = len(visual)
                hidden, position_ids, positions, sources, visual = self._select(
                    hidden, position_ids, positions, sources, visual, selected, values)
                events.append(dict(after_layer=0, visual_before=before, visual_after=len(visual), position_sources=sources.tolist()))
            self._score_source_positions = sources.detach().clone()
            if self._requires_query_binding:
                self._beta_source_positions = sources.detach().clone()
        # Unpadded prefill uses native causal SDPA; one-query decode can attend
        # all keys in each layer's independently sized physical KV cache.
        embeddings = text_model.rotary_emb(hidden, position_ids)
        for i, layer in enumerate(text_model.layers):
            if prefill:
                counts.append(len(visual))
            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=None,
                           position_ids=position_ids, past_key_values=past_key_values,
                           use_cache=True, cache_position=positions, **kwargs)
            if not prefill:
                continue
            cut = i + 1
            before = len(visual)
            changed = False
            if self.method in ('fastv', 'mico') and cut == self.cut and n2 < before:
                if self.record_s2_hidden_digest:
                    raw_hidden = hidden.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    self._s2_pre_hidden = dict(shape=list(hidden.shape), dtype=str(hidden.dtype),
                                              sha256=hashlib.sha256(raw_hidden).hexdigest())
                quality = self._quality(visual)
                if self.method == 'fastv':
                    selected = torch.topk(quality, n2).indices.sort().values
                else:
                    from mico_selector import coverage_select
                    selected = coverage_select(hidden[0].index_select(0, visual), quality, n2, normalize_weights=True)
                hidden, position_ids, positions, sources, visual = self._select(
                    hidden, position_ids, positions, sources, visual, selected)
                changed = True
            elif self.method == 'apet' and cut == self.apet_cut and n2 < before:
                from mm_methods import apet_select_merge
                values, selected = apet_select_merge(hidden[0].index_select(0, visual), n2, seed=0, stage='s2')
                hidden, position_ids, positions, sources, visual = self._select(
                    hidden, position_ids, positions, sources, visual, selected, values)
                changed = True
            elif cut in self.context['schedule']:
                from sparse_selector import score_visual_tokens, select_and_recycle, compact_visual_spans
                q, k = self._rotated_probe()
                quality = score_visual_tokens(q, k, visual, raters)
                kept, merged = select_and_recycle(hidden, visual, quality, self.context['schedule'][cut][0])
                hidden, _, pos_source, visual, _, old_to_new = compact_visual_spans(hidden, visual, [kept], [merged])
                # Recycled centroids inherit the last retained visual anchor.
                # Text raters instead follow the exact (non-monotonic) source map.
                position_ids, positions, sources = position_ids[..., pos_source], positions[pos_source], sources[pos_source]
                raters = old_to_new[raters]
                if bool((raters < 0).any()):
                    raise RuntimeError('SparseVLM removed a text rater')
                changed = True
            if changed:
                events.append(dict(after_layer=cut, visual_before=before, visual_after=len(visual), position_sources=sources.tolist()))
                embeddings = text_model.rotary_emb(hidden, position_ids)
        self._captured = {}
        self._linear_capture = {}
        hidden = text_model.norm(hidden)
        if prefill:
            original_visual = self.context['visual'].tolist()
            visual_set = set(original_visual)
            original_text = [p for p in range(self.context['original_length']) if p not in visual_set]
            visual_mask = torch.zeros(len(sources), dtype=torch.bool, device=sources.device)
            visual_mask[visual] = True
            retained_text = sources[~visual_mask].tolist()
            if retained_text != original_text or any(p not in visual_set for p in sources[visual].tolist()):
                raise RuntimeError('pruning changed text order or mixed visual/text anchors')
            avg = sum(counts) / self.num_layers
            self.last_record = dict(
                method=self.method, target_tokens=self.target_tokens,
                alpha_num=self.alpha_num, alpha_den=self.alpha_den,
                score_beta_mode=self.score_beta_mode, score_query_scope=self.score_query_scope,
                stage1_visual_tokens=n1, stage2_visual_tokens=n2,
                two_stage_eligible=(self.method == 'mico' and n0 > n1 > n2),
                actual_prune_stages=sum(e['visual_after'] < e['visual_before'] for e in events),
                record_s2_hidden_digest=self.record_s2_hidden_digest,
                s2_pre_hidden=self._s2_pre_hidden,
                cut_after_layer=self.cut,
                score_source_layer=self.score_source_layer, drop_after_layer=self.drop_after_layer,
                score_source_type=self.score_source_type, effective_beta_mode=self.effective_beta_mode,
                score_capture=self._score_evidence,
                quality_mode=self.quality_mode, gate_capture=self._gate_evidence,
                effective_target_tokens=self.context['target'],
                original_visual_tokens=n0, original_visual_positions=original_visual,
                original_text_tokens=len(original_text), original_prompt_tokens=self.context['original_length'],
                num_layers=self.num_layers, layer_visual_counts=counts,
                actual_layer_average_tokens=avg, actual_retention_ratio=avg / n0,
                budget_rounding_error_tokens=avg - self.context['target'],
                final_visual_tokens=len(visual), text_tokens_preserved=True, decode_steps=0,
                stage_events=events, final_position_sources=sources.tolist(),
                sparse_prune_layers=list(self.sparse_cuts) if self.method == 'sparsevlm' else [],
                apet_seed=0 if self.method == 'apet' else None,
            )
            if self._requires_query_binding:
                self.last_record.update(native_hybrid_capture=self._score_evidence,
                    linear_beta_capture=self._beta_evidence,
                    score_query_binding=dict(self._score_query_evidence),
                    question_binding=dict(self._question_evidence))
        else:
            self.last_record['decode_steps'] += 1
        self.last_record['full_attention_cache_lengths'] = {
            str(i): past_key_values.get_seq_length(i) for i in past_key_values.transformer_layers}
        if prefill and self.record_callback is not None:
            self.record_callback(self.last_record)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


def install_pruning(model, method, target_tokens=None, record_callback=None, *, cut_after_layer=4, quality_mode='qk',
                    alpha_num=2, alpha_den=1, score_beta_mode=None, score_query_scope=None,
                    record_s2_hidden_digest=False):
    return PruningController(model, method, target_tokens, record_callback, cut_after_layer=cut_after_layer,
        quality_mode=quality_mode, alpha_num=alpha_num, alpha_den=alpha_den,
        score_beta_mode=score_beta_mode, score_query_scope=score_query_scope,
        record_s2_hidden_digest=record_s2_hidden_digest)
