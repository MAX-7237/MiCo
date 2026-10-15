# coding=utf-8
"""ApET (two-stage FPS-basis reconstruction-residual pruning) for Qwen3-VL.

Ported from the curated single-image reference
    /path/to/local-assets
(Maqkccx/ApET @70cc01b). This Qwen3 adaptation shares a rank-order, FP32 merge helper
between S1 and S2. Unlike the reference, S1 receives post-merger image embeddings,
does not assign basis slots in Boolean-index order, and does not merge in FP16.
S2 also omits the reference's final FP16 round-trip and passes FP32 features to FPS.
These existing choices are preserved when resuming this suite's partial results.
The decoder integration uses the Qwen3-VL FastV/SparseVLM skeleton:

  * P2 -- recompute RoPE from the pruned position_ids (Qwen3-VL cos/sin are [B,S,head_dim]).
  * P3 -- rebuild the post-prune causal mask with attention_mask/past_key_values/position_ids
    dropped together (packed-sequence poisoning guard).
  * P5 -- gather the DeepStack side-features at the same retained original positions.
  * P6 -- explicit set_visual_token_positions() index vector (no fixed offset window).

ApET mechanism in this Qwen3 adaptation:
  S1 (post-merger, in modeling_qwen3_vl.py's selection block): apet_stage1 keeps `stage1 = 3*T//2`
     tokens -- FPS pick `basis` seeds, solve the Gram system, rank by reconstruction residual,
     keep the top-`stage1`, cosine-merge the discarded remainder into the non-basis kept tokens.
     The kept indices drive stage-1 compaction; merged features are cast to image_embeds.dtype.
  S2 (decoder, this module): at layer `prune_after = L//2` (18 for Qwen3-VL's 36 layers),
     apet_stage2 runs the SAME math on the visual hidden states, pruning `stage1 -> stage2 = T//2`.

Budget contract (matrix_contract.py:20-24): for a final per-image budget T,
     stage1 = 3*T//2,  stage2 = T//2,  prune_after = L//2,  basis = 10.
The LAYER-AVERAGE visual token count is 1.5T*(0.5) + 0.5T*(0.5) = T, matching the "Retain T"
row semantics used by VisionZip/HoloV/SparseVLM in this suite.
"""

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask as _create_causal_mask_impl
    _BASE_MODEL_CLS = Qwen3VLModel
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as _BASE_MODEL_CLS
    _create_causal_mask_impl = None

logger = logging.get_logger(__name__)

APET_REV = "qwen3vl-apet-20260916/P2P3P5P6/cosine-tiled"

_FALSEY = ('', '0', 'false', 'False', 'FALSE', 'no', 'off')


def _env_on(name: str, default: str = '0') -> bool:
    return str(os.environ.get(name, default)).strip() not in _FALSEY


APET_BASIS = int(os.environ.get('APET_BASIS', '10'))


# ---------------------------------------------------------------------------------------------
# Budget contract (matrix_contract.py:configuration).  T -> (stage1, stage2, prune_after)
# ---------------------------------------------------------------------------------------------
def apet_budget(target_tokens: int, n_layers: int) -> Tuple[int, int, int]:
    T = max(1, int(target_tokens))
    stage1 = 3 * T // 2
    stage2 = T // 2
    prune_after = n_layers // 2
    return stage1, stage2, prune_after


# ---------------------------------------------------------------------------------------------
# This suite's existing S1 and S2 share the reference S2's rank-order FP32 merge.
# Keep that adaptation unchanged when fixing memory use, so completed scores stay comparable.
# ---------------------------------------------------------------------------------------------
def _fps(x: torch.Tensor, count: int, seed: Optional[int]) -> torch.Tensor:
    """Farthest-point sampling on [B,N,D] -> Long[B,count]. Matches reference apet.py:27-40."""
    b, n, _ = x.shape
    generator = None
    if seed is not None:
        generator = torch.Generator(device=x.device).manual_seed(seed)
    indices = torch.zeros(b, count, dtype=torch.long, device=x.device)
    distance = torch.full((b, n), 1e10, device=x.device)
    farthest = torch.randint(0, n, (b,), device=x.device, generator=generator)
    for i in range(count):
        indices[:, i] = farthest
        centroid = x[torch.arange(b, device=x.device), farthest].unsqueeze(1)
        distance = torch.min(distance, ((x - centroid) ** 2).sum(-1))
        farthest = distance.max(1)[1]
    return indices


def _nearest_cosine_targets(rest: torch.Tensor, targets: torch.Tensor,
                            max_pair_bytes: int = 256 * 1024 * 1024) -> torch.Tensor:
    """Bound each [rest, target, feature] cosine temporary without changing its math."""
    # HallB created a 90.74 GiB broadcast tensor. Slice only rest rows, retaining
    # every target and the original feature reduction/argmax tie order per row.
    # The cap is per broadcast temporary, not total peak memory of cosine_similarity.
    row_bytes = targets.shape[0] * targets.shape[1] * rest.element_size()
    if row_bytes == 0 or max_pair_bytes < row_bytes:
        raise ValueError("cosine workspace must hold at least one row of target pairs")
    row_chunk = max_pair_bytes // row_bytes
    nearest = torch.empty(rest.shape[0], dtype=torch.long, device=rest.device)
    for start in range(0, rest.shape[0], row_chunk):
        similarity = F.cosine_similarity(
            rest[start:start + row_chunk].unsqueeze(1), targets.unsqueeze(0), dim=2)
        nearest[start:start + row_chunk] = similarity.argmax(dim=1)
    return nearest


@torch.no_grad()
def apet_merge(features: torch.Tensor, keep: int, basis: int,
               seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """[1,M,D] float -> (merged FP-work [1,M,D], sorted kept indices [1,keep]).

    This suite's FP32 merge, based on reference apet_stage2, applied to a [1,M,D] span:
      FPS `basis` seeds -> solve Gram(seeds) x = seeds@work^T -> reconstruction residual norm
      -> rank top-`keep` -> write seeds into the basis slots -> cosine-merge the NON-selected
      remainder into the non-basis kept targets (FP32 merge, count-normalised).
    Returns the full-length work tensor (only the kept rows matter) plus the sorted kept indices;
    the caller gathers work[kept] to obtain the [keep,D] compact block.
    """
    if features.dim() != 3 or features.shape[0] != 1:
        raise ValueError(f"[ApET] apet_merge needs [1,M,D], got {tuple(features.shape)}")
    _, count, dim = features.shape
    keep = max(1, min(int(keep), count))
    basis = max(1, min(int(basis), keep - 1)) if keep > 1 else 1
    if not (0 < basis < keep <= count):
        # degenerate span (tiny image): identity, keep everything in order.
        return features.float().clone(), torch.arange(count, device=features.device)[None]

    fps_indices = _fps(features, basis, seed)
    seeds = features.gather(1, fps_indices.unsqueeze(-1).expand(-1, -1, dim))[0].float()  # [basis,D]
    work = features[0].float().clone()                                                    # [M,D]
    gram = torch.matmul(seeds, seeds.t())
    gram.diagonal().add_(1e-5)
    coefficients = torch.linalg.solve(gram, torch.matmul(seeds, work.t())).t()
    rank = torch.topk(torch.norm(work - torch.matmul(coefficients, seeds), dim=-1), k=keep)[1]
    work[rank[-basis:]] = seeds
    selected = torch.zeros(count, dtype=torch.bool, device=work.device).scatter_(0, rank, True)
    targets = work[rank[:-basis]]
    rest = work[~selected]
    if rest.shape[0]:
        nearest = _nearest_cosine_targets(rest, targets)
        counts = torch.zeros(keep - basis, device=work.device, dtype=torch.int)
        merged = torch.zeros_like(targets)
        merged += targets
        merged.scatter_add_(0, nearest.unsqueeze(-1).expand(-1, dim), rest)
        counts.scatter_add_(0, nearest, torch.ones_like(nearest, dtype=counts.dtype))
        merged /= 1 + counts.unsqueeze(1)
        work[rank[:-basis]] = merged
    indices = torch.sort(rank)[0]
    return work.unsqueeze(0), indices[None]


@torch.no_grad()
def apet_stage1(features: torch.Tensor, keep: int, basis: int = 10,
                seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Post-merger S1: [N,D] -> (merged [N,D] in ORIGINAL dtype, kept idx [keep]).

    Called from modeling_qwen3_vl.py's selection block. The merged features overwrite image_embeds
    at the kept positions; the returned indices become the stage-1 keep mask.
    """
    if features.dim() != 2:
        raise ValueError(f"[ApET] apet_stage1 needs [N,D], got {tuple(features.shape)}")
    work, idx = apet_merge(features.unsqueeze(0), keep, basis, seed)
    return work[0].to(features.dtype), idx[0]


def _past_length(cache) -> int:
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except Exception:
        return 0


class ApETQwen3VLModel(_BASE_MODEL_CLS):
    """Qwen3 ApET adaptation: post-merger S1, single-layer decoder S2."""

    def __init__(self, config, visual_token_num: int):
        super().__init__(config)
        lm = self.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.rotary_emb = lm.rotary_emb
        self._lm_update_causal_mask = getattr(lm, '_update_causal_mask', None)

        # P6: explicit visual-position window.
        self.visual_token_positions: Optional[torch.LongTensor] = None
        self._visual_positions_fresh = False

        try:
            target = int(visual_token_num)
        except (TypeError, ValueError):
            raise ValueError(f"visual_token_num must be an integer, got {visual_token_num!r}")
        if target <= 0:
            raise ValueError(
                f"ApET needs a positive visual_token_num (got {visual_token_num!r}); "
                "-1 means 'no budget' and the S1/S2 budget split would be degenerate."
            )
        self.target = target
        self.basis = APET_BASIS
        self.seed = None if os.environ.get('APET_SEED') is None else int(os.environ['APET_SEED'])

        self.n_layers = len(self.layers)
        self.stage1, self.stage2, self.prune_after = apet_budget(self.target, self.n_layers)
        # prune AFTER completing layer `prune_after` (1-based) == before 1-based layer prune_after+1.
        self.prune_at_layer = self.prune_after  # 1-based index of the layer whose OUTPUT triggers prune
        self.debug = _env_on('ENABLE_DEBUG', '0')

        print(f"[ApET Init] rev={APET_REV} Qwen3-VL L={self.n_layers} target T={self.target} "
              f"stage1(S1 post-merger)={self.stage1} stage2(S2 dec)={self.stage2} "
              f"prune_after_layer={self.prune_after} basis={self.basis} seed={self.seed}")
        self.reset_state()

    def reset_state(self):
        self.prefill_done = False
        self.original_visual_length: Optional[int] = None
        self.current_visual_length: Optional[int] = None
        self.retain_token_indices: Optional[torch.LongTensor] = None
        self.visual_token_num = 0.0
        self.visual_token_len = 0.0

    def set_visual_token_positions(self, positions: Optional[torch.LongTensor]) -> None:
        if positions is None:
            self.visual_token_positions = None
            self._visual_positions_fresh = False
            return
        self.visual_token_positions = positions.detach().reshape(-1).to(torch.long)
        self._visual_positions_fresh = True

    def _get_causal_mask(self, input_tensor, attention_mask, cache_position, past_key_values, position_ids=None):
        if position_ids is not None and attention_mask is None and past_key_values is None:
            position_ids = None
        if _create_causal_mask_impl is not None:
            return _create_causal_mask_impl(
                self.config, input_tensor, attention_mask, cache_position,
                past_key_values=past_key_values, position_ids=position_ids
            )
        elif self._lm_update_causal_mask is not None:
            return self._lm_update_causal_mask(
                attention_mask, input_tensor, cache_position, past_key_values, False
            )
        return None

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
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        past_seen_tokens = _past_length(past_key_values)
        if past_seen_tokens == 0:
            self.reset_state()

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else getattr(self.config, 'output_hidden_states', False)
        )
        use_cache = use_cache if use_cache is not None else getattr(self.config, 'use_cache', True)
        return_dict = return_dict if return_dict is not None else getattr(self.config, 'use_return_dict', True)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if self.is_gradient_checkpointing and self.training and use_cache:
            use_cache = False
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        seq_length = inputs_embeds.shape[1]
        is_prefill = (past_seen_tokens == 0) and (seq_length > 1)

        # Qwen3-VL 4D position_ids: [4,B,S] (text + mrope T/H/W).
        text_position_ids = None
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.dim() == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]

        causal_mask = self._get_causal_mask(
            inputs_embeds, attention_mask, cache_position, past_key_values, position_ids=text_position_ids
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None

        # Resolve this sample's visual window (post-S1 positions pushed from the main file).
        visual_positions: Optional[torch.LongTensor] = None
        n_visual = 0
        do_prune = False
        if is_prefill:
            self.prefill_done = True
            pushed = self.visual_token_positions if self._visual_positions_fresh else None
            self._visual_positions_fresh = False
            if pushed is not None:
                pushed = pushed.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
                if pushed.numel() and int(pushed.max()) >= seq_length:
                    raise RuntimeError(
                        f"[ApET] stale visual_token_positions: max {int(pushed.max())} >= seq_len {seq_length}"
                    )
                visual_positions = pushed
                n_visual = int(pushed.numel())
            if n_visual > 0:
                self.original_visual_length = n_visual
                self.current_visual_length = n_visual
                # S2 prunes stage1 -> stage2. Only meaningful if the window is larger than stage2.
                do_prune = n_visual > self.stage2
                if self.debug:
                    print(f"\n[ApET Qwen3-VL] rev={APET_REV} prefill seq={seq_length} visual N={n_visual} "
                          f"stage2_target={self.stage2} prune_after={self.prune_after} "
                          f"prune={'yes' if do_prune else 'no'} "
                          f"deepstack={0 if deepstack_visual_embeds is None else len(deepstack_visual_embeds)}",
                          flush=True)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            layer_idx = decoder_layer.self_attn.layer_idx + 1  # 1-based; this layer just ran

            # ── P5a: DeepStack injection after text layers 0/1/2 (upstream UP:929) ──
            _ds = layer_idx - 1
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and 0 <= _ds < len(deepstack_visual_embeds)
            ):
                assert deepstack_visual_embeds[_ds].shape[0] == int(visual_pos_masks.sum()), \
                    "deepstack rows out of sync with visual_pos_masks"
                hidden_states = self.language_model._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[_ds]
                )

            # ── S2: ApET prune once, right AFTER completing layer `prune_after` ──
            if (
                is_prefill and do_prune and visual_positions is not None
                and layer_idx == self.prune_at_layer
            ):
                device = hidden_states.device
                orig_seq_len = hidden_states.shape[1]
                pre_vis_positions = visual_positions
                n_pre_vis = int(pre_vis_positions.numel())

                keep = max(1, min(self.stage2, n_pre_vis))
                # Run ApET math on the CURRENT visual hidden states.
                visual_hs = hidden_states[0].index_select(0, pre_vis_positions).unsqueeze(0)  # [1,Nv,D]
                work, kept_local = apet_merge(visual_hs.float(), keep, self.basis, self.seed)
                kept_local = kept_local[0]                      # [keep] indices into pre_vis_positions
                merged_visual = work[0].index_select(0, kept_local).to(hidden_states.dtype)  # [keep,D]

                # Overwrite the kept visual rows in-place with their merged values, then compact:
                # keep only [non-visual tokens] + [kept visual rows]. ApET merges INTO kept rows
                # (no appended block), so there is no separate merged-tail to place.
                kept_abs = pre_vis_positions.index_select(0, kept_local.sort().values)
                # apply merged values at kept_abs (align to sorted order)
                sort_perm = kept_local.sort().indices
                hidden_states = hidden_states.clone()
                hidden_states[0, kept_abs] = merged_visual.index_select(0, sort_perm)

                retain_mask = torch.ones(orig_seq_len, dtype=torch.bool, device=device)
                retain_mask[pre_vis_positions] = False
                retain_mask[kept_abs] = True
                retain_token_indices = retain_mask.nonzero(as_tuple=False).squeeze(-1)
                hidden_states = hidden_states.index_select(1, retain_token_indices).contiguous()
                new_len = hidden_states.shape[1]

                visual_positions = torch.searchsorted(retain_token_indices, kept_abs)

                # Position bookkeeping (P2): gather kept positions, recompute RoPE.
                position_ids = position_ids[:, :, retain_mask]
                if text_position_ids is not None:
                    text_position_ids = text_position_ids[:, retain_mask]
                cache_position = cache_position[:new_len]
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                # P3: rebuild causal mask with everything-None trigger.
                causal_mask = self._get_causal_mask(hidden_states, None, cache_position, None, position_ids=None)

                # P5b: prune DeepStack rows 1:1 with kept originals (no merged-tail here).
                if deepstack_visual_embeds is not None:
                    _pre_keep = torch.zeros(n_pre_vis, dtype=torch.bool, device=device)
                    _pre_keep[torch.searchsorted(pre_vis_positions, kept_abs)] = True
                    deepstack_visual_embeds = [d[_pre_keep] for d in deepstack_visual_embeds]
                if visual_pos_masks is not None:
                    _new_mask = torch.zeros(
                        visual_pos_masks.shape[0], new_len,
                        dtype=visual_pos_masks.dtype, device=visual_pos_masks.device
                    )
                    _new_mask[:, visual_positions] = True
                    visual_pos_masks = _new_mask

                self.visual_token_positions = visual_positions
                self.retain_token_indices = retain_token_indices
                self.current_visual_length = int(visual_positions.numel())
                if self.debug:
                    print(f"[ApET PRUNE] after layer {layer_idx}: visual {n_pre_vis} -> "
                          f"{self.current_visual_length} (S2 keep={keep}) seq {orig_seq_len} -> {new_len}",
                          flush=True)

        hidden_states = self.norm(hidden_states)

        if is_prefill and n_visual > 0:
            # LAYER-AVERAGE visual tokens, matching the suite's schedule-average accounting:
            # stage1-window for layers <= prune_after, stage2 after.
            pre = self.original_visual_length
            post = self.current_visual_length or pre
            total = pre * self.prune_after + post * (self.n_layers - self.prune_after)
            self.visual_token_num = total / self.n_layers
            self.visual_token_len = float(post)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
        )
