import math
import types
from typing import List, Optional, Tuple, Union
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_flash_attention_utils import is_flash_attn_available
from transformers.utils import logging
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_flashatt,
    apply_rotary_pos_emb_vision,
)

if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
logger = logging.get_logger(__name__)


def _is_mico(method: str) -> bool:
    prefix = "mico_"
    if not isinstance(method, str) or not method.startswith(prefix):
        return False
    body = method[len(prefix) :]
    if "_s1m" in body:
        body = body.rsplit("_s1m", 1)[0]
    return body.isdigit()


def _mico_s1_mult(method: str) -> int:
    """S1 budget multiple of T, resolved method-name-first.

    `<...>_llm_<K>_s1m<M>` pins it per arm (so one run's arm cannot leak into
    the next); otherwise fall back to the MICO_STAGE1_MULT env knob. Default 2
    preserves the published `mico_K` behaviour exactly.
    """
    from vlmeval.vlm.mico_coverage_select import mico_s1_mult as _env_mult

    if isinstance(method, str):
        head, _, tail = method.rpartition("_s1m")
        if head and tail.isdigit() and (int(tail) >= 1):
            return int(tail)
    return _env_mult()


def _is_mico_s1(method: str) -> bool:
    return method == "mico_s1" or _is_mico(method)


try:
    from vlmeval.utils.memory_debug import (
        log_memory_stats,
        validate_attention_memory,
        log_tensor_info,
    )

    MEMORY_DEBUG_ENABLED = True
except ImportError:
    MEMORY_DEBUG_ENABLED = False
    print("Warning: Memory debugging utilities not available")


class Qwen2_5_VLForConditionalGenerationPruning(Qwen2_5_VLForConditionalGeneration):
    _supports_flash_attn_2 = False

    def __init__(self, config, pruning_method="vanilla", visual_token_num=-1):
        super().__init__(config)
        if _is_mico(pruning_method):
            if visual_token_num <= 0:
                raise ValueError("MiCo requires a positive token budget")
            from .modeling_qwen2_5_vl_mico import MiCoQwen2_5_VLModel
            from vlmeval.vlm.mico_coverage_select import mico_s2_keep

            n_layers = (
                getattr(config, "num_hidden_layers", None) or config.text_config.num_hidden_layers
            )
            k_layer = int(pruning_method.rsplit("_s1m", 1)[0].rsplit("_", 1)[1])
            if not 1 <= k_layer < n_layers:
                raise ValueError("MiCo decoder cut must be inside the language model")
            multiplier = _mico_s1_mult(pruning_method)
            schedule = {k_layer: mico_s2_keep(int(visual_token_num), n_layers, k_layer, multiplier)}
            self.model = MiCoQwen2_5_VLModel(
                config,
                int(visual_token_num),
                pruning_layers=schedule,
                mico=True,
                dynamic_r=True,
                s1_mult=multiplier,
            )
        else:
            self.model = Qwen2_5_VLModel(config)
        self.pruning_method = pruning_method
        self.target_visual_token_num = visual_token_num
        self.visual_token_num = visual_token_num

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        stage1_metadata = {}
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):

                    def forward_with_attn(
                        self,
                        hidden_states: torch.Tensor,
                        cu_seqlens: torch.Tensor,
                        rotary_pos_emb: Optional[torch.Tensor] = None,
                        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    ) -> torch.Tensor:
                        if MEMORY_DEBUG_ENABLED:
                            log_memory_stats("Attention Forward: Start", verbose=True)
                            log_tensor_info(hidden_states, "Input hidden_states")
                        seq_length = hidden_states.shape[0]
                        print(
                            f"[DEBUG][MiCo][forward_with_attn] hidden_states: {hidden_states.shape}, dtype={hidden_states.dtype}"
                        )
                        q, k, v = (
                            self.qkv(hidden_states)
                            .reshape(seq_length, 3, self.num_heads, -1)
                            .permute(1, 0, 2, 3)
                            .unbind(0)
                        )
                        print(
                            f"[DEBUG][MiCo][forward_with_attn] q: {q.shape}, k: {k.shape}, v: {v.shape}"
                        )
                        if MEMORY_DEBUG_ENABLED:
                            log_memory_stats(
                                "Attention Forward: After QKV projection", verbose=True
                            )
                            print(f"  Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
                            print(f"  Dtype: {q.dtype}")
                        if position_embeddings is None:
                            logger.warning_once(
                                "The attention layers in this model are transitioning from computing the RoPE embeddings internally through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be removed and `position_embeddings` will be mandatory."
                            )
                            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                            cos = emb.cos()
                            sin = emb.sin()
                        else:
                            cos, sin = position_embeddings
                        self.q, self.k = (q.clone(), k.clone())
                        q, k = apply_rotary_pos_emb_flashatt(
                            q.unsqueeze(0), k.unsqueeze(0), cos, sin
                        )
                        q = q.squeeze(0)
                        k = k.squeeze(0)
                        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                        attn_output = flash_attn_varlen_func(
                            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
                        ).reshape(seq_length, -1)
                        attn_output = self.proj(attn_output)
                        if MEMORY_DEBUG_ENABLED:
                            log_memory_stats(
                                "Attention Forward: Before standard attention weights", verbose=True
                            )
                        q, k = apply_rotary_pos_emb_vision(self.q, self.k, cos, sin)
                        q = q.transpose(0, 1)
                        k = k.transpose(0, 1)
                        print(
                            f"[DEBUG][MiCo][forward_with_attn] q after RoPE+transpose: {q.shape}, k: {k.shape}"
                        )
                        seq_len = q.shape[1]
                        if MEMORY_DEBUG_ENABLED:
                            print(f"\n!!! Computing attention weights for pruning !!!")
                            print(f"  Q shape after transpose: {q.shape}")
                            print(f"  K shape after transpose: {k.shape}")
                            print(f"  Sequence length: {seq_len}")
                            estimated_gb = validate_attention_memory(
                                q.shape, k.shape, dtype=q.dtype, location="VisionZip Attention"
                            )
                        SINGLE_IMAGE_TOKENS = 1296
                        MAX_SEQ_LEN_FOR_FULL_ATTENTION = 5000
                        _seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                        real_num_images = len(_seg_lens)
                        estimated_num_images = real_num_images
                        tokens_per_image = seq_len / max(real_num_images, 1)
                        is_likely_multi_image = real_num_images > 1
                        if is_likely_multi_image:
                            print(
                                f"\n📸 Detected multi-image sequence: {real_num_images} images (cu_seqlens lens={_seg_lens})"
                            )
                            print(f"  Sequence length: {seq_len} tokens")
                            print(f"  Estimated tokens per image: {tokens_per_image:.1f}")
                        if seq_len > MAX_SEQ_LEN_FOR_FULL_ATTENTION:
                            print(
                                f"\n⚠️  WARNING: Sequence length {seq_len} exceeds maximum {MAX_SEQ_LEN_FOR_FULL_ATTENTION}"
                            )
                            if is_likely_multi_image:
                                print(
                                    f"  Processing as {real_num_images} separate images to avoid OOM"
                                )
                                all_attn_weights = []
                                offset = 0
                                for seg in _seg_lens:
                                    start = offset
                                    end = offset + int(seg)
                                    offset = end
                                    if end > start:
                                        q_img = q[:, start:end, :]
                                        k_img = k[:, start:end, :]
                                        attn_img = torch.matmul(
                                            q_img, k_img.transpose(1, 2)
                                        ) / math.sqrt(q.shape[-1])
                                        attn_img = torch.nn.functional.softmax(attn_img, dim=-1)
                                        all_attn_weights.append(attn_img)
                                print(
                                    f"  Creating compressed attention representation for {len(all_attn_weights)} images"
                                )
                                num_heads = all_attn_weights[0].shape[0]
                                max_dim = max((w.shape[-1] for w in all_attn_weights))
                                avg_attn = torch.zeros(
                                    num_heads, max_dim, max_dim, device=q.device, dtype=q.dtype
                                )
                                for attn_block in all_attn_weights:
                                    block_size = attn_block.shape[-1]
                                    avg_attn[:, :block_size, :block_size] += attn_block
                                avg_attn = avg_attn / len(all_attn_weights)
                                self.attn_weights = avg_attn
                                self.attn_blocks = all_attn_weights
                                self.num_images = estimated_num_images
                                print(
                                    f"  Compressed attention: {avg_attn.shape} (from {seq_len}×{seq_len})"
                                )
                                print(
                                    f"  Memory saved: ~{(seq_len * seq_len - max_dim * max_dim) * 2 / 1024 ** 3:.2f} GB"
                                )
                            else:
                                print(
                                    f"  Using lightweight attention probe for non-standard sequence"
                                )
                                q_probe = q.mean(dim=1, keepdim=True)
                                attn_scores = torch.matmul(q_probe, k.transpose(1, 2)) / math.sqrt(
                                    q.shape[-1]
                                )
                                attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
                                self.attn_weights = attn_weights.expand(-1, seq_len, -1)
                            if MEMORY_DEBUG_ENABLED:
                                log_memory_stats(
                                    "After multi-image attention computation", verbose=True
                                )
                        else:
                            if MEMORY_DEBUG_ENABLED:
                                print(f"  Attempting to allocate full attention matrix...")
                            try:
                                attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
                                    q.shape[-1]
                                )
                                if MEMORY_DEBUG_ENABLED:
                                    log_memory_stats(
                                        "Attention Forward: After attention matmul", verbose=True
                                    )
                                    log_tensor_info(attn_weights, "Attention weights matrix")
                                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                                self.attn_weights = attn_weights
                                print(
                                    f"[DEBUG][MiCo][forward_with_attn] attn_weights (standard): {attn_weights.shape}"
                                )
                            except torch.cuda.OutOfMemoryError as e:
                                if MEMORY_DEBUG_ENABLED:
                                    print(f"\n{'!' * 70}")
                                    print(f"!!! OOM at attention weight computation !!!")
                                    print(f"{'!' * 70}")
                                    print(f"  Tried to compute: {q.shape} @ {k.shape}.T")
                                    print(
                                        f"  Expected shape: [{q.shape[0]}, {q.shape[1]}, {k.shape[1]}]"
                                    )
                                    log_memory_stats("OOM Details", show_tensors=True, verbose=True)
                                raise e
                        return attn_output

                    self.visual.blocks[-1].attn.forward = types.MethodType(
                        forward_with_attn, self.visual.blocks[-1].attn
                    )
                if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6") and (
                    not getattr(self, "_mico_merger_hook_set", False)
                ):

                    def _cap_merger_in(mod, inp):
                        self._mico_merger_input = inp[0]

                    self.visual.merger.register_forward_pre_hook(_cap_merger_in)
                    self._mico_merger_hook_set = True
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):
                    attn_last = self.visual.blocks[-1].attn
                    if hasattr(attn_last, "num_images") and attn_last.num_images > 1:
                        all_patch_scores = []
                        for attn_block in attn_last.attn_blocks:
                            all_patch_scores.append(attn_block.mean(dim=0).mean(dim=0))
                        all_patch_scores = torch.cat(all_patch_scores)
                        print(f"[DEBUG][MiCo][vision] all_patch_scores: {all_patch_scores.shape}")
                        token_scores = []
                        offset = 0
                        for thw in image_grid_thw:
                            _, H, W = (thw[0].item(), thw[1].item(), thw[2].item())
                            t = H * W
                            img_scores = all_patch_scores[offset : offset + t]
                            img_token_scores = img_scores.view(-1, 4).mean(dim=1)
                            win_idx, _ = self.visual.get_window_index(thw.unsqueeze(0))
                            rev_idx = torch.argsort(win_idx)
                            img_token_scores = img_token_scores[rev_idx]
                            print(
                                f"[DEBUG][MiCo][vision] image H={H} W={W} patches={t} tokens={img_token_scores.shape[0]}"
                            )
                            token_scores.append(img_token_scores)
                            offset += t
                        image_attns = torch.cat(token_scores)
                        print(
                            f"[DEBUG][MiCo][vision] image_attns (multi-image): {image_attns.shape}, N={image_embeds.shape[0]}"
                        )
                    else:
                        seq_length = pixel_values.shape[0]
                        attn_weights = attn_last.attn_weights
                        print(
                            f"[DEBUG][MiCo][vision] attn_weights (single-image): {attn_weights.shape}, seq_length={seq_length}"
                        )
                        window_index, _ = self.visual.get_window_index(image_grid_thw)
                        reverse_indices = torch.argsort(window_index)
                        image_attns = attn_weights.view(
                            -1, seq_length // 4, 4, seq_length // 4, 4
                        ).mean(dim=(2, 4))
                        image_attns = image_attns[:, reverse_indices, :][:, :, reverse_indices]
                        print(
                            f"[DEBUG][MiCo][vision] image_attns (single-image, after window restore): {image_attns.shape}"
                        )
                N, D = image_embeds.shape
                device = image_embeds.device
                selected_masks = torch.ones(N, dtype=torch.bool, device=device)
                if self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6"):
                    from vlmeval.vlm.mico_coverage_select import inner_weighted_raw_cls_select

                    keep_2t = self.pruning_method in ("mico_2", "mico_4", "mico_6")
                    use_vit = False
                    _s1m = int(getattr(self.model, "s1_mult", 2) or 2) if keep_2t else 1
                    stage1_keep_num = min(max(1, self.target_visual_token_num * _s1m), N)
                    if image_attns is not None:
                        if image_attns.dim() == 3:
                            cls_attn = image_attns.mean(dim=1).mean(dim=0)
                        elif image_attns.dim() == 2:
                            cls_attn = image_attns.mean(dim=0)
                        else:
                            cls_attn = image_attns
                    else:
                        cls_attn = image_embeds.norm(dim=-1)
                    if use_vit:
                        vit_feat = getattr(self, "_mico_merger_input", None)
                        if vit_feat is not None and vit_feat.shape[0] // 4 == N:
                            _wi, _ = self.visual.get_window_index(image_grid_thw)
                            _rev = torch.argsort(_wi)
                            cov_feat = (
                                vit_feat.float()
                                .reshape(vit_feat.shape[0] // 4, 4, -1)
                                .mean(dim=1)[_rev, :]
                            )
                            _covsrc = "vit"
                        else:
                            cov_feat = image_embeds.float()
                            _covsrc = "proj(fallback)"
                    else:
                        cov_feat = image_embeds.float()
                        _covsrc = "proj"
                    mico = self.pruning_method in ("mico_s1", "mico_2", "mico_4", "mico_6")
                    selected_masks = inner_weighted_raw_cls_select(
                        cov_feat, cls_attn, stage1_keep_num, mico=mico
                    )
                    s2 = f"S2={self.model.pruning_layers}, avg T" if keep_2t else "no LLM prune"
                    print(
                        f"[{self.pruning_method}] S1 raw_visual_attn{'*|h|'}*coverage cov_space={_covsrc}: N={N} → keep {int(selected_masks.sum().item())} ({(f'{_s1m}T' if keep_2t else 'T')}={stage1_keep_num}, {s2})"
                    )
                self.visual_token_len = selected_masks.sum().item()
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                mask = input_ids == self.config.image_token_id
                if self.pruning_method in ("mico_2", "mico_4", "mico_6"):
                    if (
                        mask.shape[0] != 1
                        or selected_masks.ndim != 1
                        or selected_masks.dtype != torch.bool
                    ):
                        raise ValueError(
                            "mico_coverage LLM pruning requires one image-token sequence and a 1D bool S1 mask"
                        )
                    image_positions = mask[0].nonzero().squeeze(-1)
                    if (
                        image_positions.numel() == 0
                        or selected_masks.numel() != image_positions.numel()
                    ):
                        raise ValueError("S1 mask must align with the actual image-token span")
                    visual_start = int(image_positions[0].item())
                    if not torch.equal(
                        image_positions,
                        torch.arange(
                            visual_start,
                            visual_start + image_positions.numel(),
                            device=image_positions.device,
                        ),
                    ):
                        raise ValueError(
                            "mico_coverage LLM pruning requires a contiguous image-token span"
                        )
                    stage1_metadata["stage1_visual_span"] = (
                        visual_start,
                        int(selected_masks.sum().item()),
                    )
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                cache_position is not None
                and cache_position[0] == 0
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                self._orig_prompt_len = int(inputs_embeds.shape[1])
                self.retain_token_indices = None
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask
                )
                self.rope_deltas = rope_deltas
                if selected_masks.sum() < selected_masks.numel():
                    system_prompt_length = int(mask.max(dim=-1).indices)
                    visual_token_length = selected_masks.shape[0]
                    visual_token_indices = selected_masks.nonzero().squeeze(-1)
                    retain_token_indices = torch.cat(
                        (
                            torch.arange(system_prompt_length, device=device),
                            visual_token_indices + system_prompt_length,
                            torch.arange(
                                system_prompt_length + visual_token_length,
                                inputs_embeds.shape[1],
                                device=device,
                            ),
                        )
                    )
                    print(
                        f"[MiCo-Variant3 Stage1] retain_token_indices length = {retain_token_indices.shape[0]}"
                    )
                    self.retain_token_indices = retain_token_indices
                    inputs_embeds = inputs_embeds[:, retain_token_indices]
                    position_ids = position_ids[:, :, retain_token_indices]
                    attention_mask = attention_mask[:, retain_token_indices]
                    cache_position = cache_position[: retain_token_indices.shape[0]]
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                retain_token_indices = getattr(self, "retain_token_indices", None)
                if attention_mask is not None and retain_token_indices is not None:
                    tail_indices = torch.arange(
                        self._orig_prompt_len,
                        attention_mask.shape[1],
                        device=attention_mask.device,
                        dtype=retain_token_indices.dtype,
                    )
                    attention_mask = attention_mask[
                        :, torch.cat((retain_token_indices.to(attention_mask.device), tail_indices))
                    ]
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **stage1_metadata,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
