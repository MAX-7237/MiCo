import argparse
import torch
import os
import json
import time
import re
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Suppress warnings if ENABLE_DEBUG is not set
    if os.environ.get('ENABLE_DEBUG', '0') != '1':
        import warnings
        warnings.filterwarnings('ignore')

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # This family keeps 2T in CLIP space, then uses exactly one dynamic S2 cut.
    # Parse K from the method name so the evaluation entry point and the model
    # cannot silently disagree about the measured LLM prune locus.
    hvit_llm_match = re.fullmatch(r"mico_([1-9][0-9]*)", args.pruning_method)

    # ========== Method Configuration ==========
    use_fastv = True if args.pruning_method == "fastv" else False
    fastv_config = {"K": 2, "T": args.visual_token_num}

    # Attn-Cov-LLM family at K=2 uses T as layer-average (FastV-matched), not
    # post-prune keep. R = (T*L - K*N)/(L-K); 7B 128/64 -> R≈98/30. Do not pass
    # R=T — that was the old exact-keep path and is not comparable to FastV.
    # merge / afterlayer012 / proj already used T_avg. NeXT scripts pass 5x.
    use_mico_coverage_llm = args.pruning_method in (
        "mico_coverage_llm", "merge_mico_coverage_llm", "mico_coverage_llm_2", "mico_coverage_llm_26",
        "mico_coverage_llm_afterlayer012", "mico_coverage_proj_llm", "mico_coverage_proj_llm_2", "mico_coverage_proj_llm_12_24", "mico_coverage_proj_llm_12", "mico_coverage_vit_proj_llm_12", "cov_proj_llm_12_24", "mico_projected_llm_12_24", "mico_coverage_proj_llm_4regions_sg", "mico_coverage_proj_llm_4regions_entropy_sg", "last_mico_coverage_proj_llm_2", "cov_llm", "mico_weighted_llm",
        "sim_cov_llm", "anti_sim_cov_llm",
    ) or hvit_llm_match is not None
    if args.pruning_method == "merge_mico_coverage_llm":
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
        }
    elif hvit_llm_match is not None:
        mico_coverage_llm_config = {
            "K": int(hvit_llm_match.group(1)),
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
            "mico": True,
        }
    elif args.pruning_method in ("mico_coverage_llm_2", "mico_coverage_llm_26"):
        mico_coverage_llm_config = {
            "K": 2,
            "K2": 6,
            "prune_layers": [2, 6],
            "adaptive_r2": True,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
        }
    elif args.pruning_method in ("mico_coverage_proj_llm_2", "mico_coverage_proj_llm_4regions_sg", "mico_coverage_proj_llm_4regions_entropy_sg"):
        # proj keep 2T or entropy-gated N1 in [1.5T, 2T], then FastV-locus K=2.
        # T is layer-avg via dynamic_R. S2 is text_attn×coverage (all text that
        # can see visuals under the causal mask).
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
        }
    elif args.pruning_method in ("mico_coverage_proj_llm_12", "mico_coverage_vit_proj_llm_12"):
        # Same S2 as proj_llm_12. prefix2_first: 7B L12 / 13B L15.
        # Keep is FastV-style R=(T*L-K*2T)/(L-K), not the prefix2 keep table.
        # vit_proj S1 is CLIP-space; proj S1 is post-mm_projector.
        mico_coverage_llm_config = {
            "K": 12,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
            "prefix2_first": True,
        }
    elif args.pruning_method == "mico_coverage_proj_llm_12_24":
        # Same S1 (CLS×coverage 2T) and S2 select as proj_2.
        # prefix2_keep overrides loci/keep to tops: 7B L12/24, 13B L15/30.
        mico_coverage_llm_config = {
            "K": 12,
            "K2": 24,
            "prune_layers": [12, 24],
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": False,
            "prefix2_keep": True,
        }
    elif args.pruning_method == "cov_proj_llm_12_24":
        # Same 2T + L12/L24 keep table; w=1 so both stages are coverage-only.
        mico_coverage_llm_config = {
            "K": 12,
            "K2": 24,
            "prune_layers": [12, 24],
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": False,
            "prefix2_keep": True,
            "coverage_only": True,
        }
    elif args.pruning_method == "mico_projected_llm_12_24":
        # Same 2T + L12/L24 keep table; w = attn * |h| at each prune site.
        mico_coverage_llm_config = {
            "K": 12,
            "K2": 24,
            "prune_layers": [12, 24],
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": False,
            "prefix2_keep": True,
            "mico": True,
        }
    elif args.pruning_method == "last_mico_coverage_proj_llm_2":
        # Same S1 (CLS×coverage 2T) and layer-avg K=2 as mico_coverage_proj_llm_2.
        # S2 weights are last-token attention, not mean post-visual text attn.
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
            "last_token": True,
        }
    elif args.pruning_method in ("mico_coverage_llm_afterlayer012", "mico_coverage_proj_llm"):
        # After layers 0/1/2 (1-based K=3). mico_coverage_proj_llm also keeps 2T at projector.
        mico_coverage_llm_config = {
            "K": 3,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
        }
    elif args.pruning_method == "cov_llm":
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
            "coverage_only": True,
        }
    elif args.pruning_method == "mico_weighted_llm":
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
            "mico": True,
        }
    elif args.pruning_method == "sim_cov_llm":
        # K=2; T is layer-average (FastV-matched). w = all-text cosine, times coverage.
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
            "sim_cov": True,
        }
    elif args.pruning_method == "anti_sim_cov_llm":
        # Same as sim_cov_llm but w = (1 - [cos]_+) * coverage.
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
            "anti_sim_cov": True,
        }
    else:
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
        }

    use_sparsevlm = True if args.pruning_method == "sparsevlm" else False
    sparsevlm_config = {"T": args.visual_token_num}

    use_d2p = True if args.pruning_method == "d2p" else False
    d2p_config = {"T": args.visual_token_num}

    use_svdvlm = True if args.pruning_method == "svdvlm" else False
    svdvlm_config = {"T": args.visual_token_num}

    use_tops_early = True if args.pruning_method == "tops_early" else False
    tops_early_config = {"T": args.visual_token_num}

    use_holov2 = True if args.pruning_method == "HoloV_2" else False
    # Official HoloV (pruning_method=HoloV/holov) is vision-side only; keep the standard Llama decoder.
    holov2_config = {"T": args.visual_token_num}

    use_idea = True if args.pruning_method == "Idea" else False
    idea_config = {"T": args.visual_token_num}

    use_tops = True if args.pruning_method in ("tops", "tops_iclr", "cover_prune_2", "cover_prune_2_rawattn", "cover_prune_2_tv") else False
    tops_config = {"T": args.visual_token_num, "disable_s2": args.pruning_method == "tops_iclr", "cover_prune_2": args.pruning_method in ("cover_prune_2", "cover_prune_2_rawattn", "cover_prune_2_tv"), "cover_prune_2_rawattn": args.pruning_method == "cover_prune_2_rawattn", "cover_prune_2_tv": args.pruning_method == "cover_prune_2_tv"}

    use_pdrop = True if args.pruning_method == "pdrop" else False
    pdrop_config = {"T": args.visual_token_num}

    # 🔥 MustDrop Configuration (Dual Attention Filter baseline)
    # Reference: MustDrop paper - training-free token pruning
    # IMPORTANT: MustDrop uses threshold-based pruning, NOT direct token count!
    # Thresholds from official MustDrop scripts (scripts/v1_5/eval/textvqa_*.sh):
    MUSTDROP_THRESHOLDS = {
        # NOTE: MustDrop minimum is ~47 tokens due to keep_rate=0.08 (47 key tokens always retained)
        # vtn=32 is architecturally impossible without modifying keep_rate
        64:  {"global_thr": 0.011,  "individual_thr": 0.01},
        128: {"global_thr": 0.0012, "individual_thr": 0.001},
        192: {"global_thr": 0.001,  "individual_thr": 0.001},
    }
    use_mustdrop = True if args.pruning_method == "mustdrop" else False
    # Get thresholds for target token count (default to 128 if not found)
    thr_config = MUSTDROP_THRESHOLDS.get(args.visual_token_num, MUSTDROP_THRESHOLDS[128])
    mustdrop_config = {
        "pruning_layers": [2, 6, 10, 14],  # 7B model layers (modelling_sparse_llama.py:134)
        "global_thr": thr_config["global_thr"],       # Threshold-based pruning
        "individual_thr": thr_config["individual_thr"],
        "keep_rate": 0.08,                  # 8% key tokens from CLS attention (clip_encoder.py:102)
        "merge_threshold": 0.8,             # Cosine similarity for token merging (clip_encoder.py:111)
        "merge_window_size": (3, 3),        # Super patch size 3x3 (clip_encoder.py:111)
        "T": args.visual_token_num,         # Target visual tokens for comparison
    }

    # ⭐ TOPSPrototype Configuration
    use_tops_prototype = True if args.pruning_method in ["tops_prototype", "tops_variant2", "tops_experimental", "tops_variant5"] else False
    tops_prototype_config = {
        "T": args.visual_token_num,  # Target visual tokens (64/128/192)
        "num_latent": args.num_latent,  # Number of latent tokens (default: 20)
        "latent_pool_size": (args.latent_pool_h, args.latent_pool_w),  # Grid size (default: 5x4)
        "mode": "cover_prune_2" if args.pruning_method == "cover_prune_2" else ("tops_variant5" if args.pruning_method == "tops_variant5" else ("tops_experimental" if args.pruning_method == "tops_experimental" else ("tops_variant2" if args.pruning_method == "tops_variant2" else "tops_prototype"))),  # COVER-Prune-2 / TOPSPrototype-V5/V3/V2/V1
        "debug": False,  # Set to True for debugging, False for performance testing
    }

    # 🔬 VScan Configuration (Tencent's Training-Free Visual Token Reduction)
    # Reference: https://github.com/Tencent/SelfEvolvingAgent/tree/main/VScan
    # Stage 1 (llava_arch): Complementary global and local scans -> stage1_tokens
    # Stage 2 (modeling_llama_vscan): Middle layer pruning -> stage2_tokens
    use_vscan = True if args.pruning_method == "vscan" else False
    vscan_config = {
        "stage1_tokens": args.visual_token_num,  # Stage 1 output (default: 96)
        "stage2_tokens": args.vscan_stage2_tokens,  # Stage 2 output (default: 32)
        "prune_layer": args.vscan_prune_layer,  # Layer for Stage 2 pruning (default: 16 for 7B)
    }

    use_text_tower = True if args.pruning_method in ("tsvol","tsvol_w","tsvol_llm","tsvol_w_llm","cover_prune_2","cover_prune_2_proj","cover_prune_2_rawattn","cover_prune_2_tv","cover_prune_3","cover_prune_3_anti","mmtok") or args.pruning_method == "trim" or "cdp3" in args.pruning_method or "thcp" in args.pruning_method or args.pruning_method == "tops_experimental" or args.pruning_method == "tops_early" else False

    # 🎯 DART: 官方 LLM 层剪枝 (第K层, key-norm pivot + 文本引导). 视觉侧不剪枝.
    use_dart_llm = True if args.pruning_method in ("dart_llm", "dart_coverage") else False
    dart_config = {"T": args.visual_token_num, "use_coverage": args.pruning_method == "dart_coverage"}

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        pruning_method=args.pruning_method,
        visual_token_num=args.visual_token_num,
        use_fastv=use_fastv, fastv_config=fastv_config,
        use_mico_coverage_llm=use_mico_coverage_llm, mico_coverage_llm_config=mico_coverage_llm_config,
        use_sparsevlm=use_sparsevlm, sparsevlm_config=sparsevlm_config,
        use_d2p=use_d2p, d2p_config=d2p_config,
        use_svdvlm=use_svdvlm, svdvlm_config=svdvlm_config,
        use_tops_early=use_tops_early, tops_early_config=tops_early_config,
        use_holov2=use_holov2, holov2_config=holov2_config,
        use_idea=use_idea, idea_config=idea_config,
        use_tops=use_tops, tops_config=tops_config,
        use_pdrop=use_pdrop, pdrop_config=pdrop_config,
        use_tops_prototype=use_tops_prototype, tops_prototype_config=tops_prototype_config,  # ⭐ TOPSPrototype
        use_mustdrop=use_mustdrop, mustdrop_config=mustdrop_config,  # 🔥 MustDrop
        use_vscan=use_vscan, vscan_config=vscan_config,  # 🔬 VScan
        use_dart_llm=use_dart_llm, dart_config=dart_config,  # 🎯 DART LLM层
        use_text_tower=use_text_tower,
    )

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # Smoke test: cap #samples via env TSVOL_MAX_SAMPLES (0/unset = all)
    _max_s = int(os.environ.get("TSVOL_MAX_SAMPLES", "0"))
    if _max_s > 0:
        questions = questions[:_max_s]
        print(f"[SMOKE] TSVOL_MAX_SAMPLES={_max_s} -> running {len(questions)} samples on chunk {args.chunk_idx}")
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    # Performance tracking variables
    total_tokens_generated = 0
    total_end_to_end_time = 0.0
    performance_samples = 0

    # Progress bar control: reads ENABLE_DEBUG from environment
    # Set ENABLE_DEBUG=1 to show progress bar, ENABLE_DEBUG=0 (or unset) for clean logs
    show_progress = os.environ.get('ENABLE_DEBUG', '0') == '1'
    data_bar = tqdm(zip(data_loader, questions), total=len(questions), disable=not show_progress)
    data_num = 0
    for (input_ids, image_tensors, image_sizes), line in data_bar:
        idx = line["question_id"]
        cur_prompt = line["text"]

        question = cur_prompt
        # question = question.split("\nReference OCR token")[0]
        question = question.replace("\nAnswer the question using a single word or phrase.", "")

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            if data_num == 50:
                model.start_latency = True
                model.track_flops = True
                torch.cuda.reset_peak_memory_stats()
            # elif data_num == 1050:
            #     model.start_latency = False
            #     model.track_flops = False
            #     break

            # Measure end-to-end time
            start_time = time.time()

            # 🔥 MustDrop: Different generate() signature
            # Reference: MustDrop model_vqa_loader.py:105-116
            if use_mustdrop:
                output_ids = model.generate(
                    mustdrop_config["global_thr"],      # First positional arg
                    mustdrop_config["individual_thr"],  # Second positional arg
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
                visual_token_num = getattr(model.model, 'visual_token_num', 576)
            else:
                output_ids, visual_token_num = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    texts=question,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            end_time = time.time()
            end_to_end_time = end_time - start_time

            if hasattr(model.model, 'visual_token_num'):
                visual_token_num = model.model.visual_token_num

            # Track performance metrics after warmup
            if data_num >= 50:
                num_tokens = output_ids.shape[1]
                total_tokens_generated += num_tokens
                total_end_to_end_time += end_to_end_time
                performance_samples += 1

            gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3

            # Calculate throughput for progress bar
            throughput_info = {}
            if performance_samples > 0 and total_end_to_end_time > 0:
                throughput_info = {
                    "vtn": visual_token_num,
                    "gpu": f"{gpu_memory_usage:.2f}GB",
                    "img/s": f"{performance_samples / total_end_to_end_time:.2f}"
                }
            else:
                throughput_info = {
                    "vtn": visual_token_num,
                    "gpu": f"{gpu_memory_usage:.2f}GB"
                }

            data_bar.set_postfix(**throughput_info)
            data_num += 1

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

    # Performance metrics summary
    print("\n" + "="*70)
    print("Performance Metrics Summary")
    print("="*70)

    # Latency
    print("\n[Latency Metrics]")
    if performance_samples > 0:
        avg_prefill = getattr(model, 'prefill_latency', 0.0) / performance_samples / 1000
        avg_decode = getattr(model, 'decode_latency', 0.0) / performance_samples / 1000
        avg_total = avg_prefill + avg_decode
        print(f"  Prefill latency (avg): {avg_prefill:.2f} ms")
        print(f"  Decode latency (avg): {avg_decode:.2f} ms")
        print(f"  Total latency (avg): {avg_total:.2f} ms")
        print(f"  End-to-end time (avg): {(total_end_to_end_time / performance_samples) * 1000:.2f} ms")
    else:
        print("  No performance data collected (need > 50 samples)")

    # GPU Memory
    print("\n[GPU Memory]")
    print(f"  Peak usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # FLOPS
    print("\n[FLOPs]")
    if model.flops_count > 0:
        avg_flops = model.total_flops / model.flops_count
        print(f"  Average per sample: {avg_flops:.2f} TFLOPs")
        if hasattr(model, 'layer_visual_tokens') and model.layer_visual_tokens:
            print(f"  Visual tokens per layer: {model.layer_visual_tokens}")
    else:
        print("  No FLOPS data collected")

    # Throughput
    print("\n[Throughput]")
    if performance_samples > 0 and total_end_to_end_time > 0:
        tokens_per_sec = total_tokens_generated / total_end_to_end_time
        images_per_sec = performance_samples / total_end_to_end_time
        sec_per_image = total_end_to_end_time / performance_samples
        print(f"  Tokens/second: {tokens_per_sec:.2f}")
        print(f"  Images/second: {images_per_sec:.2f}")
        print(f"  Seconds/image: {sec_per_image:.2f}")
        print(f"  Total samples measured: {performance_samples}")
    else:
        print("  No throughput data collected")

    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--pruning_method", type=str, default=None)
    parser.add_argument("--visual_token_num", type=int, default=576)

    # ⭐ TOPSPrototype specific arguments
    parser.add_argument("--num_latent", type=int, default=20,
                        help="Number of latent tokens for TOPSPrototype (default: 20)")
    parser.add_argument("--latent_pool_h", type=int, default=5,
                        help="Height of latent pooling grid for TOPSPrototype (default: 5)")
    parser.add_argument("--latent_pool_w", type=int, default=4,
                        help="Width of latent pooling grid for TOPSPrototype (default: 4)")

    # 🔬 VScan specific arguments
    parser.add_argument("--vscan_stage2_tokens", type=int, default=32,
                        help="VScan Stage 2 target tokens (default: 32)")
    parser.add_argument("--vscan_prune_layer", type=int, default=16,
                        help="VScan Stage 2 pruning layer (default: 16 for 7B, use 20 for 13B)")

    args = parser.parse_args()

    eval_model(args)
