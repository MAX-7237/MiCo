import argparse
import torch
import os
import json
import re
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # Keep the S2 locus encoded in the method identifier; this is paired with
    # the generic H-ViT S1 route in llava_arch.py.
    hvit_llm_match = re.fullmatch(r"mico_([1-9][0-9]*)", args.pruning_method)

    use_fastv = True if args.pruning_method == "fastv" else False
    fastv_config = {"K": 2, "T": args.visual_token_num}
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
        # proj keep 2T or entropy-gated N1; K=2; T is layer-avg. S2 matches proj_2.
        mico_coverage_llm_config = {
            "K": 2,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
        }
    elif args.pruning_method in ("mico_coverage_proj_llm_12", "mico_coverage_vit_proj_llm_12"):
        # Same S2 as proj_llm_12. prefix2_first: 7B L12 / 13B L15.
        # Keep is FastV-style avg T. vit_proj S1 is CLIP-space.
        mico_coverage_llm_config = {
            "K": 12,
            "T_avg": args.visual_token_num,
            "T": args.visual_token_num,
            "dynamic_R": True,
            "prefix2_first": True,
        }
    elif args.pruning_method == "mico_coverage_proj_llm_12_24":
        # Same S1/S2 as proj_2. prefix2_keep: 7B L12/24, 13B L15/30 keep table.
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

    # VScan Configuration
    use_vscan = True if args.pruning_method == "vscan" else False
    vscan_config = {
        "stage1_tokens": args.visual_token_num,  # Stage 1 output
        "stage2_tokens": args.vscan_stage2_tokens,  # Stage 2 output
        "prune_layer": args.vscan_prune_layer,  # Layer for Stage 2 pruning
    }

    # MustDrop Configuration (aligned with model_vqa_loader.py)
    MUSTDROP_THRESHOLDS = {
        64:  {"global_thr": 0.011,  "individual_thr": 0.01},    # ✅ Fixed: was 0.002/0.0015
        128: {"global_thr": 0.0012, "individual_thr": 0.001},
        192: {"global_thr": 0.001,  "individual_thr": 0.001},
    }
    use_mustdrop = True if args.pruning_method == "mustdrop" else False
    thr_config = MUSTDROP_THRESHOLDS.get(args.visual_token_num, MUSTDROP_THRESHOLDS[128])
    mustdrop_config = {
        "pruning_layers": [2, 6, 10, 14],
        "global_thr": thr_config["global_thr"],
        "individual_thr": thr_config["individual_thr"],
        "keep_rate": 0.08,                  # ✅ Added: 8% key tokens
        "merge_threshold": 0.8,             # ✅ Added: cosine similarity for merging
        "merge_window_size": (3, 3),        # ✅ Added: super patch size
        "T": args.visual_token_num,         # ✅ Added: target token count
    }

    # TOPSPrototype Configuration
    use_tops_prototype = True if args.pruning_method in ["tops_prototype", "tops_variant2", "tops_experimental"] else False
    tops_prototype_config = {
        "T": args.visual_token_num,
        "num_latent": 20,
        "latent_pool_size": (5, 4),
        "mode": "cover_prune_2" if args.pruning_method == "cover_prune_2" else ("tops_experimental" if args.pruning_method == "tops_experimental" else ("tops_variant2" if args.pruning_method == "tops_variant2" else "tops_prototype")),
        "debug": True,
    }

    use_text_tower = True if args.pruning_method in ("tsvol","tsvol_w","tsvol_llm","tsvol_w_llm","cover_prune_2","cover_prune_2_proj","cover_prune_2_rawattn","cover_prune_2_tv","cover_prune_3","cover_prune_3_anti","mmtok") or args.pruning_method == "trim" or "cdp3" in args.pruning_method or "thcp" in args.pruning_method or args.pruning_method == "tops_experimental" or args.pruning_method == "tops_early" else False
    # 🎯 DART: 官方 LLM 层剪枝, 视觉侧不剪枝.
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
        use_tops_prototype=use_tops_prototype, tops_prototype_config=tops_prototype_config,
        use_mustdrop=use_mustdrop, mustdrop_config=mustdrop_config,
        use_vscan=use_vscan, vscan_config=vscan_config,
        use_dart_llm=use_dart_llm, dart_config=dart_config,
        use_text_tower=use_text_tower,
    )

    # Data
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_bar = tqdm(questions.iterrows(), total=len(questions))
    for index, row in data_bar:
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            try:
                image = load_image_from_base64(row['image'])
            except Exception as e:
                # LMUData tsv 里少数 circular 变体行的 image 字段是 index 引用而非 base64，
                # 非 LLaVA 官方口径行，跳过以免整个 chunk 崩溃。
                print(f"[skip] index={idx} bad image ({e})")
                continue
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                    # cur_prompt = cur_prompt + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                    # cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            question = cur_prompt
            # question = row['question']

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                # MustDrop uses different generate() signature
                if use_mustdrop:
                    output_ids = model.generate(
                        mustdrop_config["global_thr"],
                        mustdrop_config["individual_thr"],
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=1024,
                        use_cache=True)
                    visual_token_num = getattr(model.model, 'visual_token_num', 576)
                else:
                    output_ids, visual_token_num = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        texts=question,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=1024,
                        use_cache=True)
                    if hasattr(model.model, 'visual_token_num'):
                        visual_token_num = model.model.visual_token_num
                data_bar.set_postfix(vtn=f"{visual_token_num}")

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

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
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--pruning_method", type=str, default=None)
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--vscan_stage2_tokens", type=int, default=32,
                        help="VScan Stage 2 target tokens (default: 32)")
    parser.add_argument("--vscan_prune_layer", type=int, default=16,
                        help="VScan Stage 2 pruning layer (default: 16 for 7B, use 20 for 13B)")
    args = parser.parse_args()

    eval_model(args)
