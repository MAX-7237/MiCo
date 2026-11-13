import argparse
import torch
import os
import json
import time
import re
from tqdm import tqdm
import shortuuid
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        return (input_ids, image_tensor, image.size)

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return (input_ids, image_tensors, image_sizes)


def create_data_loader(
    questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def eval_model(args):
    if os.environ.get("ENABLE_DEBUG", "0") != "1":
        import warnings

        warnings.filterwarnings("ignore")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    match = re.fullmatch("mico_([1-9][0-9]*)", args.pruning_method)
    if args.pruning_method not in ("vanilla", "mico_s1") and match is None:
        raise ValueError("Unsupported MiCo configuration")
    config = (
        {
            "K": int(match.group(1)),
            "T": args.visual_token_num,
            "T_avg": args.visual_token_num,
            "dynamic_R": True,
            "mico": True,
        }
        if match
        else None
    )
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        pruning_method=args.pruning_method,
        visual_token_num=args.visual_token_num,
        use_mico_coverage_llm=match is not None,
        mico_coverage_llm_config=config,
    )
    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and ("mmtag" not in args.conv_mode)
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )
    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config
    )
    total_tokens_generated = 0
    total_end_to_end_time = 0.0
    performance_samples = 0
    show_progress = os.environ.get("ENABLE_DEBUG", "0") == "1"
    data_bar = tqdm(zip(data_loader, questions), total=len(questions), disable=not show_progress)
    data_num = 0
    for (input_ids, image_tensors, image_sizes), line in data_bar:
        idx = line["question_id"]
        cur_prompt = line["text"]
        question = cur_prompt
        question = question.replace("\nAnswer the question using a single word or phrase.", "")
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device="cuda", non_blocking=True)
        with torch.inference_mode():
            if data_num == 50:
                model.start_latency = True
                model.track_flops = True
                torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
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
                use_cache=True,
            )
            end_time = time.time()
            end_to_end_time = end_time - start_time
            if hasattr(model.model, "visual_token_num"):
                visual_token_num = model.model.visual_token_num
            if data_num >= 50:
                num_tokens = output_ids.shape[1]
                total_tokens_generated += num_tokens
                total_end_to_end_time += end_to_end_time
                performance_samples += 1
            gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3
            throughput_info = {}
            if performance_samples > 0 and total_end_to_end_time > 0:
                throughput_info = {
                    "vtn": visual_token_num,
                    "gpu": f"{gpu_memory_usage:.2f}GB",
                    "img/s": f"{performance_samples / total_end_to_end_time:.2f}",
                }
            else:
                throughput_info = {"vtn": visual_token_num, "gpu": f"{gpu_memory_usage:.2f}GB"}
            data_bar.set_postfix(**throughput_info)
            data_num += 1
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()
    print("\n" + "=" * 70)
    print("Performance Metrics Summary")
    print("=" * 70)
    print("\n[Latency Metrics]")
    if performance_samples > 0:
        avg_prefill = getattr(model, "prefill_latency", 0.0) / performance_samples / 1000
        avg_decode = getattr(model, "decode_latency", 0.0) / performance_samples / 1000
        avg_total = avg_prefill + avg_decode
        print(f"  Prefill latency (avg): {avg_prefill:.2f} ms")
        print(f"  Decode latency (avg): {avg_decode:.2f} ms")
        print(f"  Total latency (avg): {avg_total:.2f} ms")
        print(
            f"  End-to-end time (avg): {total_end_to_end_time / performance_samples * 1000:.2f} ms"
        )
    else:
        print("  No performance data collected (need > 50 samples)")
    print("\n[GPU Memory]")
    print(f"  Peak usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
    print("\n[FLOPs]")
    if model.flops_count > 0:
        avg_flops = model.total_flops / model.flops_count
        print(f"  Average per sample: {avg_flops:.2f} TFLOPs")
        if hasattr(model, "layer_visual_tokens") and model.layer_visual_tokens:
            print(f"  Visual tokens per layer: {model.layer_visual_tokens}")
    else:
        print("  No FLOPS data collected")
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
    print("\n" + "=" * 70)


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
    args = parser.parse_args()
    eval_model(args)
