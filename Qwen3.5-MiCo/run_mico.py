"""Single-image MiCo inference with the recovered Qwen3.5 QK-at-layer-12 recipe."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', default='Qwen/Qwen3.5-9B')
    p.add_argument('--image', required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--tokens', type=int, choices=[128, 256], default=128)
    p.add_argument('--variant', choices=['mico', 'mico_s1', 'baseline'], default='mico')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--trace-file')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.max_new_tokens < 1:
        p.error('--max-new-tokens must be positive')
    print(json.dumps({'model': a.model_path, 'method': a.variant, 'tokens': a.tokens,
                      'cut_after_layer': 12 if a.variant == 'mico' else 4,
                      'quality_mode': 'qk', 'attention': 'sdpa'}, indent=2))
    if a.dry_run:
        return
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
    from unified_core import install_pruning
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the 9B inference entry point')
    image = Path(a.image).expanduser().resolve(strict=True)
    processor = AutoProcessor.from_pretrained(a.model_path, min_pixels=802816, max_pixels=802816)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.model_path, torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image.as_uri()}, {'type': 'text', 'text': a.question}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors='pt').to(model.device)
    ctl = install_pruning(model, a.variant,
        target_tokens=None if a.variant == 'baseline' else a.tokens,
        cut_after_layer=12 if a.variant == 'mico' else 4, quality_mode='qk')
    try:
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
        generated = output[:, inputs['input_ids'].shape[1]:]
        print(processor.batch_decode(generated, skip_special_tokens=True)[0])
        if a.trace_file:
            path = Path(a.trace_file).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(ctl.last_record, indent=2) + '\n', encoding='utf-8')
    finally:
        ctl.remove()


if __name__ == '__main__':
    main()
