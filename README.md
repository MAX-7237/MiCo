# MiCo

MiCo source release for visual token pruning in image and video language models.

**Status: source preview.** The implementations and evaluation entry points are included, but clean-environment installation, GPU inference, and benchmark reproduction have not yet been completed for this release. Dependency files are candidate pinned configurations, not validated lockfiles. Qwen3.5-9B is not included: its implementation and runnable environment have not been recovered.

## Implementations

| Directory | Models | MiCo internal method |
| --- | --- | --- |
| `LLaVA-MiCo` | LLaVA-1.5-7B / 13B | `attn_H_cov_vit_proj_llm_7` / `_8` |
| `LLaVA-MiCo` | LLaVA-NeXT (Vicuna) 7B / 13B | `attn_H_cov_vit_proj_llm_7` / `_14` |
| `LLaVA-Video-MiCo` | LLaVA-Video-7B-Qwen2 | `attn_H_cov_proj_llm_12` |
| `VLMEvalKit-MiCo` | Qwen2.5-VL-7B-Instruct / InternVL3-8B | `attn_H_cov_vit_proj_llm_2` / `_4` |
| `VLMEvalKit-MiCo-Qwen3` | Qwen3-VL-8B-Instruct | `attn_H_cov_vit_proj_llm_6` |

The public `run_mico.py` launchers select these existing implementation IDs. The suffix identifies the pruning layer; it is not the token budget. Image-model S1-only evaluation uses `--variant mico-s1`. Video S1-only is not exposed. Other historical method names and baselines remain where required by framework imports; they should not be treated as aliases for MiCo.

For image models, stage 1 retains up to `min(N, 2T)` visual tokens. Stage 2 performs a post-layer-K cut with target `R = (T*L - K*N_enter)/(L-K)`, subject to implementation rounding and clamping. Video uses its own per-frame token layout and budget.

## Environments

Use Linux x86_64 with an NVIDIA GPU and a CUDA-compatible driver. Keep these environments separate; the framework versions are incompatible with one another.

| Directory | Environment name | Python | PyTorch / torchvision | Transformers | FlashAttention |
| --- | --- | --- | --- | --- | --- |
| `LLaVA-MiCo` | `fastv` | 3.10 | 2.1.2 / 0.16.2 | 4.37.2 | Not present in recorded environment |
| `VLMEvalKit-MiCo` | `vlmevalkit-qwen2.5vl` | 3.11 | 2.7.0+cu126 / 0.22.0+cu126 | 4.48.1 | 2.8.3 |
| `VLMEvalKit-MiCo-Qwen3` | `vlmevalkit-qwen3vl` | 3.12 | 2.7.0 / 0.22.0 | 5.2.0 | Not present in recorded environment |
| `LLaVA-Video-MiCo` | `llavavid` | 3.10 | 2.1.2+cu121 / 0.16.2+cu121 | 4.41.2 | 2.6.3 |

The video profile is a candidate recovered from an existing environment and still needs runtime verification. Each directory provides its own `requirements.txt`; install only the one for the model you need. For example:

```bash
git clone git@github.com:MAX-7237/MiCo.git
cd MiCo
conda create -n vlmevalkit-qwen2.5vl python=3.11 -y
conda activate vlmevalkit-qwen2.5vl
cd VLMEvalKit-MiCo
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-flash-attn.txt
python -m pip check
```

For other profiles, substitute the environment name, Python version, and directory from the table. Install FlashAttention only for profiles with a pinned version; build it after PyTorch using a compatible CUDA toolkit and compiler. Do not use `--no-deps` to bypass dependency conflicts. The launchers add the local source tree to `PYTHONPATH`, so editable installation is not required for those launchers. Upstream packaging metadata is retained for reference and may contain older dependency constraints.

Candidate dependency adjustments from the recovered snapshots: video Triton 2.1.0 and HTTPcore 0.16.3/H11 0.14.0; Qwen2.5 OpenCV 4.11.0.86 for NumPy 1.26; image LLaVA PEFT 0.4.0 for its older Transformers API. These adjustments still require full runtime validation. Dataset-specific adapters may need additional dependencies.

## Evaluation

Download the model weights and datasets separately, accepting their respective licenses. Set `HF_HOME` and `LMUData` if you want custom cache and VLMEval data locations. The repository contains no model weights or datasets. Any `/path/to/local-assets` values in legacy adapters are placeholders that must be configured before using those adapters.

Run commands from the indicated directory. Start with `--dry-run` to inspect the selected method and command; remove it to execute. Dry runs do not load model weights or validate GPU execution.

### LLaVA image models

```bash
cd LLaVA-MiCo
python run_mico.py --model llava-1.5-7b --tokens 128 \
  --model-path liuhaotian/llava-v1.5-7b \
  --question-file /path/to/questions.jsonl --image-folder /path/to/images \
  --answers-file outputs/llava15/answers.jsonl --dry-run
```

Supported aliases: `llava-1.5-7b`, `llava-1.5-13b`, `llava-next-7b`, `llava-next-13b`. Typical LLaVA-1.5 budgets are 128/64/32; NeXT total budgets are 640/320/160. `--tokens` is passed unchanged. The launcher generates predictions through `llava.eval.model_vqa_loader`; use the appropriate dataset scorer afterwards. MMBench and ScienceQA also have dedicated modules under `llava/eval`.

### Qwen2.5-VL and InternVL3

```bash
cd VLMEvalKit-MiCo
python run_mico.py --model qwen2.5-vl-7b --tokens 128 --dataset AI2D_TEST \
  --nproc 1 --work-dir outputs/qwen25 --dry-run
python run_mico.py --model internvl3-8b --tokens 256 --dataset AI2D_TEST \
  --nproc 1 --work-dir outputs/internvl3 --dry-run
```

### Qwen3-VL

```bash
cd VLMEvalKit-MiCo-Qwen3
python run_mico.py --model qwen3-vl-8b --tokens 128 --dataset AI2D_TEST \
  --nproc 1 --work-dir outputs/qwen3 --dry-run
```

VLMEval launchers use the model registry in `vlmeval/config.py`; change that registry if using local weights. Their `--model-path` argument is not supported. Consult the dataset adapter for preparation and scoring requirements, including optional external judge models. Clear any previous stage-1 ablation settings (`H_VIT_S1_MULT`, `H_VIT_NO_H`, `H_VIT_NO_COV`, `H_VIT_NO_ATTN`, `MICO_STAGE1_MULT`) when running the default configuration.

### LLaVA-Video

```bash
cd LLaVA-Video-MiCo
python run_mico.py --model llava-video-7b --tokens 32 --dataset videomme \
  --model-path lmms-lab/LLaVA-Video-7B-Qwen2 \
  --nproc 1 --work-dir outputs/video --dry-run
```

Video evaluation includes the vendored `lmms-eval` adapter, uses the Qwen conversation template and a maximum of 64 frames, and interprets the token budget per frame. Prepare datasets according to the selected task configuration under `lmms-eval/lmms_eval/tasks`.

## Code layout and provenance

- Image LLaVA: `llava/model/llava_arch.py` and `llava/model/language_model/modeling_llama_attn_cov_llm.py`.
- Video: `llava/model/llava_arch.py`, `llava/model/language_model/llava_qwen.py`, and the `lmms_eval/models/llava_vid.py` adapter.
- Qwen2.5/InternVL3: `vlmeval/vlm/attn_cov_select.py` and the respective `qwen2_5_vl` / `internvl3` model packages.
- Qwen3: `vlmeval/vlm/qwen3_vl/modeling_qwen3_vl.py` and its pruning implementations.

This repository builds on LLaVA, LLaVA-NeXT/LLaVA-Video, VLMEvalKit, lmms-eval, Transformers, and retained baseline implementations. Existing license files and source attribution remain with their components. Refer to those licenses when reusing or redistributing code; this preview does not assign a new blanket license to third-party sources.

## Validation status

Source syntax and all eight launcher dry runs are checked before the initial commit. Full dependency installation, model imports, GPU inference, and benchmark score reproduction remain pending. This release does not claim validated accuracy or speed results. Known legacy adapters may have unresolved optional imports; use the documented MiCo paths and report failures with the selected environment and model.
