# MiCo

MiCo source release for visual token pruning in image and video language models.

## Implementations

| Directory | Models | MiCo configuration |
| --- | --- | --- |
| `LLaVA-MiCo` | LLaVA-1.5-7B / 13B | `mico_7` / `_8` |
| `LLaVA-MiCo` | LLaVA-NeXT (Vicuna) 7B / 13B | `mico_7` / `_14` |
| `LLaVA-Video-MiCo` | LLaVA-Video-7B-Qwen2 | `mico` |
| `VLMEvalKit-MiCo` | Qwen2.5-VL-7B-Instruct / InternVL3-8B | `mico_2` / `_4` |
| `VLMEvalKit-MiCo-Qwen3` | Qwen3-VL-8B-Instruct | `mico_6` |

The public `run_mico.py` launchers select these existing implementation IDs. For image models, the suffix identifies the pruning layer; it is not the token budget. Video uses `mico` with decoder refinement after layer K=6. Image-model S1-only evaluation uses `--variant mico-s1`. Video S1-only is not exposed. Only the documented MiCo methods and their unpruned/single-stage controls are exposed.

For image models, stage 1 retains up to `min(N, 2T)` visual tokens. Stage 2 performs a post-layer-K cut with target `R = (T*L - K*N_enter)/(L-K)`, subject to implementation rounding and clamping. Video uses its own per-frame token layout and budget.

## Qwen3.5-9B

See [Qwen3.5-MiCo](../Qwen3.5-MiCo/README.md) for the QK-at-layer-12 adapter, single-image inference, and core tests. It shares the `vlmevalkit-qwen3vl` environment (Python 3.12 / PyTorch 2.7.0 / Transformers 5.2.0).

## Environments

Use Linux x86_64 with an NVIDIA GPU and a CUDA-compatible driver. Keep these environments separate; the framework versions are incompatible with one another.

| Directory | Environment name | Python | PyTorch / torchvision | Transformers | FlashAttention |
| --- | --- | --- | --- | --- | --- |
| `LLaVA-MiCo` | `llava` | 3.10 | 2.1.2 / 0.16.2 | 4.37.2 | Not required |
| `VLMEvalKit-MiCo` | `vlmevalkit-qwen2.5vl` | 3.11 | 2.7.0+cu126 / 0.22.0+cu126 | 4.52.3 | 2.8.3 |
| `VLMEvalKit-MiCo-Qwen3` | `vlmevalkit-qwen3vl` | 3.12 | 2.7.0 / 0.22.0 | 5.2.0 | Not required |
| `LLaVA-Video-MiCo` | `llavavid` | 3.10 | 2.1.2+cu121 / 0.16.2+cu121 | 4.41.2 | 2.6.3 |

Each directory provides its own `requirements.txt`; install only the one for the model you need. For example:

```bash
# Download this repository and extract it into the MiCo directory.
cd MiCo
conda create -n vlmevalkit-qwen2.5vl python=3.11 -y
conda activate vlmevalkit-qwen2.5vl
cd VLMEvalKit-MiCo
python ../setup_environment.py --component VLMEvalKit-MiCo
```

For other profiles, substitute the environment name, Python version, and directory from the table. Install FlashAttention only for profiles with a pinned version; build it after PyTorch using a compatible CUDA toolkit and compiler. Do not use `--no-deps` to bypass dependency conflicts. The launchers add the local source tree to `PYTHONPATH`, so editable installation is not required for those launchers. The component package metadata uses the same pinned dependency file.

The installer installs the component's pinned dependencies, configures video-decoding compatibility when needed, installs its FlashAttention dependency after PyTorch, and runs dependency checks. Use `python ../setup_environment.py --component <component-directory> --check` to verify an existing environment. Run it from the selected component directory. Set `PYTHONNOUSERSITE=1` before installation and evaluation.

## Evaluation

Download the model weights and datasets separately, accepting their respective licenses. Set `HF_HOME` and `LMUData` if you want custom cache and VLMEval data locations. The repository contains no model weights or datasets. Replace `/path/to/...` placeholders with your local data and checkpoint paths.

Run commands from the indicated directory with the environment, datasets, and model weights configured. The examples execute inference or evaluation directly.

### LLaVA image models

```bash
cd LLaVA-MiCo
python run_mico.py --model llava-1.5-7b --tokens 128 \
  --model-path liuhaotian/llava-v1.5-7b \
  --question-file /path/to/questions.jsonl --image-folder /path/to/images \
  --answers-file outputs/llava15/answers.jsonl
```

For LLaVA MME, use the prepared benchmark questions with this exact answer instruction:

```text
<original MME question>
Answer the question using a single word or phrase.
```

Keep `--conv-mode vicuna_v1`, greedy decoding (`temperature=0`), and the benchmark's image/question pairing. Do not replace the answer instruction with `Please answer yes or no.` when comparing with the paper protocol. This format applies to the LLaVA MME input JSONL; other model families use their native benchmark adapters. Use one benchmark process per GPU.

Supported aliases: `llava-1.5-7b`, `llava-1.5-13b`, `llava-next-7b`, `llava-next-13b`. Typical LLaVA-1.5 budgets are 128/64/32; NeXT total budgets are 640/320/160. `--tokens` is passed unchanged. The launcher generates predictions through `llava.eval.model_vqa_loader`; use the appropriate dataset scorer afterwards. MMBench and ScienceQA also have dedicated modules under `llava/eval`.

### Qwen2.5-VL and InternVL3

```bash
cd VLMEvalKit-MiCo
python run_mico.py --model qwen2.5-vl-7b --tokens 128 --dataset AI2D_TEST \
  --nproc 1 --work-dir outputs/qwen25
python run_mico.py --model internvl3-8b --tokens 256 --dataset AI2D_TEST \
  --nproc 1 --work-dir outputs/internvl3
```

### Qwen3-VL

```bash
cd VLMEvalKit-MiCo-Qwen3
python run_mico.py --model qwen3-vl-8b --tokens 128 --dataset AI2D_TEST \
  --nproc 1 --work-dir outputs/qwen3
```

VLMEval launchers use the model registry in `vlmeval/config.py`; change that registry if using local weights. Their `--model-path` argument is not supported. Consult the dataset adapter for preparation and scoring requirements, including optional external judge models. Clear any previous stage-1 ablation settings (`MICO_STAGE1_MULT`, `MICO_NO_NORM`, `MICO_NO_COVERAGE`, `MICO_NO_ATTENTION`) when running the default configuration.

### LLaVA-Video

```bash
cd LLaVA-Video-MiCo
python run_mico.py --model llava-video-7b --tokens 32 --dataset videomme \
  --model-path lmms-lab/LLaVA-Video-7B-Qwen2 \
  --nproc 1 --work-dir outputs/video
```

Video evaluation includes the vendored `lmms-eval` adapter, uses the Qwen conversation template and a maximum of 64 frames, and interprets the token budget per frame. Prepare datasets according to the selected task configuration under `lmms-eval/lmms_eval/tasks`.

## Source layout and licenses

See the [source map](STRUCTURE.md) for the retained implementation files. The repository builds on LLaVA, LLaVA-NeXT/LLaVA-Video, VLMEvalKit, lmms-eval and Transformers. Original component licenses and required copyright notices are preserved.

Run `python -m pip check` before evaluation. Each benchmark process should have an exclusive GPU. Model weights and benchmark data are configured separately.
