# MiCo

## Supported models

| Model family | Implementation | Environment |
| :--- | :--- | :--- |
| LLaVA-1.5 / LLaVA-NeXT · 7B & 13B | [LLaVA-MiCo](LLaVA-MiCo/run_mico.py) | `llava` |
| LLaVA-Video · 7B | [LLaVA-Video-MiCo](LLaVA-Video-MiCo/run_mico.py) | `llavavid` |
| Qwen2.5-VL · 7B / InternVL3 · 8B | [VLMEvalKit-MiCo](VLMEvalKit-MiCo/run_mico.py) | `vlmevalkit-qwen2.5vl` |
| Qwen3-VL · 8B | [VLMEvalKit-MiCo-Qwen3](VLMEvalKit-MiCo-Qwen3/run_mico.py) | `vlmevalkit-qwen3vl` |
| Qwen3.5 · 9B | [Qwen3.5-MiCo](Qwen3.5-MiCo/run_mico.py) | `vlmevalkit-qwen3vl` |

## Quick start

Choose the model family below. Each group includes environment setup and a runnable launcher example. Use **Linux with an NVIDIA GPU**, and download the model weights and benchmark data separately.

```bash
# Download this repository and extract it into the MiCo directory.
cd MiCo
export REPO_ROOT="$PWD"
export PYTHONNOUSERSITE=1
```

Run the setup block for the component you need. If its Conda environment already exists, skip `conda create` and activate it. Keep the environment profiles separate: image LLaVA, video LLaVA, and the Qwen families use different Transformers versions. Dependency pins and installation caveats are documented in the [environment guide](docs/RUNNING.md#environments).

The commands below run inference or evaluation directly, assuming the environment, datasets, and model weights are configured. Replace `/path/to/...` with your local paths.

### 1. LLaVA image models — four backbones

```bash
cd "$REPO_ROOT/LLaVA-MiCo"
conda create -n llava python=3.10 -y
conda activate llava
python -m pip install -r requirements.txt
python -m pip check
```

| Backbone | `--model` | `--tokens` choices |
| :--- | :--- | :--- |
| LLaVA-1.5-7B | `llava-1.5-7b` | `128`, `64`, `32` |
| LLaVA-1.5-13B | `llava-1.5-13b` | `128`, `64`, `32` |
| LLaVA-NeXT-7B (Vicuna) | `llava-next-7b` | `640`, `320`, `160` |
| LLaVA-NeXT-13B (Vicuna) | `llava-next-13b` | `640`, `320`, `160` |

MiCo passes the NeXT total budget directly to the implementation: use `640/320/160`.

Prepare the benchmark question JSONL and image directory, then choose one of the four commands: For MME, preserve the original question followed by `\nAnswer the question using a single word or phrase.`; see the [benchmark input format](docs/RUNNING.md#llava-image-models).

```bash
python run_mico.py --model llava-1.5-7b --tokens 128 \
  --question-file /path/to/questions.jsonl --image-folder /path/to/images \
  --answers-file outputs/llava-1.5-7b/answers.jsonl

python run_mico.py --model llava-1.5-13b --tokens 128 \
  --question-file /path/to/questions.jsonl --image-folder /path/to/images \
  --answers-file outputs/llava-1.5-13b/answers.jsonl

python run_mico.py --model llava-next-7b --tokens 640 \
  --question-file /path/to/questions.jsonl --image-folder /path/to/images \
  --answers-file outputs/llava-next-7b/answers.jsonl

python run_mico.py --model llava-next-13b --tokens 640 \
  --question-file /path/to/questions.jsonl --image-folder /path/to/images \
  --answers-file outputs/llava-next-13b/answers.jsonl

```

The launcher selects the default Hugging Face model ID for each alias; use `--model-path /path/to/weights` for local image-model weights. These commands write predictions. Run the relevant benchmark scorer afterwards; MMBench and ScienceQA also provide dedicated evaluation modules under `llava/eval`.

### 2. LLaVA-Video — 7B Qwen2

Video has its own environment; do not install its dependencies into `llava`.

```bash
cd "$REPO_ROOT/LLaVA-Video-MiCo"
conda create -n llavavid python=3.10 -y
conda activate llavavid
python -m pip install -r requirements.txt
python ../tools/repair_decord_wheel.py
python -m pip install --no-build-isolation -r requirements-flash-attn.txt
python -m pip check

python run_mico.py --model llava-video-7b --tokens 32 \
  --model-path lmms-lab/LLaVA-Video-7B-Qwen2 \
  --dataset videomme --nproc 1 --work-dir outputs/video
```

The adapter uses the Qwen conversation template and samples up to **64 frames**. The token budget is **per frame** (`32/16`). Prepare VideoMME according to `lmms-eval/lmms_eval/tasks/videomme`, or select another registered video task with `--dataset`. Predictions and evaluation artifacts go under `--work-dir`. The video environment passed real 64-frame GPU inference and pruning-budget checks at both budgets; full video accuracy reproduction has not been tested.

### 3. Qwen2.5-VL and InternVL3

Both backbones share `vlmevalkit-qwen2.5vl` and the VLMEvalKit entry point.

```bash
cd "$REPO_ROOT/VLMEvalKit-MiCo"
conda create -n vlmevalkit-qwen2.5vl python=3.11 -y
conda activate vlmevalkit-qwen2.5vl
python -m pip install -r requirements.txt
python ../tools/repair_decord_wheel.py
python -m pip install --no-build-isolation -r requirements-flash-attn.txt
python -m pip check

# Qwen2.5-VL-7B-Instruct
python run_mico.py --model qwen2.5-vl-7b --tokens 256 \
  --dataset AI2D_TEST --nproc 1 --work-dir outputs/qwen25

# InternVL3-8B
python run_mico.py --model internvl3-8b --tokens 256 \
  --dataset AI2D_TEST --nproc 1 --work-dir outputs/internvl3
```

Use `--tokens 256` or `128`. Set `LMUData` to your dataset/cache directory when needed, and follow the selected VLMEvalKit dataset adapter's preparation instructions. To use local model weights, edit the matching entry in `vlmeval/config.py`; these VLMEval launchers do not accept `--model-path`. Some benchmarks require an external judge model in addition to inference.

FlashAttention must be built **after PyTorch**, with a compatible CUDA toolkit and compiler. See the [environment guide](docs/RUNNING.md#environments) for the pinned versions.

### 4. Qwen3-VL and Qwen3.5

These components share `vlmevalkit-qwen3vl` (Python 3.12, PyTorch 2.7.0, Transformers 5.2.0). Their entry points differ: Qwen3-VL uses VLMEvalKit, while Qwen3.5 provides single-image inference.

**Qwen3-VL-8B benchmark evaluation**

```bash
cd "$REPO_ROOT/VLMEvalKit-MiCo-Qwen3"
conda create -n vlmevalkit-qwen3vl python=3.12 -y
conda activate vlmevalkit-qwen3vl
python -m pip install -r requirements.txt
python ../tools/repair_decord_wheel.py
python -m pip check

python run_mico.py --model qwen3-vl-8b --tokens 128 \
  --dataset AI2D_TEST --nproc 1 --work-dir outputs/qwen3
```

Use budgets `256/128`. Dataset preparation and local-weight configuration follow the same VLMEvalKit conventions as the preceding group.

**Qwen3.5-9B single-image inference**

Activate the shared environment created above. If you only need Qwen3.5, create it with `conda create -n vlmevalkit-qwen3vl python=3.12 -y` first.

```bash
cd "$REPO_ROOT/Qwen3.5-MiCo"
conda activate vlmevalkit-qwen3vl
python -m pip install -r requirements.txt
python ../tools/repair_decord_wheel.py
python -m pip check

python run_mico.py --model-path Qwen/Qwen3.5-9B \
  --image /path/to/image.jpg --question 'What is shown in the image?' \
  --tokens 128 --trace-file outputs/trace.json
```

The default is the MiCo QK recipe at layer 12, with SDPA, one image, batch size one, and cached greedy decoding. Use budgets `128/256`. The answer is printed and pruning diagnostics are saved to `--trace-file`. This is a single-image inference entry, not a complete benchmark scoring pipeline. See the [Qwen3.5 guide](Qwen3.5-MiCo/README.md) for controls and core tests.

See the [full evaluation guide](docs/RUNNING.md#evaluation) for additional configuration and scoring details.

## Validation

The four environment profiles have been exercised with real model weights. Image-model checks use MME; video checks use 64-frame inference at both supported budgets. Paper-score reproduction remains under verification. See [validation records](VALIDATION.json) for scope and limitations.

## Acknowledgements

Built on LLaVA, LLaVA-NeXT/LLaVA-Video, VLMEvalKit, lmms-eval, and Transformers. Component licenses and original third-party attribution are preserved in their respective directories.
