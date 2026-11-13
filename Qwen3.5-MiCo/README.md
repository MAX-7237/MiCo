# Qwen3.5-9B MiCo

MiCo inference for one unpadded image, batch size one, SDPA attention and cached greedy decoding. Stage 1 retains up to `2T` visual tokens; stage 2 prunes after layer 12 to target the layer-average budget `T`.

Use Python 3.12 and the pinned dependencies in `requirements.txt`:

```bash
conda create -n vlmevalkit-qwen3vl python=3.12 -y
conda activate vlmevalkit-qwen3vl
python ../setup_environment.py --component Qwen3.5-MiCo
python run_mico.py --model-path /path/to/Qwen3.5-9B --image /path/to/image.jpg \
  --question 'What is shown in the image?' --tokens 128 --trace-file outputs/trace.json
python -m unittest discover -s core/tests -p 'test_*.py' -v
```

Budgets are `128/256`. `--variant mico_s1` retains `T` tokens in a single stage; `--variant baseline` leaves visual tokens unpruned. The entry performs single-image inference; benchmark input preparation and scoring are separate.

The core preserves token positions and per-layer cache lengths. CPU tests exercise the pruning schedule and the native architecture using small random weights.
