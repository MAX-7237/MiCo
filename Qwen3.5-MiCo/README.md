# Qwen3.5-9B MiCo

Recovered native Transformers 5.2 implementation for one unpadded image, batch size one, SDPA attention and cached greedy decoding. The public launcher selects the `qk` layer-12 recipe: stage 1 retains up to `2T` tokens and stage 2 targets the layer-average budget `T`. It is distinct from the later task-query hybrid experiments retained in the core API. Do not interchange these recipes when comparing results.

Use the same `vlmevalkit-qwen3vl` environment as the Qwen3 component: Python 3.12, PyTorch 2.7.0, Transformers 5.2.0. `requirements.txt` contains the shared candidate pins. The implementation uses Transformers internal APIs; upgrading Transformers requires revalidation.

```bash
conda create -n vlmevalkit-qwen3vl python=3.12 -y
conda activate vlmevalkit-qwen3vl
python -m pip install -r requirements.txt
python -m pip check
python run_mico.py --model-path Qwen/Qwen3.5-9B --image /path/to/image.jpg \
  --question 'What is shown in the image?' --tokens 128 --trace-file outputs/trace.json
python -m unittest discover -s core/tests -p 'test_*.py' -v
```

`--variant baseline` and `--variant mico_s1` expose controls. The stage-1-only control retains `T` tokens; it does not retain `2T` like the first stage of the two-stage method. This directory provides single-image inference and core tests, not an eight-benchmark scoring pipeline. Weights and datasets are external.

The core also preserves `native_hybrid_task_v1` with task-token binding and native DeltaNet beta weighting. Advanced callers must derive and bind the exact question-token mask before generation; the simple public launcher intentionally exposes only the QK recipe. Random-weight CPU tests exercise the native 32-layer architecture, cache lengths, budget accounting, cross-image resets, and position preservation. They do not establish 9B benchmark accuracy or GPU performance. The optional historical-source parity test is skipped unless `QWEN35_LEGACY_CORE` is supplied.
