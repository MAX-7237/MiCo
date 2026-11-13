# Source layout

- [Image launcher](../LLaVA-MiCo/run_mico.py): LLaVA-1.5 and LLaVA-NeXT, 7B/13B.
- [Image stage 1](../LLaVA-MiCo/llava/model/llava_arch.py): visual-token selection before the projector.
- [Image stage 2](../LLaVA-MiCo/llava/model/language_model/modeling_llama_mico_coverage_llm.py): decoder pruning and budget calculation.
- [Video launcher](../LLaVA-Video-MiCo/run_mico.py): LLaVA-Video-7B.
- [Video selector](../LLaVA-Video-MiCo/llava/model/language_model/mico_coverage_video.py): frame-aware MiCo selection.
- [Qwen2.5/InternVL launcher](../VLMEvalKit-MiCo/run_mico.py) and [registry](../VLMEvalKit-MiCo/vlmeval/config.py).
- [Qwen3 launcher](../VLMEvalKit-MiCo-Qwen3/run_mico.py) and [registry](../VLMEvalKit-MiCo-Qwen3/vlmeval/config.py).
- [Qwen3.5 launcher](../Qwen3.5-MiCo/run_mico.py) and [controller](../Qwen3.5-MiCo/core/unified_core.py).
- [Environment setup](../setup_environment.py): pinned dependency installation and validation.
- [Evaluation guide](RUNNING.md): data formats, budgets and launch commands.

Pretrained weights, benchmark data, generated predictions and private experiment records are external to this repository. Third-party framework code and licenses are retained where required by the supported inference/evaluation paths.
