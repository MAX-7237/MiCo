<div align="center">

<img src="assets/mascot.png" width="96" alt="MiCo mascot" />

# MiCo

### Mutual Information Coverage Optimization through Semantic Erasure Modeling<br>for Efficient MLLM Inference

**Training-free visual token pruning · Image & video understanding · 7B–13B models**

[Paper](https://arxiv.org/abs/2609.34330) · [Method](#method) · [Results](#results) · [Quick start](#quick-start) · [Evaluation guide](docs/RUNNING.md)

</div>

---

MiCo selects a compact set of visual evidence through **two stages of mutual information coverage optimization**: a representative visual candidate pool before the language model, followed by task-aware refinement inside it.

<table>
<tr>
<th align="center">94.4% visual tokens pruned</th>
<th align="center">97.5% relative performance</th>
<th align="center">2.31× video speedup</th>
</tr>
<tr>
<td align="center">LLaVA-NeXT-13B · T = 160</td>
<td align="center">NeXT-13B · ten-benchmark aggregate</td>
<td align="center">LLaVA-Video-7B · T = 16/frame</td>
</tr>
</table>

<p align="center">
<img src="assets/teaser-process.png" width="30%" alt="Original paper illustration of MiCo's two-stage token selection" />
<img src="assets/teaser-results.png" width="34%" alt="Original cross-model relative-performance radar from the MiCo paper" />
<img src="assets/teaser-efficiency.png" width="29%" alt="Original NeXT-13B POPE quality versus one-token speedup plot" />
</p>

<p align="center"><sub>Original paper teaser panels: visual evidence selection, cross-model performance, and the quality–speed trade-off.</sub></p>

> Results shown here are reported in the paper. Repository verification and paper benchmark reproduction are tracked separately in [Validation](#validation).

## Method

<a href="assets/method.pdf"><img src="assets/method.png" width="100%" alt="MiCo's original framework figure from the LaTeX project" /></a>

1. **Visual candidate selection.** Use feature magnitude, visual attention, and coverage to retain representative evidence before LLM processing.
2. **Task-aware refinement.** Use text-to-visual attention at the selected decoder layer to refine that candidate pool under a token budget.

Both stages greedily optimize an observable weighted coverage objective. The paper establishes monotonicity and submodularity for this surrogate. The stage-2 budget is chosen to match the target layer-average token count.

## Results

### Across model families

<a href="assets/results-table.pdf"><img src="assets/results-table.png" width="100%" alt="Original MiCo cross-model results table, compiled from the paper LaTeX" /></a>

The original table reports relative performance against each model's unpruned reference. It covers LLaVA, Qwen2.5-VL, Qwen3-VL, Qwen3.5, and InternVL3 at the paper's token budgets.

### Quality and inference efficiency

<a href="assets/efficiency.pdf"><img src="assets/efficiency.png" width="100%" alt="Original MiCo performance-efficiency plots and latency breakdown" /></a>

**Image and video protocols differ:** NeXT plots use one-token latency; video plots use complete-answer latency. See the paper for the benchmark and generation protocol behind each point.

<details>
<summary><strong>View the original LLaVA-NeXT-13B results table</strong></summary>

<a href="assets/next13b-table.pdf"><img src="assets/next13b-table.png" width="100%" alt="Original ten-benchmark MiCo results on LLaVA-NeXT-13B" /></a>

</details>

## Supported models

| Model family | Implementation | Environment |
| :--- | :--- | :--- |
| LLaVA-1.5 / LLaVA-NeXT · 7B & 13B | [LLaVA-MiCo](LLaVA-MiCo) | `fastv` |
| LLaVA-Video · 7B | [LLaVA-Video-MiCo](LLaVA-Video-MiCo) | `llavavid` |
| Qwen2.5-VL · 7B / InternVL3 · 8B | [VLMEvalKit-MiCo](VLMEvalKit-MiCo) | `vlmevalkit-qwen2.5vl` |
| Qwen3-VL · 8B | [VLMEvalKit-MiCo-Qwen3](VLMEvalKit-MiCo-Qwen3) | `vlmevalkit-qwen3vl` |
| Qwen3.5 · 9B | [Qwen3.5-MiCo](Qwen3.5-MiCo) | `vlmevalkit-qwen3vl` |

## Quick start

Each component has its own pinned dependency profile. Start with the [environment guide](docs/RUNNING.md#environments), then use the public MiCo launcher:

```bash
git clone https://github.com/MAX-7237/MiCo.git
cd MiCo/VLMEvalKit-MiCo
conda activate vlmevalkit-qwen2.5vl

python run_mico.py --model qwen2.5-vl-7b --tokens 128 \
  --dataset AI2D_TEST --work-dir outputs/mico --dry-run
```

Remove `--dry-run` after preparing the environment, weights, and dataset. See the [full evaluation guide](docs/RUNNING.md#evaluation) for image, video, and model-specific commands; Qwen3.5 has a [dedicated single-image entry](Qwen3.5-MiCo/README.md).

## Validation

This is a **source-preview release**. Static checks and launcher dry runs pass; the recovered Qwen3.5 core also passes seven random-weight CPU model tests. Full clean-environment installation and real-weight GPU benchmark reproduction remain pending. See [validation records](VALIDATION.json) and [Qwen3.5 checks](Qwen3.5-MiCo/VALIDATION.json).

The README uses original figures and LaTeX tables from the paper project. PDF images are rasterized for GitHub display; click them for the vector originals. [Asset provenance](assets/SOURCES.json) records source files and hashes.

## Citation

```bibtex
@article{wang2026mico,
  title={MiCo: Mutual Information Coverage Optimization through Semantic Erasure Modeling for Efficient MLLM Inference},
  author={Wang, Tinghao and Guo, Yichen and others},
  journal={arXiv preprint arXiv:2609.34330},
  year={2026}
}
```

## Acknowledgements

Built on LLaVA, LLaVA-NeXT/LLaVA-Video, VLMEvalKit, lmms-eval, and Transformers. Component licenses and original third-party attribution are preserved in their respective directories.
