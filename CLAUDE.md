# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Response formatting (IMPORTANT)

- **Math must be plain text, never LaTeX.** The user's terminal does NOT render LaTeX/MathJax — `$...$`, `\(...\)`, `\[...\]`, `\frac`, `\sum`, `\hat`, etc. show up as raw unreadable source. Write all equations in plain ASCII/Unicode instead.
  - Use spelled-out or Unicode operators: `sum_over_c`, `prod`, `sqrt(x)`, `<=`, `>=`, `~=`, `->`.
  - Subscripts/superscripts with `_` and `^`: `F_c`, `q_hat`, `x^2`, `Fbar^{-1}`.
  - Greek letters spelled out (`alpha`, `lambda`, `tau`) or Unicode (α, λ, τ) — not `\alpha`.
  - Use fenced code blocks for multi-line derivations so spacing is preserved.
  - Define every symbol inline in a `where ...` clause.

## Project Overview

SSL + Conformal Prediction research: applying Full Conformal Prediction on top of Self-Supervised Learning (SSL) image embeddings for uncertainty quantification. Core pipeline: extract embeddings -> run conformal prediction -> analyze results.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, timm (for SSL models), scikit-learn, matplotlib, tqdm, Pillow.

## Common Commands

### Download datasets (ImageFolder format)
```bash
python src/download_datasets.py --dataset cifar100 --output_dir data/cifar100 --num_per_class 25
# Supported: cifar10, cifar100, stl10, eurosat
```

### Extract SSL embeddings
```bash
python src/extract_features.py --data_dir data/cifar10 --output_name embeddings_cifar10.pt --model dinov2-base
# Model presets: dinov2-base/large/giant, clip-base/large, beit-base, beitv2-base, mae-base/large, ssl-resnet50, swsl-resnet50
# For large datasets on 4GB GPU: use --input_size 336 --batch_size 32
```

### Run conformal prediction experiment
```bash
python src/run_conformal_experiment.py --embeddings_path output/embeddings.pt --alpha 0.1 --ncm geodesic_topk_mean --k 5

# Compare CP methods (Full CP vs Split CP vs CV+)
python src/run_conformal_experiment.py --embeddings_path output/embeddings.pt --compare_cp_methods --n_trials 3
```

## Architecture

### Data flow
1. `data/` -- Raw images in ImageFolder structure (`class_name/image.jpg`)
2. `src/extract_features.py` -- SSL model via `timm`, saves `output/<name>.pt` as `{"embeddings": Tensor, "labels": Tensor}`
3. `src/conformal_prediction.py` -- Core library; loaded as local module via `sys.path.insert(0, "src")`
4. `src/run_conformal_experiment.py` -- Orchestrates CP experiments and multi-method comparisons
5. `output/` -- All results: `.pt` files (data), `.png` files (plots)

### `conformal_prediction.py` -- key components

**Nonconformity Measures** (all subclass `NonconformityMeasure`):
- `MahalNNRatio` -- Mahalanobis-whitened NN ratio (baseline)
- `WhitenedGeodesicNNRatio` -- whitened geodesic NN ratio
- `GeodesicTopKMeanNCM` -- top-k averaged geodesic ratio (symmetric/asymmetric variants, main NCM)
- `SoftmaxNonconformity` -- softmax head for split CP baseline

**Predictors**:
- `FullConformalPredictor` -- Full (transductive) CP
- `SoftmaxSplitCP` -- Inductive/split CP with softmax classifier (THR score)
- `SemiCP` -- Semi-supervised CP with NNM augmentation (THR/APS/RAPS scores; Zhou et al. 2025)
- `CrossValidationPlusPredictor` -- CV+/Jackknife+

**Score functions** (used by `SemiCP`):
- `compute_cp_scores(probs, y_indices, score_fn)` -- batch score computation
- `compute_cp_sets(probs, q_hat, score_fn)` -- prediction set construction
- Score types: `THR` (1 - p(y)), `APS` (cumulative sorted probs), `RAPS` (APS + rank penalty)

**Utilities**:
- `create_ncm(ncm_type, k)` -- factory function
- `cal_test_split` / `stratified_cal_test_split` -- dataset splitting

### NCM choices (`--ncm` argument)
`mahal_nn_ratio`, `whitened_geodesic`, `geodesic_topk_mean`, `geodesic_topk_asym`, `softmax`

**Best NCM**: `geodesic_topk_asym` (asymmetric: 1-NN numerator / mean-k denominator). Use `geodesic_topk_mean` for theoretical simplicity.

### Exchangeability policy (DEFAULT = exact guarantee)
The fully-exchangeable path is the default; any variant that breaks FCP's
exchangeability — even at O(1/n) — must be explicitly approved and emits a
validity warning otherwise. Soft gate (warn-only): the variant still runs and is
numerically unchanged, it just warns.
- **Exact (default):** fit every data-dependent transform (whitening / PCA /
  k-means / RBF bandwidth) on an INDEPENDENT unlabeled pool via
  `exchangeable_features.UnlabeledTransform`, and use an unwhitened NCM
  (`unwhitened_topk_mean` / `unwhitened_topk_asym`). `GeodesicTopKMeanNCM` now
  defaults to `whiten=False`. See `src/exchangeable_fcp_experiment.py`.
- **Non-exchangeable (must approve):** cal-fit whitening (`whiten=True`, and
  always-on for `mahal_nn_ratio` / `whitened_geodesic`), cal-fit RBF bandwidth
  (`rbf_density` with `sigma=None`), cal-fit PCA/AE (`pca_experiment.py
  --pca_source cal|pool`), and the MS-CS fixed-ŷ/M penalty
  (`run_fcp_with_mscs(exchangeable=False)`). Pass `allow_nonexchangeable=True`
  (to `create_ncm`, the NCM ctor, or `run_fcp_with_mscs`) to approve + silence.
- Helper: `conformal_prediction.warn_nonexchangeable(...)` +
  `ExchangeabilityWarning`. The named whitened NCMs (`geodesic_topk_*`,
  `whitened_geodesic`, `mahal_nn_ratio`, `rbf_density`) keep their historical
  cal-fit numeric behavior but require this approval to run warning-free.

### Active experiment scripts
- `src/macs_experiment.py` -- MA-CS binary superclass penalty (Fargion et al. 2025)
- `src/mscs_unlabeled_experiment.py` -- MS-CS with unlabeled data (k-means clustering)
- `src/semicp_experiment.py` -- SemiCP vs FCP comparison (Zhou et al. 2025)
- `src/pca_experiment.py` -- PCA dimensionality reduction
- `src/compare_ncms.py` -- NCM comparison
- `src/compare_backbones.py` -- SSL backbone comparison
- `src/run_multi_dataset_experiments.py` -- multi-dataset FCP vs CV+ vs SCP

### Archived (in `src/archive/`)
Completed/failed investigations: pool augmentation, LATA score smoothing, split audit, few-shot, old SSL comparison, rank diagnostic, old plots.

## Notes

- Run from repo root or ensure `src/` is on Python path.
- `output/` and `data/` are gitignored; `.pt` files are not tracked.
- GPU used automatically via `torch.cuda.is_available()`.
- **Stratified sampling**: All experiment scripts must use stratified (balanced) splits for cal/test/unlabeled — equal samples per class. Random splits cause class imbalance that distorts FCP results, especially with many classes (K>=100). Use `stratified_cal_test_split` from `conformal_prediction.py` or equivalent stratified logic.
