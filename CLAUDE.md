# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSL + Conformal Prediction research: applying Full Conformal Prediction on top of Self-Supervised Learning (SSL) image embeddings for uncertainty quantification. Core pipeline: extract embeddings → run conformal prediction → analyze results.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, timm (for SSL models), scikit-learn, matplotlib, tqdm, Pillow.

## Common Commands

### Download datasets (ImageFolder format)
```bash
python src/download_datasets.py --dataset cifar10 --output_dir data/cifar10 --num_per_class 200
python src/download_datasets.py --dataset eurosat --output_dir data/eurosat --num_per_class 200
# Supported: cifar10, cifar100, stl10, eurosat
```

### Extract SSL embeddings
```bash
python src/extract_features.py --data_dir data/cifar10 --output_name embeddings_cifar10.pt --model dinov2-base
# Model presets: dinov2-base/large/giant, clip-base/large, beit-base, beitv2-base, mae-base/large, ssl-resnet50, swsl-resnet50
# Or use a full timm model name as --model
# For large datasets on 4GB GPU: use --input_size 336 --batch_size 32 (5.4h per 50k images)
# Warning: 518x518 at batch_size=32 saturates 4GB GPU (~600s/batch); use 336x336 or batch_size=8
```

### Run conformal prediction experiment
```bash
python src/run_conformal_experiment.py --embeddings_path output/embeddings.pt --alpha 0.1 --ncm knn --k 5

# Compare CP methods (Full CP vs Split CP vs CV+)
python src/run_conformal_experiment.py --embeddings_path output/embeddings.pt --compare_cp_methods --n_trials 3

# Compare across k values
python src/run_conformal_experiment.py --embeddings_path output/embeddings.pt --compare_k
```

### Compare SSL models
```bash
python src/compare_ssl_models.py --data_dir data/cifar100 --num_per_class 100 --output_dir output/ssl_comparison
python src/compare_ssl_models.py --data_dir data/cifar100 --skip_extraction --output_dir output/ssl_comparison
```

### Visualize embeddings
```bash
python src/visualize_embeddings.py --input_path output/embeddings.pt --method tsne
```

## Architecture

### Data flow
1. `data/` — Raw images in `torchvision.datasets.ImageFolder` structure (`class_name/image.jpg`)
2. `src/extract_features.py` — Loads a pretrained SSL model via `timm`, extracts embeddings, saves `output/<name>.pt` as `{"embeddings": Tensor, "labels": Tensor}`
3. `src/conformal_prediction.py` — Core library; loaded as a local module by other scripts via `sys.path.insert(0, "src")`
4. `src/run_conformal_experiment.py` — Orchestrates CP experiments, plotting, and multi-method comparisons
5. `output/` — All results: `.pt` files (data), `.png` files (plots)

### `conformal_prediction.py` — key components

**Nonconformity Measures** (all subclass `NonconformityMeasure`):
- `KNNNonconformity` — min-distance ratio (same/other class), supports Full CP updates
- `SimplifiedKNNNonconformity` — sum of k-NN distances to same class (faster, Cherubin et al. 2021)
- `CentroidNonconformity` / `RelativeCentroidNonconformity` — centroid-based
- `RidgeRegressionNonconformity` — LS-SVM/ridge regression NCM (best results on EuroSAT)
- `FastNNRatio` / `HypersphericalNNRatio` — ratio-based NCMs
- `SoftmaxNonconformity` — trains a softmax head; used for split CP baseline

**Predictors**:
- `FullConformalPredictor` — Full (transductive) CP; for each test point and candidate label, augments calibration set and computes p-value
- `SoftmaxSplitCP` — Inductive/split CP with a softmax classifier trained on training split
- `CrossValidationPlusPredictor` — CV+/Jackknife+ style conformal predictor

**Utilities**:
- `create_ncm(ncm_type, k, lambda_reg)` — factory function mapping NCM name string to class
- `cal_test_split` / `train_cal_test_split` — stratified dataset splitting
### NCM choices (`--ncm` argument)
`knn`, `simplified_knn`, `centroid`, `relative_centroid`, `ridge`, `nn_ratio`, `geodesic_nn_ratio`, `geodesic_topk_mean`, `geodesic_topk_asym`, `softmax`

`geodesic_topk_mean` (symmetric: top-k on both numerator and denominator) is the current default NCM for FCP experiments. Easier to explain theoretically than the asymmetric variant.

## Notes

- Scripts in `src/` import `conformal_prediction` as a local module — run from the repo root or ensure `src/` is on the Python path.
- `output/` and `data/` are gitignored; `.pt` files are not tracked.
- GPU is used automatically if available via `torch.cuda.is_available()`.
- Default NCM for all new experiments: `geodesic_topk_mean` (symmetric top-k geodesic ratio — best balance of performance and explainability).
