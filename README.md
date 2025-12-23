# SSL + Conformal Prediction Research

Combining Self-Supervised Learning (SSL) representations with Full Conformal Prediction for reliable uncertainty quantification.

## Overview

This project implements **Full Conformal Prediction** on top of **DINOv2** self-supervised embeddings, following:
- **SSL**: DINOv2 (Oquab et al., 2023) - https://arxiv.org/pdf/2304.12210
- **Conformal Prediction**: Cherubin et al., 2021 - https://proceedings.mlr.press/v139/cherubin21a/cherubin21a.pdf

## Project Structure

```
ssl-cp-research/
├── data/                    # Your image datasets (classA/, classB/, etc.)
├── output/                  # Generated embeddings and results
├── src/
│   ├── extract_features.py           # Extract DINOv2 embeddings
│   ├── visualize_embeddings.py       # Visualize embeddings (t-SNE/PCA)
│   ├── conformal_prediction.py       # Full CP implementation
│   └── run_conformal_experiment.py   # Main experiment script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Extract SSL Features

Extract DINOv2 embeddings from your image dataset:

```bash
python src/extract_features.py \
    --data_dir data/ \
    --output_name embeddings.pt \
    --model_name vit_base_patch14_dinov2.lvd142m \
    --batch_size 32
```

This saves embeddings to `output/embeddings.pt`.

### Step 2: (Optional) Visualize Embeddings

```bash
python src/visualize_embeddings.py \
    --input_path output/embeddings.pt \
    --method tsne
```

### Step 3: Run Conformal Prediction

Run Full CP with k-NN nonconformity measure:

```bash
python src/run_conformal_experiment.py \
    --embeddings_path output/embeddings.pt \
    --alpha 0.1 \
    --ncm simplified_knn \
    --k 5 \
    --cal_ratio 0.5 \
    --save_predictions
```

**Key Arguments:**
- `--alpha`: Significance level (e.g., 0.1 = 90% coverage guarantee)
- `--ncm`: Nonconformity measure (`knn` or `simplified_knn` - simplified is faster)
- `--k`: Number of neighbors for k-NN
- `--cal_ratio`: Calibration set size (default: 0.5, rest goes to test)

## Outputs

After running the experiment, you'll get:

1. **Metrics** (printed to console):
   - Coverage (should be ≥ 1-α)
   - Average prediction set size
   - Singleton rate and accuracy

2. **Visualizations** (`output/`):
   - `set_size_distribution.png`: Distribution of prediction set sizes
   - `coverage_vs_alpha.png`: Coverage and efficiency across α values
   - `per_class_analysis.png`: Per-class coverage and set sizes

3. **Saved Results** (`output/`):
   - `cp_results.pt`: Metrics and per-class statistics
   - `predictions.pt`: Prediction sets and p-values (if `--save_predictions`)

## Key Concepts

### Full Conformal Prediction

For each test example `x` and candidate label `y`:
1. Compute nonconformity score assuming label `y`
2. Compare with calibration scores
3. Compute p-value: `P(y|x) = (# cal scores ≥ test score) / (# cal examples)`
4. Include `y` in prediction set if `p-value > α`

**Guarantee**: With probability ≥ 1-α, the prediction set contains the true label.

### Nonconformity Measures

1. **k-NN**: Average distance to k nearest neighbors of the same class
   - Lower distance = more conforming = higher confidence

2. **Inverse k-NN**: Inverse distance to other classes
   - Closer to other classes = less conforming


## References

1. Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision"
2. Cherubin, G., et al. (2021). "Exact and Approximate Conformal Inference for Multi-Output Regression"
3. Vovk, V., et al. (2005). "Algorithmic Learning in a Random World"

## License

MIT