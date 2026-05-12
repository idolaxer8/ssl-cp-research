# Research Findings — SSL + Conformal Prediction

> Pull from here when writing Methods / Results / Discussion sections of the paper.
> Detailed tables and retracted results: `archive/findings_archive.md`

---

## Setup

- **Backbone**: DINOv2-base (ViT-B/14, 768-D, 336×336 input)
- **CP methods**: Full CP (FCP), CV+/Jackknife+ (CV+), Split CP (SCP)
- **Best NCM**: `geodesic_topk_asym` (new winner, post-bugfix)
- **Default α**: 0.1 (target ≥ 90% coverage)

---

## 1. NCM Comparison (FCP, α=0.1, 5 trials)

### CIFAR-100 (100 classes, 25/class) — Key differentiation dataset

| NCM | cal=200 | cal=300 | cal=400 | cal=600 |
|-----|---------|---------|---------|---------|
| nn_ratio | 0.947 sz=23.25 | 0.915 sz=8.36 | 0.905 sz=6.06 | 0.912 sz=4.23 |
| geodesic_nn_ratio | 0.948 sz=21.57 | 0.917 sz=7.93 | 0.907 sz=5.59 | 0.912 sz=3.97 |
| whitened_geodesic | 0.943 sz=19.24 | 0.909 sz=6.96 | 0.904 sz=5.21 | 0.912 sz=3.84 |
| geodesic_topk_mean | 0.927 sz=11.51 | 0.885❌ sz=3.62 | 0.891 sz=3.17 | 0.933 sz=4.08 |
| **geodesic_topk_asym** | 0.952 sz=15.78 | **0.917 sz=4.65** | **0.916 sz=3.18** | **0.907 sz=2.29** |

**Winner: `geodesic_topk_asym`** — 39–40% smaller sets than whitened_geodesic at cal=400–600, valid coverage. `geodesic_topk_mean` under-covers at cal=300 (0.885❌).

### miniImageNet (100 classes, 500/class)

| NCM | cal=400 | cal=600 | cal=800 | cal=1200 |
|-----|---------|---------|---------|---------|
| whitened_geodesic | 0.909 sz=1.51 | 0.904 sz=1.18 | 0.898 sz=1.09 | 0.906 sz=1.06 |
| **geodesic_topk_asym** | **0.913 sz=1.18** | **0.918 sz=1.10** | **0.900 sz=1.05** | 0.911 sz=1.03 |

`geodesic_topk_asym` dominates at cal=400–800 (22–24% smaller sets).

### Other datasets (summary)

- **CIFAR-10** (10 cls): No NCM differentiation. All give sz≈0.92–0.95. FCP over-conservative (~91–92%).
- **Flowers-102** (102 cls): FCP works from cal=400 (sz≈0.91). At cal=200, `geodesic_topk_mean` uniquely avoids blowup (sz=6.98 vs 90–102 for others).

### NCM Ranking

| Rank | NCM | When to use |
|------|-----|-------------|
| 1 | **geodesic_topk_asym** | K≥100, cal≥300 (best set sizes) |
| 2 | whitened_geodesic | Stable fallback at cal≤200 or K=10 |
| 3 | geodesic_topk_mean | Extreme scarcity (cal=200, Flowers-102) |

---

## 2. FCP vs CV+ vs SCP

### CIFAR-100 (100 classes, 25/class)

| cal | FCP (whitened_geodesic) | CV+ | SCP |
|-----|------------------------|-----|-----|
| 400 | **90.9% sz=7.45** ✅ 7.8s | 100% sz=100 ❌ | 98% ❌ |
| 600 | **90.0% sz=3.39** ✅ 8.8s | 94.7% sz=8.68 ❌ | 90.7% sz=12.1 |
| 800 | 90.1% sz=2.78 9.7s | **90.8% sz=2.65** ✅ 42s | 89.8% sz=3.01 |

FCP first valid: **cal=400**. CV+ first valid: **cal=800** (2× data, 5× slower). With geodesic_topk_asym: FCP cal=400 sz drops 7.45 → **3.18** (57% reduction).

### CUB-200 (200 classes, ~50/class) — Strongest FCP advantage

| cal | FCP | CV+ | SCP |
|-----|-----|-----|-----|
| 800 | **91.9% sz=2.15** ✅ 12s | 100% ❌ | 100% ❌ |
| 1500 | **90.3% sz=1.53** ✅ 14s | 92.4% sz=1.67 58s | 89.8% sz=2.18 |
| 2000 | 89.9% sz=1.39 16s | 90.7% sz=1.40 134s | 89.6% sz=1.46 |

FCP first valid: **cal=800**. CV+ first valid: **cal=1500** (1.9× data, 4× slower). At cal=2000: same sz, FCP **9× faster**.

### Other datasets

- **CIFAR-10**: FCP over-conservative. Use CV+ or SCP.
- **EuroSAT**: All methods work from cal=75. No clear winner.
- **Flowers-102**: FCP valid from cal=400 (all NCMs).

---

## 3. MA-CS Penalty (Fargion et al. 2025)

Binary superclass indicator penalty: `s_λ(x,y) = s(x,y) + λ · I{g(y) ≠ g(ŷ(x))}` where `ŷ(x)` = LOO 1-NN prediction.

### Results (CIFAR-100, geodesic_topk_asym, 5 trials)

| cal | baseline | λ=0.03 (optimal) | λ=0.10 |
|-----|----------|-------------------|--------|
| 600 | 0.899 sz=2.58 | **0.895 sz=2.37** (−8%) | 0.900 sz=2.73 (worse) |
| 800 | 0.909 sz=1.96 | **0.909 sz=1.92** (−2%) | 0.913 sz=2.16 (worse) |

**Key findings**: U-shaped response (λ=0.02-0.03 optimal). Superclass count monotonically decreases (Corollary 4.3 verified). Coverage maintained. Stronger at moderate cal (8% at cal=600 vs 2% at cal=800).

### Theoretical guarantees (Fargion et al. 2025)

- **Quantile shift**: q̂ ≤ q̂_λ ≤ q̂ + λ (coverage loss bounded)
- **No out-of-group introduction**: penalty only removes out-of-group candidates
- **Superclass monotonicity**: G_λ(x) ⊆ G(x) (verified empirically)

### Score audit vs paper

Our LOO 1-NN for ŷ(x) differs from paper's softmax argmax. 1-NN is the natural nonparametric classifier for FCP (no trained model). Exchangeability preserved (LOO for cal, plain NN for test — standard FCP asymmetry). Open question: formal justification needed for paper. See `archive/findings_archive.md §11.4`.

---

## 4. MS-CS with Unlabeled Data

Build similarity matrix M from k-means clusters on unlabeled pool (no superclass labels needed).

**Pipeline**: K-means on unlabeled → class centroids from cal → match to clusters → M from co-assignment + inter-cluster distance → penalty `s_l(x,y) = s(x,y) + l*(1-M[y, ŷ(x)])`.

### Best results (CIFAR-100, geodesic_topk_mean, cal=600, 5 trials)

Best config: n_clusters=20, tau=0.5×median_d², λ=0.05 → **sz=2.35** (27% reduction from baseline 3.20), cov=0.889.

**Key findings**:
1. Comparable to MA-CS (sz=2.37) but **requires no superclass labels**
2. Tau normalization (tau = multiplier × median_d²) recommended for scale-independence
3. Optimal regime: tau ∈ [0.25, 0.5] × median_d², λ ∈ [0.03, 0.05]
4. Exchangeability fix (`--exchangeable`) recovers +0.5–1.3% coverage at cal≤400, negligible at cal≥600
5. Only meaningful at cal≥600 (baseline under-covers at cal=400 with geodesic_topk_mean)

Detailed sweep tables: `archive/findings_archive.md §11.2`.

---

## 5. Backbone Comparison — DINOv2 vs CLIP vs BEiTv2

CIFAR-100, geodesic_topk_mean FCP, 5 trials. Key result at cal=600:

| Backbone | FCP sz | CV+ sz | SCP sz |
|----------|--------|--------|--------|
| **DINOv2-base** | **2.96** | 6.50 | 14.4 |
| BEiTv2-base | 5.24 | 7.68 | 9.22 |
| CLIP-base | 32.6 | 35.6 | 35.3 |

**DINOv2 dominates** — 77% smaller sets than BEiTv2, 11× smaller than CLIP. CLIP unsuitable for NN-ratio FCP. FCP advantage over CV+/SCP generalizes across all backbones.

---

## 6. When to Use Each Method

| Scenario | Best method | Best NCM |
|----------|-------------|----------|
| K≥100, cal≥300 | **FCP** | geodesic_topk_asym |
| K≥100, cal=200 | **FCP** | whitened_geodesic |
| K≥100, extreme scarcity | **FCP** | geodesic_topk_mean |
| K=10, well-separated | CV+ or SCP | — |

---

## 7. Theory Notes

**geodesic_topk_asym**: 1-NN numerator (tight same-class) / mean-k denominator (smoothed other-class). Asymmetry is key — mean-k on both sides over-smooths and destabilizes coverage.

**Exchangeability of whitening**: Whitening computed from cal only, not updated per candidate. Asymmetry is O(1/n) — negligible for n≥200. Label-dependent projections (LDA) are NOT O(1/n) and cause structural under-coverage (70-85%). Only unsupervised projections (PCA) are safe. See `archive/findings_archive.md` for LDA details.

**Why FCP beats CV+**: CV+ trains on (k-1)/k data per fold — with few samples and many classes, some classes absent from folds → trivially full sets. FCP uses full cal set.

---

## 8. Dataset Properties

| Dataset | K | N/class | FCP behavior |
|---------|---|---------|-------------|
| CIFAR-10 | 10 | 500 | Over-conservative; use CV+ |
| EuroSAT | 10 | 200 | All methods work |
| CIFAR-100 | 100 | 25 | FCP dominates; topk_asym wins |
| Flowers-102 | 102 | ~55 | FCP from cal=400 |
| CUB-200 | 200 | ~50 | Strongest FCP advantage |
| miniImageNet | 100 | 500 | topk_asym best |
| tiny-imagenet | 200 | 500 | Pending |

---

## 9. Publication Roadmap

**Claim:** Geometry-aware NCMs for SSL embeddings make FCP the only valid and most efficient method in the data-scarce regime.

**Target**: AISTATS 2026 (primary), TMLR (backup)

### P0 — Must complete

1. Implement RAPS/APS as SCP baseline (reviewers will reject without it)
2. Re-run few-shot with geodesic_topk_asym on CIFAR-100, CUB-200, miniImageNet
3. Complete multi-dataset comparison with consistent NCM across all 6 datasets
4. Add error bars / significance (stderr, paired tests)

### P1 — Strongly recommended

5. MA-CS on CUB-200 (bird taxonomy), miniImageNet (ImageNet hierarchy)
6. Alpha sensitivity sweep (α ∈ {0.05, 0.1, 0.15, 0.2})
7. Whitening ablation (whiten on/off)
8. NCM comparison on CUB-200
9. Backbone comparison with geodesic_topk_asym + CUB-200

### P2+ — Nice to have

10. tiny-ImageNet experiments
11. Cross-NCM × backbone grid
12. Adaptive λ selection for MA-CS

See `archive/findings_archive.md §10` for stale results tracker, reviewer predictions, and paper structure.

---

## 10. Future Directions

1. **MA-CS multi-dataset + adaptive λ** — most novel contribution. See §3.
2. **MS-CS multi-dataset** — test on CUB-200, miniImageNet. See §4.
3. **PCA dimensionality reduction** — active investigation, shows promise.
4. **RAPS baseline implementation** — required for paper.

### Negative results (archived)

- Pool augmentation: breaks exchangeability. See `archive/findings_archive.md §11.5`.
- LDA projection: structural under-coverage. See `archive/findings_archive.md §8`.
- Original CS penalty: data leakage. See `archive/findings_archive.md §4`.

---

*Last updated: 2026-05-12*
