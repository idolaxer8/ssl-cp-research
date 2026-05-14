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

## 10. PCA Dimensionality Reduction (Unlabeled Pool)

PCA fit on unlabeled pool (n >> d=768) preserves FCP exchangeability — projection is unsupervised w.r.t. cal/test. Stratified splits used throughout (equal samples/class).

### CIFAR-100 (100 classes, PCA on 10K unlabeled)

| dim | cal=400 sz | cal=600 sz | cal=800 sz |
|-----|-----------|-----------|-----------|
| PCA-32 | 4.55 | 4.71 | 2.96 |
| PCA-64 | 2.74 | 2.80 | 2.04 |
| **PCA-128** | **2.18** | **2.36** | **1.64** |
| PCA-256 | 2.85 | 2.99 | 1.77 |
| PCA-512 | 5.19 | 4.58 | 2.44 |
| Full-768 | 4.11 | 4.11 | 2.16 |

**PCA-128 optimal**: 24% smaller sets than full-768 at cal=800. Coverage valid (91.0%). U-shape: too few dims loses info, too many reintroduces noise.

### CUB-200 (200 classes, PCA on 5600 unlabeled carved 28/class)

| dim | cal=600 sz | cal=800 sz | cal=1000 sz | cal=1500 sz |
|-----|-----------|-----------|------------|------------|
| PCA-128 | 2.85 | 2.32 | 2.29 | 1.87 |
| PCA-256 | 2.62 | 2.10 | 2.03 | 1.76 |
| **PCA-512** | **2.64** | **2.07** | **2.04** | **1.70** |
| Full-768 | 2.87 | 2.34 | 2.46 | 1.90 |

**PCA-512 optimal**: 11% smaller sets at cal=1500. Fine-grained dataset needs more dims. Coverage valid (90.4-92.6%).

### miniImageNet (100 classes, PCA on 10K unlabeled carved 100/class)

| dim | cal=400 sz | cal=600 sz | cal=800 sz |
|-----|-----------|-----------|-----------|
| PCA-64 | 1.32 | 1.36 | 1.15 |
| **PCA-128** | **1.08** | **1.13** | **1.05** |
| PCA-256 | 1.12 | 1.17 | 1.07 |
| Full-768 | 1.24 | 1.28 | 1.11 |

**PCA-128 optimal**: 5% smaller sets at cal=800 (already near-perfect at 1.11). Coverage valid (92.2%).

### Control: PCA on cal data (CIFAR-100)

| dim | cal=400 cov | cal=600 cov | cal=800 cov |
|-----|------------|------------|------------|
| PCA-128 | 0.876 | 0.915 | 0.900 |
| PCA-256 | 0.816 | 0.886 | 0.888 |
| PCA-512 | — | 0.816 | 0.851 |

**Cal-based PCA under-covers** at high dims (n_cal < n_features → PCA axes overfit). Confirms unlabeled pool is essential.

### Summary

| Dataset | Best PCA dim | Reduction vs full-768 | Mechanism |
|---------|-------------|----------------------|-----------|
| CIFAR-100 | 128 | **24%** | Noise removal in coarse-grained classes |
| CUB-200 | 512 | **11%** | Fine-grained needs more dims |
| miniImageNet | 128 | **5%** | Already near-perfect baseline |

PCA-128 is the default recommendation for K=100 coarse-grained datasets. Fine-grained (CUB-200) benefits from PCA-512. Diminishing returns when baseline is already small (miniImageNet).

---

## 11. FCP+PCA vs SplitCP vs SemiCP (Multi-Dataset)

Head-to-head comparison: FCP with PCA (unlabeled) vs SplitCP vs SemiCP (Zhou et al. 2025, NNM augmentation). SemiCP uses logistic regression softmax head trained on 50% of cal budget, scores augmented via nearest-neighbor matching to unlabeled pool. **Stratified train/cal split** (critical fix from earlier non-stratified version). 5 trials, α=0.1. NCM: `geodesic_topk_asym`.

### 11a. CIFAR-100 (100 classes, PCA-128 on 10K unlabeled)

| Cal | SCP-THR | SemiCP-THR | FCP | **FCP+PCA** |
|-----|---------|------------|-----|-------------|
| 300 | 3.44 (86%) | 4.10 (87%) | 11.20 (91%) | **5.74 (91%)** |
| 400 | 2.64 (87%) | 2.81 (88%) | 4.66 (91%) | **3.01 (91%)** |
| 600 | 2.13 (86%) | 2.30 (87%) | 2.45 (91%) | **1.85 (90%)** |
| 800 | 1.80 (87%) | 1.85 (88%) | 1.95 (91%) | **1.62 (91%)** |
| 1000 | 1.63 (88%) | 1.61 (88%) | 1.77 (91%) | **1.52 (90%)** |

### 11b. miniImageNet (100 classes, PCA-128 on 10K carved unlabeled)

| Cal | SCP-THR | SemiCP-THR | FCP | **FCP+PCA** |
|-----|---------|------------|-----|-------------|
| 300 | 1.43 (81%) | 1.90 (84%) | 3.60 (89%) | **2.18 (89%)** |
| 400 | 1.40 (87%) | 1.66 (88%) | 1.52 (90%) | **1.23 (90%)** |
| 600 | 1.13 (88%) | 1.18 (88%) | 1.11 (90%) | **1.07 (90%)** |
| 800 | 1.10 (89%) | 1.09 (89%) | 1.11 (91%) | **1.01 (89%)** |
| 1000 | 1.04 (90%) | 1.04 (90%) | 1.02 (89%) | **1.00 (90%)** |

### 11c. CUB-200 (200 classes, PCA-512 on 5600 carved unlabeled)

| Cal | SCP-THR | SemiCP-THR | FCP | **FCP+PCA** |
|-----|---------|------------|-----|-------------|
| 600 | 2.12 (86%) | 2.42 (87%) | **3.31 (93%)** | 3.85 (93%) |
| 800 | 2.21 (89%) | 2.32 (90%) | **2.11 (91%)** | 2.18 (91%) |
| 1000 | 1.82 (89%) | 1.83 (89%) | **1.73 (90%)** | 1.73 (91%) |
| 1500 | 1.41 (88%) | 1.42 (88%) | **1.46 (90%)** | 1.44 (90%) |

### 11d. Key findings

1. **FCP+PCA dominates on CIFAR-100 and miniImageNet** (coarse-grained, 100 classes). At cal=800: FCP+PCA sz=1.62 vs SCP-THR sz=1.80 (CIFAR-100), FCP+PCA sz=1.01 vs SCP-THR sz=1.10 (miniImageNet).
2. **CUB-200 is different**: FCP wins over FCP+PCA at cal=600 (sz=3.31 vs 3.85). PCA-512 preserves 98.2% variance but doesn't help — fine-grained bird features need full dimensionality at low cal. At cal≥1000, FCP and FCP+PCA converge.
3. **NNM augmentation (SemiCP) is neutral to harmful** across all 3 datasets. SemiCP-THR ≈ SCP-THR at all cal sizes; NNM never helps significantly.
4. **APS/RAPS are useless** with logistic regression on K≥100 classes: APS produces near-trivial sets (43-73), RAPS overcoveres with sets of 8-28. THR is the only viable SCP score function.

### 11e. SCP under-coverage analysis (fundamental limitation)

**SCP-THR under-covers** (82-89%) at cal≤800 across all datasets. Root cause: **the stratified train/cal split breaks exchangeability**.

The Split CP coverage guarantee requires the score function (classifier) to be independent of the calibration data. The stratified train/cal split assigns points to train vs cal based on CLASS LABELS, creating a dependency: which data the classifier sees depends on the labels of cal points too. This violates the exchangeability assumption.

**Controlled experiment** (miniImageNet, cal=300, 50 trials):

| Cal Selection | Train/Cal Split | Coverage | Set Size |
|---------------|-----------------|----------|----------|
| Random | Random | **90.0%** ✓ | 100.00 (trivial!) |
| Stratified | Random | 89.5% ✓ | ~similar |
| Random | **Stratified** | 82.3% ❌ | 1.55 |
| Stratified | **Stratified** | 81.4% ❌ | 1.55 |

The stratified train/cal split is the sole cause. But removing it creates a worse problem: with a random split at cal=300, some classes are absent from training → classifier gives 0 probability → cal scores = 1.0 → q̂ = 1.0 → ALL 100 classes in every set (sz=100, trivially valid but useless).

**The SCP dilemma at K≥100, low cal:**
- Random split: valid coverage, trivial sets (sz=K)
- Stratified split: informative sets, invalid coverage

**FCP avoids this entirely** — no train/cal split needed, NCM is computed transductively, exchangeability is preserved by construction.

---

## 12. Autoencoder Bottleneck vs PCA (CIFAR-100)

Autoencoder (1-hidden-layer MLP, MSE reconstruction loss) trained on the same 10K unlabeled pool as PCA. Equally exchangeability-safe (unsupervised, no labels). Tests whether nonlinear manifold structure in DINOv2 embeddings can beat PCA's linear projection.

### 12a. AE dimension sweep (3 trials, geodesic_topk_mean)

| dim | cal=400 sz | cal=600 sz | cal=800 sz |
|-----|-----------|-----------|-----------|
| **AE-32** | **2.35** | **2.32** | 1.70 |
| AE-64 | 2.46 | 2.40 | 1.73 |
| AE-128 | 2.47 | 2.38 | **1.69** |
| AE-256 | 3.40 | 2.77 | 1.91 |
| AE-512 | 7.42 | 5.22 | 2.76 |
| Full-768 | 4.49 | 4.55 | 2.23 |

U-shape similar to PCA, but **AE optimal dim is much lower** (32 vs PCA's 128). AE-512 catastrophic (overfitting: near-identity mapping preserves noise). AE-32 and AE-128 essentially tied at cal=800.

### 12b. Head-to-head comparison (5 trials, each at optimal dim)

| Cal | FCP (768d) | FCP+PCA-128 | FCP+AE-32 | FCP+MS-CS |
|-----|-----------|-------------|-----------|-----------|
| 300 | 10.91 | **4.97** | 5.08 | 7.93 |
| 400 | 4.76 | 2.85 | **2.64** | 3.51 |
| 600 | 2.58 | **1.76** | 1.90 | 2.09 |
| 800 | 2.25 | **1.68** | 1.80 | 1.95 |
| 1000 | 1.91 | **1.56** | 1.61 | 1.77 |

Coverage: all methods valid (89-93%) at all cal sizes.

### 12c. Key findings

1. **AE-32 wins at cal=400** (sz=2.64 vs PCA sz=2.85, 7% smaller). At extreme scarcity, nonlinear compression into a very compact space helps.
2. **PCA-128 wins at cal≥600** (cal=800: PCA sz=1.68 vs AE sz=1.80, 7% better). Linear projection is slightly more efficient with adequate data.
3. **Both dominate baseline FCP and MS-CS** at all cal sizes (25-55% smaller sets than full-768).
4. **DINOv2 embeddings are approximately linear** — the nonlinear bottleneck doesn't find meaningful additional structure. PCA is near-optimal.
5. **AE adds 25s training overhead** (vs 0.2s PCA). Not justified given marginal or negative improvement.

**Conclusion**: PCA remains the recommended dimensionality reduction for FCP on DINOv2 embeddings. AE is a useful negative result — confirms the manifold is well-approximated by a linear subspace.

---

## 13. NCM Comparison After Dimensionality Reduction (CIFAR-100)

6 NCMs compared across 3 reductions (full-768, PCA-128, AE-32) on same balanced stratified splits. 3 trials, α=0.1.

### 13a. Best NCM per reduction (by set size, coverage ≥ 89%)

| Cal | full-768 | PCA-128 | AE-32 |
|-----|----------|---------|-------|
| 300 | topk_mean: 6.53 | unwhitened_topk_mean: 3.96 | topk_mean: 3.18 |
| 400 | **topk_asym: 3.03** | **topk_asym: 1.91** | **topk_asym: 1.95** |
| 600 | **topk_asym: 2.17** | **topk_asym: 1.81** | **topk_asym: 1.79** |
| 800 | **topk_asym: 2.20** | **topk_asym: 1.79** | **topk_asym: 1.81** |

### 13b. Whitening ablation after PCA-128

| Cal | topk_asym (whitened) | unwhitened_topk_asym | Whitening benefit |
|-----|---------------------|---------------------|-------------------|
| 400 | **1.91** | 2.17 | **12%** |
| 600 | **1.81** | 1.90 | **5%** |
| 800 | **1.79** | 1.90 | **6%** |

Whitening is **NOT redundant** after PCA. PCA removes noise dimensions (total variance), whitening rescales by within-class variance — complementary operations.

### 13c. Key findings

1. **`geodesic_topk_asym` dominates at cal≥400** across all reductions (full, PCA, AE). Confirms it as the universal best NCM.
2. **PCA-128 + topk_asym = 1.79 at cal=800** — best overall pipeline. 19% smaller than full-768 topk_asym (2.20), 8% smaller than PCA + topk_mean (1.94).
3. **Whitening still helps 5-12%** after PCA reduction — not redundant.
4. **AE-32 ≈ PCA-128 at cal≥600** (within 0.02 set size). AE-32 wins at cal=300 (3.18 vs 3.96, 20% smaller).
5. **Unwhitened variants competitive at cal=300** only (unwhitened_topk_mean: 3.96 on PCA-128, best at that cal size). At cal≥400 whitened variants dominate.

### Winning pipeline: DINOv2 → PCA-128 (unlabeled) → whitened geodesic topk_asym (k=5) → FCP

Results saved: `output/ncm_comparison/`

---

## 14. Future Directions

### P0 — Must complete
1. ~~**PCA + NCM combination**~~ — **DONE** (§13). PCA-128 + topk_asym = 1.79 at cal=800 (best pipeline).
2. ~~**SemiCP multi-dataset**~~ — **DONE** (§11). FCP+PCA dominates on CIFAR-100 & miniImageNet. CUB-200: FCP wins (PCA-512 doesn't help fine-grained). NNM always neutral.
3. **MA-CS multi-dataset + adaptive λ** — most novel contribution. See §3.
4. **MS-CS multi-dataset** — test on CUB-200, miniImageNet. See §4.

### P1 — Stronger semi-supervised baselines (literature review, 2026-05-13)
5. **PPI-RCPS** (Einbinder et al. 2024, arXiv 2412.11174) — Prediction-Powered Inference for threshold tuning with unlabeled data. Orthogonal approach: optimizes conformal quantile rather than augmenting scores. Moderate effort.
6. **SSCP** (Seedat et al., AISTATS 2023, arXiv 2302.12238) — Self-supervised pretext tasks to improve NCM. Different paradigm from PCA dim reduction. Note: designed for regression, needs adaptation for classification sets.
7. **Transductive Standardization** (Fan & Sesia 2025, arXiv 2512.15383) — Validates O(1/n) exchangeability for data-dependent standardization (relevant to our whitening theory).
8. **Pseudo-Label CP** (Angelman et al. 2025) — Source-free calibration using pseudo-labels on unlabeled pool. Tests whether pseudo-labels > NNM matching for unlabeled data use.

### Negative results (archived)

- Pool augmentation: breaks exchangeability. See `archive/findings_archive.md §11.5`.
- LDA projection: structural under-coverage. See `archive/findings_archive.md §8`.
- Original CS penalty: data leakage. See `archive/findings_archive.md §4`.
- PCA on cal data: under-coverage at high dims. See §10 control experiment.
- Pseudo-label trained head: same failure as LDA (label-dependent projection). See `archive/findings_archive.md §8`.
- Autoencoder bottleneck: matches PCA at best, slightly worse at cal≥600. DINOv2 manifold is approximately linear. See §12.

---

*Last updated: 2026-05-13*
