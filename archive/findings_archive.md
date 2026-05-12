# Archived Findings — Completed/Retracted Investigations

> These sections were moved from `findings.md` on 2026-05-12 to reduce clutter. They document completed investigations, retracted results, and abandoned approaches.

---

## 4. Class Similarity Penalty — RETRACTED

> **RETRACTED (2026-04-23)**: All CS penalty results in this section are invalid due to **data leakage** in `build_class_similarity_matrix()`. The similarity matrix M was computed on the full dataset (cal+test), meaning test data leaked into the scoring function. When M is computed on calibration data only (the only deployable option), CS under-covers badly. Additionally, the CS penalty is **redundant** with NN-ratio NCMs in the low-data regime. See details below.

### Retraction Details

**Bug**: `build_class_similarity_matrix(embeddings, labels)` was called on the FULL dataset (cal+test) in every script. The matrix M included test data — not deployable in practice.

**Sanity check** (CIFAR-100, geodesic_topk_asym, 3 trials, alpha=0.1):

| cal | No CS | CS (M=cal-only) | CS (M=full) LEAKED |
|-----|-------|-----------------|-------------------|
| 300 | 0.888 sz=10.50 | 0.828 sz=3.23 | 0.888 sz=4.57 |
| 400 | 0.872 sz=4.04 | 0.841 sz=2.58 | 0.883 sz=3.32 |
| 600 | 0.914 sz=2.59 | 0.872 sz=2.20 | 0.904 sz=2.52 |

- **CS with cal-only M under-covers badly** (82.8-87.2% vs 90% target) — invalid.
- **CS with leaked M barely helps** at large cal (sz=2.52 vs 2.59 at cal=600).
- The prior "51-61% improvement" was an artefact of test data leakage.

**Redundancy argument**: In the low-data regime, the NN-ratio NCM already computes point-to-point distances to all class members, capturing the same geometric structure that M's centroid-based similarity summarizes. CS adds a coarser version of information the NCM already has.

### 4.1 CS with whitened_geodesic (prior results) — RETRACTED

Results below are **invalid** due to data leakage described above. Retained for historical record only.

| Dataset | Effect on FCP set size |
|---------|----------------------|
| CIFAR-10 | ~~Hurts — over-covers (94–99%). Classes already well-separated.~~ RETRACTED |
| **CIFAR-100** | ~~**Huge benefit**: cal=400 sz: 7.45 → 3.19 (57% reduction).~~ RETRACTED — leaked M |
| EuroSAT | ~~Slight hurt. Classes already distinct.~~ RETRACTED |

### 4.2 CS with geodesic_topk_mean — CIFAR-100 — RETRACTED

Results below are **invalid** due to data leakage. Retained for historical record only.

| cal | No-CS cov | No-CS sz | +CS cov | +CS sz | Change |
|-----|-----------|----------|---------|--------|--------|
| 300 | 93.4% | 10.78 | 93.0% | 4.21 | ~~-61%~~ RETRACTED |
| 400 | 90.7% | 6.10 | 90.1% | 2.99 | ~~-51%~~ RETRACTED |
| 500 | 89.0% | 3.34 | 91.1% | 2.70 | ~~-19%~~ RETRACTED |
| 600 | 90.0% | 3.14 | 89.7% | 2.22 | ~~-29%~~ RETRACTED |
| 700 | 89.2% | 2.42 | 89.8% | 2.15 | ~~-11%~~ RETRACTED |
| 800 | 89.4% | 2.42 | 89.9% | 2.11 | ~~-12%~~ RETRACTED |
| 1000 | 89.4% | 1.87 | 89.8% | 1.90 | ~~~0%~~ RETRACTED |

---

## 11.1 Data Preprocessing Audit — Split Strategy & Data Sources

**Data source per dataset**:
| Dataset | Downloaded split | Source |
|---------|-----------------|--------|
| CIFAR-10/100 | `--split train` (default) | Train split only (50K pool) |
| Flowers-102 | All splits merged (train+val+test) | Single combined pool |
| CUB-200 | All splits merged | Single combined pool |
| miniImageNet | All splits merged | Single combined pool |
| EuroSAT | Single dataset (no split) | Full dataset |

**Issue**: Both calibration and test are carved from the same downloaded pool (typically train split). Test points are really train data — problematic for scientific rigor. For proper evaluation, test data should come from a held-out split never seen during calibration.

**Resolution**: Use `download_datasets.py --split both` to download train and test separately. Extract embeddings for each. Load calibration from train embeddings, test from test embeddings. This ensures proper train/test separation.

**Split balance audit**:
| Script | Split method | Balanced? |
|--------|-------------|-----------|
| `conformal_prediction.py:cal_test_split()` | Random permutation | NO |
| `conformal_prediction.py:train_cal_test_split()` | Random permutation | NO |
| `conformal_prediction.py:stratified_cal_test_split()` | Exact k/class (new) | YES |
| `run_multi_dataset_experiments.py` | >=1/class, rest random | Partial |
| `run_few_shot_experiment.py` | Exactly k/class | YES |
| `macs_experiment.py` | >=1/class, rest random | Partial |

**New**: Added `stratified_cal_test_split(balanced=True)` to `conformal_prediction.py` and `--balanced` flag to `run_conformal_experiment.py`. Guarantees exactly `cal_size // n_classes` samples per class.

**Experimental Results** (CIFAR-100, geodesic_topk_mean, α=0.1, 5 trials):

| cal | Mode A (Unbalanced, single pool) | Mode B (Balanced, single pool) | Mode C (Balanced, train/test separated) |
|-----|---|---|---|
| 300 | cov=0.939 sz=11.50 | cov=0.915 sz=6.56 | cov=0.918 sz=6.23 |
| 400 | cov=0.910 sz=4.97 | cov=0.904 sz=3.97 | cov=0.899 sz=3.72 |
| 600 | cov=0.900 sz=2.64 | cov=0.930 sz=3.67 | cov=0.929 sz=4.33 |

**Findings**:
1. **Balanced split helps enormously at small cal**: At cal=300, sz drops 11.50 → 6.56 (43% reduction) from ensuring 3/class instead of random.
2. **CV+ benefits even more**: Unbalanced cal causes fold collapse (min class size=0-1 → 2 folds with inf scores → sz=100). Balanced split enables CV+ from cal=300 (sz=9.79).
3. **Train/test separation (C) vs same-pool (B)**: Nearly identical — expected since DINOv2 embeddings are frozen (no training involved).
4. **At cal=600, unbalanced appears better (sz=2.64 vs 3.67)** but with lower coverage (0.900 vs 0.930). The unbalanced split over-represents easy classes, making FCP look artificially tighter but with more variable coverage.
5. **SCP completely fails** in all modes at cal=300-400 (sz=80-87) — 150 training samples for 100 classes is insufficient.

**Recommendation**: Use balanced split for all future experiments at cal≤600 on K≥100 datasets. At cal≥800 the difference is negligible.

---

## 11.3 Pre-trained Head for Dimensionality Reduction (Deferred)

**Idea**: Use a pre-trained classification head to reduce embedding dimensionality before CP.

**Problem**: DINOv2-base's ImageNet head maps 768-D -> 1000-D (increases dims, not reduces). The idea only makes sense with a head outputting fewer dims than features. No such pre-trained head exists for our scarce-data setting (can't train one without overfitting).

**Future directions** (if revisited):
- Models with compact heads (e.g., DINO with 256-D projection head)
- PCA as unsupervised dim reduction baseline (preserves FCP coverage; see §7)
- Fine-tuned models for fewer classes (requires sufficient data)

---

## 11.5 Semi-Supervised CP: Using Unlabeled Data More Effectively

**Motivation**: We already use unlabeled data for MS-CS (building the cluster similarity matrix M). Can we do more? The semi-supervised CP literature offers several approaches for leveraging unlabeled data to improve conformal prediction, potentially addressing our key limitations: (1) small calibration sets, (2) conservative prediction sets, (3) finite-sample undercoverage.

#### Literature Landscape

We identify **three families** of approaches for using unlabeled data in CP:

**Family 1: Augmenting calibration with pseudo-scores (SemiCP, SFCP)**
- **SemiCP** (Zhou et al., CVPR 2026): Defines a Nearest Neighbor Matching (NNM) score for unlabeled samples — estimates their nonconformity by matching to the closest labeled point with the same pseudo-label. Achieves O(1/√N) convergence in coverage gap as unlabeled count N grows. With 20 labeled + 4000 unlabeled, reduces coverage gap by 77%.
- **SFCP** (Angelman et al., COPA 2025): Uses pseudo-labels from source model to estimate conformal thresholds entirely without labeled cal data. Guarantee: P(Y∈C) ≥ 1-α-β for model with accuracy 1-β.

**Family 2: Score function refinement via unlabeled data (SSCP, ECP/EACP, LATA)**
- **SSCP** (Seedat et al., AISTATS 2023): Trains auxiliary SSL pretext task; uses SSL reconstruction error as normalization for nonconformity scores, making them adaptive to instance difficulty.
- **ECP/EACP** (Kasa et al., UAI 2025): Entropy-scaled CP — adjusts scores using model's own uncertainty on unlabeled test data. Handles distribution shift without labels.
- **LATA** (Bozorgtabar et al., 2026): Smooths zero-shot probabilities over image-image kNN graph built on joint cal+test pool. Training-free, label-free. Preserves CP validity via deterministic transformation.

**Family 3: Hyper-parameter tuning via unlabeled data (PPI-RCPS, StCP)**
- **PPI-RCPS** (Einbinder, Ringel & Romano, 2024): Uses prediction-powered inference to tune RCPS hyper-parameters with unlabeled data, reducing conservatism. Directly applicable to our λ selection in MS-CS.
- **StCP** (Min et al., 2026): Labeled source + unlabeled target for set stability. Reduces variance of conditional set size.

#### Applicability Analysis for Our Setting

Our setting: **Full CP (transductive) + frozen SSL embeddings + few-shot calibration (3-6/class) + available unlabeled pool (~10K images)**

| Approach | Applicable? | Adaptation needed | Expected benefit | Difficulty |
|----------|-------------|-------------------|-----------------|------------|
| **SemiCP (NNM)** | Partially | NNM designed for Split CP; need LOO-compatible variant for FCP | +++ (augments cal from 600→5000+) | High |
| **SFCP** | Yes | Use kNN pseudo-labels as proxy cal scores | ++ (eliminates label requirement) | Medium |
| **LATA (graph smoothing)** | **Yes** | Smooth NCM scores instead of softmax probs on kNN graph | +++ (transductive, training-free) | **Low** |
| **PPI-RCPS** | **Yes** | Apply to MS-CS λ selection | ++ (principled λ, less conservative) | **Low** |
| **SSCP** | Partially | Need pretext task residuals from DINOv2 (not trivially available) | + (adaptive normalization) | High |
| **ECP** | No | Requires softmax model (we use non-parametric NCMs) | — | N/A |
| **StCP** | Yes | Use large unlabeled pool as "source" for stability | + (reduced variance) | Medium |

#### Most Promising Approaches (Ranked)

**1. LATA-style Transductive NCM Smoothing** (Priority: HIGH)

The most natural fit for our FCP pipeline. Key idea:
- Build kNN graph G on **joint pool** = {cal embeddings} ∪ {unlabeled embeddings} ∪ {test point}
- For each test candidate (x_test, y), compute base NCM score s(x_test, y) as usual
- **Smooth** the score using graph Laplacian: propagate information from nearby cal points' scores
- Calibration scores are similarly smoothed against the graph
- The graph construction is **deterministic** given the data → preserves exchangeability

Why this works for us:
- We already have SSL embeddings for unlabeled data (same feature space)
- kNN graph construction is O(n log n) with approximate NN libraries
- The smoothing acts as a denoiser for NCM scores at small cal — exactly our weak spot
- Training-free, no hyperparameters beyond k_graph and number of smoothing iterations
- Compatible with any NCM (geodesic_topk_mean, geodesic_topk_asym, etc.)

Potential concern: computational overhead. FCP already evaluates K candidates per test point. Adding graph smoothing per candidate could be expensive. Mitigation: pre-compute graph once, only smooth the candidate's score (O(k_graph) per candidate).

**2. PPI-based λ Selection for MS-CS** (Priority: HIGH)

Direct application of Einbinder et al. to our existing MS-CS pipeline:
- Current approach: manual λ sweep, pick best λ on cal data (circular!)
- PPI approach: use unlabeled data to estimate the risk (expected set size) at each λ, then select λ that minimizes set size subject to coverage constraint
- The unlabeled data provides unbiased gradient information about how λ affects predictions on unseen data
- No additional embedding extraction needed — we already have the unlabeled pool

**3. Pseudo-Score Augmentation (SemiCP-style for FCP)** (Priority: MEDIUM)

Adapt NNM idea for transductive FCP:
- For each unlabeled point x_u, assign pseudo-label ŷ_u = 1-NN(x_u, cal)
- Compute pseudo-nonconformity score: s(x_u, ŷ_u) using the NCM on {cal ∪ (x_u, ŷ_u)}
- **Augment calibration pool**: treat (x_u, ŷ_u, s_u) as additional calibration points
- p-value computation now uses n_cal + n_pseudo scores instead of just n_cal
- Coverage guarantee becomes: P(Y∈C) ≥ 1-α - O(β/√n_pseudo) where β = pseudo-label error rate

Challenge: In FCP, calibration scores are LOO by construction. Pseudo-labeled points should also be LOO — but LOO with respect to what? Need careful theoretical treatment.

**4. SFCP for Extreme Few-Shot** (Priority: LOW)

When labels are extremely scarce (k=1-2/class), even FCP struggles. SFCP idea:
- Use kNN classifier on the tiny labeled set to pseudo-label the unlabeled pool
- Compute conformal scores on pseudo-labeled data
- Accept 1-α-β guarantee where β = kNN error rate on SSL embeddings

Less interesting because: our DINOv2 kNN accuracy is already ~75-85% on CIFAR-100, so β≈0.15-0.25, giving a weak guarantee (1-0.1-0.2 = 0.70). Only useful if we can drive β down.

#### Proposed Implementation Plan

**Phase 1** (immediate): LATA-style graph smoothing
- Build kNN graph on cal ∪ unlabeled pool (use sklearn NearestNeighbors, k_graph=10-20)
- Implement score smoothing: `s_smooth(x) = (1-γ)*s(x) + γ * mean(s(neighbors of x))`
- Test on CIFAR-100 at cal=400,600 with geodesic_topk_mean
- Compare: baseline FCP vs graph-smoothed FCP vs MS-CS vs graph-smoothed MS-CS

**Phase 2**: PPI λ selection
- Implement PPI framework for MS-CS λ: use unlabeled data to estimate E[set_size|λ] without labels
- Replace manual λ sweep with data-driven selection
- Validate: does PPI-selected λ match our empirically-optimal λ=0.05?

**Phase 3** (if Phase 1 succeeds): Pseudo-score augmentation
- Implement NNM-style pseudo-calibration for FCP
- Theoretical analysis of LOO compatibility
- Compare effective coverage at cal=200,300 with and without pseudo augmentation

#### Key Open Questions

1. **Does graph smoothing preserve exchangeability in FCP?** The graph is built from cal+unlabeled (fixed), but the test point joins it. If the graph construction is deterministic given the embeddings, and the test point is treated the same as cal points in the smoothing, exchangeability should hold. Need formal argument.

2. **How much unlabeled data is enough?** SemiCP shows O(1/√N) convergence. For our CIFAR-100 setting with 600 cal points, even 2500 unlabeled (our remaining pool) should provide significant benefit.

3. **Does smoothing help the NCM or hurt it?** Risk: smoothing might blur class boundaries in embedding space, making the NCM less discriminative. Counter-argument: at small cal (3-4/class), the NCM is already noisy — smoothing should help more than hurt.

4. **Interaction with MS-CS**: Graph smoothing and MS-CS both modify scores. Are they complementary or redundant? Hypothesis: complementary — MS-CS adjusts for class structure (macro), graph smoothing adjusts for local geometry (micro).

#### Experimental Results: Pool Augmentation (Completed 2026-05-12)

We implemented and tested three modes of pseudo-labeled pool augmentation on CIFAR-100 (100 classes, DINOv2-base, geodesic_topk_mean, α=0.1).

**Mode 1: Full augmentation** — augment both same-class and other-class pools.
```
cal=600 (6/class), 3 trials, mode=full:
  Baseline         cov=0.922±0.020  sz=3.70±0.32
  SemiSup(N=100)   cov=0.912±0.023  sz=3.18±0.21  (+14%)
  SemiSup(N=200)   cov=0.893±0.022  sz=2.34±0.19  (+37%)  ← barely valid
  SemiSup(N=300)   cov=0.881±0.030  sz=2.18±0.29  (+41%)  ← BROKEN
  SemiSup(N=500)   cov=0.869±0.041  sz=1.78±0.31  (+52%)  ← BROKEN
```
Oracle test (true labels, N=2000): cov=0.884 — confirms structural issue, not pseudo-label noise.

**Mode 2: Denominator-only** — pseudo-labeled points enrich other-class pool only.
```
cal=600, 5 trials, mode=denom_only:
  Baseline         cov=0.925±0.019  sz=3.66±0.26
  SemiSup(N=500)   cov=0.909±0.021  sz=3.71±0.52  (-1%)
  SemiSup(N=1000)  cov=0.897±0.015  sz=3.79±0.56  (-3%)
  SemiSup(N=2000)  cov=0.872±0.014  sz=3.50±0.68  ← BROKEN
```

**Mode 3: Whitening-only** — use augmented pool for variance estimation, fit NCM on cal only.
```
cal=600, 3 trials, mode=whiten_only:
  Baseline         cov=0.922±0.020  sz=3.70±0.32
  SemiSup(N=500+)  cov=0.926±0.019  sz=3.88±0.33  (-5%)  ← saturates
```

**Key finding: fundamental impossibility.** Set-size reduction requires enriching the same-class neighbor pool (numerator), which creates a feedback loop that breaks exchangeability. Removing one cal point z_i from the LOO doesn't remove its influence on the pseudo-labels of nearby unlabeled points, so z_i's same-class neighbors are biased (too close) → LOO scores biased low → test scores relatively too high → p-values too low → undercoverage. The bias scales as O(N/n_cal), making it worse with more pseudo-labeled points.

| Mode | Coverage | Set Size | Exchangeability |
|------|----------|----------|----------------|
| Full (N=100-200) | Marginal (0.89-0.91) | -14 to -37% | Broken O(N/n) |
| Denom-only | Preserved (N≤500) | +1 to +5% (worse) | Weak O(N/n²) |
| Whiten-only | Preserved | -5% (worse, saturates) | Preserved |

**Conclusion**: Pool augmentation is a dead end for exchangeability-preserving FCP improvement. The valid uses of unlabeled data remain:
1. **MS-CS** (already implemented): build class similarity matrix M from unlabeled clusters
2. **LATA-style score smoothing** (next priority): post-hoc score refinement, not pool augmentation
3. **PPI-based λ selection** (next priority): principled hyperparameter tuning

**Status**: Pool augmentation investigation complete (negative result). Pivoting to LATA-style score smoothing.

---

## Section 8 — Archived Directions

### Direction 3 — Theoretical Validity of CS Penalty — RETRACTED

The CS penalty has been retracted due to confirmed data leakage and redundancy with the base NCM. The formal treatment previously in `theory_cs_validity.md` has been deleted.

### Direction 4 — Per-Class Variance NCM — Low Priority

LDA projection was investigated and abandoned (see Direction 1 in findings.md §8). Per-class whitening (replacing pooled diagonal whitening with class-specific variance estimates) remains a potential extension for datasets with highly heterogeneous class spread (CUB-200, Flowers-102), but carries the same exchangeability concerns as LDA — class-specific statistics computed from calibration data create asymmetric overfitting. Only viable if the per-class variance estimates are stable enough that the O(1/n) approximation holds.

### LDA Projection — Investigated and Abandoned

We investigated Shrinkage-LDA Projection (768D → K-1 dims via Fisher Linear Discriminant) as an alternative to diagonal whitening. While theoretically motivated (projects away noise dimensions, reduces hubness), **LDA is fundamentally incompatible with Full Conformal Prediction**:

- **Structural under-coverage**: LDA optimizes class separation on calibration data, making calibration nonconformity scores artificially low (mean 0.16 vs 0.71 for test). This causes 70-85% actual coverage vs 90% target — a *structural* bias, not O(1/n).
- **Persists at all scales**: Tested at cal=300 to cal=20000 (miniImageNet), n_components=9 to 99, across CIFAR-10/100/miniImageNet. Under-coverage is universal.
- **LOO-LDA fixes the bias but is prohibitive**: Recomputing LDA excluding each calibration point shifts cal scores to match test (0.16→0.71), confirming the diagnosis. But at 394ms/point it's computationally intractable for FCP's O(n) inner loop.
- **PCA preserves coverage**: Unsupervised PCA at any dimensionality maintains valid FCP coverage, confirming the issue is specifically *label-dependent* projection creating asymmetric overfitting.
- **Conclusion**: Only *symmetric* data-dependent preprocessing (where adding one point changes statistics by O(1/n)) is safe for FCP. LDA's optimization of class boundaries violates this.

---

## Section 10 — Publication Roadmap Details

### Stale Results Tracker

These existing results use outdated NCMs and should be re-run:

| Section | Current NCM | Should Use | Impact |
|---------|-------------|------------|--------|
| §2 (FCP vs CV+ vs SCP) | whitened_geodesic FCP | geodesic_topk_asym | Set sizes will improve ~39-40% on K>=100 datasets |
| Few-shot (removed §3) | whitened_geodesic FCP | geodesic_topk_asym | Set sizes at k=7-20 should improve ~40% |
| §5 (Backbone comparison) | geodesic_topk_mean | geodesic_topk_asym | Rankings likely unchanged but absolute numbers will improve |

### Reviewer Prediction Table

| Likely Question | Our Answer | Action Needed |
|----------------|------------|---------------|
| "Why not compare against RAPS/APS?" | Standard Split CP score for classification | P0 #1 — must implement |
| "Are improvements statistically significant?" | We have 5 trials per config | P0 #4 — add error bars |
| "Why not test on ImageNet-scale?" | FCP targets low-data regime (cal < 1000). ImageNet has 1.28M training samples — not our setting. | None (explain in paper) |
| "How does this compare to meta-learning CP?" | Fisch et al. require auxiliary tasks and episode training. Our approach is inference-only on frozen features — simpler, faster. | P3 #13 (theoretical argument) |
| "What about distribution shift?" | Out of scope. FCP assumes exchangeability. | None (future work) |
| "Why only DINOv2-base?" | 4GB GPU constraint. Methodology is model-agnostic. | P3 #14 (future work) |
| "Is geodesic NCM better than RAPS?" | Different paradigms: NCM for FCP, RAPS for SCP. We compare FCP+geodesic vs SCP+RAPS. | P0 #1 enables this comparison |
| "Why not class-conditional coverage?" | K>=100 makes per-class calibration impossible at small cal. Cite Ding et al. (2023). | None (discuss in paper) |

### Proposed Paper Structure

| Section | Pages | Content |
|---------|-------|---------|
| 1. Introduction | 1.5 | Motivation, gap (all prior work = Split CP), contribution summary |
| 2. Background | 1.5 | CP (Full/Split/CV+), SSL embeddings, hyperspherical geometry |
| 3. Geodesic NCMs | 2.0 | Design principles, whitened geodesic, top-k averaging, O(N) updates |
| 4. When Does FCP Dominate? | 2.0 | 6-dataset comparison, coverage validity threshold, runtime, few-shot |
| 5. MA-CS in FCP Setting | 1.0 | Binary superclass penalty + geodesic NCMs, theoretical framework |
| 6. Backbone Geometry | 0.5 | DINOv2 vs CLIP vs BEiTv2, why self-distillation wins |
| 7. Negative Result: LDA | 0.5 | Exchangeability violation, O(1/n) vs structural |
| 8. Experiments | 2.0 | Tables, figures, ablations |
| 9. Discussion | 0.5 | Limitations (O(nK) scaling), when to prefer Split CP |
| **Total** | **~12 pages** | Suitable for AISTATS/ICML |

---

## Section 11.2 — MS-CS Detailed Results

### Initial Results (tau=1.0, geodesic_topk_asym, 3 trials)

| n_clusters | λ=0 (baseline) | λ=0.01 | λ=0.02 | λ=0.03 | λ=0.05 | λ=0.10 |
|---|---|---|---|---|---|---|
| 10 | 0.914 sz=2.59 | 0.913 sz=2.55 | 0.911 sz=2.53 | 0.910 sz=2.51 | 0.910 sz=2.47 | 0.907 sz=2.42 |
| 20 | 0.914 sz=2.59 | 0.913 sz=2.54 | 0.910 sz=2.50 | 0.909 sz=2.48 | 0.909 sz=2.44 | 0.904 sz=2.34 |
| 50 | 0.914 sz=2.59 | 0.912 sz=2.55 | 0.911 sz=2.51 | 0.909 sz=2.47 | 0.909 sz=2.42 | 0.913 sz=2.46 |

### Absolute Tau Sweep (n_clusters=20, geodesic_topk_mean, 3 trials)

| tau | cal | λ=0 (baseline) | λ=0.01 | λ=0.02 | λ=0.03 | λ=0.05 | λ=0.10 |
|---|---|---|---|---|---|---|---|
| 0.1 | 400 | 0.880❌ sz=3.41 | 0.877❌ sz=3.33 | 0.873❌ sz=3.28 | 0.870❌ sz=3.27 | 0.876❌ sz=3.26 | 0.880❌ sz=3.26 |
| 0.1 | 600 | 0.901 sz=3.27 | 0.899 sz=3.08 | 0.899 sz=2.82 | 0.898 sz=2.63 | 0.900 sz=2.38 | 0.898 sz=2.73 |
| 0.3 | 600 | 0.901 sz=3.27 | 0.900 sz=3.13 | 0.899 sz=2.93 | 0.899 sz=2.77 | 0.900 sz=2.49 | 0.897 sz=2.44 |
| 0.5 | 600 | 0.901 sz=3.27 | 0.901 sz=3.17 | 0.899 sz=3.02 | 0.900 sz=2.88 | 0.900 sz=2.63 | 0.897 sz=2.42 |
| 1.0 | 600 | 0.901 sz=3.27 | 0.900 sz=3.19 | 0.900 sz=3.09 | 0.900 sz=3.00 | 0.902 sz=2.82 | 0.902 sz=2.67 |

### Normalized Tau Sweep (tau = multiplier × median_d², median_d²=0.288)

| tau multiplier | cal | λ=0 (baseline) | λ=0.01 | λ=0.02 | λ=0.03 | λ=0.05 | λ=0.10 |
|---|---|---|---|---|---|---|---|
| 0.25 | 600 | 0.901 sz=3.27 | 0.900 sz=3.08 | 0.899 sz=2.83 | 0.897 sz=2.63 | 0.901 sz=2.39 | 0.897 sz=2.73 |
| 0.50 | 600 | 0.901 sz=3.27 | 0.900 sz=3.07 | 0.899 sz=2.80 | 0.897 sz=2.60 | 0.900 sz=2.37 | 0.899 sz=2.67 |
| 1.0 | 600 | 0.901 sz=3.27 | 0.900 sz=3.12 | 0.899 sz=2.91 | 0.899 sz=2.74 | 0.900 sz=2.48 | 0.897 sz=2.41 |
| 2.0 | 600 | 0.901 sz=3.27 | 0.900 sz=3.17 | 0.899 sz=3.01 | 0.900 sz=2.87 | 0.899 sz=2.61 | 0.897 sz=2.42 |

### Exchangeability Fix Results (n_clusters=20, tau=0.5×median_d², geodesic_topk_mean, 3 trials)

| cal | λ | Non-Exch cov | Non-Exch sz | Exch cov | Exch sz | Δcov |
|---|---|---|---|---|---|---|
| 300 | 0.05 | 0.873 | 7.77 | 0.878 | 8.13 | +0.5% |
| 300 | 0.10 | 0.871 | 6.29 | 0.878 | 6.93 | +0.7% |
| 400 | 0.05 | 0.864 | 3.76 | 0.872 | 3.88 | +0.8% |
| 400 | 0.10 | 0.847 | 3.42 | 0.860 | 3.66 | +1.3% |
| 600 | 0.05 | 0.900 | 2.37 | 0.901 | 2.40 | +0.1% |
| 600 | 0.10 | 0.898 | 2.64 | 0.900 | 2.69 | +0.2% |

### Pooled Whitening Ablation

Removing whitening (`unwhitened_topk_mean`) has negligible effect (+0.001 coverage). The undercoverage at small cal is a finite-sample effect from having only 3-4 same-class points for top-k averaging, not an exchangeability violation from whitening.

### Consolidated 5-Trial Results

**cal=600** (5 trials, n_clusters=20, tau=0.5×median_d²):

| λ | Non-Exch cov/sz | Exch cov/sz |
|---|---|---|
| 0.00 | 0.891 / 3.20 | 0.891 / 3.20 |
| 0.02 | 0.895 / 2.74 | 0.895 / 2.76 |
| 0.05 | 0.889 / 2.35 | 0.890 / 2.38 |
| 0.10 | 0.888 / 2.55 | 0.890 / 2.61 |

**cal=400** (5 trials):

| λ | Non-Exch cov/sz | Exch cov/sz |
|---|---|---|
| 0.00 | 0.889 / 5.49 | 0.889 / 5.49 |
| 0.05 | 0.881 / 3.88 | 0.887 / 4.01 |
| 0.10 | 0.873 / 3.79 | 0.881 / 4.09 |

### Section 11.4 — LOO 1-NN vs Softmax for y_hat

See §4.3.4 in findings.md for the comparison table. Our MA-CS implementation uses LOO 1-NN for y_hat(x) while Fargion et al. use argmax softmax. Key questions:

1. **Theoretical justification**: Is 1-NN a valid substitute? It's the natural nonparametric classifier in embedding space, consistent with the distance-based NCM logic.
2. **Exchangeability**: LOO for calibration, plain NN for test — same asymmetry as FCP itself (calibration scores are always LOO by construction). Should be fine.
3. **Alternative**: Use NCM's own scores: y_hat = argmin_y s(x,y). More principled but requires computing all K scores before applying penalty (slower).

Status: Open question. Current implementation works empirically (valid coverage maintained). Formal justification needed for paper.
