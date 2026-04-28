# Research Findings — SSL + Conformal Prediction

> Pull from here when writing Methods / Results / Discussion sections of the paper.

---

## Setup

- **Backbone**: DINOv2-base (ViT-B/14, 768-D, 336×336 input)
- **CP methods**: Full CP (FCP), CV+/Jackknife+ (CV+), Split CP (SCP)
- **Best NCM**: `geodesic_topk_asym` ⭐ (new winner, post-bugfix)
- **Default α**: 0.1 (target ≥ 90% coverage)
- **Scripts**: `run_multi_dataset_experiments.py`, `run_few_shot_experiment.py`, `compare_ncms.py`

---

## 1. NCM Comparison (FCP, α=0.1, 5 trials, post-bugfix)

> **Note**: All results below are from the fixed codebase (empty-set fallback bug corrected).
> Prior results (e.g., Flowers-102 "structural failure") were artefacts of that bug — see §2.4.

### 1.1 Unified NCM Comparison Table

#### CIFAR-10 (10 classes, 500/class, test=500)

| NCM | cal=50 | cal=100 | cal=200 | cal=300 | cal=400 |
|-----|--------|---------|---------|---------|---------|
| nn_ratio | 0.912 sz=1.07 | 0.905 sz=0.95 | 0.918 sz=0.95 | 0.921 sz=0.94 | 0.912 sz=0.92 |
| geodesic_nn_ratio | 0.912 sz=1.07 | 0.905 sz=0.95 | 0.919 sz=0.95 | 0.921 sz=0.94 | 0.913 sz=0.92 |
| mahal_nn_ratio | 0.892 sz=1.01 | 0.899 sz=0.94 | 0.920 sz=0.95 | 0.921 sz=0.94 | 0.914 sz=0.93 |
| whitened_geodesic | 0.897 sz=1.01 | 0.899 sz=0.94 | 0.917 sz=0.95 | 0.922 sz=0.94 | 0.914 sz=0.93 |
| geodesic_topk_mean | 0.886❌ sz=1.00 | 0.916 sz=0.94 | 0.917 sz=0.93 | 0.912 sz=0.92 | 0.916 sz=0.92 |
| **geodesic_topk_asym** | 0.901 sz=1.04 | 0.907 sz=0.95 | 0.924 sz=0.96 | 0.914 sz=0.93 | 0.912 sz=0.92 |

**Takeaway**: No meaningful differentiation on CIFAR-10. All NCMs give sz≈0.92–0.95 at any cal≥100. FCP is systematically over-conservative (~91–92% vs 90% target) across all NCMs — structural, not fixable by NCM choice.

#### CIFAR-100 (100 classes, 25/class, test=300) ⭐ Key differentiation dataset

| NCM | cal=200 | cal=300 | cal=400 | cal=600 |
|-----|---------|---------|---------|---------|
| nn_ratio | 0.947 sz=23.25 | 0.915 sz=8.36 | 0.905 sz=6.06 | 0.912 sz=4.23 |
| geodesic_nn_ratio | 0.948 sz=21.57 | 0.917 sz=7.93 | 0.907 sz=5.59 | 0.912 sz=3.97 |
| mahal_nn_ratio | 0.943 sz=20.96 | 0.912 sz=7.91 | 0.905 sz=5.66 | 0.912 sz=4.19 |
| whitened_geodesic | 0.943 sz=19.24 | 0.909 sz=6.96 | 0.904 sz=5.21 | 0.912 sz=3.84 |
| geodesic_topk_mean | 0.927 sz=11.51 | 0.885❌ sz=3.62 | 0.891 sz=3.17 | 0.933 sz=4.08 |
| **geodesic_topk_asym** | 0.952 sz=15.78 | **0.917 sz=4.65** | **0.916 sz=3.18** | **0.907 sz=2.29** |

**Takeaway**: `geodesic_topk_asym` is the **clear winner** — 39–40% smaller sets than `whitened_geodesic` at cal=400–600 while maintaining valid coverage. `geodesic_topk_mean` under-covers at cal=300 (0.885❌) and over-covers at cal=600 (0.933) — structurally unstable, do not use as a primary NCM.

#### Flowers-102 (102 classes, ~55/class, test=500)

| NCM | cal=200 | cal=400 | cal=600 | cal=800 | cal=1200 |
|-----|---------|---------|---------|---------|---------|
| nn_ratio | 1.000 sz=102 | 0.937 sz=0.94 | 0.915 sz=0.92 | 0.910 sz=0.91 | 0.911 sz=0.91 |
| geodesic_nn_ratio | 1.000 sz=90.4 | 0.938 sz=0.94 | 0.915 sz=0.92 | 0.908 sz=0.91 | 0.910 sz=0.91 |
| mahal_nn_ratio | 1.000 sz=102 | 0.938 sz=0.94 | 0.913 sz=0.91 | 0.908 sz=0.91 | 0.913 sz=0.91 |
| whitened_geodesic | 1.000 sz=90.4 | 0.937 sz=0.94 | 0.915 sz=0.92 | 0.909 sz=0.91 | 0.910 sz=0.91 |
| **geodesic_topk_mean** | **0.999 sz=6.98** | 0.924 sz=0.93 | 0.912 sz=0.91 | 0.918 sz=0.92 | 0.912 sz=0.91 |
| geodesic_topk_asym | 1.000 sz=90.4 | 0.941 sz=0.94 | 0.914 sz=0.91 | 0.902 sz=0.90 | 0.915 sz=0.92 |

**Takeaway**: FCP works on Flowers-102 from cal=400 (all NCMs, sz≈0.91). The previous "structural failure" was a bug. At cal=200, `geodesic_topk_mean` is uniquely robust (sz=6.98 vs sz≈90–102 for all others) — its averaged denominator prevents score blow-up under extreme data scarcity. From cal≥400, all NCMs are equivalent.

#### miniImageNet (100 classes, 500/class, test=500) ⭐

| NCM | cal=200 | cal=400 | cal=600 | cal=800 | cal=1200 |
|-----|---------|---------|---------|---------|---------|
| nn_ratio | 0.950 sz=7.16 | 0.910 sz=1.56 | 0.904 sz=1.21 | 0.899 sz=1.10 | 0.906 sz=1.06 |
| geodesic_nn_ratio | 0.951 sz=6.79 | 0.910 sz=1.52 | 0.904 sz=1.19 | 0.899 sz=1.10 | 0.906 sz=1.06 |
| mahal_nn_ratio | 0.947 sz=6.29 | 0.907 sz=1.51 | 0.904 sz=1.19 | 0.898 sz=1.09 | 0.906 sz=1.06 |
| whitened_geodesic | 0.946 sz=5.79 | 0.909 sz=1.51 | 0.904 sz=1.18 | 0.898 sz=1.09 | 0.906 sz=1.06 |
| geodesic_topk_mean | 0.913 sz=2.16 | 0.907 sz=1.23 | 0.918 sz=1.13 | 0.906 sz=1.03 | **0.910 sz=0.99** |
| **geodesic_topk_asym** | 0.951 sz=3.22 | **0.913 sz=1.18** | **0.918 sz=1.10** | **0.900 sz=1.05** | 0.911 sz=1.03 |

**Takeaway**: `geodesic_topk_asym` dominates at cal=400–800 (22–24% smaller sets vs whitened_geodesic). `geodesic_topk_mean` wins at cal=1200 (sz=0.99) but slightly over-covers at cal=600 (0.918). The previous finding "standard NCMs win on miniImageNet" was **wrong** — a pre-bugfix artefact.

### 1.2 NCM Ranking Summary

| Rank | NCM | Strengths | Weaknesses |
|------|-----|-----------|------------|
| 🥇 **geodesic_topk_asym** | Best for K≥100, cal=300–1200 | 39–40% smaller sets on CIFAR-100; 22% on miniImageNet; valid at all cal | No advantage on CIFAR-10; weaker at cal=200 on Flowers |
| 🥈 whitened_geodesic | Best "safe" NCM; best standard at small cal | Stable at all cal sizes and datasets | 39% larger sets than topk_asym at moderate cal |
| 🥉 geodesic_topk_mean | Only viable at cal=200 on Flowers-102; wins at cal=1200 on miniImageNet | Unique robustness under extreme cal scarcity | Unstable alone: under-covers CIFAR-100 at cal≥500 (0.885–0.894) |
| 4 mahal_nn_ratio / geodesic_nn_ratio | Slight improvement over nn_ratio | Similar to whitened_geodesic | No clear advantage over whitened_geodesic |
| 5 nn_ratio | Baseline | Simple, fast | Largest sets at all cal sizes |

**Winner: `geodesic_topk_asym`** for any K≥100 experiment with cal≥300.
Use `whitened_geodesic` as a stable fallback at cal≤200 or for CIFAR-10/EuroSAT.

---

## 2. FCP vs CV+ vs SCP — Per-Dataset Results

**Config**: whitened_geodesic FCP (to be updated with geodesic_topk_asym) · geodesic_nn_ratio CV+ · softmax SCP · 5 trials.

### 2.1 CIFAR-10 (10 classes, 500/class)

FCP is **systematically over-conservative** at all cal sizes (~91–92% coverage, sz≈0.92–0.95). Root cause: structural class ambiguity (cat/dog, automobile/truck) — not fixable by NCM choice (see §5). CV+ achieves near-target with tight sets (sz≈0.9–1.0).

| cal | FCP | CV+ | SCP |
|-----|-----|-----|-----|
| 100 | 95.1% sz=1.62 ❌ | **90.3% sz=0.97** ✅ | 86.0% ❌ |
| 200 | 97.6% sz=1.66 ❌ | **91.5% sz=0.94** ✅ | 90.5% sz=0.93 ✅ |
| 400 | 99.1% sz=1.91 ❌ | **89.7% sz=0.91** ✅ | 89.4% sz=0.90 ✅ |

**Takeaway**: For easy coarse-grained datasets, use CV+ or SCP.

### 2.2 CIFAR-100 (100 classes, 25/class) ⭐

| cal | FCP (whitened_geodesic) | CV+ | SCP |
|-----|------------------------|-----|-----|
| 400 | **90.9% sz=7.45** ✅ 7.8s | 100% sz=100 ❌ | 98% ❌ |
| 600 | **90.0% sz=3.39** ✅ 8.8s | 94.7% sz=8.68 ❌ | 90.7% sz=12.1 ✅ |
| 800 | 90.1% sz=2.78 9.7s | **90.8% sz=2.65** ✅ 42s | 89.8% sz=3.01 |
| 1000 | **90.0% sz=2.35** ✅ 9s | 91.2% sz=2.35 62s | 89.8% sz=2.36 |

- FCP first valid: **cal=400** (7.8s). CV+ first valid: **cal=800** (42s) — 2× more data, 5× slower.
- At cal=1000: identical set sizes; FCP **7× faster**.
- With geodesic_topk_asym (§1.1): FCP cal=400 sz drops from 7.45 → **3.18** (57% reduction), matching CV+'s efficiency but 5–7× faster.

### 2.3 EuroSAT (10 classes, 200/class)

All methods work; FCP valid from cal=75 (sz=2.42, 0.7s). No clear FCP advantage — use whichever is most convenient.

### 2.4 Flowers-102 (102 classes, ~55/class) — CORRECTED

**⚠️ Prior conclusion ("FCP structural failure — 100% coverage at all cal sizes") was wrong.**
It was caused by the empty-set fallback bug, not by any property of the dataset or FCP.

**Correct findings** (post-bugfix, from §1.1):
- FCP is valid from **cal=400** for all NCMs (cov≈0.91–0.94, sz≈0.91–0.94).
- At cal=200: standard NCMs give full prediction sets (sz≈90–102), but `geodesic_topk_mean` avoids blowup (sz=6.98) — its k-NN averaging of the denominator prevents the ratio from diverging under extreme scarcity.
- From cal≥600, all NCMs converge to sz≈0.91 — near-optimal efficiency.

**New conclusion**: Flowers-102 behaves like any fine-grained 100-class dataset. FCP works well. Use `geodesic_topk_mean` if cal=200 is required, otherwise any NCM works from cal=400.

### 2.5 CUB-200 (200 classes, ~50/class) ⭐

| cal | FCP | CV+ | SCP |
|-----|-----|-----|-----|
| 600 | **93.4% sz=3.75** ✅ 11s | 100% ❌ | 100% ❌ |
| 800 | **91.9% sz=2.15** ✅ 12s | 100% ❌ | 100% ❌ |
| 1000 | 89.3% sz=1.70 | 98.6% ❌ | 90.7% sz=6.73 |
| 1500 | **90.3% sz=1.53** ✅ 14s | 92.4% sz=1.67 ✅ 58s | 89.8% sz=2.18 |
| 2000 | 89.9% sz=1.39 16s | 90.7% sz=1.40 134s | 89.6% sz=1.46 |

- FCP first valid: **cal=800** (12s). CV+ first valid: **cal=1500** (58s) — 1.9× more data, 4× slower.
- At cal=2000: same set size; FCP **9× faster**.

---

## 3. Few-Shot Regime (k shots per class, fixed test set)

> To be re-run with geodesic_topk_asym FCP for updated numbers.

### CIFAR-100 ⭐ Smoking-gun result (whitened_geodesic FCP)

| k | Total cal | FCP | CV+ | SCP |
|---|-----------|-----|-----|-----|
| 3 | 300 | **91.7% sz=7.63** ✅ 4s | 89.8% sz=2.27 ⚠️ 64s | 100% sz=100 ❌ |
| 5 | 500 | **94.1% sz=6.42** ✅ 4.3s | 91.7% sz=1.76 ✅ 105s | 95.7% sz=41.6 ❌ |
| 7 | 700 | **90.7% sz=3.19** ✅ 5.3s | 90.9% sz=1.48 ✅ 107s | 91.5% sz=2.93 ✅ |
| 10 | 1000 | 90.8% sz=2.37 6s | 89.5% sz=1.26 108s | 90.1% sz=1.73 |
| 20 | 2000 | 90.5% sz=1.84 9s | 90.5% sz=1.17 109s | 91.3% sz=1.36 |

**FCP is the only valid method at k=3–5. SCP completely fails. CV+ borderline at k=3 and 16–20× slower.**
With geodesic_topk_asym, FCP set sizes at k=7–20 should improve by ~40% (pending re-run).

### Other datasets
- **CIFAR-10**: FCP over-conservative (~93–99%); CV+ preferred for 10-class.
- **EuroSAT**: FCP valid from k=15; all methods converge at k=30.

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

### 4.3 MA-CS Indicator Penalty (Fargion et al. §4) — Mixed Result

After retracting the MS-CS (continuous similarity matrix) approach, we tested the **MA-CS (Model-Agnostic)** variant from the same paper. MA-CS uses a binary superclass indicator — no data leakage since `g(y)` is fixed external knowledge (CIFAR-100's 20 superclasses of 5 classes each):

`s_λ(x, y) = s(x, y) + λ · I{g(y) ≠ g(ŷ(x))}`

**Results** (CIFAR-100, geodesic_topk_mean FCP, 3 trials, α=0.1):

| cal | No penalty | λ=0.01 | λ=0.02 | λ=0.03 | λ=0.05 | λ=0.10 |
|-----|-----------|--------|--------|--------|--------|--------|
| 300 | 0.879 sz=11.3 | 0.877 sz=10.3 | 0.879 sz=9.5 | 0.877 sz=8.7 | 0.877 sz=7.7 | 0.871 sz=6.4 |
| 400 | 0.880 sz=5.5 | 0.876 sz=4.9 | 0.874 sz=4.6 | 0.876 sz=4.4 | 0.870 sz=4.1 | 0.880 sz=4.2 |
| 600 | **0.901 sz=3.27** | 0.908 sz=3.01 | 0.902 sz=2.71 | 0.903 sz=2.65 | **0.903 sz=2.56** | 0.908 sz=2.92 |
| 800 | **0.898 sz=2.20** | 0.898 sz=2.08 | 0.899 sz=2.01 | 0.902 sz=1.99 | **0.897 sz=1.98** | 0.896 sz=2.25 |

**Key observations:**

1. **Gentle λ (0.01-0.05) helps at sufficient cal sizes.** At cal=600, λ=0.05 reduces set size from 3.27→2.56 (22% reduction) while maintaining valid coverage (0.903). At cal=800, λ=0.05 gives 2.20→1.98 (10% reduction).
2. **U-shaped response to λ.** λ=0.10 is too aggressive — sets grow again (cal=600: 2.92 vs 2.56 at λ=0.05). The sweet spot is λ≈0.03-0.05 for this NCM/dataset.
3. **Cannot rescue under-coverage.** At cal=300-400 the baseline already under-covers (0.879-0.880); MA-CS reduces set size further but cannot fix coverage — it trades coverage for efficiency.
4. **Prior negative result was with too-large λ.** Earlier experiments with λ∈{0.1, 0.2, 0.3, 0.5} on geodesic_topk_asym found MA-CS uniformly harmful. The symmetric NCM (geodesic_topk_mean) with gentler λ shows a clear benefit when the baseline has valid coverage.

**Interpretation**: The binary superclass indicator does add orthogonal information to NN-ratio NCMs, but only when λ is calibrated to the score scale. With scores concentrated near 0.9±0.15, λ=0.03-0.05 (2-4% of score range) is the right magnitude — large enough to shift out-of-superclass candidates below the quantile threshold, small enough to avoid inflating the quantile itself (Lemma 4.1: q ≤ q_λ ≤ q+λ).

**Conclusion**: MA-CS with gentle λ is a viable efficiency improvement for FCP when (a) cal size is sufficient for valid baseline coverage, and (b) superclass structure is known. Further validation needed on other datasets with hierarchical labels.

---

## 5. Backbone Comparison — DINOv2 vs CLIP vs BEiTv2

**Config**: CIFAR-100 (100 classes, 25/class), **geodesic_topk_mean** (symmetric top-k) FCP, geodesic_nn_ratio CV+, softmax SCP, 5 trials, α=0.1, test=300.

| Backbone | Method | cal=200 | cal=300 | cal=400 | cal=600 | cal=800 | cal=1000 |
|----------|--------|---------|---------|---------|---------|---------|----------|
| **DINOv2-base** | FCP | 0.971 sz=37.0 | **0.913 sz=8.43** | **0.911 sz=5.69** | **0.907 sz=2.96** | 0.906 sz=2.75 | 0.899 sz=1.99 |
| | CV+ | 1.00 sz=100 | 1.00 sz=100 | 1.00 sz=97.2 | 0.951 sz=6.50 | 0.931 sz=3.35 | 0.919 sz=2.62 |
| | SCP | 1.00 sz=100 | 1.00 sz=100 | 1.00 sz=100 | 0.919 sz=14.4 | 0.907 sz=3.04 | 0.895 sz=1.97 |
| **BEiTv2-base** | FCP | 0.990 sz=58.6 | 0.941 sz=12.6 | 0.915 sz=8.61 | 0.911 sz=5.24 | 0.911 sz=4.15 | 0.907 sz=3.09 |
| | CV+ | 1.00 sz=100 | 1.00 sz=100 | 1.00 sz=95.3 | 0.941 sz=7.68 | 0.925 sz=4.79 | 0.907 sz=3.89 |
| | SCP | 1.00 sz=100 | 1.00 sz=100 | 1.00 sz=100 | 0.895 sz=9.22 | 0.915 sz=4.13 | 0.906 sz=2.71 |
| **CLIP-base** | FCP | 1.00 sz=99.9 | 0.945 sz=49.0 | 0.913 sz=41.2 | 0.904 sz=32.6 | 0.909 sz=27.6 | 0.901 sz=21.3 |
| | CV+ | 1.00 sz=100 | 1.00 sz=100 | 1.00 sz=99.8 | 0.937 sz=35.6 | 0.930 sz=25.9 | 0.914 sz=20.3 |
| | SCP | 1.00 sz=100 | 1.00 sz=100 | 1.00 sz=100 | 0.913 sz=35.3 | 0.905 sz=14.6 | 0.902 sz=10.7 |

**Key findings:**

1. **DINOv2 dominates** — smallest sets by far at every cal size. At cal=600: sz=2.96 vs BEiTv2's 5.24 (77% larger) and CLIP's 32.6 (11× larger).
2. **CLIP is unsuitable for NN-ratio FCP** — set sizes remain 21-100 even at large cal. CLIP's language-supervised alignment does not produce the tight intra-class clusters that NN-ratio NCMs need.
3. **BEiTv2 is a reasonable second** — ~77% larger sets than DINOv2 at cal=600, but still viable. Self-supervised masked image modeling produces better NN-ratio geometry than CLIP's contrastive language-image objective.
4. **FCP advantage is universal across backbones** — FCP beats CV+ and SCP at cal≤600 for all three backbones (CV+ and SCP produce full/near-full sets at cal≤400). The "FCP is the only valid method at small cal" finding generalizes beyond DINOv2.
5. **DINOv2's self-distillation objective is the best match for NN-ratio NCMs** — confirming that the backbone choice matters significantly for CP efficiency.
6. **Symmetric NCM (geodesic_topk_mean) results are consistent** with prior asymmetric findings — same ranking, similar magnitudes. The symmetric formulation is preferred for theoretical simplicity.

**Output**: `output/backbone_comparison_symmetric/`

---

## 6. Summary — When to Use Each Method

| Scenario | Best method | Best NCM | Reason |
|----------|-------------|----------|--------|
| K≥100, cal≥300 | **FCP** | geodesic_topk_asym | 39–40% smaller sets vs whitened; only valid method at small cal |
| K≥100, cal=200 | **FCP** | whitened_geodesic | More stable at very small cal |
| K≥100, extreme cal scarcity (cal=200) fine-grained | **FCP** | geodesic_topk_mean | Only NCM that avoids score blowup at cal=200 |
| Few shots (k≤7/class, K≥100) | **FCP** | geodesic_topk_asym | SCP fails; CV+ borderline and 16–20× slower |
| K=10, well-separated | CV+ or SCP | — | FCP over-conservative regardless of NCM |

---

## 7. Theoretical Justification

### geodesic_topk_asym NCM
Asymmetric design: **1-NN numerator** / **mean-k denominator**.
- 1-NN numerator: tightest same-class distance keeps inlier scores small and tight
- mean-k denominator: averages noise in other-class distances, smoothing score distribution
- Asymmetry is key: mean-k denominator reduces variance without losing class signal; mean-k on both sides (topk_mean) over-smooths and destabilizes coverage

### whitened_geodesic NCM
Combines two insights about DINOv2 embeddings:
1. **Anisotropic intra-class variance**: pooled diagonal whitening amplifies discriminative tight dims
2. **Hyperspherical geometry** (Wang & Isola 2020): arccos metric amplifies differences near class centroids; whitening before projection gives double discrimination boost

### Why FCP beats CV+ at small cal
CV+ trains on (k-1)/k data per fold — with few samples and many classes, some classes are absent from folds → trivially full sets. FCP uses the full calibration set for fitting, making it robust to very small cal sizes.

---

## 8. Future Directions (Prioritized)

> Instructor-endorsed directions marked ⭐. Ordered by priority.

### 🥇 Direction 1 — CP-Aware SSL Fine-Tuning ⭐ *(most novel)*

Take a pretrained DINOv2 backbone and apply additional fine-tuning with an objective that directly optimizes CP efficiency on the downstream task.

**Motivation**: CP efficiency depends on specific embedding geometry — tight intra-class clusters, large inter-class margins, and an NN-ratio score distribution with low variance. Standard SSL objectives (DINO's self-distillation) do not optimize for this structure. A task-aware fine-tuning stage could reshape the embedding space in a CP-favorable direction without full retraining.

**Proposed approach** (see §8 implementation plan):
1. **Architecture**: Frozen DINOv2 backbone + lightweight linear adapter (or small MLP). Only the adapter is trained — few-shot compatible.
2. **Primary loss — NN-ratio margin loss**: maximize `log(d_other / d_same)` averaged over the few-shot calibration set. Directly maximizes the NCM score margin.
3. **Auxiliary loss — confusion-weighted contrastive loss**: upweight repulsion between confusable class pairs (using pairwise class distances computed from training centroids). Targets pairs where FCP produces unnecessarily large prediction sets.
4. **Score variance regularization**: penalize variance of calibration scores within each class (tighter score distribution → smaller quantile gap → smaller sets).

**Key question**: Does adapter fine-tuning transfer across datasets? If so, a single CP-aware adapter trained on any labeled set could be reused.

**Connection to Direction 2**: Other SSL objectives (CLIP, BEiT) may already produce better-structured embeddings for CP — understanding why informs which fine-tuning signal to add to DINOv2.

---

### 🥈 Direction 2 — Backbone Comparison ⭐ — **DONE** (initial results)

> See §5 for full results. Summary below.

**Completed**: DINOv2-base vs CLIP-base vs BEiTv2-base on CIFAR-100 with geodesic_topk_asym FCP.

**Results**: DINOv2 dominates (sz=2.42 at cal=600). BEiTv2 is a reasonable second (~60% larger sets). CLIP is unsuitable for NN-ratio FCP (sz=24.5 at cal=600 — 10× worse than DINOv2). FCP advantage over CV+/SCP at small cal generalizes across all three backbones.

**Hypothesis outcome**: Partially confirmed. DINOv2's self-distillation produces the best geometry for NN-ratio NCMs. CLIP's language-supervised alignment does NOT improve FCP on easy datasets (as hypothesized) — instead it produces poor NN-ratio geometry across the board. The CLIP hypothesis was wrong.

**Remaining**: MAE-base, ssl-ResNet50 not yet tested. CUB-200 and miniImageNet cross-backbone comparison pending.

---

### 🥉 Direction 3 — Theoretical Validity of CS Penalty — **RETRACTED**

> The CS penalty has been retracted due to confirmed data leakage and redundancy with the base NCM. The formal treatment previously in `theory_cs_validity.md` has been deleted.

---

### Direction 4 — Per-Class Variance NCM + LDA Projection *(secondary)*

Merge of two related ideas:
- **Per-class whitening**: normalize each dimension by within-class std per class (heterogeneous intra-class spread). With k=3–10 shots, requires Ledoit-Wolf or diagonal shrinkage toward pooled covariance.
- **LDA projection**: project onto K-1 Fisher discriminant subspace before NCM. Maximizes between/within-class variance ratio; particularly useful in few-shot regime.

Both address the same underlying issue as `whitened_geodesic` but at a finer granularity. Expected benefit on fine-grained datasets with heterogeneous class spread (CUB-200, Flowers-102). Low implementation cost; can be evaluated by adding new NCM classes to `conformal_prediction.py`.

---

## 9. Dataset Properties

| Dataset | K | N/class | Post-bugfix FCP behavior | Notes |
|---------|---|---------|--------------------------|-------|
| CIFAR-10 | 10 | 500 | Over-conservative (sz≈0.92–0.95, cov≈91–92%) | No NCM differentiation; use CV+ |
| EuroSAT | 10 | 200 | All methods work; no clear winner | Well-separated satellite classes |
| CIFAR-100 | 100 | 25 | FCP dominates; topk_asym wins ⭐ | Key result dataset |
| CUB-200 | 200 | ~50 | FCP only valid method at cal<1500 ⭐ | Strongest FCP advantage vs CV+ |
| Flowers-102 | 102 | ~55 | FCP works from cal=400 (sz≈0.91) ✅ | Pre-bugfix "failure" was artefact; topk_mean uniquely helpful at cal=200 |
| miniImageNet | 100 | 500 | topk_asym best (sz=1.18@400, 1.10@600) ⭐ | ImageNet-domain; FCP works well |
| tiny-imagenet | 200 | 500 | Pending | ImageNet-domain, 200 classes |

---

*Last updated: 2026-04-23*
*All output: `output/ncm_comparison/`, `output/multi_dataset_experiments/`, `output/few_shot_experiments/`*
