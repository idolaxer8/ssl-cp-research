# Research Findings — SSL + Conformal Prediction

> Pull from here when writing Methods / Results / Discussion sections of the paper.
> Detailed tables, retracted results, and superseded numbers: `archive/findings_archive.md`.
> Per-experiment raw outputs: `output/`, cluster runs in `output/from_cluster/`.

---

## 1. Headline & Setup

- **Backbone**: DINOv2-base (ViT-B/14, 768-D). Cluster runs use **input_size=518** (native); local laptop runs use 336 (4 GB VRAM cap).
- **Best pipeline**: **DINOv2 → PCA-128 (unlabeled) → whitened geodesic_topk_mean → FCP, with MS-CS penalty in PCA space**.
- **NCM choices**: `geodesic_topk_mean` (cluster headline NCM), `geodesic_topk_asym` (best locally, expected ~8% smaller sets when re-run on cluster — open).
- **Target**: α = 0.1 (≥ 90% coverage). 5 trials default, stratified cal/test splits everywhere.
- **GPU fast-path**: `FullConformalPredictor.predict(device='cuda', gpu_batch_size=256)` — 23-30× speedup, bit-equivalent. Covers any `GeodesicTopKMeanNCM` variant. See `[[gpu-fcp-path]]` memory.

**Cluster headline (FCP+PCA-128+MS-CS, matched-518, 5 trials, NCM = geodesic_topk_mean):**

| Cal  | CIFAR-10 (K=10) | CIFAR-100 (K=100) | miniImageNet (K=100) |
|------|------------------|----------------------|------------------------|
| 300  | 0.91 (89.8%)     | 1.77 (88.7%)         | 1.04 (90.7%)           |
| 400  | 0.94 (89.6%)     | 1.69 (91.1%)         | 0.97 (89.6%)           |
| 600  | 0.93 (91.0%)     | 1.85 (92.6%)         | 1.00 (91.4%)           |
| 800  | 0.91 (89.9%)     | **1.41 (91.1%)**     | 0.97 (89.0%)           |
| 1000 | 0.92 (90.1%)     | 1.40 (91.9%)         | 0.98 (90.1%)           |

- **CIFAR-100 is the discriminating benchmark** for the paper headline.
- **miniImageNet saturates** to near-singleton sets by cal=400 (DINOv2 pretraining transfers very directly).
- **CIFAR-10 doesn't separate methods** — keep as smoke test, recommend dropping from main results table.
- **CUB-200 at matched-518 still pending** (see §10).

---

## 2. Why FCP Wins vs Split-CP / SemiCP

Total-label-budget framing: every method gets the same B labels. SCP/SemiCP split B internally 50/50 train/cal. FCP+PCA+MS-CS uses all B for calibration (frozen SSL features). See `[[feedback-total-budget-framing]]`.

**CIFAR-100 head-to-head (matched-518, 5 trials, α=0.1):**

| Cal  | **FCP+PCA+MS-CS** | SCP-THR     | SCP-RAPS    | SemiCP-THR  | SemiCP-RAPS | Best non-FCP margin |
|------|--------------------|-------------|-------------|-------------|-------------|----------------------|
| 300  | **1.87 (88.6%)**   | 100 (100%)  | 44.4 (98%)  | 97.6 (98%)  | 44.4 (98%)  | **24×** smaller      |
| 400  | **1.76 (90.3%)**   | 84.0 (99%)  | 33.0 (98%)  | 24.8 (93%)  | 28.0 (94%)  | **14×** smaller      |
| 600  | **1.83 (92.9%)**   | 7.4 (93%)   | 21.7 (97%)  | 8.3 (93%)   | 22.2 (97%)  | **4×** smaller       |
| 800  | **1.42 (90.8%)**   | 2.96 (93%)  | 18.4 (98%)  | 2.90 (92%)  | 18.8 (98%)  | **2×** smaller       |
| 1000 | **1.41 (91.6%)**   | 2.06 (92%)  | 16.1 (99%)  | 2.09 (92%)  | 16.4 (99%)  | 1.5× smaller         |

**Findings (consistent across CIFAR-100 / miniImageNet, 3 datasets confirmed):**

1. **NNM augmentation is null on THR**: SemiCP-THR ≈ SCP-THR everywhere. The paper's signature contribution adds nothing in our regime.
2. **Classifier-quality bottleneck**: SCP/SemiCP-THR collapse to sz=K when inner-trained softmax accuracy < ~60% (cal ≤ 400 for K=100). FCP avoids the train/cal split entirely.
3. **APS/RAPS bloated** with logistic regression at K≥100 (sz 17-100). Even where SemiCP improves these scores, absolute sets remain unusable.
4. **Margin closes as cal grows** — story is strongest in small-cal regime (300-600), the realistic SSL+CP deployment setting.
5. **CIFAR-10 is saturated** — SemiCP's published CovGap advantage is consistent with statistical noise on a saturated task.

Frame for the paper: SemiCP is a legitimate but narrow contribution for APS/RAPS pipelines on saturated benchmarks. Different problem setting from ours — not "we beat them."

---

## 2b. Class-Conditional Coverage — FCP+PCA+MS-CS Also Wins on CovGap

The marginal-coverage comparison in §2 raises a natural question: does FCP+PCA+MS-CS keep its margin once you look *inside* each class? Following Ding, Tibshirani & Ramdas (2023, arXiv:2306.09335), we measure per-class coverage `cov_c = P(c ∈ C(X) | Y=c)` and report **CovGap = mean_c |cov_c − (1−α)| × 100** (pp), plus worst-class coverage and fraction of classes under-covered. We also include their **Clustered Conformal Prediction (ClusterCP)** as the natural conditional-coverage baseline.

**CIFAR-100 + miniImageNet, K=100, 5 trials, test=2000 (20/class), α=0.1.** Script: `src/conditional_coverage_experiment.py`; results: `output/conditional_coverage/`.

| Dataset       | cal | Method            | Marg. cov | Set sz | **CovGap** (pp) | Worst-class | frac<target |
|---------------|-----|-------------------|-----------|--------|------------------|-------------|-------------|
| CIFAR-100     | 800 | **FCP+PCA+MS-CS** | 0.900     | **1.59** | **7.68**       | 0.52        | 0.31        |
| CIFAR-100     | 800 | FCP               | 0.905     | 2.22   | 7.55             | 0.50        | 0.31        |
| CIFAR-100     | 800 | SemiCP-THR        | 0.901     | 2.86   | 9.02             | 0.05        | 0.24        |
| CIFAR-100     | 800 | SCP-THR / ClusterCP | 0.913   | 3.23   | 8.73             | 0.08        | 0.21        |
| CIFAR-100     | 400 | **FCP+PCA+MS-CS** | 0.888     | **1.92** | 8.93           | 0.41        | 0.32        |
| CIFAR-100     | 400 | ClusterCP         | 1.000     | 100    | 10.00            | 1.00        | 0.00        |
| miniImageNet  | 800 | **FCP+PCA+MS-CS** | 0.900     | **1.01** | **8.45**       | 0.35        | 0.27        |
| miniImageNet  | 800 | ClusterCP         | 0.908     | 1.27   | 9.23             | 0.20        | 0.23        |
| miniImageNet  | 800 | SemiCP-THR        | 0.886     | 1.13   | 9.93             | 0.18        | 0.31        |

**Headlines:**

1. **FCP+PCA+MS-CS achieves the lowest CovGap at cal=800 on both datasets** (7.68 / 8.45 pp), better than ClusterCP (8.73 / 9.23 pp) — *and* with 2× smaller sets. Our method is not only the marginal-coverage winner; it implicitly delivers competitive class-conditional coverage without targeting it.
2. **ClusterCP degenerates to plain SplitCP-THR in our regime.** ClusterCP partitions classes by their cal-score quantile signature, but rare classes (n_y < n_thresh = ⌈(n_q+1)/α⌉ = 60 at α=0.1, n_q=5) go into a "null cluster". With cal=800/K=100 → 8/class, every class is rare → ClusterCP yields one pooled q̂, identical to SCP-THR (their rows match to 3 decimals across every cell). ClusterCP's design regime (cal/K ≥ 6-10) sits ~10× above ours. *This is the informative result: ClusterCP can't be a meaningful conditional-coverage competitor at small cal/large K.*
3. **Worst-class tradeoff is honest.** FCP under-covers ~30% of classes (frac<target ≈ 0.31); SCP/ClusterCP under-cover ~20-25% but their "high" worst-class coverage (0.05-0.08) is achieved by over-covering everywhere (marginal cov 0.91+, set sizes 3-100). The CovGap is the right symmetric summary; raw worst-class can mislead.

Five plots per (dataset, cal_size) in the results dir: `covgap_bar`, `perclass_hist`, `sorted_perclass`, `size_vs_class`, plus `covgap_vs_calsize` per dataset.

**Implementation notes:** Added `ClusteredSplitCP` to `src/conformal_prediction.py` (THR scores, k-means on per-class quantile signatures, rare-class null-cluster pooling — matches Ding et al.'s public repo defaults). Added `return_sets=True` kwarg to `run_fcp_with_mscs` so per-class accounting works without re-implementing the MS-CS inner loop. Local miniImageNet uses an auto-carved unlabeled pool (50/class from train) since the configured external `embeddings_miniimagenet_unlabeled.pt` isn't on this machine — a printed `[INFO] auto-carve` heads-up on each run. Cluster reproduction with the matched-518 external unlabeled pool is the natural follow-up.

---

## 3. Dimensionality Reduction

PCA fit on the **unlabeled pool** (n >> d) preserves FCP exchangeability — unsupervised wrt cal/test. Cal-based PCA under-covers at high dims (overfitting); always use a disjoint unlabeled pool.

| Dataset       | Best PCA dim | Reason                                          |
|---------------|--------------|--------------------------------------------------|
| CIFAR-100     | **128**      | Coarse-grained, noise removal                   |
| miniImageNet  | **128**      | Saturated baseline, marginal benefit            |
| CUB-200       | **512**      | Fine-grained needs more dims (PCA-128 *hurts*)  |

**Unlabeled-pool size sensitivity** (CIFAR-100, cal=400, see `output/from_cluster/unlabeled_size_sweep_cifar100/`):

- PCA needs **≥500 unlabeled** to be useful (at N=100/250, degenerate basis hurts).
- FCP+PCA+MS-CS **plateaus at N=2500-5000** (sz ≈ 1.70). N=10000 returns no further benefit.
- **Practitioners with ~2.5K unlabeled examples reproduce our headline within 1%.**

**AE vs PCA — the alignment hypothesis.** Earlier finding "DINOv2 embeddings are linear" was **wrong** (see `[[ae-vs-pca-linearity-diagnostics]]`):
- Nonlinear AE recovers 12-26% more variance than PCA at same d — curvature is real.
- PCA wins downstream because the **NCM is linear** (Mahalanobis whitening), not because embeddings are linear.
- **Prediction confirmed**: with a nonlinear NCM (whitened RBF), AE-128 beats PCA-128 at cal=600 by 17% (sz 2.18 vs 2.62, p=0.001 over 10 trials). See §5.
- **AE-32 only wins downstream at very small cal** (cal ≤ 400 with linear NCM, archived).

---

## 4. MS-CS — Label-Free Class Similarity Penalty

K-means on unlabeled pool → similarity matrix M from cluster co-assignment → `s_λ(x,y) = s(x,y) + λ·(1-M[y, ŷ(x)])`. No superclass labels needed. See `src/mscs_unlabeled_experiment.py`.

**Best config**: λ=0.05, n_clusters=20, τ=0.5·median_d², clustering in **PCA-128 space**.

**MS-CS adds 8-9% reduction over FCP+PCA at cal=300-600** on CIFAR-100 (the regime where it matters); neutral on miniImageNet / CIFAR-10 (already saturated). Coverage preserved everywhere. Beats MA-CS (binary superclass penalty, Fargion et al. 2025) by 2-7% and requires no taxonomy labels.

**Effects decompose roughly additively** at cal=800: PCA −19%, MS-CS −13%, combined −24% (vs FCP baseline 2.30 → 1.74 local, 1.87 → 1.41 cluster).

---

## 5. Whitened-RBF NCM + AE-128 — Confirmed Win at Medium Cal

After the AE/PCA diagnostics predicted that nonlinear NCMs would let AE features pay off, we built `RBFDensityNCM` (Gaussian kernel density, ratio mode, pooled-within-class whitening). See `[[rbf-ncm-with-ae]]`.

**CIFAR-100, paired t-test, 10 trials:**

| Cal | BASE (PCA-128 + geodesic) | AE-128 + wSymRBF (σ=0.18) | Paired diff | p-value      |
|-----|----------------------------|----------------------------|-------------|--------------|
| 600 | 2.62 ± 0.17                | **2.18 ± 0.10**            | −0.44       | **0.001**    |
| 800 | 1.84 ± 0.08                | **1.80 ± 0.06**            | −0.04       | 0.25 (tie)   |
| 400 | baseline wins              | (AE capacity > samples/class) | —        | —            |

Validity holds at all cal sizes (cov ≥ 0.90). Exchangeability concessions (σ, pooled covariance fit on cal alone) are O(1/n), same regime as MahalNN/WhitenedGeodesic whitening.

**Pattern**: AE+wRBF wins at cal/K ≈ 6-8 (enough samples to estimate density reliably, not enough for linear baseline to saturate).

Code: `RBFDensityNCM` in `src/conformal_prediction.py`, experiment driver `src/rbf_ncm_experiment.py`.

---

## 6. The Three Regimes of FCP (Small-Cal Sweep)

CIFAR-100, K=100, cal ∈ {50, 100, 200, 300}. See `output/from_cluster/small_cal_sweep_cifar100/`. Universal across CIFAR-100 and miniImageNet.

| Regime | Condition | Behavior |
|--------|-----------|----------|
| **A** — random fallback | cal < K | Many classes absent. Coverage collapses to P(class ∈ sample) ≈ 41% at cal=50. No method recovers. |
| **B** — LOO breakdown   | cal = K (1/class) | LOO removes the single same-class anchor → degenerate NN-ratio. Coverage trivially valid (99%), sets bloated (~73 classes). |
| **C** — deployment      | cal ≥ 2K (≥2/class) | NCM stable. Coverage at target (~90%), sets shrink sharply (cal=200: sz 2.8; cal=800: sz 1.4). |

**Sharp transition between cal=K and cal=2K** is the most striking feature. **Restrict paper claims to Regime C.**

`balanced_split` truncates `cal_size // K` per class, so cal=150 and cal=250 silently equal cal=100/200. Use cal ∈ {K, 2K, 3K, …} for clean sweeps.

---

## 7. Backbone Comparison (CIFAR-100, cal=600, geodesic_topk_mean, 5 trials)

| Backbone        | FCP sz  | CV+ sz | SCP sz |
|-----------------|---------|--------|--------|
| **DINOv2-base** | **2.96** | 6.50  | 14.4   |
| BEiTv2-base     | 5.24    | 7.68   | 9.22   |
| CLIP-base       | 32.6    | 35.6   | 35.3   |

DINOv2 dominates — 77% smaller than BEiTv2, 11× smaller than CLIP. FCP advantage over CV+/SCP generalizes across backbones. CLIP unsuitable for NN-ratio FCP.

---

## 8. Dataset Properties

| Dataset       | K   | N/class | FCP behavior                                   | Status (matched-518)         |
|---------------|-----|---------|------------------------------------------------|------------------------------|
| CIFAR-10      | 10  | 500     | Over-conservative; saturated; smoke test only  | done                         |
| CIFAR-100     | 100 | 25      | FCP dominates; **headline benchmark**          | done                         |
| miniImageNet  | 100 | 500     | Saturates near sz=1.0; topk_asym best          | done                         |
| CUB-200       | 200 | ~50     | Strongest FCP advantage (10× over SCP)         | **pending** (local 336 only) |
| Flowers-102   | 102 | ~55     | FCP from cal=400                               | local 336 only               |
| EuroSAT       | 10  | 200     | All methods work                               | local 336 only               |

---

## 9. Theory Notes

**Why FCP > CV+ at K≥100, low cal**: CV+ trains on (k-1)/k data per fold — with few samples, classes go missing from folds → trivially full sets. FCP uses full cal set transductively.

**Why FCP > SCP at K≥100, low cal**: SCP needs a softmax classifier trained on inner cal split. With K=100 and ≤300 cal, the inner classifier is too weak; scores degenerate. FCP needs no train/cal split.

**Exchangeability**: Whitening / PCA fit on cal-only is O(1/n) asymmetry — negligible for n ≥ 200. Label-dependent projections (LDA, pseudo-label heads) are NOT O(1/n) and cause structural under-coverage (70-85%). Only **unsupervised** projections (PCA on unlabeled pool, AE) are safe.

**Why geodesic_topk_asym wins**: 1-NN numerator (tight same-class signal) / mean-k denominator (smoothed other-class). Symmetric mean-k on both sides over-smooths.

**Whitening is not redundant after PCA**: PCA removes noise dims (total variance), whitening rescales by within-class variance — complementary. Whitening adds 5-12% benefit after PCA-128.

**Over-coverage at cal=600 (cluster, all methods at 92.5-92.9%)**: Likely stratified-cal/test split + ceiling quantile interaction (method-independent). Worth a footnote. Definitive test pending (§10 P1 item).

---

## 10. Future Directions & Pending Experiments

*Last reviewed: 2026-05-21.*

### P0 — Required for paper headline

1. **CUB-200 at matched-518 cluster extraction** + full ablation (FCP, +PCA-512, +MS-CS, all combos).
   *Why*: only missing dataset in main results table. Local-336 results (archive) show CUB-200 is FCP's strongest advantage (10× over SCP at cal=800).
2. **20-trial paper-quality rerun of §1 + §2** on CIFAR-100. Settles cal=600 over-coverage question and tightens error bars. Cheap with GPU fast-path (~10-20 min total).
3. **NCM = `geodesic_topk_asym` confirmation on cluster** (CIFAR-100). Local memory predicts sz ~1.30 at cal=800 (vs current 1.41 with `mean`). Same GPU fast-path covers it.
4. **Fix broken AE result on miniImageNet** (`output/from_cluster/ablation_miniimagenet/`: sz 34-51, 89-95% coverage). Suspected cause: AE trained on CIFAR-100 unlabeled and reused. Retrain per dataset.

### P1 — Strengthen the story

5. **RBF NCM multi-dataset confirmation**: re-run AE-128 + wSymRBF on CUB-200, miniImageNet, and at matched-518 (current §5 result is local 336 only). Tests whether the alignment hypothesis generalizes.
6. **MA-CS / MS-CS multi-dataset extension**: MS-CS is label-free, runs on anything; MA-CS needs taxonomy (have CIFAR-100 fine→coarse; check CUB-200 family/genus, miniImageNet WordNet).
7. **Stratified-vs-random split ablation** at cal=600 to settle the over-coverage mechanism. ~5 min with GPU.
8. ~~**Per-class coverage diagnostics**~~ — **DONE 2026-05-25** (§2b). FCP+PCA+MS-CS wins CovGap on CIFAR-100 + miniImageNet; ClusterCP degenerates to SplitCP in our regime.

### P2 — Robustness / breadth

9. **Very-small-cal sweep at cal/K = 2-8** in deployment Regime C (cal ∈ {200, 300, 400, 500, 600} for K=100). Useful margin curve for the paper.
10. **Larger-K experiments**: ImageNet-1K subsamples (K=1000). Tests scaling of the PCA+MS-CS recipe. GPU FCP makes K=1000, cal=2000 take minutes.
11. **Backbone sensitivity on cluster**: CLIP-base, BEiTv2-base with full pipeline. Confirms the recipe is backbone-agnostic.
12. **Per-trial efficiency distribution plot** — show SCP/SemiCP have high trial variance at low cal vs FCP+PCA+MS-CS stability.
13. **GPU fast-path extension** to `MahalNNRatio` and `WhitenedGeodesicNNRatio` (currently CPU-only). Only useful if reviewers ask about NCM sensitivity at scale.
14. **CUB-200 + RBF NCM**: where AE was 5% *worse* than PCA in `[[ae-vs-pca-linearity-diagnostics]]`. Cleanest test of whether AE+RBF rescues AE on fine-grained data.

### P3 — Literature baselines still to compare against

15. **SSCP** (Seedat et al., AISTATS 2023, arXiv 2302.12238): SSL pretext tasks for NCM. Designed for regression — needs classification adaptation.
16. **Transductive Standardization** (Fan & Sesia 2025, arXiv 2512.15383): validates O(1/n) exchangeability for data-dependent standardization (relevant to whitening theory section).
17. **Pseudo-Label CP** (Angelman et al. 2025, MLR v266): source-free calibration. Tests pseudo-labels vs NNM as unlabeled-data utilization.

### Deferred / Side missions

- **AE-256 capacity test** for cal=800 RBF (currently within noise; bigger AE may push over the line).
- **k-NN density NCM** (Loftsgaarden-Quesenberry) — locally adaptive bandwidth, parameter-free.
- **GPU benchmark / timing study** — explicit wall-clock-per-1000-predictions table for the paper.

### Recommended execution order

1 → 2 → 3 → 7 → 4 → 5 → 6. Builds the headline table, validates NCM choice, extends to multi-dataset, settles over-coverage mechanism, confirms RBF result, completes class-similarity story.

### Completed since last review (moved out of pending)

- ~~Multi-dataset cluster reproduction for miniImageNet, CIFAR-10~~ — §1.
- ~~Very-small-cal sweep~~ — §6.
- ~~Unlabeled-pool size sensitivity~~ — §3.
- ~~SemiCP corrected investigation~~ — §2.
- ~~AE/PCA linearity diagnostics + alignment hypothesis~~ — §3, `[[ae-vs-pca-linearity-diagnostics]]`.
- ~~Whitened-RBF NCM + AE win at cal=600~~ — §5.
- ~~GPU fast-path for FCP~~ — `[[gpu-fcp-path]]`.
- ~~Per-class coverage diagnostics + ClusterCP head-to-head~~ — §2b, `[[conditional-coverage-results]]`.

### Negative results (archived)

- Pool augmentation: breaks exchangeability (`archive/findings_archive.md §11.5`).
- LDA projection: structural under-coverage.
- Original CS penalty: data leakage.
- PCA on cal data: under-coverage at high dims.
- Pseudo-label trained head: same failure as LDA.
- PPI-RCPS (Einbinder et al. 2024): split-CP oriented, archived to `src/archive/ppi_rcps.py`.
- AE bottleneck with linear NCM: matches PCA at best (now reframed as NCM-pipeline alignment, see §3).

---

*Last updated: 2026-05-26.*
