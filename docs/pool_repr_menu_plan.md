# Pool-fit representation for conformal prediction: denoise -> whiten -> discriminate

**What this document is.** A self-contained explanation of the pool-fit
representation line (2026-08-03..05): the theory that motivates it, the
three-stage pipeline, and the experiments behind every claim. Raw results
and figures under `output/pool_repr_menu/`; per-round lab history in this
file's git log. The previous truncate-then-whiten stages are kept as
Appendix A (comparison baseline).

---

## 1. Setting and main claim

Full conformal prediction (FCP) for K-way classification (K >= 100) on
frozen SSL embeddings (DINOv2, 768-d), with a tiny labeled calibration set
(200-800 points) and a large UNLABELED pool (3-10k points). The question:
**what is the most the unlabeled pool can buy, and where in the pipeline
should it be spent?**

Answer: spend all of it in the REPRESENTATION, before any label is
touched. One fixed map, fit once on the pool:

```
T(x)  =  E^T  W0  ( D(x) - mu )

D    stage 1  denoise    pull x toward its pool neighbors
mu            center     pool mean
W0   stage 2  whiten     equalize the within-cluster (noise) covariance
E    stage 3  discriminate  keep the d' directions that survive whitening
                            with the most variance (= where cluster means
                            spread most)
```

- **Exact coverage for free** (sec 2) — the architectural point and the
  main novelty claim: the representation slot for unlabeled data in CP is
  empty in the literature (existing uses act on scores or thresholds:
  SemiCP, SNAPS, unsupervised calibration; `literature.md` sec 9), and we
  show the representation-level use strictly subsumes the score-level one
  (sec 4).
- **Best or near-best known set sizes** with ONE construction across
  regimes: cifar100 @ cal 200 drops 4.17 -> 2.27; CUB-200 @ 400 drops
  3.65 -> 2.73; aircraft ties its champion. Two knobs total: d'
  (regime-tracked) and the denoiser strength (safe default, sec 6).

## 2. Why the pool-fit phase gives exact validity (the one theory fact)

CP needs the calibration + test scores to be exchangeable. A transform fit
ON calibration data breaks that symmetry; a transform that is a fixed
function of the POOL alone treats every cal and test point identically —
T = A(pool) is measurable w.r.t. data independent of the bag, so
Proposition 2 (`docs/theory.md` sec 2) applies and coverage is exactly
1-alpha for ANY such transform. Intuition: the pool is frozen scenery —
whatever geometry we carve out of it, cal and test walk through the same
carved landscape, so their ranks stay exchangeable.

Consequences: (1) validity never constrains the search, only the
information set — the whole design question is efficiency; (2) this is why
the representation is the right home for pool information: the score-level
alternative (our SNAPS adaptation) cost a leak repair, an O(1/n) validity
gate, and a full-CP wrapping proof for LESS gain (sec 4). Why sets can
shrink at all: expected set size is a separation functional of the
true-vs-false score populations (`docs/conformal_metric_objective.md`
sec 2; sec 5 there also shows accuracy-style surrogates are structurally
misaligned — hence every experiment below is judged by true CP size only).

## 3. The three stages (brief, term by term)

### Stage 1 — D: pool-neighbor denoising (alpha-QE)

Ported from image retrieval (Radenovic, Tolias & Chum, TPAMI 2019, Sec 3.5
+ 5.3; database-side variant: Turcot & Lowe 2009, Gordo et al. 2017). With
s_i = max(cos(x, u_i), 0) over the k nearest pool points u_i:

```
classic:  D(x) = L2norm( x + sum_i s_i^alpha u_i )              k=10, alpha=3
safe:     D(x) = L2norm( (1-b) x + b * weighted-neighbor-mean )  b=0.3, k=5
```

Averaging a point with its nearest pool neighbors shrinks off-manifold
noise ~1/sqrt(k) while keeping the class-carrying position; s^alpha
down-weights far (likely wrong-class) neighbors. A cut-the-pipeline
ablation located the mechanism: INPUT denoising (the cal/test points are
the noisy objects at 2-8 shots/class, not the pool statistics). The safe
form caps the neighbor mass — it is not-worse-than-baseline in every cell
we tested, including the adversarial fine-grained regime (sec 6).

### Stage 2 — W0: within-cluster whitening (equalize the noise)

k-means the pool (C clusters, **C >= K** — see below); treat clusters as
label-free stand-ins for classes:

```
Sigma_W = (1/N) sum_u (u - m_c(u)) (u - m_c(u))^T    within-cluster scatter
          (Ledoit-Wolf shrunk: well-conditioned at any N/d)
W0      = Sigma_W^{-1/2}                             ZCA whitening root
```

Intuition: Sigma_W estimates "variation that does NOT change identity"
(the analogue of within-class noise). W0 makes that noise isotropic, so
the angular NCMs stop being dominated by a few large nuisance directions.
The C >= K rule is structural: with fewer clusters than classes, clusters
merge classes and class signal leaks INTO Sigma_W — whitening then
suppresses exactly the signal (measured: CUB with C=100 < K=200
degenerates; C=300 fixes it). C is set by known constants, no estimation.

### Stage 3 — E: discriminative truncation (keep what survives whitening)

```
E = top-d' eigenvectors of  Cov( W0 (U - mu) )       (whitened pool)
```

One sentence: after the noise is equalized to ~unit size in every
direction, the directions that STILL have large variance are the ones
where cluster means spread — i.e. identity-carrying directions. Formally,
by the exact scatter decomposition Sigma_T = Sigma_W + Sigma_B, the
whitened total covariance is I + W0 Sigma_B W0, so ranking by whitened
total variance selects the same subspace as ranking by between-cluster
spread (eigenvalue shift; exact pre-shrinkage, certified empirically
post-shrinkage — the explicit between-ranked form reproduces these
results within ~1 SE). d' is the one regime knob: 128 on separable data,
512 on fine-grained (tracked by the pool participation ratio, Appendix A).

Lineage: this stage IS the "learned whitening" of Radenovic et al.
Sec 3.4 (originally Mikolajczyk & Matas) with one substitution — their
matching/non-matching pairs from SfM geometry become our k-means
pseudo-clusters, keeping the construction fully label-free and pool-only.

### Order

D runs first and in the RAW cosine metric — the only metric trusted a
priori (SSL aligned it); whitening AMPLIFIES low-variance directions, so
denoising must precede it (equalize-then-smooth was tested: catastrophic
exactly where whitening is full-rank — aircraft 34.6 vs 28.2). W0 before
E is definitional (E is computed in the whitened space).

## 4. Experiments behind each claim (condensed; figures by path)

All: balanced splits, 10-20 trials, coverage verified ~0.90 everywhere (as
sec 2 guarantees), judged on mean set size.

- **Stage 1 discovery + kills** (`transform_controls_*.png`): a
  pre-registered menu of four pool transforms; qe was the survivor
  (cifar100 4.17 -> 2.34 @ cal 200 on the then-pipeline), Yeo-Johnson
  Gaussianization a no-op, LPP killed with mechanism (low-PR pool kNN
  graphs are majority wrong-class — pool NEIGHBORHOOD structure is
  unreliable at low PR, only second-order statistics are), soft
  per-cluster whitening dropped by the pre-registered rule.
- **Representation subsumes scores** (`snaps_stack/stack_corners.png`,
  `stack_mechanism.png`, `runtime_qe_vs_snaps.png`): 2x2 vs our SNAPS
  score correction — qe alone >= SNAPS alone everywhere; SNAPS's marginal
  gain collapses post-qe (-41% -> -6.6% at the most starved cell, ~0
  elsewhere) with its best mixing weight shrinking toward 0; and it costs
  174-410 ms of machinery per recalibration vs qe's fixed 0.4 ms/point.
- **Knobs + safe mode** (`qe_knobs/qe_knob_heatmaps.png`): flat surface on
  separable data; k tracks pool shots/class; alpha=0 (plain AQE) is worse
  than alpha=1 in 23/27 cells (the similarity guard is a small consistent
  refinement); beta=0.3 removes the last harm regime — safe mode is
  not-worse-than-incumbent in all 13 cells tested (7 wins, 6 ties).
- **Stages 2-3 adoption** (`ldapool/` JSONs): the discriminant transform
  beats the old truncate-then-whiten ordering pre-qe on separable data
  (2.73 vs 4.17 @ cifar-200), ties the aircraft champion at d'=512, and
  with C >= K wins CUB among qe arms (2.73/1.62/1.37 @ C=300). The
  between-ranked certification form reproduces it within ~1 SE.
  Completion runs in flight: safe-qe composition on aircraft/CUB, mini,
  and a d' fine-scan on the two cells where Appendix A's menu still leads.

## 5. Bottom line (10-trial standings; * = completion runs in flight)

```
                       cal 200        cal 400        cal 800        cal 1600
cifar100 (proto)     4.17 -> 2.27   1.64 -> 1.43   1.30 -> 1.31*      -
cub200   (proto)          -         3.65 -> 2.73   1.73 -> 1.62   1.31 -> 1.37*
aircraft (geo)      29.01 -> 28.69* 24.61 -> 24.66* 22.03 -> 22.37*    -
miniImageNet             *              *              *               -
(left value = best no-pool-repr incumbent; right = unified pipeline;
 the two starred gap cells are where Appendix A's per-cell menu still
 leads: cifar@800 1.23, CUB@1600 1.23)
```

## 6. Deployment and the (performance-only) gate

Safe defaults need NO estimated quantity: stage 1 in safe mode (k=5,
alpha=3, b=0.3), C = max(100, 1.5 K), d' by the regime rule. Validity is
exact regardless; safe mode has no known harm cell. The only estimated
input is a label-free homophily proxy hom_hat, used purely for
performance tuning: switch stage 1 to classic (bigger small-cal gains,
10-15%) when hom_hat >= ~0.75 and cal <= ~800. Wrong-switch cost is
bounded and asymmetric (conservative thresholds make the costly error
unreachable). Estimator panel = open work.

## 7. Status

- DONE: stage-1 line (discovery, mechanism, SNAPS subsumption, knobs,
  safe mode, order), stages 2-3 replacement (formalization,
  certification, C >= K rule, adoption decision).
- OPEN: (a) completion cells (sec 5 stars); (b) hom_hat estimator panel;
  (c) 50-trial cluster hardening before write-up; (d) side anomaly:
  prototype + pca512 @ aircraft cal-800 = 14.90, vs the "prototype bloats
  on fine-grained" scope limit.

---

## Appendix A — the previous stages 2-3: truncate-then-whiten menu

What the pipeline used before the discriminant transform (and the
comparison baseline throughout `output/pool_repr_menu/`):

```
T_old(x) = W_cw * P_d'^T * (D(x) - mu)

P_d'  top-d' eigenvectors of the RAW pool covariance (plain PCA);
      d' by regime: 128 separable / 512 mid / none on collapsed spectra
W_cw  per-dimension within-cluster whitening in the projected space
      (diagonal); on collapsed spectra instead: full-rank Ledoit-Wolf
      ZCA at 768 with NO truncation (lw_cluster768)
```

I.e. a per-regime MENU of three arms {pca128_cw, pca512_cw,
lw_cluster768}, selected by the pool participation ratio
PR = (sum l_j)^2 / sum l_j^2 (cifar/mini ~240 -> 128; CUB ~58 -> 512;
aircraft ~16 -> no truncation). Its justification record: JL and
tail-probe controls (PCA's data-adaptivity is the lever, reduction per se
is not; aircraft's signal really is in the low-variance tail),
Gavish-Donoho spiked-spectrum anchors (valid where truncation works,
provably inapplicable where it does not).

Differences to the adopted stages 2-3: the menu RANKS directions by raw
total variance and equalizes afterwards; the discriminant equalizes first
and ranks by what survives. The orders coincide when the top spectrum is
signal (separable data, hence near-ties there with qe) and diverge when
signal hides in low-variance directions. Where the menu still leads
(pre-completion-runs): cifar@800 (1.23 vs 1.31) and CUB@1600 via
prototype+pca512 no-qe (1.23 vs 1.37). Where it loses: cifar/CUB small
cal pre-qe (4.17 vs 2.73; degeneracy-free behavior needs three different
arms), and it needs a per-regime selection rule the single construction
does not.

[Reserved: if the per-cell menu is later re-adopted for the last few
tenths in the gap cells, it goes here as an OPTIMIZATION EXTENSION of the
single construction — same ingredients, per-regime re-ordering.]
