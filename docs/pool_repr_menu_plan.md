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
  Headline view: `champion/champion_lines.png`.

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
- **NCM re-audit in T-space** (`ncm_audit/` JSONs): prototype-softmax is
  the matched NCM of the new geometry — it dominates the geodesic
  variants on cifar (2.29 vs 6.08 @200) and CUB, and **overturns the
  "prototype bloats on fine-grained" scope limit on aircraft at
  cal >= 400: 21.91 / 19.81 vs the geodesic 24.29 / 22.49 — new best
  aircraft cells (-11% / -10% vs the old champion)**. Prototype still
  degenerates at aircraft cal-200 (2 shots/class), where geodesic mean
  remains the fallback; saturated mini slightly favors geodesic at high
  cal (0.98 vs 1.04). CovGap fine print: the aircraft prototype win
  costs ~2pp CovGap vs geodesic.
- **Stage ablation, 2^3 over {D, W, T}**
  (`stage_ablation/pipeline_stage_ablation.png`): stage contributions
  are regime-dependent and the interactions match the theory — on cifar
  every stage helps and D is the biggest single stage (raw 10.1 -> 3.5
  @200); on aircraft W is the whole story (raw 57.7 -> 28.9) with D and
  T useless ALONE (both ~raw) and D adding only on top of W; on CUB, W
  alone HURTS (+13% @400) and pays only combined with T. No stage is
  redundant, and no stage works in every regime alone — the composition
  is what carries.
- **Completion runs (the gap cells)**: safe-qe composes cleanly with the
  discriminant — aircraft 27.38 / 24.29 / 22.49, matching the menu's best
  arm at every cal; CUB safe-qe @ C300 reaches 1.24 @ 1600 (crown 1.23 —
  closed) plus a new best 1.56 @ 800; the cifar d' scan closes the 800
  cell at d' = 192-256 (1.27 vs 1.23, ~1 SE; d' = 192 is near-optimal at
  every cifar cal: 2.29 / 1.43 / 1.27). miniImageNet: ldapool ties the
  menu pre-qe; with qe a small residual deficit remains on this saturated
  dataset (1.24 / 1.06 / 1.04 vs 1.19 / 1.03 / 0.99).

## 5. Bottom line (10-trial standings, completion runs folded in)

```
                       cal 200        cal 400        cal 800        cal 1600
cifar100 (proto)     4.17 -> 2.27   1.64 -> 1.43   1.30 -> 1.27       -
cub200   (proto)          -         3.65 -> 2.73   1.73 -> 1.56   1.31 -> 1.24
aircraft            29.01 -> 27.38  24.61 -> 21.91  22.03 -> 19.81     -
  (geo @200 / PROTOTYPE @400+ after the T-space NCM re-audit, sec 4)
miniImageNet         1.35 -> 1.24   1.05 -> 1.06   1.00 -> 1.04        -
(left = best incumbent without the pool representation; right = the
 unified pipeline, best qe-mode per cell. Knobs behind the right column:
 d' = 192 cifar / 512 CUB (C=300) / 512 aircraft / 128 mini; qe classic
 at small cal, safe at high cal / low homophily — the sec-6 gate rule.
 vs Appendix A's per-cell menu: within ~1 SE everywhere (cifar@800 1.27
 vs 1.23, CUB@1600 1.24 vs 1.23, aircraft@800 22.49 vs 22.03) except
 mini's saturated cells (-0.03..-0.05 absolute).)
```

Figures for this table (`champion/`):
- `champion_lines.png` — set size vs cal, all four datasets, four series
  (raw embeddings / old menu best / old + SNAPS best / new champion);
  the small-cal separation is the paper's positioning in one picture, and
  the aircraft panel carries the SNAPS-harm annotation in place of a
  curve.
- `champion_gains.png` — % set-size change of BOTH levers vs the old
  champion: the representation-level lever is deeper than the score-level
  one in every cell where the latter works at all, and still delivers
  where it is gated off (aircraft cal-200: -6%).
- `champion_covgap.png` — the conditional-coverage fine print: CovGap
  ties-or-improves on cifar but costs a mild +0.5-1.2 pp on
  aircraft / mini / CUB@1600. This is the one known caveat of the adopted
  pipeline (marginal coverage stays exact everywhere by sec 2); whether
  the safe-qe knob modulates it is an open follow-up.

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
- OPEN: (a) hom_hat estimator panel; (b) 50-trial cluster hardening of
  the sec-5 table before write-up; (c) the aircraft anomaly DEEPENED:
  prototype + pca512_cw (old-style space, no qe) hit 14.90 @ cal-800 in
  round 1 — still far below the new T-space prototype champion (19.81).
  If it replicates, the aircraft-optimal space for the prototype NCM is
  not the discriminant one, and the sec-5 aircraft cells have ~25% more
  headroom. Top-priority follow-up.

---

## Appendix A — the previous pool-fit pipeline (PCA + whitening, before alpha-QE)

What the pool bought us BEFORE this line of work: no denoising stage at
all — raw embeddings went straight into a linear map fit on the pool
(same exact-validity argument, sec 2):

```
T_old(x) = W_cw * P_d'^T * (x - mu)

P_d'  plain PCA: the top-d' eigenvectors of the raw pool covariance —
      "keep the biggest directions"
W_cw  whitening AFTER the cut: per-dimension within-cluster variance
      equalization in the projected space (k-means pseudo-classes)
```

Simply put: rank directions by raw variance, keep the big ones, then
equalize the noise inside what was kept. Because raw variance is not
always where class signal lives, this needed a per-regime MENU: pca128_cw
on separable data, pca512_cw on mid (CUB), and on collapsed spectra
(aircraft) truncation had to be abandoned entirely in favor of full-rank
Ledoit-Wolf ZCA whitening at 768 (lw_cluster768). The regime was selected
by the pool participation ratio PR = (sum l_j)^2 / sum l_j^2 (cifar/mini
~240 -> 128; CUB ~58 -> 512; aircraft ~16 -> no truncation). Its
justification record: JL and tail-probe controls (PCA's data-adaptivity
is the lever, reduction per se is not; aircraft's signal really is in the
low-variance tail), Gavish-Donoho spiked-spectrum anchors (valid where
truncation works, provably inapplicable where it does not).

**How the adopted pipeline differs — three changes:**

1. **A denoising stage now exists (stage 1, alpha-QE).** The old pipeline
   fed raw points into the linear map; the binding noise at 2-8 shots per
   class — in the cal points and test queries themselves — was untreated.
   This stage alone is most of the small-cal gain.
2. **The order of the two linear operations flipped.** Old:
   rank-by-raw-variance, THEN equalize (truncate-then-whiten). New:
   equalize the within-cluster noise FIRST, then rank by what survives
   (whiten-then-discriminate). The orders coincide when the top spectrum
   IS the signal (separable data — hence near-ties there) and diverge
   exactly when signal hides in low-variance directions, which is the
   regime that forced the old menu's special full-rank arm.
3. **One construction instead of a menu.** The old pipeline needed three
   different arms plus a PR selection rule; the new one is a single
   formula whose only regime knob is d' (plus C >= K, a known-constants
   rule). After the completion runs the menu's remaining lead is statistical noise
on the big datasets (cifar@800 1.23 vs 1.27; CUB@1600 1.23 vs 1.24;
aircraft@800 22.03 vs 22.49 — all ~1 SE) plus a small real deficit on
saturated miniImageNet (0.99 vs 1.04 @800). Where it loses: cifar/CUB
small cal pre-qe (4.17 vs 2.73), and it needs three different
constructions plus a per-regime selection rule where the adopted pipeline
needs one construction and a d' knob.

[Reserved: if the per-cell menu is later re-adopted for the last few
tenths in the gap cells, it goes here as an OPTIMIZATION EXTENSION of the
single construction — same ingredients, per-regime re-ordering.]
