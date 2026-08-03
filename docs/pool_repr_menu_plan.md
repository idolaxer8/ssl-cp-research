# Pool-fit representation menu — plan (goal 2, week 08-03)

**Status**: plan + pilot round 1 (2026-08-03). Successor to the G1
conformal-metric attempt; this document states the rewind rationale, the new
menu arms with full per-stage math, the benchmark protocol, and kill criteria.

---

## 1. Rewind rationale — what G1 taught us, what we keep

G1 tried to LEARN the pool transform (a continuous spectral filter) by
minimizing a label-free surrogate of conformal set size. The landscape gate
failed: the surrogate cannot rank candidate transforms (Spearman vs true FCP
size: -0.45 aircraft, ~0 cifar100, 0.75 cub200 only). The postmortem
(`docs/conformal_metric_objective.md` on `worktree-conformal-metric`)
localizes the failure in the ESTIMATOR (pseudo-label contamination below
homophily ~0.7 + quantile MC noise), not the functional.

What we keep from G1:
- The theory: E|C| = coverage + (K-1) * E_{U~Beta(k,n+1-k)}[R(U)] with
  R(u) = F_false(F_true^{-1}(u)) — expected set size IS a two-population
  separation functional; accuracy-style within-x objectives are structurally
  blind to it (the formal account of the +89% accuracy-selection regret).
- The corollary that matters here: **menu selection and metric learning are
  the same problem at |Theta| finite vs continuous.** Ranking a continuum
  label-free is what failed; ranking a MENU of ~10 qualitatively different
  fixed transforms is a far easier statistical problem — and for a finite
  menu we can even select WITH labels we already have, validly (sec 5).

The rewind (= instructor goal 2 as written): extend the pool-fit transform
menu along the two known failure axes — nonlinearity and local structure —
with FIXED maps fit once on the unlabeled pool, benchmarked by TRUE FCP, no
surrogate objective anywhere.

## 2. Starting map (established, `transform_control_experiment.py`)

Three regimes, tracked by pool participation ratio PR = (sum ev)^2 / sum ev^2:

```
cifar/mini   PR ~ 240   signal in top-128       -> pca128_cw  (truncate + diag cluster whiten, fine k)
CUB-200      PR ~ 58    signal extends to 512   -> pca512_cw
aircraft     PR ~ 16    top band nuisance,      -> lw_cluster768 (no truncation,
                        signal at bands 129-512     full-cov LW whitening, coarse k)
```

Menu today = {truncate-top d', whiten (diag-cluster / LW-full), drop-top
tail probes}. All linear, all global (one metric for the whole space).
Known negatives already on file: AE bottleneck on low-PR (nonlinearity via a
learned code is NOT the missing ingredient on aircraft), ICA/negentropy
(killed), JL random projection (adaptivity is the lever, not reduction).

The 2026-08-03 lit sweep (memory `litsweep-simple-poolfit-2026-08-03`) adds
external evidence: nonlinear DR payoff on embeddings is regime-dependent
(arXiv 2403.14001 — linear near-optimal on isotropic/contrastive spaces), 
DINOv2 fine-grained structure is nonlinearly decodable (attentive > linear
probing; arXiv 2302.00294 curvature), UMAP-family disqualified as a metric
source (density distortion). So: expect new arms to matter on aircraft/CUB,
not cifar — which is exactly where the menu is weakest.

## 3. The four new arms (round 1) — per-stage math

Notation: pool U = {u_1..u_N} in R^768 (DINOv2, L2-normalized), pool matrix
X_U (N x 768). Every arm is a map T = A(U): R^768 -> R^d fit on U alone and
applied identically to every cal/test point, so Proposition 2 (theory.md
sec 2) gives EXACT coverage unchanged — the whole comparison is efficiency.
Implementation: `exchangeable_features.UnlabeledTransform`, new stages
`pre in {yj, qe}` (before projection), `projection = lpp`, and
`whiten = lw_cluster_soft`.

### Arm E — elementwise Yeo-Johnson Gaussianization (`pre='yj'`, nonlinearity axis, rung 0)

Per dimension j, the Yeo-Johnson power transform (handles negative values,
unlike Tukey's ladder used on post-ReLU features in the few-shot lit):

```
YJ(z; lam) =  ((z+1)^lam - 1) / lam                 z >= 0, lam != 0
              log(z + 1)                            z >= 0, lam  = 0
              -(((-z+1)^(2-lam) - 1) / (2-lam))     z <  0, lam != 2
              -log(-z + 1)                          z <  0, lam  = 2
```

Fit: lam_j = argmax of the YJ profile log-likelihood on pool column j
(scipy.stats.yeojohnson MLE; 768 one-dim fits on N pool values). Then
standardize each transformed dim by its pool mean/sd (so downstream PCA is
not dominated by scale artifacts of the power map), then re-L2-normalize
rows (the NCMs are angular). Downstream stages (PCA/whiten/k-means) are
refit on the TRANSFORMED pool.

Parameter justification: lam_j per-dim by MLE = zero free knobs. The
standardize + renorm steps restore the two invariances the rest of the
pipeline assumes (per-dim scale comparability; unit-norm inputs).

Why it could pay: whitening and Mahalanobis-style metrics are optimal for
elliptical data; marginal Gaussianization is the cheapest move toward
ellipticity. Lit: Yang et al., "Free Lunch for Few-Shot Learning:
Distribution Calibration", ICLR 2021 (arXiv 2101.06395) — power-transformed
deep features become near-Gaussian per class and improve prototype
classifiers; Laparra et al. 2011 (RBIG) is the iterated version (parked —
try the one-step map first). Honest prior: DINOv2 dims are already fairly
symmetric post-L2; this is the control rung of the nonlinearity axis — if
YJ is a no-op AND the heavier nonlinear arms also fail, the axis dies
cheaply.

### Arm S — pool-neighbor feature smoothing (`pre='qe'`, local axis, denoising)

alpha-query-expansion / database-side augmentation, ported verbatim from
retrieval (Radenovic, Tolias & Chum, TPAMI 2019, arXiv 1711.02512; DBA:
Gordo et al., IJCV 2017; the de facto standard re-ranking pre-step for
frozen-feature retrieval). For ANY point x (L2-normed), let
NN_k(x) = its k nearest pool points by cosine (exact self-matches excluded),
s_i = max(cos(x, u_i), 0):

```
T(x) = L2norm( (x + sum_{i=1..k} s_i^alpha * u_i) / (1 + sum_i s_i^alpha) )
```

Parameters k=10, alpha=3 — the retrieval-canonical values (Radenovic 2019);
alpha concentrates weight on the closest neighbors (s^3 halves the weight of
a neighbor at cos .8 vs one at cos .63), which is the built-in guard against
wrong-class averaging. Pool points themselves are smoothed the same way
(neighbors = their k nearest OTHER pool points) before downstream stages are
refit on the smoothed pool.

Mechanism: local manifold denoising — the average over a coherent
neighborhood shrinks the noise component orthogonal to the local manifold by
~1/sqrt(k) while preserving the tangential (class-relevant) position. This
is SNAPS's information source (pool neighbors) moved from the SCORE level to
the FEATURE level, where it composes with everything downstream and where
the sim^alpha weighting is gentler than score averaging.

Exchangeability: T is a deterministic function of (x, U) applied pointwise;
cal and test points NEVER enter each other's smoothing (only pool vectors
are averaged). A(U)-measurable fixed map -> Prop 2, exact.

Honest risk: the SNAPS harm regime (homophily < ~0.7 on aircraft/cars) may
reappear at the feature level. That is a FINDING either way: if feature-level
smoothing survives where score-level failed, the axis matters; if both fail
identically, low-homophily harm is information-theoretic, not mechanical.

### Arm L — Locality Preserving Projection + cluster whitening (`projection='lpp'`, local axis, the sharp bet)

He & Niyogi, "Locality Preserving Projections", NIPS 2003 — the linear
approximation of Laplacian eigenmaps. Fit on the centered pool
Xc = X_U - mu (mu = pool mean):

1. kNN graph on U by cosine: W_ij = 1 if u_j in kNN_g(u_i) OR u_i in
   kNN_g(u_j), else 0 (symmetrized binary; g = 15). Binary weights = zero
   bandwidth knobs (the heat-kernel variant adds one; parked unless binary
   shows signal).
2. Degree D_deg = diag(sum_j W_ij), graph Laplacian L = D_deg - W.
3. Generalized eigenproblem on 768 x 768 matrices:

```
   (Xc^T L Xc) f  =  lam * (Xc^T D_deg Xc + eps*I) f,    eps = 1e-4 * tr(Xc^T D_deg Xc)/768
```

4. F = [f_1 .. f_d'] = the d' eigenvectors with SMALLEST lam;
   T(x) = F^T (x - mu). Then k-means + within-cluster diagonal whitening in
   LPP space (existing machinery, unchanged).

Objective being solved: min_F sum_ij W_ij ||F^T x_i - F^T x_j||^2 subject to
F^T (Xc^T D_deg Xc) F = I — keep directions in which GRAPH NEIGHBORS stay
close, normalized by degree-weighted variance. Contrast with PCA, which
keeps directions of maximal GLOBAL variance. LPP is therefore
variance-blind: a low-variance direction that separates local clumps is
retained; a high-variance nuisance direction (pose/lighting) that scatters
neighbors is discarded. This is a direct, closed-form attack on the aircraft
mechanism ("top spectral band = nuisance, class signal at bands 129-512") —
the one arm in round 1 built specifically for the low/mid-PR regimes.

Parameters: g=15 (standard kNN-graph range; the graph is a smoothing
parameter, not a resolution one — results are flat in g over 10..20 in the
LPP lit), eps ridge = spectral-scale-relative 1e-4 (numerical only),
d' in {128, 512} = the two truncation points the regime map already uses,
giving arms lpp128_cw and lpp512_cw.

Cost: one N x N cosine top-g query (chunked), two 768 x 768 Gram matrices
(L is sparse: <= 2 g N entries), one generalized eigh(768) — seconds.

### Arm M — per-cluster soft Ledoit-Wolf whitening (`whiten='lw_cluster_soft'`, local axis, metric field)

The incumbent lw_cluster768 fits ONE pooled within-cluster covariance (a
single global metric). This arm fits a LOCAL metric per cluster and
interpolates — the k-means/Ledoit-Wolf instantiation of Mixtures of
Probabilistic PCA (Tipping & Bishop, Neural Comp. 1999); published precedent
for cluster-conditional Mahalanobis on SSL features: SSD (Sehwag et al.,
ICLR 2021, arXiv 2103.12051).

Fit (pool only):
1. k-means, C clusters (existing stage; C = n_clusters_whiten).
2. Per cluster c: mean mu_c; LW-shrunk covariance of its members
   Sigma_c = (1-rho_c) S_c + rho_c (tr S_c / 768) I with rho_c analytic
   (Ledoit & Wolf 2004 — handles n_c ~ N/C < 768); ZCA whitener
   A_c = Sigma_c^{-1/2} via eigendecomposition.
3. Softness scale: tau^2 = mean over pool points of their squared distance
   to the NEAREST cluster center (a pool statistic — no knob).

Transform:

```
r_c(x)  = softmax_c( -||x - mu_c||^2 / (2 tau^2) )        soft responsibilities
T(x)    = sum_c r_c(x) * A_c (x - mu_c)
```

Smooth (softmax), piecewise-linear-in-the-limit, globally NONLINEAR map —
the mildest possible nonlinearity: locally it is exactly the whitening we
already trust, globally it lets the metric vary over the manifold.
Parameter justification: C inherited from the existing whitening stage (not
a new knob); tau from the pool assignment-distance scale (the natural unit
making r_c neither one-hot nor uniform); LW shrinkage analytic.

Risk: with C=100, n_c ~ 100 members per cluster at d=768 pushes LW toward
heavy shrinkage -> A_c collapses toward isotropy and the arm degenerates to
soft centering. If the pilot shows this, rerun with C=20 (coarse), which the
aircraft k-sweep independently prefers.

### Explicitly parked (round 2, only if round 1 shows the axis pays)
- Diffusion maps + Nystrom out-of-sample (Coifman-Lafon 2006; the canonical
  nonlinear+local arm; one real knob = kernel bandwidth) — heavier, ~100M
  affinity entries.
- RFF kernel-PCA whitening (cosine-Gaussian kernel; NeurIPS 2024 OOD-kPCA,
  arXiv 2505.15284) — the global-nonlinear arm.
- RBIG iterated Gaussianization (if YJ alone moves anything).
- Hubness-corrected NCM scores (CSLS/MP/NNN) — user-deferred; NCM stage, not
  transform stage.

## 4. Benchmark protocol (pilot, local 4GB GPU)

`transform_control_experiment.py` extended with the new arms; everything
else inherited unchanged (exact FCP, missing-class fix, GPU fast path).

```
datasets   cifar100 (PR 235, high), cub200 (PR 58, mid), aircraft (PR 16, low)
arms       raw768, pca128_cw, pca512_cw, lw_cluster768        (baselines/incumbents)
           yj_pca128_cw, yj_lw768                             (E x champion)
           qe_pca128_cw, qe_lw768                             (S x champion)
           lpp128_cw, lpp512_cw                               (L)
           lwsoft768                                          (M)
split      balanced_both (default split policy; random arm deferred to the
           confirm run — validity is settled by Prop 2, not at issue here)
cal        cifar100/aircraft {200,400,800}; cub200 {400,800,1600} (m_cal>=2)
ncm        unwhitened_topk_mean (geodesic) + prototype_softmax (T auto per arm)
trials     10 (pilot; SE reported), n_clusters_whiten=100 (script default,
           = prior transform-control setting; the k dial is a known second
           knob — NOT re-swept here, noted as a caveat on lw arms)
alpha      0.1
output     output/pool_repr_menu/ (main repo)
```

Decision metric: mean set size at matched coverage (all arms exactly valid
by construction; coverage col is a sanity check), CovGap secondary.

## 5. Selection (stage C — after the menu settles)

- Label-free: extend the existing selector (margin-tail + PR-band regime
  fallback, `transform_selection_pilot.py`) over the grown menu; CUB is the
  open mid-PR test.
- The G1 salvage: for a FINITE menu, selection does not need the label-free
  surrogate at all — a micro selection fold D_sel (m_sel ~ 100-200 labels
  from budget B, disjoint from cal) ranks ~10 arms by empirical set size /
  closed-form J_hat; conditional on (pool, D_sel) the chosen T is fixed, so
  Prop 2 applies with A(pool, D_sel) and validity stays EXACT. Lit anchors:
  Yang & Kuchibhotla (arXiv 2104.13871); Liang, Zhu & Barber (arXiv
  2408.07066, same-cal selection with validity corrections) as the
  no-extra-labels alternative.

## 6. Kill criteria

Per arm: DROP if it fails to beat the per-dataset incumbent by > 2 pooled SE
at >= 2 cal budgets on >= 1 dataset (and is not the best on any dataset).
Per axis: the nonlinearity axis dies if YJ ~ 0 everywhere AND round-2
nonlinear arms are not triggered by any round-1 signal; the local axis dies
if qe, lpp, AND lwsoft all fail on aircraft + CUB (the regimes they were
built for).
Program: if NO new arm beats incumbents anywhere, goal 2 closes with a
defensible completeness claim — "{truncate-top, whiten-all} spans the useful
pool-fit space; the three-regime map + selector is the contribution" — plus
the negative-results section (AE, ICA, JL, and now these four).

Expected shape of a WIN: any arm beating lw_cluster768 on aircraft or
pca512_cw on CUB by > 2 SE — that would be the first non-{PCA, whitening}
member of the pool-fit menu and the first result attacking the regime where
the current menu is weakest.

---

## 7. Round 1 verdict (2026-08-03, 10 trials, balanced, k_whiten=100; JSONs + figs `output/pool_repr_menu/`)

Baseline sanity: incumbents replicate history (cifar proto pca128_cw 1.30
@800 ~ rung-3 1.27-1.30; CUB geodesic pca512_cw 1.51 @1600 = the recorded
champion number). Aircraft incumbent here (topk_mean, k_whiten=100) is
weaker than the true champion config (asym, coarse k) — internal
comparisons only on that dataset.

**qe = the round-1 discovery — a small-cal lever with the SNAPS homophily
fingerprint.**
- cifar100 prototype: qe_pca128_cw 2.34+-0.11 / 1.44+-0.04 / 1.23+-0.03 @
  cal 200/400/800 vs incumbent 4.17 / 1.64 / 1.30 (-44% / -12% / -5%),
  coverage 0.90-0.93. At cal 200 this is BELOW the SNAPS-corrected
  best-known (2.57, eta x k sweep 08-02) — achieved by a pure exchangeable
  TRANSFORM, i.e. potentially stackable with the SNAPS score correction.
- cub200 prototype: 2.77+-0.22 @400 (incumbent 5.62) and 1.66+-0.04 @800,
  but WORSE @1600 (1.42 vs 1.23) — gain shrinks and reverses as cal grows.
- aircraft: qe_lw768 ties the incumbent at cal 200 (28.2 vs 29.0,
  geodesic) then HURTS at 400/800 (+8% / +12%); qe_pca128_cw hurts badly.
- Pattern: help at homophily ~0.8 (cifar), help-then-crossover at mid
  homophily (CUB), harm at 0.25 (aircraft) == the SNAPS regime map. One
  label-free homophily gate should serve BOTH levers (goal 1's blocker and
  this one are the same component). Also NCM-specific: the benefit
  concentrates in prototype/centroid scores (denoised class means); the
  geodesic kNN-ratio is ~unchanged on cifar and hurt on aircraft.

**yj = clean no-op everywhere** (every cell within noise of its parent).
The cheap nonlinearity rung dies; no round-1 signal triggers the heavier
nonlinear round-2 arms.

**lpp = KILL, with a mechanism.** Catastrophic exactly where it was aimed:
aircraft geodesic 79.9 / 71.2 / 63.1 vs incumbent 29.0 / 24.6 / 22.0;
lpp512 is terrible at small cal on every dataset (kept noise dims dominate
the whitened metric); lpp128 only ever ties. Mechanism: LPP maximizes
fidelity to the pool kNN GRAPH; on collapsed-spectrum data that graph has
label homophily ~0.25, so locality preservation faithfully preserves
wrong-class adjacency. Same failure axis as SNAPS/qe, now at the projection
level. Map update: on low-PR data, pool NEIGHBORHOOD structure is
unreliable — only pool second-order statistics are; full-rank whitening
remains the only trustworthy lever there.

**lwsoft = dropped by the pre-registered rule** (beats an incumbent > 2 SE
at one budget only: CUB proto 4.53 @400 vs 5.62). Ties-to-slightly-beats
its parent lw_cluster768 on cifar/CUB small cal, loses on aircraft
(37.0 vs 29.0) — where the aircraft pool (3333 / 20 clusters ~ 167 pts per
cluster at d=768) makes the local covariances shrinkage-dominated. Noted,
not pursued.

Side observation for follow-up: prototype + pca512_cw @ cal 800 on
aircraft = 14.90 (cov 0.912) — the best aircraft cell in the run,
against the standing "prototype bloats on fine-grained" scope limit.

**Round 1.5 (running):** qe on miniimagenet (homophily 0.92 — predicted
largest gain) + qe under the champion asym NCM on cifar100/cub200.
**Round 2 (replanned):** graph-based nonlinear arms (diffusion maps) are
DEPRIORITIZED — they share the graph-trust failure axis lpp just exposed;
the productive successors are (a) the shared label-free homophily gate,
(b) qe knob robustness (k, alpha) + qe x SNAPS stacking (feature-level +
score-level neighbor information), (c) the aircraft prototype+pca512
anomaly.
