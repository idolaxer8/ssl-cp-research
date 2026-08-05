# Pool-fit representation for conformal prediction: denoise -> project -> whiten

**What this document is.** A self-contained explanation of the QE line of
work (2026-08-03/04): the theory that motivates it, the pipeline it produced,
and the experiments behind every claim. Detailed derivations and the full
lab-notebook history live elsewhere (references inline; raw results and
figures under `output/pool_repr_menu/`, git history of this file for the
per-round verdicts).

---

## 1. Setting and main claim

We run full conformal prediction (FCP) for K-way classification (K >= 100)
on frozen SSL embeddings (DINOv2, 768-d), with a tiny labeled calibration
set (200-800 points) and a large UNLABELED pool (3-10k points). The question
this work answers: **what is the most that the unlabeled pool can buy us,
and where in the pipeline should it be spent?**

Answer: spend all of it in the REPRESENTATION, before any label is touched.
The result is a three-stage, fully label-free feature map, fit once on the
pool:

```
x  ->  qe-denoise(x)  ->  PCA-d' projection  ->  (cluster / Ledoit-Wolf) whitening  ->  NCM  ->  CP
       stage 1             stage 2               stage 3
```

- **Exact coverage for free** at every stage (sec 2 — this is the
  architectural point, and the main novelty claim).
- **Best known set sizes**: cifar100 @ cal 200 drops 4.17 -> 2.34 (below the
  previous best-known 2.57 that needed a score-level correction); CUB-200
  @ 400 drops 3.65 -> 2.77; miniImageNet @ 200 drops 1.35 -> 1.19.
- **A "safe mode"** of stage 1 that is not-worse-than-incumbent on every
  dataset x cal cell we tested, including the adversarial fine-grained
  regime (aircraft) — so deployment needs no protective gate (sec 6).

Novelty position (lit sweep 2026-07-06 + 2026-08-03, `literature.md` sec 9):
no published work fits a projection/metric/denoiser on an INDEPENDENT
unlabeled pool and runs full/transductive CP with distance or prototype
scores in that space. Existing uses of unlabeled data in CP act on the
SCORES or the THRESHOLD (SemiCP, SNAPS, unsupervised calibration); the
representation slot is empty — and sec 5 shows the representation-level use
strictly subsumes the score-level one.

## 2. Why the pool-fit phase gives exact validity (the one theory fact)

CP's coverage guarantee needs the calibration + test scores to be
exchangeable. Any transform fit ON the calibration set breaks that symmetry
(the old cal-fit-whitening under-coverage). But a transform that is a fixed
function of the POOL alone treats every cal and test point identically —
formally, T = A(pool) is measurable w.r.t. data independent of the bag, so
Proposition 2 (`docs/theory.md` sec 2) applies verbatim and coverage is
exactly 1-alpha at ANY transform, however aggressive.

Intuition: the pool is "frozen scenery". Whatever geometry we carve out of
it, cal and test walk through the same carved landscape, so their ranks —
the only thing CP consumes — stay exchangeable.

Two consequences worth stating to a reviewer:

1. **Validity never constrains the search, only the information set.** The
   entire design question is efficiency (set size). We are free to try any
   pool statistic; nothing can break coverage.
2. **This is why the representation is the right place for pool
   information.** The alternative — correcting SCORES with pool neighbors
   (our SNAPS adaptation) — mixes pool quantities into a cal-fit score
   function, which cost us a leak repair (LOO re-prototyping), an O(1/n)
   validity gate, and a full-CP wrapping proof. The feature-level route gets
   exactness with zero machinery.

Why can a transform shrink sets at all? Expected set size is exactly a
separation functional between the score distributions of true and false
labels (`docs/conformal_metric_objective.md` sec 2 for the identity; that
document also proves accuracy-style objectives are structurally the wrong
surrogate, sec 5 — which is why every experiment below is judged by TRUE CP
set size and nothing else).

## 3. The three stages and why each is justified

The whole pipeline in one formula. With pool U = {u_1..u_N}:

```
T(x)  =  W * P^T * ( D(x) - mu )

D    stage 1: one-step pool-neighbor denoiser (nonlinear, per-point)
mu   pool mean
P    stage 2: top-d' pool eigenvectors (P = I when not truncating)
W    stage 3: whitener from the within-cluster pool covariance
```

Every parameter (D's neighbor bank, mu, P, W, the k-means clusters) is a
deterministic function of U alone, so T = A(U) and sec 2 applies to the
composite map exactly as to each stage.

### Stage 1 — qe denoising (the new ingredient)

alpha-query-expansion, ported verbatim from image retrieval (Radenovic,
Tolias & Chum, TPAMI 2019, arXiv 1711.02512 — method in their Sec 3.5,
parameters and robustness argument in Sec 5.3; applying it to database
points too is "database-side augmentation", Turcot & Lowe 2009 / Gordo et
al. IJCV 2017). For any point x, with s_i = max(cos(x, u_i), 0) over its k
nearest pool points u_i:

```
classic:    T(x) = L2norm( x + sum_i s_i^alpha * u_i )            k=10, alpha=3
safe mode:  T(x) = L2norm( (1-b) x + b * weighted-neighbor-mean ) b=0.3, k=5
```

Intuition: average a point with its closest pool neighbors. The component of
x orthogonal to its local manifold (noise) shrinks like 1/sqrt(k); the
tangential, class-carrying position survives. The s^alpha weights make the
effective neighborhood adaptive — far (likely wrong-class) neighbors get
negligible weight — which is the paper's own robustness argument for
alpha-QE over plain averaging.

Where does its gain come from? A cut-the-pipeline ablation (smooth inputs
only / refit metric only / both) localized the mechanism: **input
denoising**. At 2-8 labeled shots per class, the binding noise is in the cal
points and test queries themselves; denoising them is worth ~90% of the
gain, refitting the downstream stages on the smoothed pool adds a small
consistency bonus, and the metric alone adds little (the 10k-pool covariance
was never the noisy part).

Exchangeability: T is a fixed per-point function of (x, frozen pool); cal
and test never enter each other's smoothing. Prop 2 applies unchanged.

### Stage 2 — PCA truncation (established before this line; kept, justified)

```
mu    = (1/N) sum_u u                                   pool mean
Sigma = (1/N) sum_u (u - mu)(u - mu)^T                  pool covariance
      = V diag(l_1 >= ... >= l_768) V^T                 eigendecomposition
P     = [v_1 .. v_d']          top-d' eigenvectors  (P = I: no truncation)
T2(z) = P^T (z - mu)
```

The pool spectrum tells us where class signal lives, and the participation
ratio PR = (sum_j l_j)^2 / (sum_j l_j^2) indexes the regime (three-regime
map, transform-control campaign):

```
cifar/mini  PR ~ 240   signal in the top-128       -> project to 128
CUB-200     PR ~ 58    signal extends to ~512      -> project to 512
aircraft    PR ~ 16    top band is NUISANCE,       -> do NOT truncate
                       signal in low-variance tail
```

Justification against the obvious alternatives was done by controls that
each remove one ingredient: JL random projection (reduction alone: useless
-> PCA's data-adaptivity is the lever), drop-top/tail probes (aircraft's
signal really is in the tail), Ledoit-Wolf shrinkage (the continuous
alternative; wins exactly where truncation is undefined). Pool-only
d'-selection anchors: Gavish-Donoho / Marchenko-Pastur (valid where the
spectrum is spiked — cifar/mini/CUB — provably inapplicable on aircraft's
collapsed spectrum, which is why the regime map is needed at all).

### Stage 3 — whitening (established; kept, justified)

The NCMs are angular (cosine/geodesic). Without equalization, a few
high-variance directions dominate every angle and wash out the
discriminative low-variance ones. Whitening flattens the retained spectrum,
using WITHIN-cluster spread (k-means pseudo-classes stand in for classes —
the label-free version of within-class whitening):

```
c(u)  = k-means cluster of pool point u  (C clusters, fit in T2-space)
r_u   = T2(u) - mean{ T2(u') : c(u') = c(u) }        within-cluster residual

separable regime (diagonal):   W = diag( (var_j(r) + eps)^(-1/2) )
collapsed regime (full-rank):  Sigma_W = (1-rho) Cov(r) + rho (tr/d) I
                               W = Sigma_W^(-1/2)    (Ledoit-Wolf ZCA;
                               rho analytic, handles n per cluster < d)
```

This is also the standard retrieval recipe (Jegou & Chum, ECCV 2012:
centering + whitening as exploiting negative evidence / co-occurrence).

### Why THIS order (denoise -> project -> whiten)

Two arguments, one experiment:

- The qe graph must be built in the raw cosine metric — the only metric we
  trust a priori (the SSL training objective aligned it; whitened metrics
  are downstream ESTIMATES).
- Whitening AMPLIFIES low-variance directions, which is where the noise
  sits. Denoise first and the equalizer amplifies cleaned directions;
  equalize first and the neighbor graph lives in amplified noise.

Order experiment (qe-post arm: fit PCA/whiten on the raw pool, smooth
afterwards in the whitened space): a wash on cifar (truncation to the top
spectrum amplifies almost nothing), **catastrophic on aircraft's full-rank
whitening** — 34.6 / 31.8 / 28.1 vs smooth-first 28.2 / 26.6 / 24.6, worse
than no qe at all. Rule: post-smoothing's damage scales with how much the
transform amplifies the spectral tail. Smooth-first is confirmed, not
assumed.

### Relation to the retrieval paper's learned whitening (their Sec 3.4)

The same paper we took qe from also replaces PCA-whitening — worth a
direct comparison since stages 2-3 are our version of that step. Their
construction (Radenovic et al. Sec 3.4, Eqs 7-8; originally Mikolajczyk &
Matas): from SfM 3D models they get MATCHING and NON-MATCHING image pairs,
build two difference covariances, and take a two-scatter discriminant
projection:

```
C_S = sum_{matching (i,j)}     (f_i - f_j)(f_i - f_j)^T     "within"
C_D = sum_{non-matching (i,j)} (f_i - f_j)(f_i - f_j)^T     "between"

P   = C_S^{-1/2} * eig( C_S^{-1/2} C_D C_S^{-1/2} )         keep top-D
applied as P^T (f - mu), then L2-normalize
```

i.e. whiten by the within covariance FIRST, then pick directions by
BETWEEN-pair variance measured IN THE WHITENED METRIC. It beats
PCA-whitening in 22/24 of their benchmark cells.

Three differences from our stages 2-3:

1. **Supervision oracle.** Their pairs come from SfM geometry — no human
   labels, but real structural supervision. We have no pair oracle; our
   within-proxy is k-means pseudo-clusters on the pool. (The estimand
   difference is cosmetic: pair-difference covariance = 2x the
   within-group covariance under the same grouping.)
2. **What ranks the kept directions.** Theirs: between-pair variance in
   the whitened space (discriminative). Ours: TOTAL variance in the raw
   space (plain PCA), whitening only afterwards.
3. **Order.** Theirs is whiten-then-truncate; ours is
   truncate-then-whiten. (Distinct from the stage-1 order question of
   sec "Why THIS order", which is about the DENOISER staying in the raw
   metric — that verdict is unaffected.)

**Can we align? Yes, label-free.** Take same-cluster pool pairs (or
qe/mutual neighbors) as pseudo-matching and random pool pairs as
non-matching. For random pairs C_D ~= 2 * Sigma_total, so their formula
collapses to: within-cluster whiten at full rank, THEN PCA-truncate by
total variance in the whitened space — i.e. **our own two ingredients
composed in the opposite order** (`lw_then_pca_d'`). That makes the
comparison a one-arm experiment. Prediction from the regime map: on
separable data ~ pca128_cw (whitening barely reorders the top spectrum);
on aircraft it could deliver the first VALID truncation — within-whitening
rescales the tail signal UP before the variance ranking, so discriminative
directions can enter the top-d'. Caveat: on collapsed spectra the
within-cluster covariance is the fragile estimate (heavy LW shrinkage
would degrade the arm toward plain PCA). Status: proposed arm, not yet
run.

## 4. Experiments, in order (what we tried, what died, what survived)

All experiments: balanced splits, 10-20 trials, coverage verified ~0.90 in
every cell (as sec 2 guarantees), judged on mean set size vs the
per-dataset incumbent (cifar: pca128_cw, CUB: pca512_cw, aircraft:
lw_cluster768). Figures referenced by path.

**Round 1 — a pre-registered menu, not a cherry-pick.** Four candidate pool
transforms with kill criteria fixed in advance, benchmarked on
cifar100 + CUB + aircraft (`transform_controls_*_balanced_both.png`):

- Yeo-Johnson per-dim Gaussianization: clean NO-OP everywhere -> killed
  (marginal nonlinearity is not a lever).
- LPP (locality-preserving projection): killed WITH a mechanism — it
  faithfully preserves the pool kNN graph, and on low-PR data that graph is
  majority wrong-class (label homophily .25), so preserving locality
  preserves the noise. This also deprioritized diffusion-map-style arms
  (same graph-trust axis) and taught us the low-PR rule: pool NEIGHBORHOOD
  structure is unreliable there, only pool second-order statistics are.
- Soft per-cluster LW whitening (MPPCA-lite): one-budget win only ->
  dropped by the pre-registered rule.
- **qe smoothing: the survivor.** cifar100 prototype 2.34 / 1.44 / 1.23 @
  cal 200/400/800 vs incumbent 4.17 / 1.64 / 1.30; confirmed on the
  champion asym NCM (cifar -18% @400; CUB -15% @800) and on miniImageNet
  (1.19 vs 1.35 @200). Gains concentrate at small cal — exactly where the
  method is positioned.

**qe vs SNAPS (the score-level competitor)** — 2x2 {qe on/off} x {SNAPS
on/off} with a full eta x k sweep, 4 bases
(`snaps_stack/stack_corners.png`, `stack_mechanism.png`): qe alone matches
or beats SNAPS alone everywhere, and SNAPS's marginal gain on top of qe
collapses (-41% -> -6.6% at cifar-200, ~0 elsewhere) with its best mixing
weight shrinking toward zero — the pre-registered cannibalization
signature. Notably, qe RAISES the neighbor purity SNAPS depends on (CUB
k=20: .64 -> .71) — the graph improves, but the correction has nothing left
to harvest, because the same neighborhood information was already consumed
at the feature level. Runtime (`snaps_stack/runtime_qe_vs_snaps.png`):
SNAPS costs 174-410 ms of machinery PER recalibration (pool scores + kNN +
LOO repair); qe costs +1.6 s once and 0.4 ms per point, with no
recalibration cost and no validity machinery. **Verdict: the pool-neighbor
lever belongs in the representation; SNAPS is subsumed** (kept only as the
paper's score-level foil and a ~6% top-up in the single most starved cell).

**Knob robustness** (`qe_knobs/qe_knob_heatmaps.png`): on separable data
the surface is flat (pre-registered k=10/alpha=3 within ~1 SE of best
everywhere). k is the one load-bearing knob and tracks the pool's per-class
budget: k-opt = 5 on CUB/aircraft (28-33 pool shots/class), hard
degeneration when k approaches shots-per-class (CUB k=50: size 42 +- 25 —
cross-class averaging destroys prototypes). The alpha=0 control (= plain
AQE, uniform weights, Chum et al. 2007) is worse than alpha=1 in 23/27
cells — per-cell small (mostly within 1 SE) but consistently signed, and
largest exactly where the similarity guard should matter (aircraft, and
k=20: +0.8-1.2). So the ladder is: AQE (uniform) -> alpha guard (small,
consistent refinement) -> beta cap (the harm remover); the smoothing
itself carries the gain.

**Removing the last harm (aircraft).** Classic qe still harmed aircraft at
cal >= 400 (+5-12%). Three candidate fixes, one winner:

- micro-k (k=1..3): bounds the harm, cannot clear cal-800 — as k -> 0 the
  gain dies with the harm. Not the right dial.
- reciprocal-neighbor gating (keep neighbor u only if x falls in u's own
  kNN radius): killed with a mechanism — at homophily .25 mutual-kNN keeps
  the tightest local clique, which is a wrong-class micro-clump, and drops
  the diluting far neighbors: it CONCENTRATES the bias.
- **explicit self-mix beta = 0.3: the fix.** The classic formula's implicit
  neighbor mass is ~0.5-0.7 at aircraft-level similarities — too
  aggressive. Capping the neighbor share at 30% gives 27.45 / 24.32 / 22.27
  vs no-qe 29.01 / 24.61 / 22.03: a > 2 SE WIN at 200 and statistical ties
  after. The same setting also cures the one other harm cell (CUB @ 1600).

## 5. The bottom line table

Safe mode (k=5, alpha=3, beta=0.3) vs the per-dataset no-qe incumbent —
**not worse in any of the 13 cells tested; 7 wins, 6 ties**:

```
                     cal 200      cal 400      cal 800      cal 1600
cifar100 (proto)   4.17 -> 2.69  1.64 -> 1.49  1.30 -> 1.24      -
cub200   (proto)        -        3.65 -> 2.97  1.73 -> 1.62  1.31 -> 1.30
aircraft (geo)    29.01 -> 27.45 24.61 -> 24.32 22.03 -> 22.27     -
(+ miniImageNet and champion-asym confirmations at classic knobs, sec 4)
```

Classic mode (k=10, alpha=3, no beta cap) buys an extra 10-15% in the
starved high-homophily cells (cifar-200: 2.34) at the cost of the aircraft
harm — which motivates the gate below.

## 6. Deployment and the (performance-only) gate

Because safe mode is never worse than the incumbent and validity is exact
regardless, the gate is NOT safety-critical. Its only job is to decide when
to switch from safe to classic for the extra efficiency:

```
inputs:  hom_hat  = label-free estimate of pool kNN label homophily   [the only estimated input]
         S        = N_pool / K   (known),   n_cal (known)
rule:    k    = 5 if hom_hat < ~0.75 else min(10, S/4);  never near S
         beta = classic if (hom_hat >= ~0.75 and n_cal <= ~800) else 0.3
```

Default = safe; switch to classic only on high-confidence high-homophily
evidence (the asymmetry: a wrong classic pick costs up to ~12%, a wrong
safe pick only forgoes a bonus — so conservative thresholds make the costly
error unreachable). Open item: the hom_hat estimator (candidate statistics
and validation protocol in memory `litsweep-simple-poolfit-2026-08-03`; we
hold a 6-dataset labeled homophily map to validate against).

## 7. Status and remaining work

- DONE: menu round + kills, qe discovery + NCM/dataset confirmations,
  mechanism ablation, SNAPS subsumption + runtime, knob sweep, harm
  removal, order experiment. All on branch `worktree-pool-repr-menu`;
  results + figures in `output/pool_repr_menu/`.
- OPEN: (a) the hom_hat estimator panel for the performance gate; (b)
  50-trial cluster hardening of the headline cells before write-up; (c) a
  side anomaly worth one look: prototype + pca512 @ aircraft cal-800 hit
  14.90 (best aircraft cell ever), against the standing "prototype bloats
  on fine-grained" scope limit; (d) the `lw_then_pca` arm (the label-free
  port of the paper's Sec 3.4 discriminant whitening — see sec 3), most
  interesting on aircraft.
