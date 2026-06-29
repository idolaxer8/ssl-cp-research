# Weekly Progress Summary

Rolling weekly summary for instructor discussions. Newest week on top. Each
entry: per-topic key points + output (graph/table) locations. Math is plain
text (no LaTeX), per repo convention.

All experiments: CIFAR-100, DINOv2 embeddings, exchangeable pipeline
(PCA-128 + cluster-whiten fit off cal), alpha = 0.1, unless noted.

---

## Week ending 2026-06-29 (work on 06-28 to 06-29)

**Context / motivation.** A novelty review concluded our headline objective —
minimize prediction-SET SIZE at 90% MARGINAL coverage — is saturating: the
prototype-softmax NCM (FCA-inspired; Silva-Rodriguez et al., IPMI 2025) already
dominates and PCA / whitening / MS-CS give diminishing returns (CIFAR-100 sz ~1.3
@ cal=800; miniImageNet near-singleton). We proposed two reframings; this week
delivered the **geometric-conditional coverage** direction end-to-end (topics 1-2,
the primary) plus a calibration-conditional reliability pilot (topic 3).

### 1. Diagnosis — marginal coverage HIDES a large geometric under-coverage gap

**Question.** Marginal coverage is an *average* over the test distribution. Is the
90% guarantee uniform across the embedding, or does it hide badly under-covered
regions? We stratify test points by a LABEL-FREE geometric covariate — local
density (k-th NN radius) or local intrinsic dimension (Levina-Bickel MLE, NeurIPS
2004) — computed on the disjoint unlabeled pool (a fixed pool-function => exactly
exchangeable, theory.md Prop 2), and measure per-stratum coverage + CovGap_geo (the
geometry-axis analogue of the class-conditional CovGap of Ding, Tibshirani &
Ramdas 2023, arXiv 2306.09335).

**Result (CIFAR-100 + miniImageNet, G=5 density strata, 10 trials).** Per-stratum
coverage is MONOTONE in local geometry (Spearman = -1.0 at every config): the dense
20% over-cover (~0.98-0.99), the sparse / high-LID 20% UNDER-cover to **0.60-0.75**
at a 0.90 marginal — a 15-40 pp conditional gap. It is NOT a split artifact
(identical on the exact RANDOM split), NCM- and MS-CS-INVARIANT (prototype,
geodesic, +-penalty overlap), GROWS with cal, and per-POINT not per-class. In
miniImageNet the sparsest stratum has mean set SIZE 0.74 (<1): the method hands
EMPTY sets to the hardest points. Mechanism: a STRONG predictor with locally
varying difficulty + a single GLOBAL threshold that cannot localize — over-spends
coverage on the easy majority, abstains/under-covers the sparse minority. Output:
output/geometric_coverage/, output/novelty_selling/geometry_{sell,comprehensive}.png.

### 2. The fix — geometry-conditional Full CP (Mondrian over geometric strata)

**Method.** Mondrian CP (Vovk): partition the input by a fixed taxonomy, use a
SEPARATE conformal threshold per group => group-conditional coverage
P(Y in C | g) >= 1-alpha. Our groups are the label-free geometric strata. Design C:
one global NCM (full neighbour pool) gives the test-score matrix + static LOO cal
scores; the THRESHOLD is per-stratum (q_g = the (1-alpha)(n_g+1) quantile of that
stratum's cal scores). Empty sets always allowed, NEVER replaced by all-K (the old
fallback over-covered above the valid bound). Exact up to O(1/n_g); per-STRATUM not
per-sample (exact feature-conditional coverage is impossible: Foygel-Barber et al.
2021). New src/geometric_conditional_cp.py + runner; one surgical engine add
return_test_scores (verified no-op).

**Result — closes the gap on BOTH datasets, BOTH splits:**
- CovGap_geo cut **87-95%** (mini 11.49 -> 0.58 pp; cifar 7.12 -> 0.89 pp);
- worst-stratum coverage 0.60/0.75 -> **~0.90** (empty-set holes filled the VALID
  way, via a higher LOCAL threshold, not by denying empties);
- marginal stays VALID on the exact random arm (0.902-0.907);
- size TAX +22-59%, all in the sparse stratum (where bigger sets are warranted) —
  the distribution-free conditional-coverage cost.

**Geometry is NECESSARY.** A confidence-conditioning control (Mondrian on the
model's own max-posterior) does NOT flatten the geometric gap (CovGap_geo 5.8-7.5
pp, ~10x worse) and over-covers to 0.96-0.98 — softmax confidence is miscalibrated
exactly on the atypical points. Density/LID is the right axis.

**Robust** across alpha {0.05,0.1,0.2} x G {3,5,10} x covariate {density,LID}:
CovGap_geo always -> 0.3-1.7 pp, worst -> 1-alpha. Gap GROWS with looser alpha
(miniImageNet alpha=0.2 sparse stratum 0.30 -> 0.80). Fig
output/novelty_selling/geometric_robustness.png.

**Scope boundary — FGVC-Aircraft.** Where DINOv2 cannot separate the classes (sets
uniformly 16-30 of K=100), the geometric gap barely exists (global CovGap_geo 1-2.5
pp) and Mondrian is a no-op — the gap is a property of a STRONG backbone with LOCAL
holes; apply geometry-conditioning only there.

**Secondary — RLCP** (Randomized Localized CP; Hore & Barber 2024, arXiv 2310.07850):
continuous Gaussian-kernel localization + a randomization that restores exact
marginal validity; interpolates global<->Mondrian via bandwidth (c0=0.25 ~ Mondrian,
c0=1.0 ~ global). Smooth (no hard-bin / small-n_g cliff) but mildly conservative;
Mondrian is the cleaner primary.

**Validity gated.** Unit tests 5/5 (design-C soundness, Mondrian + RLCP validity on
the exact split, empty-set invariant); GPU-parity suite 7/7. Output:
output/geometric_conditional{,_conf,_sweep,_rlcp}/, docs/novelty_pilots_findings.md.

### 3. (Secondary) PAC / calibration-conditional reliability pilot

At few-shot the calibration DRAW dominates variance. Over 100 draws at fixed budget
B, full CP rides the Beta(n=B) coverage-SD floor (split-CP coverage law, Vovk 2012)
while matched-budget Split-CP-THR degenerates to size-K sets at B<=400 ("turns on"
at B=800) and rides Beta(n=B/2); the set-size cost of a 95%-reliable 0.90 guarantee
(SSBC small-sample Beta correction) is ~12x cheaper for full CP. Figs
output/novelty_selling/reliability_{sell,convergence_cifar100}.png.

---

## Week ending 2026-06-22 (work on 06-21 to 06-22)

### 1. Centroid (cal-only) vs Cluster (unlabeled pool) MS-CS — does the cluster M update correctly per test point?

**Question.** In transductive Full CP every test point x is added to the bag
once per candidate label yc. The MS-CS similarity matrix M must update for that
augmented point. We wanted to confirm the **cluster M** (from the unlabeled
pool's k-means) updates correctly when the test point enters — i.e. does not
silently break exchangeability or produce degenerate sets.

**Mechanism verified.** Adding one point labelled yc shifts only class yc's
centroid, so the LOO update touches **only row/col yc of M** (O(K); O(1) for
cluster-M if yc's assigned cluster is unchanged), and the yhat update touches
only column yc. Exact vs brute-force LOO semantics.

**Empirical confirmation** (balanced cal+test, 10 trials, lambda=0.05,
tau = 0.5 * median_d^2). Six arms = 2 feature sources (separate 10k **pool** vs
**transductive** cal+test fit) x {cluster M, centroid M, no-penalty FCP}. Set
size (coverage in parens):

| arm | cal=200 | cal=400 | cal=800 |
|-----|---------|---------|---------|
| cluster-pool         | 7.21 (.943) | **2.05** (.907) | **1.58** (.921) |
| centroid-pool        | 8.50 (.941) | 2.15 (.909) | 1.59 (.920) |
| FCP-pool (lam=0)     | 8.93 (.943) | 2.28 (.911) | 1.63 (.920) |
| cluster-transductive | 12.52 (.942)| 2.58 (.908) | 1.72 (.920) |
| centroid-transductive| 14.42 (.943)| 2.77 (.905) | 1.72 (.920) |
| FCP-transductive     | 14.85 (.944)| 2.99 (.907) | 1.77 (.920) |

**Key points:**
- **Cluster update is correct.** All six arms stay valid (coverage 0.90-0.94,
  over-covering as expected for the balanced split — never under). A broken
  per-test update would show up as under-coverage or set blow-up; neither
  appears. The cluster-pool penalty cleanly beats its FCP-pool baseline
  (1.58 vs 1.63, ~3% @ cal=800; 2.05 vs 2.28, ~10% @ cal=400).
- **Cluster ~= centroid from cal>=400.** The unlabeled pool's value *for the M
  source* is a small-cal insurance only: cluster beats centroid ~15% @ cal=200
  (7.21 vs 8.50) but only ~1% @ cal=800. PCA denoises the cal class-centroids,
  so the cal-only centroid M recovers almost all the gain at cal>=400 — and
  needs **no pool** (pool-free, still exchangeable via the per-test centroid
  update; small change in `src/exchangeable_fcp_experiment.py`).
- **Separate pool > transductive (cal+test) features** on this balanced
  protocol (1.58 vs 1.72 @ cal=800) — the 10k pool earns its keep through the
  feature transform, not the M source.

**Output:** `output/balanced_mscs_source/balanced_mscs_source.png` + `results.json`.

### 2. K-shot test — effect of balancing cal+test, and the new default split

**What we did.** Definitive 3-arm A/B over the k-shot (C x k) protocol, 30
trials, post-missing-class-fix pipeline:
- `random` — label-blind split (exactly-exchangeable reference)
- `balanced_cal` — balanced cal, random test
- `balanced_both` — balanced cal AND test, equal shots m_cal == m_test (new)

| shots (cal=test) | random cov / sz | balanced_both cov / sz |
|------------------|-----------------|------------------------|
| 2 (200) | .9055 / 20.96 | .9198 / **5.36** |
| 4 (400) | .9033 / 3.36  | .9053 / **2.15** |
| 6 (600) | .9011 / **1.96** | .9299 / 2.18 |
| 8 (800) | .9012 / 1.68  | .9159 / 1.62 |

**Key points:**
- **Random is exactly tight** at every cal (0.901-0.906) even at cal=200 with
  13-15 classes missing from cal — the exact guarantee holds.
- **Balanced over-covers, peak +3.1pp @ cal=600** (conservative, never under).
  Balancing the *test* set too does NOT remove it => the over-coverage is the
  **bag-dependence / anchor-count channel**, not marginal label-shift (clean
  mechanism result). The cal=600 spike reproduces the old "92.5% @ cal=600"
  anomaly; conjecture: peaks where m-1 = NCM's k (5 same-class LOO anchors).
- **Set size flips with regime:** balanced wins big at small cal (-40 to -73%
  @ cal<=400), ties/loses ~+18% @ cal=600, ties @ 800. CovGap: balanced always
  better (tail-class protection).
- **DECISION (2026-06-21): default = balanced cal + balanced test, equal
  shots/class, fresh each trial.** Rationale: comparability with few-shot CP
  lit (jusiro FCA / SCA-T, the C x k protocol), every class present in both,
  smaller sets, better CovGap. Cost is mild and safe-direction (over-covers
  ~1-3pp). **Always also report the random arm** for the exact-validity claim.
  Implemented as `balanced_both` split + pilot-fixed-T in
  `src/exchangeable_fcp_experiment.py`.

**Output:** `output/split_ablation/` — `split_ablation.png` (3-panel:
coverage-vs-band / log set size / CovGap), `split_ablation_3arm.png`,
`split_ablation_results.json`, `balanced_both_results.json`.

### 3. FCA-inspired prototype-softmax NCM — the new best method

**Idea.** FCA (Silva-Rodriguez / jusiro, arXiv:2506.06076 — conformal for medical
VLMs) makes a softmax classifier head valid for Full CP by **re-fitting it in
closed form for each candidate label**. Their probe is a class-mean prototype
blended with a zero-shot text anchor, scored by LAC = 1 - p(y|x). We have no text
encoder (pure SSL, DINOv2), so we keep just the **class-mean prototype**: score
each label by a softmax over cosine similarities to the per-class means. New NCM
`PrototypeSoftmaxNCM`, `create_ncm("prototype_softmax", ...)`.

**Algorithm — prototype-softmax NCM inside Full CP** (plain text; `<.,.>` = inner
product; `n_c` = #cal points in class c; `n` = #cal points; `K` = #classes):

```
Setup (once)
  z(x)  = embedding of x after the exchangeable transform (PCA-128 +
          cluster-whiten, fit on the unlabeled pool), L2-normalised.
  mu_c  = mean of the calibration embeddings in class c   (class "prototype").
  T     = softmax temperature, fixed once on a pilot draw (never re-fit on cal).

NCM score  s(x, y)  -- "how nonconforming is label y for input x"
  f_c(x)  = < z(x), mu_c >              for every class c     (cosine similarity)
  p(y|x)  = softmax_c( f_c(x) / T )                           (class posterior)
  s(x,y)  = 1 - p(y | x)            (LAC/THR: small = typical, large = atypical)

Prediction set for a test point x   (transductive Full CP, exact)
  for each candidate label y in {1..K}:
    1. add (x,y) to the calibration bag; update ONLY class y's prototype:
         mu_y'         = (n_y * mu_y + z(x)) / (n_y + 1)
    2. leave-one-out re-score every bag point i on its OWN class y_i, using the
       bag-minus-i prototype (closed form, no matrix inverse):
         mu_{y_i}^(-i) = (n_{y_i} * mu_{y_i} - z(x_i)) / (n_{y_i} - 1)
         s_i           = 1 - softmax( < z(x_i), mu^(-i) > / T )[y_i]
    3. p_value(y)    = ( #{ i : s_i >= s(x, y) } + 1 ) / (n + 1)
    4. keep y  iff  p_value(y) > alpha
  output  { y : p_value(y) > alpha }
```

**Why it is exactly valid.** Every point in the augmented bag — the test point
AND each calibration point — is scored by the *same* leave-one-out rule, so the
n+1 scores are exchangeable and marginal coverage >= 1 - alpha holds for any fixed
T. No model is trained on the calibration labels (the "probe" is just class
means), and the leave-one-out update is closed form, so the per-candidate re-fit
is cheap; a bit-exact GPU path runs the whole sweep on the cluster. Verified: fast
path vs brute-force leave-one-out = 3.55e-15; small balanced cal **over-covers
with bloated sets, never under-covers**.

**Cluster results** (50 trials, balanced cal+test, CIFAR-100 + miniImageNet,
cal 200-1800, PCA-128 + cluster-whiten). CIFAR-100 set size (coverage), vs our
geodesic NCMs:

| cal | prototype_softmax | geodesic asym | geodesic mean | proto vs best geo |
|-----|-------------------|---------------|---------------|-------------------|
| 200 | **4.58** (.949) | 8.63 (.954) | 5.65 (.921) | **-19%** |
| 400 | **1.64** (.913) | 2.29 (.923) | 2.12 (.905) | **-22%** |
| 800 | **1.31** (.904) | 1.63 (.909) | 1.63 (.915) | **-19%** |
| 1800| **1.20** (.903) | 1.38 (.903) | 1.27 (.906) | **-6%** |

**Key points:**
- **Prototype is the TIGHTEST method at every cal on CIFAR-100** (-6% to -22%
  vs the best geodesic; up to -47% vs asym at cal=200), valid throughout
  (cov 0.90-0.95), and **CovGap best-or-tied** with asym (~5.8-6.3pp) — tighter
  *without* worse class-conditional coverage.
- **miniImageNet saturates** (DINOv2 separates it ~perfectly, sets ~1): prototype
  wins at small cal (cal=200: 1.39 vs 1.44 mean / 1.81 asym) and ties within
  ~1-4% at cal>=400. Prototype best/tied on CovGap (geo_mean worst, ~8pp).
- **Validity as designed:** small balanced cal **over-covers with bloated sets,
  never under-covers** (CIFAR-100 cal=200: sz 4.58 at cov 0.949) — the 50-trial
  confirmation of the bloat-not-undercoverage property.
- **Shipped to main:** NCM, GPU path, unit tests (incl. CUDA parity), and the
  cluster script.

**Output:** `output/from_cluster/fca_family_cluster/{results_cifar100,
results_miniimagenet,results_all}.json` + `compare_*_balanced_both.png`.
Script `src/fca_family_cluster_experiment.py` (default = prototype vs geodesic,
balanced, 2 datasets, 50 trials, GPU). NCM in `src/conformal_prediction.py`,
tests `tests/test_prototype_softmax_ncm.py`, theory `docs/theory.md` §4.1.

### Cross-cutting note

Topics 1-3 now all run on the **same balanced cal+test default** (topic 2's
decision), which is exactly the few-shot protocol FCA uses (topic 3) — so the
centroid/cluster MS-CS and the prototype-softmax results are directly
lit-comparable. The week's headline: the **prototype-softmax NCM** is the tightest
valid NCM on CIFAR-100 across the whole cal range, beating our geodesic NCMs.
Standing rule: pair every balanced headline with the random arm for the
exact-validity statement.
