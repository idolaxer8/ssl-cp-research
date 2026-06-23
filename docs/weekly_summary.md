# Weekly Progress Summary

Rolling weekly summary for instructor discussions. Newest week on top. Each
entry: per-topic key points + output (graph/table) locations. Math is plain
text (no LaTeX), per repo convention.

All experiments: CIFAR-100, DINOv2 embeddings, exchangeable pipeline
(PCA-128 + cluster-whiten fit off cal), alpha = 0.1, unless noted.

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
