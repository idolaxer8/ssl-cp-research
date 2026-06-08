# Unlabeled-data contribution to MS-CS: cal-only centroid M vs k-means-on-unlabeled M

**Question:** how much do we lose if the class-similarity matrix M is built *without*
the unlabeled pool — just `M[y,y'] = exp(-||mu_y - mu_y'||^2 / tau)` from class
centroids estimated on the calibration set?

**Setup:** CIFAR-100, DINOv2 matched-518 embeddings (recovered from cluster
`/storage`), full-768 (no PCA), NCM `geodesic_topk_mean`, alpha=0.1, test=300,
5 trials, tau = 0.5*median_d^2, n_clusters=20 (cluster mode only), non-exchangeable
penalty. Labeled pool 10000 (100/class), disjoint 10000 unlabeled pool. Identical
stratified splits across both modes (same seeds) -> controlled comparison; the
lam=0 baselines are byte-identical across modes (sz 5.87 / 3.50 / 2.49). Script:
`src/mscs_unlabeled_experiment.py --similarity {cluster,centroid}`.

## Full results (cov, sz over 5 trials)

CENTROID (cal-only M, NO unlabeled):

| cal | baseline | lam=0.03 | lam=0.05 | lam=0.10 |
|-----|----------|----------|----------|----------|
| 400 | 5.87, .885 | 5.46, .881 | 5.21, .881 | 4.59, .877 |
| 600 | 3.50, .895 | 3.14, .891 | 3.04, .890 | 2.73, .886 |
| 800 | 2.49, .912 | 2.35, .913 | 2.23, .911 | 2.06, .910 |

CLUSTER (k-means on 10K unlabeled):

| cal | baseline | lam=0.03 | lam=0.05 | lam=0.10 |
|-----|----------|----------|----------|----------|
| 400 | 5.87, .885 | 4.74, .890 | 4.13, .885 | 3.73, .887 |
| 600 | 3.50, .895 | 2.96, .891 | 2.98, .897 | 2.85, .895 |
| 800 | 2.49, .912 | 2.17, .907 | 2.08, .907 | 2.19, .910 |

## Best valid (smallest sz with cov >= 0.89) and the unlabeled contribution

| cal | centroid best | cluster best | degradation from dropping unlabeled |
|-----|---------------|--------------|-------------------------------------|
| 400 | none valid (5.46 @ .881) | 4.74 @ .890 | LARGE — centroid can't hold coverage; cluster ~15% smaller + valid |
| 600 | 3.04 @ .890 | 2.85 @ .895 | ~6% larger sets without unlabeled |
| 800 | 2.06 @ .910 | 2.08 @ .907 | ~0% — tie (centroid 1% smaller) |

## Takeaways
- **cal=800: unlabeled contributes ~nothing.** Cal-only centroid M matches/beats the
  cluster M. ~8 samples/class -> reliable class centroids, clustering adds no info.
- **cal=600: unlabeled buys ~6%** smaller sets + a little more coverage margin.
- **cal=400: unlabeled earns its keep.** Cal-only centroid M can't reach valid
  coverage (~4 samples/class centroids too noisy); cluster M stays valid (.890) with
  ~15% smaller sets. The unlabeled pool is a small-calibration insurance policy: a
  stable class-similarity prior when cal is too small to estimate one.

**Caveat:** cal=400 is cal/K=4 < 5, where the non-exchangeable penalty is itself
invalid (baseline already under-covers at .885). That row is partly entangled with
the approximation; a `--exchangeable` re-run would sharpen it. Qualitative story
(unlabeled matters only at small cal) holds. Note cal=800 baseline over-covers
(.912) — the known stratified-split + ceiling-quantile effect, not a bug.

## Small-cal extension (cal=200, 300) — unlabeled contribution grows

Non-exchangeable, so coverage is INVALID here (cal/K = 2-3; even lam=0 under-covers).
Sizes are still informative about the relative unlabeled contribution.

| cal | baseline | centroid best λ | cluster best λ |
|-----|----------|-----------------|----------------|
| 200 | 29.40, .843 | 28.96 (λ.10, .847) — barely moves | 20.03 (λ.10, .835) — −32% |
| 300 | 10.85, .875 | 8.04 (λ.10, .865) | 5.87 (λ.10, .868) — ~2× the reduction |

At cal=200 the cal-only centroid M is nearly useless (centroids from ~2 samples/class
are noise) while the unlabeled-cluster M still cuts a third — the "unlabeled =
small-cal insurance" story strengthens as cal shrinks.

## y_hat: variant A (NCM argmax top-k sim) vs legacy raw 1-NN

5 trials, cluster MS-CS, identical splits, geodesic_topk_mean. Best valid (cov>=.89):

| cal | variant A (ncm) | 1nn (legacy) | winner |
|-----|-----------------|--------------|--------|
| 400 | 3.84 @ .893 (λ.10) | 4.74 @ .890 (λ.03) | A, −19% |
| 600 | 2.79 @ .892 (λ.05) | 2.85 @ .895 (λ.10) | A (−6% at matched λ.05) |
| 800 | 2.06 @ .908 (λ.05) | 2.08 @ .907 (λ.05) | tie |

Variant A wins or ties everywhere over 5 trials (a single split had 1nn ahead — noise).
Mechanism: the more accurate ŷ keeps the true class unpenalized more reliably
(ŷ=y* more often → zero penalty on the truth), so coverage holds even at λ=0.10,
letting the penalty prune harder for smaller *valid* sets. Biggest gain at small cal.
Variant A is now the default (`--yhat_mode ncm`); `--yhat_mode 1nn` keeps the legacy
path for A/B. Implementation: GeodesicTopKMeanNCM.predict_class /
predict_class_augmented_cal / _ensure_cal_yhat (no raw distance matrices).

## Comprehensive cal=200–800, MS-CS with variant-A y_hat (5 trials)

CIFAR-100, 518 embeddings, geodesic_topk_mean, `--yhat_mode ncm`, test=300,
n_clusters=20, tau=0.5*median_d^2, non-exchangeable, lam in {0,.02,.03,.05,.10}.
(* = coverage invalid: cal/K < 5, non-exchangeable; even the lam=0 baseline
under-covers there.)

CLUSTER MS-CS (k-means on 10K unlabeled), best valid (cov >= .89, else best size):

| cal | baseline (lam0) | best MS-CS | reduction |
|-----|-----------------|------------|-----------|
| 200 | 29.40 (.843*) | 19.84 (λ.10, .836*) | 33%* |
| 300 | 10.85 (.875*) | 5.61 (λ.10, .865*)  | 48%* |
| 400 | 5.87 (.885*)  | 3.84 (λ.10, .893)   | 35% |
| 600 | 3.50 (.895)   | 2.79 (λ.05, .892)   | 20% |
| 800 | 2.49 (.912)   | 2.06 (λ.05, .908)   | 17% |

CENTROID (cal-only) best:

| cal | best | note |
|-----|------|------|
| 800 | 2.00 (λ.10, .906) | edges cluster (2.06) — unlabeled redundant |
| 600 | 3.31 (λ.02, .889) | under-covers for λ>=.03; cluster (2.79) clearly better+valid |
| <=400 | barely moves | invalid; cluster cuts much more |

Takeaways under the new y_hat:
- **cal=800: unlabeled REDUNDANT** — centroid (2.00) even slightly beats cluster (2.06).
- **cal=600: unlabeled HELPS** — cluster holds valid coverage at 2.79; centroid under-covers.
- **cal<=400: unlabeled ESSENTIAL** — centroid nearly useless.
- **vs legacy 1nn**: variant A improves cluster best-valid at cal=400 (4.74->3.84, −19%),
  cal=600 (2.85->2.79), cal=800 tie — gain concentrated at small cal.
- Only cal>=600 (and cal=400 at λ.10) hold valid coverage; cal=200/300 need
  `--exchangeable` for a coverage-valid result.

Logs: output/mscs_centroid_contrib/comprehensive_{cluster,centroid}_ncm.log

## Exchangeable pass for cal<=400 — does it restore small-cal validity? NO.

Ran cluster + centroid MS-CS with --exchangeable, new y_hat, cal=200/300/400,
5 trials. Key finding: **exchangeability does NOT fix the cal=200/300 under-coverage.**

CLUSTER, exch vs non-exch (best per cal):

| cal | baseline cov | non-exch best | exch best | valid? |
|-----|--------------|---------------|-----------|--------|
| 200 | .843 | 19.84 (λ.10, .836) | 22.51 (λ.10, .841) | both INVALID |
| 300 | .875 | 5.61 (λ.10, .865)  | 6.32 (λ.10, .868)  | both INVALID |
| 400 | .885 | 3.84 (λ.10, .893)  | 4.08 (λ.10, .895)  | both valid; non-exch tighter |

Why: the lam=0 plain-FCP **baseline itself under-covers** at cal=200 (.843) and
cal=300 (.875) — identical in both runs since lam=0 short-circuits to plain FCP.
The MS-CS penalty (exchangeable or not) only re-ranks scores; it cannot lift the
base predictor's coverage. So the small-cal under-coverage is a property of FCP at
cal/K <= 3 (noisy 768-d whitening + adaptive k=2 + ~2 same-class scores per test
point), NOT an artifact of the non-exchangeable penalty.

Exchangeable is simply more conservative: marginally larger sets + marginally
higher coverage at every cell, but it never crosses the validity threshold at
cal=200/300, and at cal=400 it is slightly worse than non-exch (4.08 vs 3.84, both
valid). Centroid exch at cal=200 even *grows* sets with lam (29.4->32.3) — a useless
M (centroids from ~2 samples/class) injects noise the penalty can only amplify.

Correction to earlier note: the right lever for small-cal validity is the BASE
predictor (e.g. PCA-128 to denoise the whitening), not penalty exchangeability.
Logs: output/mscs_centroid_contrib/exch_{cluster,centroid}_ncm.log
