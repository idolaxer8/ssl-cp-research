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
