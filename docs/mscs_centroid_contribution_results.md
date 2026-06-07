# Unlabeled-data contribution to MS-CS: cal-only centroid M vs k-means-on-unlabeled M

**Question:** how much do we lose if the class-similarity matrix M is built *without*
the unlabeled pool — just `M[y,y'] = exp(-||mu_y - mu_y'||^2 / tau)` from class
centroids estimated on the calibration set?

**Setup:** CIFAR-100, DINOv2 full-768 embeddings (no PCA), NCM `geodesic_topk_mean`,
alpha=0.1, test=300, 5 trials, tau = 0.5*median_d^2, n_clusters=20 (cluster mode only).
Non-exchangeable penalty. Identical stratified splits across both modes (same seeds),
so this is a controlled comparison. Script: `src/mscs_unlabeled_experiment.py`
with new `--similarity {cluster,centroid}` flag.

## Centroid (cal-only M, NO unlabeled data) — COMPLETED

| cal | baseline (lam=0) | lam=0.03 | lam=0.05 | lam=0.10 |
|-----|------------------|----------|----------|----------|
| 400 | sz 5.49, cov .889 | 4.75, .883 | 4.45, .881 | 3.81, .872 |
| 600 | sz 3.20, cov .891 | 2.79, .887 | 2.54, .887 | 2.26, .879 |
| 800 | sz 2.31, cov .899 | 2.10, .900 | 2.01, .899 | 1.85, .898 |

Observations:
- The cal-only centroid M *does* shrink sets: at cal=800, lam=0.05 -> 2.01 vs 2.31
  baseline (**13%**) with coverage held (.899); lam=0.10 -> 1.85 (**20%**), cov .898.
- At cal=600, lam=0.05 -> 2.54 vs 3.20 (**21%**) but coverage dips to .887 (just under
  the .89 target) and erodes further with lam.
- At cal=400 (cal/K = 4 < 5) coverage degrades with lam (.889 -> .872): the
  non-exchangeable approximation is invalid in this regime — should use
  `--exchangeable`. Set-size "wins" here are partly bought with undercoverage.

## Cluster (k-means on 10K external unlabeled) — PENDING

NOT YET RUN at this config. The run crashed because the `output/` directory
(all embeddings + result subdirs) was wiped externally mid-run (see session notes).
Re-run once embeddings are restored:

```
python src/mscs_unlabeled_experiment.py \
  --embeddings_path output/embeddings_cifar100.pt \
  --unlabeled_path output/embeddings_cifar100_unlabeled.pt \
  --similarity cluster --ncm geodesic_topk_mean \
  --cal_sizes 400 600 800 --test_size 300 --n_trials 5 \
  --lambdas 0.0 0.03 0.05 0.1 --taus 0.5 --tau_normalize --n_clusters 20
```

Preliminary signal (cal=200 smoke, 1 trial): cluster M cut sets 40.7 -> 26.2 while
centroid M only 40.7 -> 39.2 — i.e. at very low cal the unlabeled clustering does
most of the work because cal class-centroids (~2 samples/class) are too noisy. The
open question is how much of the gap closes by cal=600-800, where the table above
shows centroid M already recovering a meaningful 13-21% reduction on its own.
