# D1 score simplification: is the softmax redundant? (goal 1, week 08-17)

**Verdict: NOT redundant. The softmax is load-bearing wherever the backbone
separates the classes (high homophily / high cal), and it INVERTS to a liability
in the starved + non-separable corner (low homophily, small cal). Theorem D1 —
stated for the plain score s_c(x) = -cos(mu_c, x) — therefore does NOT cover the
deployed champion `prototype_softmax` verbatim; the gap lives entirely in the
softmax normalizer (cross-class coupling), and its SIGN tracks the qe/homophily
regime map.**

Code: `src/d1_softmax_ablation.py` (+ `PrototypeCosineNCM` in
`src/conformal_prediction.py`). Results/plot:
`output/d1_softmax_ablation/{results_*.json, d1_ablation_balanced_both.png}`.

## The two arms (a real ablation, not a reparam)

Both arms use IDENTICAL class-mean prototypes mu_c, the IDENTICAL exact
leave-one-out logit machinery (`F_base` / `_augmented_col_logit`), the IDENTICAL
exchangeable pool transform (PCA-128 + cluster-whiten fit off cal), and — per
trial — the SAME cal/test split (paired). f_c = cos(mu_c^{(-i)}, x) is the cosine
logit. The ONLY difference is the map logits -> nonconformity score:

```
  prototype_softmax  (champion) :  s_i = 1 - softmax_c(f / T)[y_i]
  prototype_cosine   (ablation) :  s_i = 1 - f_{y_i} = 1 - cos(mu_{y_i}, x_i)
```

This is a genuine ablation, not a reparameterization: softmax at a fixed T is NOT
a monotone per-class transform, because its denominator sum_c exp(f_c / T) couples
every class into each score. So the two produce different prediction sets. Note
also `prototype_cosine` has NO temperature at all — a common monotone rescale of a
per-class score leaves CP sets unchanged (invariance), so the cosine arm cannot be
"tuned" to close the gap: whatever difference we see is purely the normalizer.

Exactness: `prototype_cosine` is strictly exchangeable with NO O(1/n) cal-fit term
whatsoever (it drops even the fixed-T hyperparameter the softmax arm needs). The
GPU fast path is bit-exact vs the CPU loop (verified: prediction sets identical).

where
  mu_c^{(-i)} = class-c prototype leaving point i out of its own class;
  T          = one pilot-fixed softmax temperature per dataset (held across
               trials -> exact); validity is T-independent, T is an efficiency knob;
  h          = kNN label homophily (the qe/DWT regime dial, k=10).

## Results (balanced split, alpha=0.1, 20 trials, DINOv2 + exchangeable pipeline)

`sz` = avg set size (lower better); `Delta` = cosine set size RELATIVE to softmax
(positive = cosine worse = softmax was helping); `csr` = correct-singleton rate
(higher better). Coverage matched and valid for both arms throughout (cosine is if
anything slightly more conservative).

```
 dataset        h     cal | sz_soft  sz_cos   Delta  | csr_soft  csr_cos
 cifar100      .81    200 |   4.25    5.84   +37.4%  |  0.373    0.047
 cifar100      .81    400 |   1.67    2.14   +28.1%  |  0.596    0.341
 cifar100      .81    800 |   1.30    1.68   +29.0%  |  0.701    0.476
 miniimagenet  .92    200 |   1.34    1.62   +21.1%  |  0.691    0.542
 miniimagenet  .92    400 |   1.04    1.09    +5.4%  |  0.820    0.761
 miniimagenet  .92    800 |   1.00    1.04    +4.6%  |  0.840    0.788
 cub200       ~.75    400 |   3.46    4.48   +29.4%  |  0.283    0.153
 cub200       ~.75    800 |   1.68    2.80   +66.8%  |  0.561    0.315
 aircraft      .26    200 |  72.86   30.15   -58.6%  |  0.001    0.000
 aircraft      .26    400 |  21.49   21.46    -0.2%  |  0.010    0.000
 aircraft      .26    800 |  17.19   19.49   +13.4%  |  0.022    0.001
 stanford_cars .46    400 | 196.00   59.87   -69.5%  |  0.000    0.000
 stanford_cars .46    800 |  24.03   28.95   +20.5%  |  0.023    0.000
```

(cal=200 is skipped for K>=196 datasets: balanced needs cal >= 2K.)

## Reading the two regimes

**Regime 1 — backbone separates the classes (high h, or low h at high cal):
softmax is load-bearing.** On cifar100 / miniimagenet / cub200 (and on
aircraft/cars once cal is large) dropping the softmax inflates sets by +5 to +67%
and roughly HALVES the correct-singleton rate (cifar100@800 0.70 -> 0.48;
cub200@800 0.56 -> 0.32). Mechanism: when prototypes are well separated, the
normalizer down-weights a candidate class whenever some OTHER class fits better —
exactly the cross-class competition the plain per-class -cos score cannot express.
The gap grows with separability (mini's +5% vs cifar100's +29% at cal=800) and is
biggest on the confident, small-set points (the csr collapse).

**Regime 2 — backbone does NOT separate (low h) AND cal is small: softmax is a
LIABILITY.** On aircraft@200 the softmax bloats to sz=72.9 of K=100 while plain
cosine holds sz=30.2 (-58.6%); on cars@400 the softmax degenerates to the FULL
label set (sz=196) while cosine keeps sz=59.9 (-69.5%). Mechanism: with noisy,
overlapping prototypes and a T piloted from a larger draw, the softmax posterior
is near-uniform, so 1 - p(y) ~= 1 - 1/K is nearly constant across classes and
across cal/test — the p-values flatten and the set balloons. The plain -cos score
still RANKS classes by raw similarity, so it stays far tighter. As cal grows the
prototypes sharpen and the softmax recovers its edge (aircraft@800 +13%,
cars@800 +21%).

The crossover sits inside the same regime map the DWT work uses: the softmax helps
in the high-h / high-separation regime that qe also likes, and hurts in the
low-h / starved corner where qe also hurts. **The softmax normalizer and the qe
denoiser share a regime dial (homophily x calibration budget).**

## Implication for the theory (D1)

D1 (`docs/dwt_denoise_theorem.md`) is proved for the plain raw-similarity score
s_c = -cos(mu_c, x). This ablation shows the DEPLOYED champion is materially
different from that plain score — by ~30% set size (up to 67%) in the very regime
we deploy in. So we cannot claim "D1 covers the method verbatim." Two honest ways
forward, in order of preference:

1. **Report the plain-cosine NCM as the theory-faithful method and pay the
   efficiency tax where it is small.** The tax is only +5% on the most separable
   data (mini) and is NEGATIVE (cosine wins) in the low-h corner — but it is ~30%
   on cifar100/cub200, our headline separable datasets, so this is a real cost.

2. **Keep the softmax and add a coupling term to the theory.** The missing piece
   is exactly the normalizer Z(x) = sum_c exp(f_c / T): D1's per-class triangle
   inequality has to be replaced by a statement about 1 - exp(f_y/T)/Z(x), i.e. a
   margin/logsumexp bound. This is the "temperature / cross-class coupling" gap;
   it is a bounded, well-understood object (softmax = smooth-max), so a D1-analogue
   for the LAC score is plausible but is NOT free.

Recommended: state D1 for the plain score, present `prototype_cosine` as the
theory-covered NCM, and cite this ablation as the quantified softmax gap + its
regime dependence (softmax = a separability-gated efficiency lever, harmful when
the backbone cannot separate). This turns a "method != theory" liability into a
finding: the normalizer buys cross-class competition that pays off exactly when,
and only when, the representation is separable.

## Exchangeability / validity

`prototype_cosine` reuses the parent's proven exact-LOO logits and only applies a
common monotone map, so it is exactly exchangeable by construction (no cal-fit
term at all). Balanced coverage is valid (over-covers ~1-5pp, as the softmax arm
does). The random-split exact-validity arm (cifar100/aircraft, 10 trials,
`output/d1_softmax_ablation/random_validity/`) confirms it lands ON target:

```
 dataset   cal | cov_softmax  cov_cosine   sz_softmax  sz_cosine
 cifar100  400 |   0.8947      0.8999        1.79        2.25
 cifar100  800 |   0.8964      0.8972        1.30        1.65
 aircraft  400 |   0.8964      0.9023       22.38       22.61
 aircraft  800 |   0.8993      0.9025       17.26       19.53
```

Both arms hit ~0.90 (cosine if anything a hair more conservative). The
softmax-arm CPU/GPU paths remain bit-exact after the refactor that routed scoring
through the `_scores_from_F` / `_test_score` hooks (cosine + dot logits verified).
The size ordering is unchanged under the random split (cifar100@800 cosine +27%),
so the "softmax load-bearing on separable data" verdict is not a
balanced-split artifact.
