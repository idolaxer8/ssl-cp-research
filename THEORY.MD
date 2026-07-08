# Theory: Valid, Tight Conformal Sets from SSL Embeddings at Small Label Budgets

## Problem

Distribution-free uncertainty quantification for image classification when
labels are the scarce resource. With K >= 100 classes and a few hundred
labeled examples total, the standard recipe — train a classifier on part of
the budget, calibrate split CP on the rest — collapses: a 100-way softmax
cannot train on ~150 examples, so split-CP sets degenerate to the trivial
cover (size = K). Self-supervised embeddings (DINOv2 etc.) carry strong class
geometry that costs zero labels; the question is how to convert that geometry
plus cheap *unlabeled* data into prediction sets that are both **valid**
(coverage >= 1-alpha, exactly) and **tight** at calibration sizes where
nothing else works.

## Operating Theory

Four load-bearing claims, each backed by experiments in `docs/findings.md` /
`docs/weekly_summary.md`:

1. **Validity is an exchangeability property of the score function, and it is
   binary.** The guarantee needs every nonconformity score to be a
   permutation-symmetric function of the calibration+test bag. Any transform
   fit on the calibration set — whitening, PCA, RBF bandwidth, a fixed ŷ in
   the CS penalty — breaks this, even at O(1/n). Consequences shaped the
   codebase: (a) all data-dependent transforms are fit on an *independent
   unlabeled pool* (`exchangeable_features.py`), or label-free on cal+test
   itself (pool_source result: exactly exchangeable, recovers most of the
   benefit); (b) asymmetries hiding in edge cases are real bugs — the
   missing-class sentinel and the MS-CS non-LOO ŷ both broke coverage until
   made symmetric.

2. **The efficiency win is NCM design + budget allocation, not transductive
   machinery.** SCP-geodesic matches FCP-geodesic at every budget; FCP's
   dominance over SCP-THR is "distance/prototype NCM that spends the whole
   budget on calibration" vs "classifier that must split it". The current
   NCM ranking: `prototype_softmax` (FCA-style class-mean cosine softmax,
   closed-form LOO refit) is tightest on separable data at every cal;
   `geodesic_topk_asym` is the more honest choice (best class-conditional
   CovGap) on hard fine-grained data. Class-level similarity penalties are
   **headroom-gated**: no M (cluster, faithful Fargion centered-cosine,
   prototype-Gram, confusion) can help a tight softmax set — the classes
   cluttering it are exactly the most-similar to ŷ, which the penalty
   protects — while M remains a modest real lever for the geodesic NCM.

3. **Unlabeled data buys efficiency through the feature transform first, the
   penalty second** (pool ablation: features 41-48% of the cut, penalty a few
   percent, all exchangeable).

4. **The transform IS the metric half of the NCM** — the current workstream.
   Whiten-then-Euclidean is Mahalanobis under the pool covariance, so pool-fit
   PCA-truncate + within-cluster whitening = a low-rank regularized estimate
   of the precision metric that both NCMs' inner products live in; d' is the
   metric's regularization knob (the d'-U-shape is bias-variance in metric
   estimation, not a coverage effect). This dissolves the "PCA is ad hoc /
   unrelated to the NCM" objection that the old "removes low-variance noise"
   line invited (falsified in-house: PCA-128 keeps only 56% of variance,
   participation ratio 232). Mechanism, per the new geometry diagnostics
   (2026-07-05): the lever is *data-adaptive subspace selection* — on the
   CIFAR-100 pool PCA-128 raises between/within class contrast ~40%
   (1.42 -> 1.97) and triples relative distance contrast (2.2 -> 7.0),
   while a Johnson-Lindenstrauss projection to the same d changes neither
   (it preserves noise as faithfully as signal) and full-768 Ledoit-Wolf
   whitening *destroys* contrast (class ratio 1.10) by equalizing pure-noise
   directions with signal. Anisotropy itself is mild in DINOv2 (mean pair
   cosine 0.06) and dies with centering — so the paper should lean on
   contrast concentration, not the NLP-style anisotropy-repair story.

The lead pipeline: DINOv2 embeddings -> pool-fit PCA-128 + cluster-whiten ->
`prototype_softmax` (or `unwhitened_topk_asym` on hard data) -> FCP GPU fast
path -> optional MS-CS penalty (geodesic NCM only, centered-cosine M).
Everything in the chain is exactly exchangeable; coverage is Monte-Carlo-tight
at ~0.90 on the random split at every cal size.

## Strategy

The campaign is "exchangeability-first": the exact-guarantee path is the
*default* and every deviation must announce itself (`warn_nonexchangeable` +
`allow_nonexchangeable=True`). New experiments are valid unless someone
explicitly opts out.

The current thrust is hardening the paper's weakest rationalization — the
transform — with controls that each sever one confound, all exchangeable by
construction (`transform_control_experiment.py`):

- **JL random projection at d'=128** (+/- the same cluster whitening):
  separates mere dimension reduction from PCA's data-adaptivity.
- **Ledoit-Wolf full-matrix whitening at 768** (global / within-cluster): the
  principled continuous alternative to hard truncation. If it loses, the rank
  cut is empirically justified against the obvious "why not shrinkage?"
  question; smoke + diagnostics already point that way.
- **Geometry diagnostic figure** (`embedding_geometry_diagnostic.py`): hubness
  skew, anisotropy, relative contrast, class contrast, 1-NN accuracy per
  space — assigns credit to the right component (hubness reduction belongs to
  whitening/scaling, not rotation, per Feldbauer & Flexer).

Adjacent active workstreams (other sessions): Multi-Dimensional CP
(Tawachi & Laufer-Goldshtein) as a possible full-CP extension of their
split-only cell-selection; geometry-conditional coverage (one-sided Mondrian
floor guarantee) as the primary novelty pilot, unmerged on
`worktree-novelty-pilots`.

## Key Discoveries (pivots)

- **"FCP wins" reframed**: SCP-geodesic == FCP-geodesic killed the machinery
  framing; the story is NCM design under a total label budget B.
- **Small-cal under-coverage was a bug, not a regime** (missing-class sentinel;
  cal=200 coverage 0.844 -> 0.91). Later mirrored by the MS-CS ŷ bug: both
  were asymmetric-function bugs, reinforcing claim 1.
- **prototype_softmax dominates on separable data** (CIFAR-100: sz 1.30 vs
  geodesic 1.58 @ cal=800; -19 to -22% at mid cal) but bloats/over-covers on
  fine-grained data where geodesic asym stays honest — a backbone-separability
  scope limit (Aircraft, Stanford Cars), stated plainly in the paper.
- **No class-level M helps a tight softmax NCM** (headroom-gating, structural);
  the faithful Fargion centered-cosine M is the better lever for the geodesic
  NCM and our old "MS-CS" was not Fargion's MS — novelty claim retracted,
  fidelity corrected.
- **PCA beats AE because the downstream NCM is linear**, not because the
  embeddings are (they are measurably curved); AE + nonlinear RBF NCM recovers
  the advantage at medium cal. First evidence for the transform-NCM coupling
  now formalized as claim 4.
- **A separate unlabeled pool is a convenience, not a requirement** (cal+test
  transduction recovers most of the gain; the pool's edge is sample count).
- **The transform-control verdict** (2026-07-05, 7-arm FCP sweep, 10 trials,
  both NCMs, both splits, 4 datasets, `output/pca_pilots/`). On separable
  many-class data (CIFAR-100, miniImageNet): (a) JL-128 is 2-3x WORSE than
  raw-768 — "mere dimension reduction" destroys NN structure, so the PCA win
  is entirely data-adaptive subspace selection; (b) rp128_cw == rp128 exactly
  — whitening pays ONLY inside the signal subspace; (c) LW-global-768 is
  catastrophic (equalizes noise with signal); (d) LW-cluster-768 beats raw
  but loses to pca128_cw everywhere (worst at cal=200: +77-88%) — hard
  truncation is load-bearing; (e) CIFAR-10 control: saturated, transform
  choice irrelevant, no harm. All arms hold ~0.90 on the random split at all
  datasets — efficiency varies 40x, validity never moves (Prop 2 visible).
- **The attribution INVERTS on the weak-backbone regime** (Aircraft): PCA-128
  alone is a no-op (matches the known vanishing-PCA-payoff scope limit), the
  in-subspace whitening becomes the active ingredient (pca128_cw cuts geodesic
  sets 21-27% vs raw), and **full-rank LW-cluster whitening BEATS truncation**
  (geodesic 22.0 vs 29.4 @ cal=800; 29.0 vs 46.7 @ cal=200; prototype ties).
  The geometry diagnostic predicted this: aircraft's pool participation ratio
  is 16 (vs 235-246 on cifar/mini) — DINOv2 collapses the variants into a
  tiny dominant subspace, and the residual fine-grained discrimination lives
  in the low-variance tail (LW-cluster 1-NN 0.52 vs PCA 0.40-0.46), which
  truncation discards and full-rank within-cluster whitening amplifies. So
  the transform's active ingredient tracks backbone separability: subspace
  selection when the signal is in the top spectrum, within-cluster variance
  equalization when it is not.

- **The whitening cluster count k is a second regime-tracking knob**
  (2026-07-06 k-sweep, k in {10..200}, cifar100+aircraft): on separable data
  pca128_cw is robust to k (flat from k=50) while lw_cluster NEEDS fine
  clusters (geodesic @800: 4.19 at k=10 -> 1.81 at k=200; prototype reaches
  1.30 = PCA level at k=200 — the earlier "shrinkage loses to truncation" gap
  was partly a cluster-resolution artifact for the prototype NCM). On
  aircraft the preference INVERTS: coarser is better for both arms
  (lw_cluster geodesic @800: 18.77 at k=10 vs 22.03 at k=100; pca128_cw
  prototype @800: 12.39 at k=10 vs 17.33 at k=100 — both new aircraft bests,
  -53%/-41% vs the old defaults). k_opt ~ PR/2..PR matches both regimes —
  another pool-only heuristic. Caveat: prototype + coarse-k full-rank
  whitening degenerates to the trivial set (conservative cov=1) — the softmax
  head needs either the PCA subspace or fine clusters.
- **The spectral-band diagnostic makes the mechanism quantitative**
  (labels-as-diagnostic, `spectral_band_diagnostic.py`): cifar/mini class
  signal peaks in pool-eigenbasis dims 17-128 (1-NN 0.82/0.93) — variance and
  signal coincide, truncation is right. Aircraft: the top-16 band carries
  75.5% of variance at 1-NN 0.27 (nuisance bulk — pose/livery), while dims
  129-512 carry 4.6% of variance at 1-NN 0.57, the best band, ABOVE the full
  space (0.41). Exactly what PCA-128 discards and whitening re-weights. Also:
  mean-distance class ratio misleads on aircraft (top band looks good via
  coarse family contrast); banded 1-NN is the honest lens for NN-based NCMs.
  Suggests a band-pass transform (drop top-16 nuisance + 513+ noise) as the
  collapsed-regime optimum — being probed (tail_r16 / tailcw_r16 / cw768).

- **Probe attribution: it is the full covariance metric, not band selection**
  (2026-07-06 probes). On aircraft, bare tail-keep (drop top-16) FAILS
  (geodesic 53.9 vs raw 40.2 @800 — the surviving bands still drown the
  129-512 signal without rescaling); tail + diagonal whitening only partially
  recovers (38.8); full-rank DIAGONAL cluster whitening also fails (36.1).
  Only the full-matrix LW-cluster whitening wins (18.8) — the nuisance bulk is
  not axis-aligned, so per-dimension rescaling cannot remove it; off-diagonal
  DECORRELATION is the operative ingredient on collapsed-spectrum data. On
  cifar the control direction holds (tail arms hurt, cw768 lands between raw
  and pca128_cw).
- **CUB-200 = the intermediate regime point, replicated end-to-end** (fresh
  336px extraction, PR=57.7, top-16 var share 44%, banded 1-NN peak at 17-128
  with a STRONG 129-512 mid-band 0.77): pca512_cw wins both NCMs @ cal=1600
  (geodesic 1.51, prototype 1.23), pca128_cw is neutral-to-harmful (1.75 vs
  raw 1.69 — replicating the historical "PCA-128 hurts CUB"), lw_cluster768
  loses (2.15). The three-regime map is now: d'_opt and metric richness track
  how deep the class signal sits in the pool spectrum — cifar/mini (PR~240,
  signal in top-128) -> truncate 128 + diag cluster whiten, fine k; CUB
  (PR~58, signal extends to 512) -> truncate 512; aircraft (PR~16, top band
  nuisance, signal at 129-512) -> no truncation, full-covariance LW-cluster
  whitening, coarse k. Pool-only regime flags: PR and top-band variance
  share; the labeled banded-1NN profile is the mechanistic explanation.

- **AE on low-PR data: nonlinearity is not the missing ingredient —
  decorrelation is** (2026-07-06 AE pilot, aircraft). Vanilla AE-128 behaves
  exactly as the variance-driven prediction said: geodesic 42.0 @ cal=800
  (~ PCA-128's no-op, worse than raw), prototype degenerate — MSE
  reconstruction spends the bottleneck on the 75%-variance nuisance bulk.
  Whiten-then-AE (LW-cluster k=10 -> AE-128) is the informative arm: it
  PRESERVES the LW metric win in a compact 128-d code (geodesic 20.5 vs
  full-rank LW 18.8) and un-breaks the prototype head that full-rank
  whitening degenerates (16.3 valid vs sz=K) — but it beats neither per-NCM
  best (geodesic 18.8 full-rank LW; prototype 12.4-13.9 pca128_cw coarse-k).
  So the AE's only value on collapsed-spectrum data is compressing an
  already-good metric, consistent with the alignment hypothesis (our
  metric/linear-friendly NCMs discard AE curvature). The one untested corner
  where AE could still matter: pairing with the nonlinear RBF NCM (the
  combination that won at medium cal on CIFAR-100) — low priority.

## Open Questions

- Pool-only surrogate for the banded-1NN profile: banded cluster-structure
  statistics remain the open candidate. The TAFSSL-inspired negentropy route
  was tested and KILLED (2026-07-08 pilot, `ica_negent_pilot.py`): (a) on
  aircraft the negentropy ranking DOES point at the right bands label-free
  (70/128 selected dims from 129-512, 0 from the dead 513-768 band) but the
  resulting compact subspace performs no better than variance selection
  (negent128_cw 27.8 ~ pca128_cw 25.9, both far behind full-rank LW 18.8) —
  refining the mechanism: aircraft's class signal is not hidden in a FEW
  identifiable tail dims, it is THINLY SPREAD across hundreds, so any
  128-dim cut loses most of it and full-rank whitening is genuinely
  necessary; (b) on CIFAR-100 negentropy selection is actively harmful
  (median selected rank 312; sets 5.9-9.4 vs PCA's 1.3-1.6) — with K=100
  balanced classes a per-dim class mixture approaches Gaussian by the CLT,
  so non-Gaussianity stops tracking class signal and instead chases
  kurtotic/rogue noise dims. TAFSSL's criterion is a FEW-WAY tool (their
  5-way mixtures are detectably non-Gaussian); it does not transfer to
  many-way pools. (c) LW-whiten -> FastICA-128 also loses (FastICA's
  internal PCA re-truncates the whitened space); on separable data it lands
  ~= pca128_cw, as expected. (d) All globally-standardized compact arms
  degenerate the prototype head (flat cosine posteriors) — prototype needs
  the scale hierarchy that within-cluster whitening of a PCA subspace
  preserves.
  Until then the heuristic map rests on PR / top-band variance share as flags.
- The 'auto' arm (pool-PR transform selection, implemented + smoke-tested) is
  PARKED by user decision (2026-07-06) until the per-regime picture is fully
  understood — heuristics first, productization later. The k knob (k ~ PR)
  and the d' rung (128/512/full) would fold into it when revived.
- Cluster high-trial confirmation of the aircraft findings (10 local trials;
  geodesic margins are >3 SE, prototype k=10 number deserves 50 trials).
- Prototype-softmax + full-rank whitening degenerates (flat cosine posteriors
  -> trivial sets, conservative): why the softmax head needs the compact
  subspace while the geodesic ratio does not is worth a short note.
- MS-CS centered-cosine M: still CPU-only, confirmed at 3 trials on the mean
  NCM only — asym + high-trial confirmation pending before it's paper-grade.
