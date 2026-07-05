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
- **Truncation is load-bearing** (2026-07-05): LW shrinkage whitening at full
  rank — the "principled" alternative — underperforms PCA-128+cw at cal=400 on
  both NCMs (prototype 2.05 vs 1.61, geodesic 2.63 vs 2.01, smoke); the JL arm
  behaves exactly as JL theory predicts, validating the control design.

## Open Questions

- Full 7-arm FCP sweep (running): does rp128_cw close any gap to pca128_cw?
  Much-closed => whitening is the lever (isotropization framing returns);
  barely-closed => adaptive subspace selection carries the paragraph.
- **d'-selection protocol** — the one honest remaining vulnerability: 128/512
  came from U-shapes on the evaluation datasets. Candidate fixes: a-priori
  default (Babenko et al. 2014's 128-d operating point) + robustness band, or
  a pool-only criterion. Not yet piloted.
- Do the transform conclusions transfer to the scope-limit datasets
  (Aircraft/Stanford Cars)? The diagnostic run there should predict the
  vanishing PCA payoff (low class contrast at any d').
- MS-CS centered-cosine M: still CPU-only, confirmed at 3 trials on the mean
  NCM only — asym + high-trial confirmation pending before it's paper-grade.
