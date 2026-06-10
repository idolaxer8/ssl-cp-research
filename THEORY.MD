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

Three load-bearing claims, each now backed by experiments in
`docs/findings.md`:

1. **Validity is an exchangeability property of the score function, and it is
   binary.** The coverage guarantee needs every nonconformity score to be a
   permutation-symmetric function of the calibration+test bag. Any transform
   fit on the calibration set — whitening, PCA, RBF bandwidth, a fixed ŷ in
   the MS-CS penalty — breaks this, even at O(1/n). Two practical
   consequences shaped the codebase: (a) all data-dependent transforms are
   fit on an *independent unlabeled pool* (`exchangeable_features.py`), or
   label-free on cal+test itself (pool_source result: exactly exchangeable,
   recovers most of the benefit); (b) asymmetries hiding in edge cases are
   real bugs, not noise — the missing-class `1e9` sentinel in `score_x`
   depressed small-cal coverage (0.844 at cal=200) until replaced by
   symmetric handling.

2. **The efficiency win is mostly the no-train-split NCM, not the
   transductive machinery.** SCP with the geodesic top-k NCM matches FCP with
   the same NCM at every budget (within 0.02 set size); FCP's apparent
   dominance over SCP-THR is really "distance-based NCM that spends the whole
   budget on calibration" vs "classifier that must split it". The honest
   paper claim is about NCM design + budget allocation; FCP vs SCP machinery
   is a cost/runtime choice (`predict(update_calibration_scores=False)` gives
   the cheap variant).

3. **Unlabeled data buys efficiency through the feature transform first, the
   penalty second.** The pool ablation decomposes the gain: pool-fit PCA-128
   + within-cluster whitening accounts for most of the 43-53% set-size
   reduction at cal=200-400; the MS-CS cluster-similarity penalty adds a few
   percent on top. MA-CS (true superclass labels) remains the strongest
   penalty (~23% at cal=600); label-free MS-CS lands within 2-4pp of it.

The lead pipeline is therefore: DINOv2 embeddings -> pool-fit PCA-128 +
cluster-whiten -> unwhitened geodesic top-k NCM -> FCP (GPU fast path) ->
optional MS-CS penalty. Everything in that chain is exactly exchangeable, so
coverage sits at the theoretical target 1 - floor(alpha(n+1))/(n+1) ~ 0.90 at
every cal size — confirmed to be Monte-Carlo-tight by the 100-trial
diagnostic.

## Strategy

The campaign is "exchangeability-first": make the exact-guarantee path the
*default* and force every deviation to announce itself. `GeodesicTopKMeanNCM`
defaults to `whiten=False`; every cal-fit variant routes through
`warn_nonexchangeable` and requires `allow_nonexchangeable=True`. This turns
a theory requirement into an API affordance — new experiments are valid
unless someone explicitly opts out.

The codebase now mirrors that theory split. `src/conformal_prediction.py` is
the transductive core (NCMs, `FullConformalPredictor` + GPU path, CV+, split
utilities, the exchangeability gate); `src/split_cp_baselines.py` holds the
inductive baselines we compare against (SoftmaxSplitCP, SemiCP,
ClusteredSplitCP) and is re-exported for compatibility; superseded one-off
drivers live in `src/archive/` with a README recording what superseded them.
Refactors of this code are constrained by reproducibility: recorded findings
must stay re-runnable, so structural changes are verbatim moves verified
bit-identical (same RNG stream -> same splits -> same numbers) and gated by
the two GPU parity tests in `tests/`.

The layout is kept ordered by a standing cadence rather than ad-hoc cleanups:
borderline scripts carry greppable `ARCHIVE-CANDIDATE (review <date>)` notes
collected in a watch list (`src/archive/README.md`), and a bi-weekly cloud
routine executes `docs/repo_cleanup_routine.md` — archiving past-due unused
candidates, flagging newly stale files, and delivering a review PR. Scripts
tied to pending `docs/findings.md` §10 items are HOLD-marked and outlive
their review date until the item closes.

Deliberately *not* unified: the three `stratified_split` variants
(pool-balanced random-boundary in macs/mscs; exact per-class in semicp;
generic N-way in pca_experiment) encode different statistical designs —
balancing cal is label-dependent and over-covers (~+1-3pp, conservative),
while the random-boundary variant is exactly exchangeable and tight. Merging
them would silently change experiment semantics.

## Key Discoveries (pivots)

- **"FCP wins" reframed**: SCP-geodesic == FCP-geodesic across datasets and
  budgets killed the original framing; the story is NCM design under a total
  label budget B.
- **Small-cal under-coverage was a bug, not a regime**: the missing-class
  sentinel fix moved cal=200 coverage 0.844 -> 0.91; pre-fix small-cal
  numbers are untrustworthy.
- **PCA beats AE because the downstream NCM is linear**, not because the
  embeddings are (they are measurably curved); pairing AE with a nonlinear
  RBF NCM recovers the advantage at medium cal.
- **A separate unlabeled pool is a convenience, not a requirement**: fitting
  the transform on cal+test (label-free, SCA-T-style) is exactly
  exchangeable and recovers most of the gain — the pool's only edge is
  sample count.
- **ClusterCP degenerates to plain SplitCP in our regime** (cal/K <= 8), so
  FCP+PCA+MS-CS also wins class-conditional CovGap, not just marginal size.

## Open Questions

- The cal=600 over-coverage (92.5%) under stratified splits: explained as
  balanced-cal conservatism + ceiling quantile + trial variance, but the
  planned stratified-vs-random ablation should settle it quantitatively.
- `geodesic_topk_asym` (the efficiency-best NCM) has not been re-confirmed on
  the cluster matched-518 setup, nor through the SCP-geodesic equivalence.
- Multi-dataset breadth for the penalty story (MA-CS/MS-CS on CUB-200,
  miniImageNet) and the very-small-cal regime (cal/K = 2-8) are the main
  unproven legs of the paper's headline table. The MA-CS-vs-MS-CS
  head-to-head driver on the exchangeable engine is
  `src/mscs_vs_macs_experiment.py` (results in `output/mscs_vs_macs{,_pca}/`).
