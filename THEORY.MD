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

2. **The efficiency win is mostly the no-train-split NCM; FCP stays
   load-bearing for exactness.** The variant once called "SCP-geodesic" is
   NOT split CP — it is a static-cal *reduction* of FCP
   (`predict(update_calibration_scores=False)`: same NCM, same fit, no
   train/cal split, no exact guarantee of its own) and it reproduces FCP
   set-for-set because the dropped per-candidate refit is O(1/n). So its tie
   with FCP is a runtime footnote, not an "SCP matches FCP" result. True
   inductive Split CP (SCP-THR, a trained head on a disjoint split) collapses
   at our budgets — FCP remains the only path to tight sets WITH the exact
   guarantee. The honest attribution stands: efficiency comes from NCM design
   + spending the whole budget on calibration; exactness comes from FCP.

3. **Unlabeled data buys efficiency through the feature transform first, the
   penalty second.** The pool ablation decomposes the gain: pool-fit PCA-128
   + within-cluster whitening accounts for most of the 43-53% set-size
   reduction at cal=200-400; the MS-CS cluster-similarity penalty adds a few
   percent on top. MA-CS (true superclass labels) remains the strongest
   penalty (~23% at cal=600); label-free MS-CS lands within 2-4pp of it.

4. **Post-hoc score correction toward external pool anchors is the live
   add-on lever; score fusion is not.** The MDCP line closed with a negative:
   multi-NCM fusion adds ~nothing once the features are right (the paper's
   self-ensemble gain dissolves into adaptive-score slack; at high budgets a
   trained softmax head overtakes the distance NCMs — "no-train-split NCM
   wins" is a small-budget theorem, which is fine because small budget is our
   regime). By contrast, the SNAPS-style pool-kNN correction — pull each
   score toward the mean score of its unlabeled-pool neighbors at the same
   label — is a new best-known on CIFAR-100 at every cal, and it survives
   *exact* full-CP wrapping at zero efficiency cost, unlike LATA's collective
   smoothing (a symmetric bag shift cancels in p-value ranks; an external
   anchor moves them). Two structural lessons came with it: any cal-fit NCM
   plus *local* neighbor score aggregation self-leaks (the ego's neighbors
   sit inside its own prototype region, deflating the quantile — repaired by
   LOO re-prototyping), and the whole effect rides on pool-neighbor label
   homophily: gain above roughly 0.7 (mini .92, cifar .8), harm below (cars
   .45, aircraft .25). The harmful regime must be excluded label-free for the
   method to be deployable.

The lead pipeline is therefore: DINOv2 embeddings -> pool-fit PCA-128 +
cluster-whiten -> unwhitened geodesic or prototype NCM -> FCP (GPU fast
path) -> optional MS-CS penalty and, where homophily allows, the SNAPS pool
correction. Everything in that chain is exactly exchangeable, so coverage
sits at the theoretical target 1 - floor(alpha(n+1))/(n+1) ~ 0.90 at every
cal size — confirmed to be Monte-Carlo-tight by the 100-trial diagnostic.

## Strategy

The campaign is "exchangeability-first": make the exact-guarantee path the
*default* and force every deviation to announce itself. `GeodesicTopKMeanNCM`
defaults to `whiten=False`; every cal-fit variant routes through
`warn_nonexchangeable` and requires `allow_nonexchangeable=True`. This turns
a theory requirement into an API affordance — new experiments are valid
unless someone explicitly opts out.

The current campaign hardens the two live levers into method components
rather than lucky points. (a) The SNAPS pool correction: show the gain is a
surface, not a point — eta x k x neighbor-weighting sweeps across datasets,
extend the homophily map with CUB-200 (the missing mid point) and CIFAR-10,
and build the label-free homophily gate (margin-tail / pool
self-consistency proxies) so the low-homophily harm regime switches itself
off; that turns the aircraft/cars scope limit from a caveat into a method
component. (b) The transform stage, reframed by the PCA audit as pool-fit
metric learning: extend the menu beyond {PCA truncation, whitening} along
the two known failure axes — nonlinearity (kernel PCA, AE encoder + RBF
NCM) and local structure (per-cluster PCA, pool-graph spectral embeddings)
— with the label-free CP-native selector choosing per-dataset (margin-tail
solves high-PR, a regime fallback covers low-PR, mid-PR still open).

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

- **"FCP wins" reframed (naming corrected 2026-08)**: the static-cal FCP
  reduction (`update_calibration_scores=False`, formerly mis-named
  "SCP-geodesic" — it is not split CP) ties FCP set-for-set, killing the
  original "transductive machinery wins" framing; the story is NCM design
  under a total label budget B, with FCP still load-bearing for the exact
  guarantee (true split CP collapses at our budgets).
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
- **Aircraft inverted the PCA story and produced the regime rule**: on
  fine-grained data the class signal lives in the low-variance tail, so PCA
  truncation is a no-op and full-rank Ledoit-Wolf cluster-whitening wins;
  the pool participation ratio (16 vs ~240) predicts which regime holds —
  the transform is metric learning, not compression.
- **MDCP closed as a fusion method, paid off as an audit**: its purity-map
  re-calibration gave the first genuine adaptivity win (same set size,
  CIFAR CovGap 9 -> 5.6), and dissecting the paper's self-ensemble showed
  the gain was adaptive-score slack, not fusion machinery.
- **SNAPS self-leak**: a cal-fit NCM plus local neighbor aggregation biases
  cal scores only (cov 0.74 at cal-200, scales 1/n_c) — validity of
  published neighbor-aggregation methods silently rides on a separately
  trained score model. The LOO repair restores exact validity, and the
  corrected method keeps its full gain inside exact full CP (no LATA-style
  cancellation).

## Open Questions

- **The label-free homophily gate is THE deployability blocker** for the
  SNAPS correction: kNN label homophily (the dial) needs labels; candidate
  proxies are margin-tail and pool self-consistency. Without it the method
  ships with a caveat instead of a switch.
- Is the headline arm (eta=0.5, k=10) genuinely robust, or does the optimal
  arm drift with homophily and cal? The cars k=10-vs-20 ordering hints harm
  grows with neighborhood size; the sweep will tell.
- The label-free transform selector has no answer at mid-PR (CUB-200), and
  it is unknown whether PR keeps selecting correctly once the menu grows
  beyond linear projections.
- Neither SNAPS nor geometry-Mondrian moves *class*-conditional coverage
  (CovGap flat, worst-class floor slightly worse under correction) — the
  hardest classes are exactly the low-local-homophily ones. The class axis
  needs class-level machinery (class-Mondrian at cal >= 9K, or ClusterCP
  groups on corrected scores).
- Multi-dataset breadth for the penalty story (MA-CS/MS-CS on CUB-200,
  miniImageNet) and the very-small-cal regime (cal/K = 2-8) remain the
  unproven legs of the paper's headline table
  (`src/mscs_vs_macs_experiment.py`, `output/mscs_vs_macs{,_pca}/`).
