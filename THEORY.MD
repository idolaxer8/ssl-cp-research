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
   method to be deployable. The 08-03/04 menu round then moved the lever one
   stage earlier: alpha-QE pool-neighbor smoothing of the *embedding* (qe)
   subsumes the score-space correction — post-qe the SNAPS gain collapses
   (−41% -> −6.6%, eta* -> 0; purity rises but nothing is left to harvest).
   Same homophily fingerprint, one shared gate; the deployable pool-neighbor
   lever is input denoising, not score diffusion.

The lead pipeline — named **DWT** (Denoise -> Whiten -> Truncate) in the
08-10 instructor session — is therefore: DINOv2 embeddings -> qe
pool-neighbor denoising -> pool-fit cluster-whitening -> PCA truncation ->
unwhitened geodesic or prototype NCM -> FCP (GPU fast path) -> optional
MS-CS penalty. Every stage is pool-fit and permutation-equivariant, hence
exactly exchangeable: coverage sits at the theoretical target
1 - floor(alpha(n+1))/(n+1) ~ 0.90 at every cal size — confirmed
Monte-Carlo-tight by the 100-trial diagnostic.

## Strategy

The campaign is "exchangeability-first": make the exact-guarantee path the
*default* and force every deviation to announce itself. `GeodesicTopKMeanNCM`
defaults to `whiten=False`; every cal-fit variant routes through
`warn_nonexchangeable` and requires `allow_nonexchangeable=True`. This turns
a theory requirement into an API affordance — new experiments are valid
unless someone explicitly opts out.

The immediate campaign (08-10 week, instructor-set) is theoretical: DWT wins
empirically but has no rigorous justification, and that is now the paper's
critical path. The role model is DAPS Theorem 2 — score diffusion provably
reduces score error under a homophily radius — but DWT intervenes on the
*representation*, so the proof is rebuilt as a three-lemma chain:
(A) a denoising lemma, the DAPS-2 analogue in embedding space — under a
mixture model x = mu_y + eps, qe smoothing contracts within-class noise by
~ (1-lam)^2 + lam^2/k while paying a wrong-class contamination bias
~ lam*(1-h(k))*Delta_mu, improving the representation iff homophily h(k) is
high (the CSBM graph-convolution separability line proves exactly this for
neighborhood averaging); (B) Lipschitz transfer of that gain through the
distance-based NCM into score concentration; (C) a quantile-step argument
converting score concentration into smaller expected sets at fixed
coverage — the proof architecture of Feature CP's Theorem 6 (Teng et al.,
ICLR 2023, "cubic conditions"), whose mechanism (the quantile step costs
less in a space where scores concentrate around their quantile) is our
Step C even though its statement (feature- vs output-space bands,
regression) does not transfer. **Step A is now delivered
(`docs/dwt_denoise_theorem.md`, 08-12): Theorem D1 mirrors DAPS-2 verbatim
(deterministic, gap-free); Theorem D3 states the operative gate at the
MARGIN level — d' changes by [1 - 2*beta*kappa*(1-h)]/rho(beta,k_eff), so
improvement iff h > h* = 1 - (1-rho)/(2*beta*kappa), a scale-free condition
with zero fitted constants.** Measured on all five datasets
(`src/dwt_gate_constants.py`): h* ~ 0.27-0.40, every tested dataset on the
predicted side; the folklore "hom >= ~0.7" gate was an interpolation across
the h in (0.46, 0.80) data gap using the SNAPS *score* correction — a
different operator. Registered discriminating prediction: qe should GAIN on
stanford_cars (h_w = 0.46, predicted d'-ratio 1.52) where folklore predicts
harm.

The 08-10 lit sweep (`docs/dwt_theory_litsweep.md`) confirmed the niche is
open — no post-DAPS work ports Theorem 2 to feature-similarity neighbors
(SNAPS's lone efficiency result assumes perfectly pure neighborhoods), and
DAPS itself flags the kNN-graph setting as future work — and supplied a
converter for each step: CSBM neighborhood-averaging separability for A;
for C, either Dhillon et al.'s exact expected-size identity (stochastic
dominance of score CDFs orders expected sizes, finite-n) or the
linear-in-score-error distance-to-oracle-length bounds (Conrad et al.,
Sadinle et al.), with Behboodi et al.'s transductive KL-linear penalty as
the FCP-arm variant. W and T ride classical optimality (RCA whitening,
spiked-covariance truncation). Notably the alpha-QE retrieval literature
has zero theory — the D stage must be argued as graph-signal denoising.

The standing empirical campaign hardens the two live levers into method
components rather than lucky points. (a) The SNAPS pool correction: show the gain is a
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
- **qe subsumes SNAPS**: smoothing the embedding over pool neighbors
  captures what score diffusion was harvesting — the gain lives in the
  representation, one stage earlier than DAPS/SNAPS operate. This is why
  the justification must be a representation-space theorem, not a score
  smoothing one.
- **The feature-space CP theory exists and its architecture transfers**:
  Teng et al. (ICLR 2023, Feature CP) prove feature-level CP beats
  output-level CP via three "cubic conditions" (length preservation,
  expansion, quantile stability); the load-bearing quantity is the mean
  distance of scores from their (1-alpha)-quantile — the "cost of the
  quantile step" — which they also validate empirically per dataset. Their
  statement compares two score spaces for one model; ours compares two
  representations under one score family, so we reuse the skeleton, not
  the theorem.
- **The homophily gate lives at the margin level, not the norm level**
  (08-12, `docs/dwt_denoise_theorem.md`). In expectation, norm-level
  "denoising" ALWAYS helps at some smoothing weight — no gate exists there
  (Theorem D2); the gate appears only when the smoothing identity is
  projected onto a confusable-pair axis, where signal damage and noise
  reduction are both first-order in beta (Theorem D3). Corollaries fell
  out for free: the gate is scale-free (sigma and Delta cancel — hence one
  threshold across datasets, the sweep's "one law"), harm below the gate
  is not tunable (no small-beta rescue), one hop is optimal (drift
  compounds, variance is already at its floor), and classic alpha-QE's
  implicit self-weight beta_hat = W/(1+W) rises with neighborhood
  similarity — adapting the WRONG way on fine-grained data (0.86 on
  aircraft vs 0.62 on cifar100). Measured with zero fitted constants,
  every tested dataset lands on the predicted side, and the measured gate
  h* ~ 0.27-0.40 says the folklore "0.7" was two operators conflated: the
  D-stage smoother and the SNAPS score correction do NOT share a
  break-even.

## Open Questions

- **The cars discriminating run** (registered 08-12 before the
  experiment): Theorem D3 predicts qe GAINS on stanford_cars (h_w = 0.46 >
  h* = 0.29, d'-ratio 1.52) while the folklore 0.7 gate predicts harm; qe
  was never run there. One champion-pipeline experiment decides which gate
  story ships. Both known (I)-violations push the true gate UP from the
  measured h*, so a cars failure has a ready mechanism (G2) — a cars
  success has none in the folklore direction.
- **Closing the two medium gaps in the DWT proof draft**
  (`docs/dwt_theory.md` + `docs/dwt_denoise_theorem.md`, gap ledger
  G1-G10). G2 — selection-noise correlation — now has SIGNED directions
  ((V1) retained-own-noise = effective beta shrink; (V2) error-aligned
  foreign selection = kappa inflation; both push h* up) but still needs
  its remainder lemma (route = mean-shift small-ball expansion). G6 —
  right-tail stochastic dominance of score CDFs (moments alone don't give
  it; route = Anderson's inequality under conditional Gaussianity, or
  name-the-condition + validate, the Feature-CP precedent). Two scope
  decisions pending: the champion geodesic *ratio* NCM is not covered
  (scale-invariance neuters naive arguments — the theorem covers the
  prototype NCM), and the FCP arm needs the transductive size identity. A
  found feature: validity pins the true-class term of E|C|, so efficiency
  is purely the false-class overlap below q_hat — and the theorem's
  assumptions fail exactly where DWT empirically fails (aircraft),
  quantitatively so since 08-12.
- **The label-free gate is THE deployability blocker** for the
  pool-neighbor lever — now with a theorem-shaped target: predict
  sign(d'-ratio) = sign of [1 - 2*beta*kappa*(1-h)] - rho without pool
  labels. beta and k_eff are label-free already; only (h_w, kappa) need
  surrogates (candidates: margin-tail, pool self-consistency, hub-skew).
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
