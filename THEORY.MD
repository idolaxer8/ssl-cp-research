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

Five load-bearing claims, each backed by experiments in `docs/findings.md` /
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
   The current NCM ranking: `prototype_softmax` (FCA-style class-mean cosine
   softmax, closed-form LOO refit) is tightest on separable data at every
   cal; `geodesic_topk_asym` is the more honest choice (best
   class-conditional CovGap) on hard fine-grained data. Class-level
   similarity penalties are **headroom-gated**: no M (cluster, faithful
   Fargion centered-cosine, prototype-Gram, confusion) can help a tight
   softmax set — the classes cluttering it are exactly the most-similar to
   ŷ, which the penalty protects — while M remains a modest real lever for
   the geodesic NCM.

3. **Unlabeled data buys efficiency through the feature transform first, the
   penalty second** (pool ablation: features 41-48% of the cut, penalty a few
   percent, all exchangeable).

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

5. **The transform IS the metric half of the NCM.** Whiten-then-Euclidean is
   Mahalanobis under the pool covariance, so pool-fit PCA-truncate +
   within-cluster whitening = a low-rank regularized estimate of the
   precision metric that both NCMs' inner products live in; d' is the
   metric's regularization knob (the d'-U-shape is bias-variance in metric
   estimation, not a coverage effect). This dissolves the "PCA is ad hoc /
   unrelated to the NCM" objection that the old "removes low-variance noise"
   line invited (falsified in-house: PCA-128 keeps only 56% of variance,
   participation ratio 232). Mechanism, per the geometry diagnostics
   (2026-07-05): the lever is *data-adaptive subspace selection* — on the
   CIFAR-100 pool PCA-128 raises between/within class contrast ~40%
   (1.42 -> 1.97) and triples relative distance contrast (2.2 -> 7.0),
   while a Johnson-Lindenstrauss projection to the same d changes neither
   (it preserves noise as faithfully as signal) and full-768 Ledoit-Wolf
   whitening *destroys* contrast (class ratio 1.10) by equalizing pure-noise
   directions with signal. Anisotropy itself is mild in DINOv2 (mean pair
   cosine 0.06) and dies with centering — so the paper should lean on
   contrast concentration, not the NLP-style anisotropy-repair story.

The lead pipeline — named **DWT** (Denoise -> Whiten -> Truncate) in the
08-10 instructor session — is therefore: DINOv2 embeddings -> qe
pool-neighbor denoising (where homophily allows) -> pool-fit
cluster-whitening -> PCA truncation (skipped at low PR: full-rank
LW-cluster whitening instead) -> unwhitened geodesic or prototype NCM ->
FCP (GPU fast path) -> optional MS-CS penalty (geodesic NCM only,
centered-cosine M). Every stage is pool-fit and permutation-equivariant,
hence exactly exchangeable: coverage sits at the theoretical target
1 - floor(alpha(n+1))/(n+1) ~ 0.90 at every cal size — confirmed
Monte-Carlo-tight by the 100-trial diagnostic.

## Strategy

The campaign is "exchangeability-first": the exact-guarantee path is the
*default* and every deviation must announce itself (`warn_nonexchangeable` +
`allow_nonexchangeable=True`). New experiments are valid unless someone
explicitly opts out.

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
different operator. The registered discriminating prediction — qe should
GAIN on stanford_cars (h_w = 0.46, predicted d'-ratio 1.52) where folklore
predicts harm — **RESOLVED 08-13 against the (I)-model: qe HARMS on cars
(+11/+28/+40% at 2/4/8 shots/class, coverage on target), and the measured
margin d'-ratio is 0.735, not 1.52. The pre-registered escape hatch fired
exactly as written: the (V1)/(V2) selection effects are first-order at
mid-homophily. The structural finding survives and sharpens — measuring
the d'-ratio directly on all five datasets (1.05/1.18/1.21/0.74/0.64 vs
predicted 1.96/2.41/2.35/1.52/0.70) shows the (I)->measured damping is
universal, the measured ratio is monotone in homophily, and its sign
calls the qe verdict 5/5. So D3's margin->size link is validated
everywhere; what needs repair is only the composition-constants
prediction of d' (the G2 remainder). Empirical gate bracket:
h_w in (0.46, 0.81).**

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
— with the label-free dials read as a descriptive regime map (ICLR-2027
decision 2026-08-18: no automatic transform-selection gate is shipped;
per-regime choice is fixed a priori, dials are the diagnosis).

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

Historical note on the selector line: the transform-control campaign (JL /
Ledoit-Wolf controls, geometry and spectral-band diagnostics, tail probes)
closed with the three-regime map; its attempted successor — G1 conformal
metric LEARNING (continuous spectral filter optimized against a label-free
set-size surrogate, branch `worktree-conformal-metric`) — was REWOUND on
2026-08-03: the surrogate cannot rank candidates (landscape Spearman -0.45
aircraft / ~0 cifar100), and the repair path was accreting machinery. Its
theory survives (`docs/conformal_metric_objective.md` there: E|C| =
Beta-weighted separation functional; within-x/accuracy objectives
structurally misaligned), and one corollary outlived it: menu SELECTION and
metric learning are the same problem at |Theta| finite vs continuous —
ranking a continuum label-free failed; ranking a small menu is easy and can
even use a micro labeled fold with EXACT validity (A(pool, D_sel)
measurability). That selector line is now FUTURE WORK per the 08-18
descoping.

The current thrust is therefore the instructor's simpler framing
(week-08-03 goal 2): EXTEND the pool-fit menu with fixed, closed-form maps
along the two failure axes, judged by true FCP only
(`docs/pool_repr_menu_plan.md`, arms in `transform_control_experiment.py`):

- **Nonlinearity axis, rung 0**: per-dim Yeo-Johnson Gaussianization
  (`yj_*`; Distribution-Calibration lineage) — the cheap control; if it and
  the local arms are no-ops, heavier nonlinear arms (diffusion maps,
  RFF-kPCA) stay parked.
- **Local axis, denoising**: alpha-QE pool-neighbor feature smoothing
  (`qe_*`; Radenovic retrieval canon) — SNAPS's information source moved
  from score level to feature level; doubles as a test of whether
  low-homophily harm is mechanical or information-theoretic.
- **Local axis, projection**: Locality Preserving Projection (`lpp*_cw`;
  He & Niyogi 2003) — variance-BLIND linear reduction; the sharp bet for
  the collapsed-spectrum regime where class signal hides in low-variance
  locally-discriminative directions (aircraft mechanism).
- **Local axis, metric field**: per-cluster soft LW whitening (`lwsoft768`;
  MPPCA-lite, SSD precedent) — the mildest global nonlinearity: locally the
  whitening we trust, metric varying over the manifold.

Round-1 verdict (2026-08-03, 10-trial pilot, `output/pool_repr_menu/`,
sec 7 of the plan doc): the win came from the axis predicted to be safest,
not the one aimed at the hard regime. **alpha-QE pool-neighbor smoothing is
the first non-{PCA, whitening} menu member** — cifar100 prototype
2.34/1.44/1.23 @ cal 200/400/800 (incumbent 4.17/1.64/1.30; the cal-200
cell undercuts even the SNAPS-corrected best-known 2.57, via a pure
exchangeable transform), CUB 2.77 @400 with a high-cal crossover, aircraft
harmed. Its gain/harm pattern reproduces the SNAPS homophily map at the
feature level — so score-level and feature-level pool-neighbor levers share
one regime dial and should share one label-free gate. YJ Gaussianization
was a clean no-op (marginal nonlinearity is not a lever). LPP was killed
WITH a mechanism that sharpens the three-regime map: it preserves the pool
kNN graph faithfully, and on collapsed-spectrum data that graph is
majority wrong-class — pool neighborhood structure is unreliable at low
PR, only pool second-order statistics are; this also deprioritizes
diffusion-map-style round-2 arms (same graph-trust axis). lwsoft dropped
by the pre-registered rule (one-budget win only). Open follow-ups: qe
under the champion asym NCM + miniimagenet (running), qe x SNAPS stacking,
the shared homophily gate, and the aircraft prototype+pca512 @800 anomaly
(14.90, best cell in the run, vs the "prototype bloats on fine-grained"
scope limit).

Adjacent active workstreams (other sessions): Multi-Dimensional CP
(Tawachi & Laufer-Goldshtein) as a possible full-CP extension of their
split-only cell-selection; geometry-conditional coverage (one-sided Mondrian
floor guarantee) as the primary novelty pilot, unmerged on
`worktree-novelty-pilots`.

## Key Discoveries (pivots)

- **"FCP wins" reframed (naming corrected 2026-08)**: the static-cal FCP
  reduction (`update_calibration_scores=False`, formerly mis-named
  "SCP-geodesic" — it is not split CP) ties FCP set-for-set, killing the
  original "transductive machinery wins" framing; the story is NCM design
  under a total label budget B, with FCP still load-bearing for the exact
  guarantee (true split CP collapses at our budgets).
- **Small-cal under-coverage was a bug, not a regime** (missing-class sentinel;
  cal=200 coverage 0.844 -> 0.91). Later mirrored by the MS-CS ŷ bug: both
  were asymmetric-function bugs, reinforcing claim 1. Pre-fix small-cal
  numbers are untrustworthy.
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
  now formalized as claim 5.
- **A separate unlabeled pool is a convenience, not a requirement**: fitting
  the transform on cal+test (label-free, SCA-T-style) is exactly
  exchangeable and recovers most of the gain — the pool's only edge is
  sample count.
- **ClusterCP degenerates to plain SplitCP in our regime** (cal/K <= 8), so
  FCP+PCA+MS-CS also wins class-conditional CovGap, not just marginal size.
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
  best. So the AE's only value on collapsed-spectrum data is compressing an
  already-good metric, consistent with the alignment hypothesis (our
  metric/linear-friendly NCMs discard AE curvature).
- **qe-upgrade pilots: neighbor SELECTION is a lever independent of dose; one
  hop is optimal** (2026-08-10, `docs/qe_upgrade_pilots.md`). (a) hub-debiased
  selection (subtract gamma * pool-point mean top-k similarity from the
  selection score, CSLS/NNN lineage; gamma~1) improves qe on 3/3 datasets —
  new best starved aircraft cell (26.55 vs 27.38 @200 geodesic), safe+hub
  dominates safe, and every miniimagenet cell wins — the first harm-regime
  lever that works through neighbor CHOICE rather than dose reduction
  (k-reciprocal filtering loses to beta-dilution everywhere). (b) 2-hop
  smoothing loses at EVERY cell in BOTH regimes: one-hop qe is not a
  truncation compromise, it is the optimum (first-order filter suffices; the
  second hop compounds wrong-class contamination quadratically — now
  Corollary C4 of Theorem D3). (c) At large gamma the hub-debiased map
  self-gates toward identity — a soft OFF switch. Related: under the
  prototype NCM on aircraft, OFF beats every smoothing mode (no-qe
  ldapool512 18.22 vs champion 19.81 @800) — any gate needs an OFF setting,
  not just dilution.
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
  break-even. CAVEAT (08-13/08-18): D3's quantitative layer OVERPREDICTS —
  the sqrt(k) averaging dividend does not exist (shared-field noise, phi ~
  0.8-0.96) — so D1 (exact, assumption-free) is the theory's grounding and
  the d'-ratio is a measured diagnostic, not a derived gate.

## Open Questions

- ~~The label-free homophily gate as THE deployability blocker~~ — RESCOPED
  (ICLR-2027 decision 2026-08-18): no automatic gate/selector is shipped;
  the dials (PR, h_w, d'-ratio) are presented as a descriptive regime map
  and the per-regime transform choice is fixed a priori. The label-free
  surrogate for (h_w, kappa) (theorem-shaped target: predict
  sign(d'-ratio) without pool labels; candidates margin-tail, pool
  self-consistency, hub-skew), the mid-PR (CUB) selector, and the 'auto'
  arm (PARKED 2026-07-06) are all FUTURE WORK. A wrong per-regime choice
  costs efficiency, never validity.
- ~~The cars discriminating run~~ **RESOLVED 08-13 (commit 84336d8,
  `src/cars_qe_gate_experiment.py`): qe harms; the (I)-model mis-sorted
  the cell (measured d'-ratio 0.735 vs predicted 1.52) via the
  pre-registered V1/V2 mechanism; the margin->size link held and the
  measured d'-ratio sign calls the verdict 5/5 across datasets.**
  ~~Where does the overprediction live?~~ **LOCALIZED same day
  (`src/dprime_overprediction_diagnostic.py`, research edition §8b):
  in the DENOMINATOR — the sqrt(k) averaging dividend does not exist
  (measured noise shrink 0.90-0.98 vs modeled 0.31-0.43), because
  Var(nu_v) ~ sigma_v^2 with corr(e_v, nu_v) ~ 0.85-0.90: the kNN mean is
  the mean-shift local mean, and within-class variance along confusable
  axes is ~80-95% shared structured field (phi ~ 0.78-0.96), only the iid
  shell averages. Numerator damage also overpredicted (S ~ 1.0-1.05 at
  high h — mode-sharpening). Revised mechanism: qe = mean-field mover,
  not variance reducer; helps iff the local field flows toward the
  own-class mode.** The G2 remainder lemma now has a concrete shape: a
  two-component noise model (shared field + iid shell) with the single
  new constant phi, replacing 1/k_eff by phi + (1-phi)/k_eff and
  re-deriving the gate. Until then the deployable gate stays empirical
  (qe OFF below h ~ 0.8).
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
  quantitatively so since 08-12. NOTE (ICLR-2027 plan §4): these gaps are
  NOT on the submission's critical path — the paper is empirical-first and
  claims only D1 + exact-core W1a/W1b/T1a + validity (decision D-1).
- Is the headline arm (eta=0.5, k=10) genuinely robust, or does the optimal
  arm drift with homophily and cal? The cars k=10-vs-20 ordering hints harm
  grows with neighborhood size; the sweep will tell.
- Neither SNAPS nor geometry-Mondrian moves *class*-conditional coverage
  (CovGap flat, worst-class floor slightly worse under correction) — the
  hardest classes are exactly the low-local-homophily ones. The class axis
  needs class-level machinery (class-Mondrian at cal >= 9K, or ClusterCP
  groups on corrected scores).
- Multi-dataset breadth for the penalty story (MA-CS/MS-CS on CUB-200,
  miniImageNet) and the very-small-cal regime (cal/K = 2-8) remain the
  unproven legs of the paper's headline table
  (`src/mscs_vs_macs_experiment.py`, `output/mscs_vs_macs{,_pca}/`).
- Pool-only surrogate for the banded-1NN profile: banded cluster-structure
  statistics remain the open candidate. The TAFSSL-inspired negentropy route
  was tested and KILLED (2026-07-08 pilot, `ica_negent_pilot.py`): the
  ranking points at the right bands on aircraft but the compact subspace
  performs no better than variance selection (signal THINLY SPREAD across
  hundreds of dims -> full-rank whitening genuinely necessary); on CIFAR-100
  negentropy chases kurtotic noise (K=100 mixtures are CLT-Gaussian per
  dim). TAFSSL's criterion is a FEW-WAY tool. Until a surrogate exists the
  heuristic map rests on PR / top-band variance share as flags.
- Cluster high-trial confirmation of the aircraft findings (10 local trials;
  geodesic margins are >3 SE, prototype k=10 number deserves 50 trials).
- Prototype-softmax + full-rank whitening degenerates (flat cosine posteriors
  -> trivial sets, conservative): why the softmax head needs the compact
  subspace while the geodesic ratio does not is worth a short note.
- MS-CS centered-cosine M: still CPU-only, confirmed at 3 trials on the mean
  NCM only — asym + high-trial confirmation pending before it's paper-grade.
