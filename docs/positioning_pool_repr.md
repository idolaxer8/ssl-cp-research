# Positioning audit: the pool-fit representation pipeline vs the field

**What this document is.** A positioning + gap audit (2026-08-06) for the
adopted pipeline of `pool_repr_menu_plan.md` — T(x) = E^T W0 (D(x) - mu)
feeding FCP — against the CP literature. Companion to `literature.md`
(esp. sec 9's novelty slot, verified 2026-07-06) with a fresh collision
spot-check (2026-08-06, none found). Gap statuses reflect user review of
the same date; gap 4 is under active testing and intentionally not
addressed here.

---

## 1. Where the novelty actually sits

Every individual transform is ported and must be cited as lineage:
alpha-QE from retrieval (Radenovic TPAMI19), within-class whitening /
learned whitening (Mikolajczyk & Matas; Radenovic sec 3.4), LDA-style
discriminant on k-means pseudo-clusters. The novelty is NOT the parts.
Three pillars carry the paper:

1. **The architectural slot.** Unlabeled data enters the REPRESENTATION,
   before any score is computed. Existing uses of unlabeled data in CP
   act on scores or thresholds (SemiCP, SNAPS/DAPS, unsupervised
   calibration, OT-robust thresholds); the representation slot is empty
   in the literature (sec 9 negative finding; re-verified by spot-check
   2026-08-06 — see sec 5 re-sweep TODO).
2. **Exact validity for free, and FCP as the only exact machinery at
   this budget.** Any pool-measurable map keeps coverage exactly 1-alpha
   (Prop 2, `docs/theory.md` sec 2) — validity never constrains the
   transform search. The exactness is delivered by full CP: the only
   *valid* competitor at tiny cal, true split CP with a trained head,
   collapses (sec 3). So "post-processing + FCP" is the correct headline
   pairing: the representation buys efficiency, FCP buys exactness.
   BUDGET QUALIFIER (G3 result, 2026-08-09): "collapses" holds at
   cal/K <= ~4 shots/class; by ~6 shots/class on saturating-geometry
   data (aircraft cal-800: probe 15.85 vs champion 19.81) a trained
   probe + split CP crosses over and wins. State the regime as a
   budget, never as "always".
3. **The subsumption result.** Representation-level use of the pool
   empirically dominates and largely subsumes the score-level use: qe
   alone >= SNAPS alone in every cell; SNAPS marginal gain collapses
   post-qe (-41% -> -6.6% at the most starved cell, ~0 elsewhere),
   eta* -> 0; runtime asymmetry is architectural (0.4 ms/point fixed map
   vs 174-410 ms per recalibration + an LOO leak repair). No other paper
   occupies both slots, so no other paper can run this comparison.

## 2. Lane-by-lane lineup

**Lane 1 — unlabeled data at the score/threshold level (direct
competitors).** SemiCP (arXiv 2505.21147), SNAPS (NeurIPS 2024) and the
DAPS graph family, Mazuelas unsupervised calibration (NeurIPS 2025,
2510.07185), Correia & Louizos OT-robust thresholds (2507.10425),
Flechsig zero-label CP (2509.10321). All inject the pool AFTER scoring.
We sit upstream and hold the head-to-head 2x2 (sec 1, pillar 3). New
adjacent entry to cite: anatomically-aware CP with random walks
(2601.18997) — diffuses conformal SCORES on a kNN graph of
foundation-model features (segmentation); reinforces that the field
keeps choosing the score slot.

**Lane 2 — "CP in feature space".** Feature CP (Teng et al., ICLR 2023,
2210.00173) is the name-collision risk: CP on the trained network's own
intermediate features, regression, band propagation, approximate. Also
CP with Learned Features (2404.17487 — learned partitions for
CONDITIONAL coverage) and CONFIDE (2604.08885 — kNN scores on LM
representations with grid-searched PCA/Mahalanobis; closest concurrent
analogue, our pool-spectrum selection is the delta). None uses an
independent unlabeled pool; none gets exact validity structurally; none
targets classification set size at tiny cal. Disarm Feature CP by name,
early.

**Lane 3 — transductive foundation-model adaptation before CP.** The
Silva-Rodriguez trilogy (FCA 2506.06076, Conf-OT 2505.24693, SCA-T
2506.17503) is closest in spirit: transform, then conformalize.
Standing differentiators (`literature.md`, Key Differentiators) hold;
one sharpening: Conf-OT transports PROBABILITIES, we re-metrize the
EMBEDDING SPACE itself — which is why the pipeline composes with any
NCM, and why the T-space NCM re-audit (the matched NCM changes with the
geometry; prototype overtakes geodesic) is a finding they cannot state.

**Lane 4 — non-CP ancestry (the biggest framing risk).** Transductive
few-shot learning does nearly identical embedding post-processing:
TAFSSL (PCA/ICA on unlabeled task data), SimpleShot, PT-MAP,
Distribution Calibration / Tukey, and especially Embedding Propagation
(Rodriguez et al., ECCV 2020) — neighbor-smoothing of features, a
functional cousin of the qe stage, currently uncited in
`literature.md`. Expected reviewer line: "standard few-shot tricks, new
metric." The defense, stated explicitly: (a) in CP these transforms
come with a THEOREM — any pool-fit map keeps coverage exact, validity
never constrains the search — a statement with no analogue in those
papers; (b) the objective is CP set size, and accuracy surrogates are
measurably misaligned with it (`conformal_metric_objective.md` sec 5 —
the negative result becomes load-bearing here); (c) the audit content
is new: order matters (smooth-then-whiten; reversed is catastrophic
exactly where whitening is full-rank), LPP killed with mechanism,
C >= K rule, stage ablation showing composition carries and no stage
suffices alone.

## 3. The FCP framing correction (user, 2026-08-06)

The historical "SCP-geodesic == FCP-geodesic" comparison is bad
NAMING, and the conclusion drawn from it ("don't stake anything on
FCP") was wrong. Code-verified 2026-08-06
(`src/conformal_prediction.py`, `predict()` docstring + loop):

- "SCP-geodesic" is `FullConformalPredictor.predict(
  update_calibration_scores=False)` — reuse the static leave-one-out
  calibration scores instead of recomputing them on each augmented bag
  {cal u (x_test, y)}. That is the ONLY difference from FCP. There is
  NO train/calibration split and NO trained classifier in that path.
- It is therefore a **static-calibration REDUCTION of FCP**, not split
  CP. The reduction is asymmetric (cal points scored against
  cal-minus-self, n-1 references; the test candidate against full cal,
  n references; the candidate never enters the cal points' reference
  sets) — so the exact finite-sample guarantee is LOST. Empirically the
  sets are indistinguishable from FCP (within 0.02 across 3 datasets,
  every budget and pipeline layer) and coverage held in all recorded
  runs, but that validity is inherited, an O(1/n)-perturbation
  observation, not a theorem.
- **True split CP at matched TOTAL budget collapses** — already
  documented, no fresh run needed: SCP-THR (softmax head trained on
  part of the budget) gives set size = K at B <= 300, 67 at B = 400,
  26.7 at B = 600 on CIFAR-100; catastrophic until B = 1200 on CUB-200;
  same pattern on miniImageNet (memory
  `scp-geodesic-isolates-ncm-vs-fcp`, 2026-05-26 multi-dataset
  extension). The failure is structural: a 100-way head cannot train on
  ~150 examples, and the split itself burns the budget.

Consequences for the paper:

- The valid-vs-valid comparison is **FCP vs true split CP**, and FCP
  wins it outright at this budget. FCP is load-bearing for the
  exactness claim — keep it as a pillar (sec 1, pillar 2).
- The static reduction survives as a RUNTIME footnote: same sets,
  2.5-3x cheaper, guarantee forfeited. Frame it as "a practical
  approximation of FCP", never as "SCP".
- Follow-up (naming debt): re-word THEORY.MD claim 2 and
  `docs/theory.md` sec 5.1 / `docs/findings.md` sec 9 — replace
  "SCP-geodesic" language with "static-calibration FCP reduction"; the
  efficiency attribution (the win is the no-train-split NCM +
  representation, not the transductive refit) stands unchanged.
- Optional cheap hardening for the paper table: re-run the SCP-THR
  collapse row on the CURRENT champion pipeline / asym NCM and at the
  smallest cal cells, so the valid-vs-valid table is fresh rather than
  2026-05 numbers.

## 4. Gap list (statuses per user review 2026-08-06)

- **G1 — stale novelty verification (open).** `literature.md` sec 9's
  "empty slot" finding predates qe / the discriminant / subsumption
  (verified 2026-07-06). Spot-check 2026-08-06 found no collision, but
  a full re-sweep keyed to the NEW claims (feature
  denoising/smoothing x CP, hubness correction x CP, pool-fit metric x
  CP) is required right before write-up. Add citations: Embedding
  Propagation (ECCV 2020), random-walk CP segmentation (2601.18997),
  fair conformal via representations (2605.12195).
- **G2 — RESOLVED by reframing (sec 3).** Not a novelty gap; remaining
  work is naming debt in THEORY.MD/theory/findings + the optional
  fresh SCP-THR row.
- **G3 — CLOSED WITH A BOUNDARY (2026-08-09; runs on
  `worktree-g3-semisup`, verdicts in `g3_semisup_baseline_plan.md`
  EXECUTIVE VERDICT + `output/pool_repr_menu/g3_semisup/`).** Both
  variants ran: arm A (self-training probe + split CP, oracle over
  ratios/rounds/lam, scaled probe) collapses structurally at
  cal <= 400 on all three datasets but CROSSES OVER on aircraft at
  cal 800 (15.85 vs champion 19.81, valid cov) — the paper's scope
  sentence becomes budget-indexed (cal/K <= ~4 shots exclusive; ~6+
  shots on saturating geometry = contested). Arm B (pool-only
  DeepCluster MLP, matched d'/C) LOSES everywhere run and is
  actively harmful on aircraft (worse than raw embeddings — neural
  collapse onto wrong-class pseudo-clusters at hom .25): the depth
  axis inherits and AMPLIFIES the homophily regime map, so the
  closed-form discriminant stands as the right depth point. Residual
  (stopped for the meeting, not verdict-relevant): CUB arm-B cells,
  cifar 3-seed MLP hardening, CUB lam=0 top-up. Original proposal
  kept below for the record:
  - *Variant A, label-dependent (self-training / pseudo-labeling with
    cal labels):* fit a probe on cal, pseudo-label the pool, retrain,
    conformalize. Under split CP this re-inherits the budget split that
    kills SCP-THR; training on all of cal and calibrating on the same
    cal is invalid; under FCP a cal-fit head is the exact
    exchangeability violation we already rejected for pseudo-label
    heads (PCA nightrun 2026-05-13). Expected to lose — but reviewers
    want to SEE it lose: one representative arm (pseudo-label linear
    probe + split CP, cal 200-800) suffices.
  - *Variant B, label-free pool-only (the fair in-slot competitor):*
    strengthen the representation with the pool alone — e.g. a small
    MLP head trained on pool k-means pseudo-labels with C >= K
    (DeepCluster-style), or SSL continue-pretraining on pool images.
    Any such map is a fixed function of the pool, so Prop 2 applies
    and exact validity is retained: this is a LEGITIMATE stronger
    competitor to the closed-form T(x) in the same architectural slot.
    The reviewer question becomes "is shallow closed-form the right
    point on the depth axis?" Proposal: ONE arm — 2-layer MLP on pool
    k-means pseudo-labels (C >= K), same NCMs on top. If it loses or
    ties: the closed-form story gains a "we tested the deeper end"
    shield (alongside the AE-encoder and yj/LPP negatives already in
    hand). If it wins: we found a better stage 3. Either outcome pays.
    Deployment argument regardless: closed-form has 2 knobs, no SGD,
    deterministic, 4GB-GPU feasible.
- **G4 — cars/aircraft regime counterexample (under active testing —
  intentionally NOT addressed here).** The within-whitening family
  loses on cars; aircraft shows ~25% headroom for prototype+pca512_cw.
  Resolution (label-free regime flag vs honest scoping) pending the
  current experiments; revisit before the headline table freezes.
- **G5 — knob provenance / benchmark overfitting (open).** d' =
  192/512/512/128 was chosen per-cell after many rounds on the same
  datasets. State the PR-based d' rule a priori and validate on data
  not used to derive it (cifar10 spot-check + 50-trial cluster
  hardening; several headline cells are ~1 SE claims).
- **G6 — conditional-coverage fine print (open).** The champion costs
  +0.5-1.2 pp CovGap on aircraft/mini/CUB@1600 and the aircraft
  prototype win costs ~2 pp. Marginal exactness is the theorem; report
  the conditional cost prominently; check whether safe-qe modulates it.
- **G7 — no efficiency theory (open).** Validity is exact by
  construction; set-size gains are entirely empirical. Even a stylized
  Gaussian-mixture lemma (when denoise/whiten/discriminate shrinks the
  expected set size) would materially strengthen the paper. Keep the
  frame: "validity assumption-free, efficiency assumption-dependent"
  (`literature.md` sec 9).
- **G8 — pool robustness + backbone breadth (open, cheap wins).**
  Validity holds for an ARBITRARY pool (only cal/test exchangeability
  matters): a pool-contamination/shift experiment (coverage flat,
  efficiency degrades gracefully) turns an assumption into a
  robustness figure. Headline is DINOv2-base-only; one CLIP or
  DINOv2-large replication guards "backbone artifact".

## 5. Write-up checklist

- Novelty sentence to defend: "unlabeled data belongs in the
  representation, not the score — exact coverage is free there
  (pool-measurability), and it empirically subsumes the score-level
  alternatives." Present qe/whitening/discriminant as PORTED WITH AN
  AUDIT, cited aggressively; the risk is "assembled known parts", the
  antidote is the mechanism results, not the parts.
- FCP pillar, corrected framing: FCP = the only exact-guarantee
  machinery that works at this budget (true split CP collapses); the
  static reduction is a runtime footnote, not a rival method.
- Cite and disarm by name: Feature CP, Conf-OT/SCA-T/FCA, SemiCP,
  SNAPS, CONFIDE, Embedding Propagation, TAFSSL, 2601.18997.
- Re-run the sec-9 novelty sweep keyed to the new claims immediately
  before submission (G1).
- Pre-register the d' rule + hom_hat panel; harden headline cells to
  50 trials; keep the random-split arm in every validity table
  (balanced splits over-cover ~+1-3 pp).
- Deployment story: unlike score-level correction (needs a gate for
  SAFETY), safe-qe needs the gate only for TUNING — no estimated
  quantity can break validity or the not-worse-than-baseline property.
- Keep the runtime figure (fixed-map vs per-recalibration machinery).

## Reference IDs (this doc)

SemiCP 2505.21147 · SNAPS NeurIPS 2024 · Mazuelas 2510.07185 ·
Correia & Louizos 2507.10425 · Flechsig 2509.10321 · Feature CP
2210.00173 · Learned Features 2404.17487 · CONFIDE 2604.08885 · FCA
2506.06076 · Conf-OT 2505.24693 · SCA-T 2506.17503 · TAFSSL 2003.06670
· SimpleShot 1911.04623 · PT-MAP 2006.03806 · Distribution Calibration
ICLR21 2101.06395 · Embedding Propagation ECCV20 (Rodriguez et al.) ·
random-walk CP segmentation 2601.18997 · fair conformal representations
2605.12195 · alpha-QE Radenovic TPAMI19 1711.02512.

*Created 2026-08-06 (positioning session; gap statuses per user review
of the same date).*
