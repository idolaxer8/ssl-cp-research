# ICLR 2027 paper — framework outline v1 (structure-first, modular)

Status: v1 (2026-08-18). Purpose: DEMONSTRATION of the paper's shape — what
is solid, what needs runs, where to focus. Every block below is
self-contained and swappable (no cross-block dependencies in the writing);
each carries a status tag:

- `[SOLID]` — backed by a paper-grade asset that exists on disk today
- `[NEEDS-RUN]` — evidence exists but below paper grade; blocked on an R-run
- `[OPEN-DECISION]` — blocked on a user/instructor decision, not on compute

Companions: `docs/iclr2027_plan.md` (claims C1-C6, calendar, risks),
`docs/findings.md`, `THEORY.MD`. Math in this doc: plain ASCII (planning
doc); the paper itself is LaTeX.

---

## Block 0 — Title `[OPEN-DECISION, low priority]`

Rule (user, 08-18): do NOT lead with "valid" — validity is table stakes for
CP. Placeholder + candidate directions (decide at wk 08-31 draft):

- *"Tight conformal sets from frozen SSL embeddings at few labels per
  class"* (placeholder; label-budget angle)
- Lead with the frozen-foundation-model angle: *"Conformal prediction on
  frozen vision foundation models: pool-fit transforms for label-starved
  classification"*
- Lead with the unlabeled-data angle: *"Unlabeled data buys tighter
  conformal sets: exchangeable pool-fit transforms for SSL embeddings"*

---

## Block 1 — §1 Introduction (~1 page)

### 1a. Opening motivation `[SOLID — writing only]` (user note 08-18)

Draft of the opening paragraph (to be polished, content fixed):

> Frozen vision SSL / foundation models (DINOv2, CLIP, ...) are now the
> default feature extractors across applied computer vision: practitioners
> take the embedding, attach a light head or a nearest-prototype rule, and
> ship. These embeddings carry strong class geometry that cost zero labels
> — but they carry NO uncertainty: a cosine similarity is not a confidence,
> and the few-label heads trained on top are miscalibrated exactly when
> labels are scarce. What is needed is distribution-free uncertainty
> quantification ON TOP of the frozen model — prediction SETS with a
> finite-sample coverage guarantee — without training, without a held-out
> split large enough to train a K-way head, and ideally exploiting the one
> resource that is free in this ecosystem: unlabeled images. Conformal
> prediction provides the guarantee; this paper shows how to make it TIGHT
> in the regime where foundation models are actually used — many classes,
> few labels per class.

### 1b. The regime, defended (high-K framing) `[SOLID — writing only]`

Draft of the framing defense (user asked this be defensible; cifar10 demoted
by it):

> The operating regime is budget-indexed: what matters is labels per class,
> cal/K, not the absolute label count. Split CP must first train a K-way
> classifier on part of the budget; its collapse is therefore governed by
> cal/K, and with K >= 100 classes even 800 labels is only 8 per class —
> label starvation is not a corner case but the realistic operating point
> of a K>=100 problem. We measure the boundary directly: below ~4-6
> labels/class every trained-head method is invalid or trivial (sets of
> 2-19x our size, or size = K), and trained probes only catch up around ~6
> labels/class. Datasets with small K (e.g. CIFAR-10: 20-80 labels/class at
> the same budgets) sit outside the regime and appear only as a saturated
> sanity control (appendix).

Evidence anchor: g3-semisup verdict (boundary 4-6 shots/class; SemiCP =
strongest baseline). Consequence: MAIN datasets = cifar100 / miniImageNet /
aircraft / stanford_cars (K = 100/100/100/196, all matched-518 embeddings);
CUB-200 appendix (local-336, footnoted); cifar10 appendix control.

### 1c. Contribution bullets `[SOLID — writing only]`

1. **Regime result (C1/C2).** At <=8 labels/class with K>=100, Full CP over
   frozen SSL embeddings with training-free NCMs is the only approach that
   is simultaneously valid and tight; trained-head split CP, SemiCP, and
   CV+ collapse (boundary measured at ~4-6 labels/class). Graceful
   degradation on fine-grained data (coverage holds, sets grow) — the
   scope limit stated plainly.
2. **Method (C3).** A family of pool-fit embedding transforms — Denoise
   (pool-kNN alpha-QE smoothing), Whiten (cluster Ledoit-Wolf), Truncate
   (PCA) — every stage fit on unlabeled data only, hence exchangeable BY
   CONSTRUCTION: the exact coverage guarantee is untouched and set sizes
   drop substantially (bind the exact % to 50-trial Table 2 cells at draft
   time — do NOT quote the 10-trial numbers in contributions). The
   practitioner-facing regime dial is the pool participation ratio —
   LABEL-FREE by construction; pool homophily and the d'-ratio are
   presented as measured (labeled) DIAGNOSTICS that explain the mechanism,
   not as shipped dials (ars-outline catch 08-18: h_w/d' need labels; the
   label-free surrogate is future work). Per-regime choice fixed a priori;
   no automatic selection gate (C4-descoped).
3. **Generality (C5).** The pipeline transfers across SSL backbones
   (dinov2 / dinov3 / clip-B / clip-L): valid + tight in all 8 tested
   backbone x dataset cells; PR predicts the transform decision in 7/8
   (state the cell count explicitly — do not imply dataset breadth beyond
   cifar100 + aircraft unless R4 runs).
4. **Theory (C6, supporting).** Validity-is-free proposition (exact:
   pool-measurable transforms preserve exchangeability — a wrong transform
   costs efficiency, never coverage) + an exact, assumption-free per-point
   denoise lemma (D1; DAPS-Theorem-2 analogue in representation space)
   whose certification rate tracks the empirical transform ranking. W/T
   phases discussed abstractly; full treatment appendix.

### 1d. Figure 1 `[NEEDS-RUN R1]`

Two-panel: (left) pipeline schematic — frozen backbone -> unlabeled pool
fits D/W/T -> NCM -> FCP -> sets, with "exchangeable by construction"
annotated on the pool arrows; (right) headline size-vs-cal curve, cifar100
+ aircraft, champion vs best baseline (from Table 2 / R1 data).

---

## Block 2 — §2 Related work (~0.75 page) `[SOLID — writing only]`

One paragraph each; families and OUR positioning (baseline / foundation /
contrast). All entries already annotated in `docs/literature.md`:

- **CP families** (split / full / CV+; Vovk; Fan-Sesia O(1/n)): foundation.
  Position: FCP is load-bearing for exactness at our budgets; the
  static-cal reduction footnote kills the "transductive machinery wins"
  misread.
- **Semi/unsupervised-data CP** (SemiCP/Zhou NNM; Ding ClusterCP): our
  strongest baselines; budget-indexed comparison is ours.
- **Graph-CP score smoothing** (DAPS, SNAPS): closest mechanism line —
  they smooth SCORES on a given graph; we denoise the REPRESENTATION with
  a pool-kNN graph, subsume the score-level correction (measured), and
  inherit/extend their theory template (D1).
- **Prototype/FCA line** (Silva-Rodriguez FCA, CAOS): the prototype NCM
  family; CAOS App-E = external support for no-train-split. Marazov cosine
  NN-ratio = external validation of ratio NCMs.
- **Few-shot metric estimation** (Simple CNAPS, Transductive CNAPS): the W
  stage is the few-shot field's Mahalanobis best practice, factored into
  an exchangeable preprocessing transform — our novelty is the validity
  discipline, not the metric.
- **SSL embedding spectra** (RankMe, dimensional-collapse): label-free
  spectral statistics predict downstream performance — the published
  precedent for our PR dial.

Space valve (ars-outline): keep the six-family structure at ~2 sentences
each; the detailed baseline-family comparison (SemiCP / ClusterCP /
APS-RAPS: guarantee type, data use, budget behavior) moves to an APPENDIX
comparison table so main text stays compressed without looking unaware.

---

## Block 3 — §3 Setting and method (~2 pages)

### 3a. FCP + guarantee `[SOLID — writing only]`

Full (transductive) CP on frozen embeddings: bag augmentation, p-value
rank, exact 1-alpha coverage; missing-class symmetric handling (the
validity bug class we fixed — one sentence); GPU fast path makes FCP
practical (prototype ~1s/trial); footnote: the static-cal reduction
(`update_calibration_scores=False`) reproduces FCP set-for-set (O(1/n)) —
a runtime option, NOT split CP (C1 naming, already fixed in THEORY.MD).
One sentence on the balanced-cal caveat: class-balanced calibration
conditions on the label-count vector (within-class exchangeability); the
random-split validity arm in every table is the empirical answer
(preempts the reviewer probe; theory.md §6 Q3).

### 3b. NCM design `[SOLID]`

Three NCMs, roles distinct:
- `prototype_softmax` (champion on separable data): class-mean cosine
  logits -> softmax LAC; exact closed-form LOO refit.
- `prototype_cosine` (theory-faithful): plain -cos(mu_c, x); NO cal-fit
  term at all; the score D1 covers verbatim. (Definitional here ONLY — the
  measured softmax gap is stated once, in §5.2; do not preview the result
  in §3, per the ars-outline duplication catch.)
- `geodesic_topk_asym` (honest on fine-grained): 1-NN / mean-k ratio; best
  class-conditional CovGap where sets are large.

### 3c. The pool-fit transform family + dials `[SOLID — writing only]`

D/W/T stages with per-regime fixed configuration:
- separable regime (high PR ~240): qe ON, PCA-128 + fine-k cluster whiten;
- fine-grained regime (low PR ~16-24): qe OFF, full-rank LW-cluster
  whiten, coarse k, NO truncation;
Dial wording (fixed by the ars-outline catch): **PR is THE practitioner
dial — label-free, from the pool spectrum alone**; pool homophily h_w and
the d'-ratio are labeled diagnostics used to EXPLAIN the regime, reported
in the analysis, never required at deployment. The recipe box: measure PR
-> pick the regime config; no automatic selector (future work).

**Table 1 -> DEMOTED to Fig 1 annotations** (ars-outline space catch): the
stage x fits-on x exchangeability content becomes labels on the pipeline
schematic; no standalone table in main text.

---

## Block 4 — §4 Theory (~0.75 page) `[SOLID for stated scope]`

Scope per user decision (08-18): FCP validity + D1 ONLY in main; W/T
abstract note; everything else appendix. Theory is NOT the heart.

### 4a. Prop 1 — validity is free

Every transform is a fixed measurable function of the pool; conditional on
the pool, exchangeability of {(T(x_i), y_i)} is preserved; FCP wrapping
stays exact. One-line proof sketch + the sentence that carries the paper's
safety story: "a wrong transform choice costs efficiency, never coverage."

### 4b. Lemma D1 — exact per-point denoise condition (proof in appendix)

Statement (for prototype_cosine / representation error):
||x_hat - mu_y|| < ||x - mu_y|| whenever
sum_u (w_u/W) r_u + (1 - h_w(x)) * Delta_F(x) < eps(x).
Framing: DAPS Theorem 2 transplanted from probability space to
representation space, graph-radius assumption replaced by a MEASURED
impurity budget. Validation table lives HERE (§5.6 merged in, ars-outline
space catch): per-dataset D1 certification rate by transform + gate
constants — certified => improved held pointwise on 5/5 datasets; the
cert-rate argmax matches 4/5 champion transform cells. Label it as
VALIDATION of the lemma's empirical relevance, NOT a derived selection
rule (else it re-opens the auto-gate door C4 closed). Source:
`output/dwt_theory/*.json` + `wt_chainfree_diagnostics` (deterministic).

### 4c. W/T abstract note (one paragraph, no numbered claims)

The whiten and truncate stages admit exact per-pair statements (optimal
metric by Cauchy-Schwarz; label-free fit exact by Sherman-Morrison;
truncation alignment identity) and a services reading — whitening raises
the homophily D1 consumes (aircraft cert rate x5.5), truncation cuts
estimated-anchor error (m/s vs d/s) — but their quantitative composition
is open; we state them in the appendix and claim only the direction.
`[OPEN-DECISION: exact wording tracks W/T progress before 08-31]`

---

## Block 5 — §5 Experiments (~3.5 pages)

Common setup line: alpha=0.1, balanced stratified splits + random-split
validity arm, cal in {200, 400, 800}, matched-518 DINOv2-base embeddings,
50-trial cluster runs (post-R1/R2), coverage always reported next to size.

### 5.1 Headline: the regime table `[NEEDS-RUN R1+R2 — THE critical path]`

**Table 2**: size (coverage) @ cal 200/400/800 x 4 main datasets. Rows:
FCP+DWT champion / FCP raw (no transform) / SCP-THR / **SCP-APS / SCP-RAPS
(added per ars-outline — standard CP baselines reviewers expect;
`split_cp_baselines.py` already implements them, near-zero cost in R2)** /
SemiCP / linear probe + CP / CV+ (**confirm the R2 driver covers CV+ —
`CrossValidationPlusPredictor` exists in the core lib; add the arm if
`g3_semisup_experiment.py` lacks it**). Message: only the FCP rows are
valid AND non-trivial below ~6 labels/class; the transform rows quantify
what the pool buys.
**Fig 2**: size-vs-cal curves per dataset (same data).
Current assets: champion cells 10-trial (`output/pool_repr_menu/`),
baselines trial-grade unknown (`output/pool_repr_menu/g3_semisup/`) ->
R1 + R2 at 50 trials.

### 5.2 NCM ablation `[SOLID — assemble only]`

**Table 3**: prototype_softmax vs prototype_cosine vs geodesic asym/mean
across datasets+cal, **WITH a CovGap column** (ars-outline catch:
class-conditional metrics must appear in main text for a CP audience; the
`fca_family_cluster` JSONs already log CovGap — assembly only). Sources on
disk: `output/from_cluster/fca_family_cluster/` (50t, 4 ds) +
`output/d1_softmax_ablation/` (20t, 5 ds, paired). Two messages: (a)
prototype dominates separable, geodesic honest on fine-grained (CovGap);
(b) the softmax normalizer is load-bearing and regime-dependent — the
quantified theory-method gap.

### 5.3 Transform menu + regime map `[NEEDS-RUN R3]`

**Table 4**: transform arms (raw / jl control / pca128_cw / pca512_cw /
lw_cluster768 / +qe) x datasets — the three-regime map. **Fig 3**: the
label-free dials — PR per dataset vs winning transform + spectral-band
signal profile (the aircraft signal-in-tail picture). Current: 10-trial,
4 ds, cars missing -> R3 at 20-30 trials + cars. Diagnostics (deterministic,
exist): `spectral_band_diagnostic`, `embedding_geometry_diagnostic`.

### 5.4 Denoise lever: gate + subsumption `[SOLID]`

**Fig 4** (two panels): (a) qe gain/harm vs measured d'-ratio sign — sign
calls the verdict 5/5 (`output/cars_qe_gate/`, `measure_dprime_all`);
(b) qe subsumes the SNAPS score correction — post-qe residual SNAPS gain
−41% -> −6.6% (`output/snaps_pool/`, 20t). Deploy wording (ars-outline
catch): the shipped config keys on PR (label-free) — low-PR regime = qe
OFF; the homophily boundary h ~ 0.8 is reported as the measured labeled
DIAGNOSTIC that explains why, not as the deployment rule.

### 5.5 Backbone transfer `[SOLID for 2 ds; optional R4]`

**Table 5**: backbone (dinov2/dinov3/clip-B/clip-L) x {cifar100, aircraft}
x cal — size(cov) + dial values; message: pipeline transfers (all valid +
tight in the 8 tested cells), PR predicts the transform decision 7/8, raw
homophily gate does NOT transfer (honest negative). Source:
`output/from_cluster/backbone_dwt_v2/` (20t). MAE = degenerate PR=2
negative pole (appendix or one line). R4 (optional): add cars per
backbone; beitv2 in-or-drop.

(§5.6 removed — the theory-validation table merged into §4b per the
ars-outline space catch.)

---

## Block 6 — §6 Discussion & limitations (~0.5 page) `[SOLID — writing only]`

### 6a. Easy vs hard datasets — the performance discussion (user note 08-18)

A dedicated discussion contrasting the two dataset families the whole paper
straddles — "easy" (cifar100, miniImageNet: DINOv2 separates the classes;
high PR ~240, homophily .8-.92) vs "hard" (aircraft, stanford_cars:
fine-grained variants the backbone does not separate; PR ~16-24, homophily
.26-.46). Content points (all measured; R1 delivers the shots-matched
head-to-head):

- **Absolute performance splits by separability, validity does not.** Easy:
  sets reach ~1.2-3 of K=100 at 8 shots/class. Hard: sets stay ~13-75 of
  K (aircraft) / up to ~60-196 (cars) — yet coverage holds ~0.90
  everywhere. The honest one-liner: the method degrades gracefully on SIZE,
  never on coverage (Prop 1 is regime-independent).
- **Every pipeline choice inverts between the families** — the regime map
  IS this contrast: (a) transform: truncation (pca128_cw) wins easy /
  is a no-op-to-harmful hard, where full-rank lw_cluster768 wins (signal
  in the low-variance spectral tail, Chang-type); (b) denoise: qe gains
  easy / harms hard (homophily below the gate; harm not tunable);
  (c) NCM: prototype_softmax dominates easy, geodesic asym is the honest
  choice hard (best CovGap; prototype over-covers/bloats at small shots);
  (d) the softmax normalizer itself flips sign — load-bearing on easy
  (+28-67% without it), a LIABILITY on hard at starved shots
  (aircraft@2-shot 72.9 -> 30.2 without softmax). One label-free statistic
  (PR) sorts every one of these decisions.
- **Why hard is hard (mechanism, not hand-waving):** DINOv2 collapses
  fine-grained variants into a tiny dominant subspace (pose/livery
  nuisance carries 75% of variance at 1-NN 0.27 on aircraft); the
  discriminant is thinly spread across hundreds of low-variance
  directions, so any truncation loses it and neighborhoods are
  majority-wrong-class (homophily .26) — which simultaneously explains
  the qe harm, the truncation harm, and the large sets.
- **Shots-response differs**: on easy data the transform payoff is largest
  at the fewest shots (T1b: anchor-estimation noise ~ m/s) and saturates;
  on hard data more shots buy little (the bottleneck is separability, not
  estimation) — R1's 2-14 shots grid makes this a figure, not a claim.
- Forward pointer: what would fix hard? A finer backbone (clip-large was
  TIGHTEST on aircraft in the backbone table — evidence it is a backbone
  property, not a method ceiling), not more labels or more pipeline.

### 6b. Remaining limitations

- Scope limit: backbone separability — on fine-grained data coverage holds
  but sets stay large (graceful degradation, stated with numbers; detailed
  in 6a).
- Dials are descriptive; automatic transform selection is future work (the
  honest answer to "why no auto gate": diagnosis vs decision, and validity
  never depends on the choice).
- Regime boundary honesty: trained probes catch up at ~6 labels/class —
  our exclusivity claim is budget-indexed, not universal.
- Theory debt: quantitative W/T composition and E|C| statements open
  (documented obstruction; pointer to appendix ledger).

---

## Block 6c — Back matter `[SOLID — writing only]` (ars-outline catch)

Mandatory ICLR statements after the conclusion (excluded from page limit):
- **Reproducibility Statement**: seeds/splits/scripts pointer, the
  exchangeability-oracle tests, the compact code artifact (plan §3b
  publishable core), embedding extraction settings.
- **Ethics Statement**: standard-benchmark, no-human-subjects short form.

## Block 7 — Appendix map `[mostly SOLID]`

A. Proof of D1 + Prop 1. B. D2/D3 + the overprediction post-mortem (the
honest (I)-model story) + chain-free W/T package (W1a/W1b/T1a statements,
cert-rate/anchor-error tables). C. CUB-200 tables (336px footnote) +
cifar10 saturated control. D. Penalty story (MS-CS/MA-CS; R5 optional).
E. qe-upgrade pilots (hub-debias; one-hop optimality). F. Exchangeability
oracle tests + reproducibility detail (seeds, splits, scripts).
G. Backbone MAE negative pole + beitv2 (if run). H. Baseline-family
comparison table (SemiCP / ClusterCP / APS-RAPS — the §2 space valve).
I. Design-choice backing from existing paper-grade assets (ars-outline
orphan catch): balanced-split ablation (`output/split_ablation/`, 30t) and
pool-size plateau (`output/from_cluster/pool_ablation_hightrial/`, 100t).

---

## Asset manifest (post ars-outline pass)

| Asset | Section | Source | Status |
|---|---|---|---|
| Fig 1: pipeline schematic (w/ stage x exchangeability annotations, ex-Table 1) + headline curve | §1 | drawing + R1 | NEEDS R1 |
| Table 2 (HEADLINE): champion + raw vs SCP-THR/APS/RAPS, SemiCP, probe, CV+ | §5.1 | 10t / unknown | **NEEDS R1+R2** |
| Fig 2: size-vs-cal curves | §5.1 | same as Table 2 | NEEDS R1+R2 |
| Table 3: NCM ablation + CovGap column | §5.2 | `fca_family_cluster` 50t + `d1_softmax_ablation` 20t | SOLID — assemble |
| Table 4 + Fig 3: transform menu + PR/spectral regime map | §5.3 | `transform_controls` 10t, no cars | NEEDS R3 |
| Fig 4: d'-sign gate + qe-subsumes-SNAPS | §5.4 | `cars_qe_gate` 20t + `snaps_pool` 20t + deterministic d' | SOLID |
| Table 5: backbone x dataset | §5.5 | `backbone_dwt_v2` 20t | SOLID (2 ds); R4 optional |
| §4b validation table: D1 cert rates + gate constants | §4 (ex-5.6) | `output/dwt_theory/*` deterministic | SOLID |
| App H: baseline-family comparison | App | text-only | write |
| App I: split ablation + pool-size plateau | App | `split_ablation` 30t, `pool_ablation_hightrial` 100t | SOLID |

## Run list (paper-grade cluster runs; wk 08-24 freeze)

| Run | What | Grade now -> target | Feeds | Priority |
|---|---|---|---|---|
| R1 | champion arms {raw, per-regime W/T, +qe gated, champion} x 3 NCMs x 4 ds x cal 200/400/800 | 10t -> 50t | Table 2, Fig 1, Fig 2 | **CRITICAL** |
| R2 | baselines at matched budgets: SCP-THR, SCP-APS, SCP-RAPS, SemiCP, probe+CP, CV+ (verify driver covers CV+ + APS/RAPS; add arms if missing) | unknown -> 50t | Table 2 | **CRITICAL** |
| R3 | transform-control menu + ADD stanford_cars | 10t/4ds -> 20-30t/5ds | Table 4, Fig 3 | high |
| R4 | backbone x cars; beitv2 decision | — | Table 5 ext | optional |
| R5 | MS-CS/MA-CS penalty tables | 5-10t -> 20t | Appendix D | optional |
| R6 | cifar10 control top-up | mostly exists | Appendix C | cheap |

Drivers exist for all (R1: `exchangeable_fcp_experiment.py` /
`transform_control_experiment.py`; R2: `g3_semisup_experiment.py`; R3:
`transform_control_experiment.py`; R4: `backbone_dwt_experiment.py`; R5:
`mscs_vs_macs_experiment.py`). No new experiment code needed for the
paper's main text — only run-scale and one dataset addition (cars in R3).

## Focus read (what this outline says)

- **Solid today**: NCM story (5.2 incl. CovGap), denoise gate + subsumption
  (5.4), backbone transfer core (5.5), theory-as-scoped (§4 incl. its
  validation table), all framing text incl. the easy-vs-hard discussion
  (6a — every content point already measured; R1 sharpens it to
  shots-matched figures), appendix design-choice backing.
- **The critical path is R1+R2** — the headline table is the only main-text
  asset below paper grade. Everything else is assembly and writing.
- **One substantive wording rule (ars-outline)**: PR is the only LABEL-FREE
  dial; homophily h_w and d'-ratio are labeled diagnostics — contributions,
  §3c, and §5.4 are written accordingly; never claim a label-free homophily
  gate.
- **Open decisions**: title (low priority); W/T abstract-note wording
  (tracks progress to 08-31); beitv2/MAE presentation (R4).
