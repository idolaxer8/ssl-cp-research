# G3 plan: the semi-supervised baseline arms (label-dependent vs pool-only)

**What this document is.** The experiment plan for closing gap G3 of
`positioning_pool_repr.md` sec 4: answer the reviewer question "with
200-800 labels and a 3-10k unlabeled pool, why not semi-supervised
learning + CP?" with two arms that bracket the design space. Written
2026-08-06; decision rules pre-registered in sec 6 BEFORE any run.

## 0. What is at stake

The paper's claim is that the pool is best spent on a CLOSED-FORM
representation (T(x) = E^T W0 (D(x) - mu)) feeding FCP. Two rival ways
to spend the same resources are untested:

- **Arm A (label-dependent):** the standard ML answer — self-training a
  classifier with cal labels + pool pseudo-labels, then split CP.
  Expected to lose structurally (budget split + shot starvation); the
  point is to SHOW it, not assume it.
- **Arm B (label-free, pool-only):** a LEARNED representation fit on the
  pool alone — same architectural slot as T(x), same exact-validity
  status (Prop 2, `docs/theory.md` sec 2), deeper on the depth axis.
  This is the scientifically live arm: it tests whether the closed-form
  pipeline is the right point on {closed-form linear ... learned
  nonlinear} rather than merely the one we built.

The two arms differ in exactly ONE resource decision each, so each
outcome is attributable:

```
                 uses cal labels in score fn?   validity
Arm A  ST-probe          yes                    valid split CP (budget split)
Arm B  pool-MLP          no                     exact FCP (Prop 2)
ours   T(x)              no                     exact FCP (Prop 2)
```

## 1. Common experimental frame

- **Datasets / cells:** cifar100 (cal 200/400/800), cub200
  (400/800/1600), aircraft (200/400/800). miniImageNet only as a final
  spot-check (saturated cells carry little information). Embeddings =
  the matched-518 finals in `output/from_cluster/embeddings/` (cub =
  `output/embeddings_cub200_all.pt` carved as in the menu runs).
- **Splits:** balanced cal + balanced test (default policy), 10 trials,
  alpha = 0.1, test size as in the menu runs. One random-split arm per
  dataset for the exact-validity report (policy: always shipped
  alongside).
- **Comparison targets per cell:** (i) raw-embedding FCP baseline,
  (ii) the adopted champion (plan doc sec 5 table), (iii) the SCP-THR
  collapse row refreshed on the same trials (this also discharges the
  "optional cheap hardening" of positioning sec 3).
- **Output:** `output/pool_repr_menu/g3_semisup/` (JSON per arm +
  figures, sec 8).

## 2. Arm A — self-training probe + split CP (ST-probe)

The strongest SIMPLE instantiation of "semi-supervised learning + CP"
on frozen features. Deliberately standard; every borrowed piece cited.

### 2.1 Stages and math

Notation: z = x / ||x|| (L2-normed DINOv2 embedding, 768-d), labeled
budget B split into train set Tr and calibration set Ca, pool U
(unlabeled, size N_u), K classes.

**(A1) Linear probe on Tr.** Multinomial logistic regression:

```
min_W  (1/|Tr|) sum_{(z,y) in Tr} CE( softmax(W^T z), y )  +  lam ||W||_F^2
```

- Probe = linear, following the standard frozen-feature evaluation
  protocol (SimCLR linear eval, Chen et al. ICML 2020); an MLP head on
  1-4 shots/class only worsens overfitting, so linear is arm A's best
  shot, not a handicap.
- lam fixed at 1e-2 (sklearn `LogisticRegression`, L2, lbfgs).
  Justification: cross-validation is impossible at 1-2 shots/class
  (nothing to hold out), so lam must be a constant; 1e-2 is the sklearn
  default scale and matched what SCP-THR used historically. lam
  sensitivity {1e-3, 1e-2, 1e-1} on ONE cell (cifar cal-400) to show
  the conclusion is not a regularization artifact.

**(A2) Self-training rounds (r = 1..R, R = 2).** Pseudo-label the pool
with the current probe; select a class-BALANCED top-q fraction per
class by confidence:

```
for each class c:  S_c^(r) = top q_r fraction of { u in U : yhat(u) = c },
                   ranked by p_c(u) = softmax_c(W^T u)
retrain the probe on  Tr  u  { (u, yhat(u)) : u in S^(r) },
pseudo-labels weighted 1.0
```

- Balanced per-class selection instead of a global confidence threshold
  (FixMatch's tau = 0.95, Sohn et al. NeurIPS 2020): at 1-2 shots/class
  the probe's confidences are uncalibrated and a global threshold
  collapses onto a few easy classes. Percentile-based curriculum
  selection follows Curriculum Labeling (Cascante-Bonilla et al., AAAI
  2021); schedule q_1 = 0.2, q_2 = 0.4 (their two-step ramp, no tuning).
- R = 2 rounds: standard self-training saturates in 1-3 rounds on
  frozen features; R is NOT tuned — reported at r = 0 (no ST, = the
  plain SCP baseline), r = 1, r = 2, so the reader sees the whole
  trajectory.
- Lineage: pseudo-labeling (Lee, ICML-WS 2013), self-training
  (Yarowsky 1995; Noisy Student, Xie et al. CVPR 2020 — sans noise,
  which needs augmentations we don't have in embedding space).

**(A3) Split CP on Ca.** THR/LAC score s(z, y) = 1 - p_y(z) (Sadinle et
al. 2019), q_hat = the ceil((n+1)(1-alpha))/n empirical quantile of cal
scores, sets {y : s(z, y) <= q_hat}. THR only: our SemiCP study showed
APS/RAPS bloat (sz 17-100) in this regime and THR is the only
competitive split score.

### 2.2 The budget-allocation sweep (arm A's fairness guarantee)

The known failure mode is the train/cal split itself. To preempt "you
chose a bad split" we sweep the allocation and score arm A by its
PER-CELL ORACLE best:

```
(|Tr|, |Ca|) in { (0.25 B, 0.75 B), (0.5 B, 0.5 B), (0.75 B, 0.25 B) }
```

reported per ratio AND as best-over-ratios (an oracle arm A is allowed;
if it loses even with the oracle, the loss is structural). 50/50 is the
tutorial default (Angelopoulos & Bates 2107.07511); the other two probe
both directions of the trade-off.

Excluded by design: "train on all of cal, calibrate on the same cal"
(invalid — no guarantee, nothing to compare); FixMatch-style
augmentation consistency (needs image-space augmentations + backbone
passes; noted as out of scope, sec 9).

### 2.3 Predictions (falsifiable)

- cal <= 400: degenerate or near-degenerate sets at every ratio (train
  half has 1-3 shots/class for K >= 100) — the SCP-THR collapse with a
  self-training band-aid.
- cal = 800+: self-training improves over plain SCP-THR (2.58 on cifar
  in the 2026-05 runs) but stays behind the champion (1.27). If ST
  closes to within 2 SE of the champion at 800/1600, that is a real
  finding and gets reported as the budget where the standard answer
  catches up.

## 3. Arm B — pool-only learned head (pool-MLP)

### 3.1 Construction

Label-free, DeepCluster-style (Caron et al., ECCV 2018): pseudo-labels
from clustering, a small head trained to predict them, penultimate
layer as the new representation.

**(B1) Pseudo-labels.** k-means on the L2-normed pool, C clusters with
the SAME C as the adopted pipeline (C = max(100, 1.5 K): 100 cifar /
300 CUB / 150 aircraft), same seed protocol. Justification: C >= K is
the pipeline's structural rule (fewer clusters merge classes and
poison the supervision — measured in the ldapool round); reusing C
removes one confound between B and T(x).

**(B2) Head + objective.** g = f2 o f1 with

```
f1: Linear(768 -> 1024) + BatchNorm + GELU
f2: Linear(1024 -> d')                      d' = 192 cifar / 512 cub /
                                                 512 aircraft  (matched!)
cls: Linear(d' -> C)  (discarded after training)

min  (1/N_u) sum_u CE( softmax(cls(g(z_u))), c(u) )  + label smoothing 0.1
```

- d' MATCHED to the adopted pipeline's per-dataset knob. This is the
  controlled-comparison requirement: representation dimension is a
  known regime knob, so it must be equal or the comparison measures
  dimension, not learnedness.
- Width 1024 (~1.3x input): the smallest standard 2-layer probe scale;
  capacity is deliberately modest — the hypothesis under test is
  "learned nonlinear beats closed-form linear", not "bigger is better".
- Label smoothing 0.1 because the targets are noisy pseudo-labels
  (standard; also stabilizes neural-collapse-onto-clusters, sec 3.3).
- Optimizer: AdamW, lr 1e-3, cosine to 0, weight decay 1e-4, batch 512,
  100 epochs. All constants standard for small MLPs on frozen
  features; convergence checked once via train-accuracy curves, not
  tuned per dataset.
- Output representation: h(z) = g(z) / ||g(z)|| (L2-norm, as the
  angular NCMs require).

**(B3) CP on top.** Identical to the pipeline's evaluation: FCP (GPU
path), NCMs = prototype_softmax + geodesic mean (+ asym on the final
grid), same trials/splits as sec 1.

**Input variants (2, pre-registered, no more):**
- B-raw: h(z) on raw embeddings — the learned REPLACEMENT of stages
  2-3.
- B-qe: h(D(z)) on qe-denoised inputs (classic small-cal / safe
  high-cal, the sec-6 rule of the plan doc) — tests whether stage 1
  composes with a learned back end the way it composes with the
  closed-form one.

### 3.2 Fitting protocol and variance accounting

The pool is FIXED per dataset (dedicated unlabeled files), so g is fit
ONCE per dataset per seed — the transform is a constant given the pool,
exactly like the pipeline's k-means/whitening; trial variance comes
from cal/test resampling only. Fit 3 head seeds (init + k-means seed)
and report across-seed SD separately from across-trial SE, so "learned
head instability" is measurable and not laundered into trial noise.

### 3.3 Predictions and known risks

- Neural collapse onto pseudo-clusters: the head compresses
  within-cluster variation — HELPFUL where clusters track classes
  (high homophily: cifar/mini), HARMFUL where clusters merge or split
  classes (aircraft, hom ~ .25). Prediction: B's regime map mirrors
  qe's homophily fingerprint. If so, that is itself a result: the
  depth axis does not escape the regime map, it inherits it.
- BN statistics are pool statistics — still Prop-2-admissible (fixed
  after training; inference in eval mode).
- If B-raw ties the closed-form pipeline on separable data but loses
  on fine-grained, the paper sentence is: "a learned pool-only head
  buys nothing over the closed-form discriminant at matched dimension"
  — the shield. If B wins >2 SE anywhere, sec 6 triggers round 2.

## 4. Validity checks (both arms)

- Arm A: valid by split-CP construction (probe fit on Tr only; Ca
  scores iid given the frozen score fn). Report coverage per cell —
  expected ~0.90 with degenerate sizes at small cal.
- Arm B: exact by Prop 2. Run the existing permutation symmetry oracle
  once on a small cell (protocol of the menu rounds, tolerance ~1e-14)
  to certify the implementation, plus the random-split coverage arm
  (expect the 0.898-0.902 band).

## 5. Implementation

- New driver `src/g3_semisup_experiment.py`, both arms behind
  `--arm {selftrain,poolmlp}`.
- Arm A reuses `SoftmaxSplitCP`/`compute_cp_scores` (THR) from
  `split_cp_baselines.py` + the balanced-split utilities; the
  self-training loop is ~60 lines around sklearn.
- Arm B: add `PoolMLPTransform` to `exchangeable_features.py`
  implementing the `UnlabeledTransform` interface (fit(pool) /
  transform(X)), torch MLP, saved per (dataset, seed) under the output
  dir so FCP runs never retrain.
- Runtime: MLP fit ~2-3 min/dataset/seed on the local 4GB GPU
  (precomputed 768-d inputs); FCP cells on the GPU path minutes each;
  arm A is CPU-trivial. Full grid comfortably < 1 day local; the
  50-trial hardening pass rides the existing cluster protocol if B is
  interesting.

## 6. Pre-registered decision rules

- **Arm A verdict = "structural collapse confirmed"** if
  best-over-ratios, best-over-rounds ST-probe is > 2x champion size at
  every cal <= 400 cell on all three datasets. If it beats 2x anywhere
  at cal <= 400, or comes within 2 SE of champion at 800/1600, report
  that cell as the budget where the standard answer revives — no
  spinning.
- **Arm B verdict per dataset:** win / tie / loss vs the adopted
  champion by the 2-SE rule per cell; dataset-level win = >2 SE in
  >= 2 cells. Any dataset-level win triggers round 2 (composition with
  safe-qe, regime map, cars); tie-or-loss everywhere closes G3 with
  the shield paragraph in `positioning_pool_repr.md` and one
  depth-axis figure in the paper.
- No knob of either arm is re-tuned after seeing CP set sizes (the
  conformal-metric lesson: surrogate tuning is fine, CP-size tuning on
  the eval cells is how benchmark overfitting starts). The only
  post-hoc freedom is WHICH cells to headline, and sec 8's figure
  shows all of them.

## 7. Execution order

1. Implement both arms + oracle check (arm B) on cifar100.
2. cifar100 pilot (all cells, 10 trials, 1 head seed) — sanity +
   runtime calibration.
3. Full grid: 3 datasets x cells x {A: 3 ratios x 3 rounds;
   B: 2 variants x 2 NCMs} x 10 trials; 3 head seeds for B.
4. Verdicts per sec 6; fold results into `positioning_pool_repr.md`
   sec 4 (G3) and the plan doc; mini spot-check only if B wins
   anywhere.

## 8. Deliverables

- `output/pool_repr_menu/g3_semisup/{arm_a,arm_b}_<ds>.json`
- **Depth-axis figure** (one per dataset): set size vs cal for raw /
  SCP-THR refresh / arm A (best ratio, r = 0..2 as light-to-dark) /
  arm B-raw / arm B-qe / closed-form champion. This figure is the
  paper artifact: the whole "why not semi-supervised?" answer in one
  panel.
- Coverage table (balanced + random arms) for both arms.
- Verdict paragraph appended to `positioning_pool_repr.md` G3.

## 9. Out of scope (recorded, not run)

- SSL continue-pretraining of the backbone on pool IMAGES (SimCLR/DINO
  on 3-10k images): the far end of the depth axis. Needs image-space
  augmentation + cluster GPU time; deferred unless arm B wins broadly
  (then it becomes round 3 on the Run:AI pod).
- FixMatch-style consistency training (same reason: image-space
  augmentations).
- Iterated DeepCluster (re-cluster in learned space, retrain): round-2
  option if one-shot B wins; adds a feedback loop that needs its own
  convergence checks, still Prop-2-safe (pool-only).

## 10. Citations

Chen et al. ICML 2020 (SimCLR linear-eval protocol) · Lee ICML-WS 2013
(pseudo-labeling) · Xie et al. CVPR 2020 (Noisy Student) · Sohn et al.
NeurIPS 2020 (FixMatch; global-threshold contrast) · Cascante-Bonilla
et al. AAAI 2021 (Curriculum Labeling; percentile schedule) · Sadinle
et al. 2019 (THR/LAC) · Angelopoulos & Bates 2107.07511 (split default)
· Caron et al. ECCV 2018 (DeepCluster) · Asano et al. ICLR 2020 (SeLa,
balanced-assignment alternative if k-means degenerates) · Zhou et al.
2505.21147 (SemiCP — the existing score-level semi-supervised
baseline) · `docs/theory.md` sec 2 Prop 2 (validity of pool-fit maps).

*Created 2026-08-06. Pre-registration: secs 2-3 constants and sec 6
rules are frozen before the first run; any deviation gets logged here
with a date.*

**Deviation log.**
- 2026-08-06 (after the cifar100 pilot): the probe spec gained a
  StandardScaler (fit on the probe's train data) in front of the
  logistic regression, and lam <= 0 now selects the historical sklearn
  default C = 1.0. Reason: without scaling, L2-normed inputs (~0.03 per
  dim) cap the ridge-penalized logits at a near-uniform softmax over
  K >= 100 classes — the pilot's arm-A collapse (sz 85-100 even at
  cal 800, vs the historical SCP-THR ~2.6-3.2 at B = 800, which DID
  scale in `SoftmaxSplitCP.fit`) was partly this artifact, not only
  the budget split. Arm A is re-run scaled and reported at its best
  lam (the sec-2.2 fairness requirement extended to the probe spec).
  The unscaled pilot rows are kept in `results_cifar100_pilot.json`
  for the record.
