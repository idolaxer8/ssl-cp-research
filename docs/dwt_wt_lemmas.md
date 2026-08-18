# Lemmas W1 and T1 — Whiten and Truncate, continued from base lemma D1

Status: v0.1 (2026-08-16, weekly goal 3). Build principle (framing decision
08-16): **D1 is the base lemma of the Denoise phase** — deterministic,
assumption-free, validated pointwise 5/5 — and the W/T phases get lemmas of
the SAME construction: an exact, gap-free statement first (W1a, T1a — both
one-line proofs), then the statistically loaded refinements as named,
conditional clauses with signed violations and a measurement instrument
(`src/wt_phase_diagnostics.py`, mirroring `src/dwt_gate_constants.py`).
Anchor stack: `docs/classification_theory_anchors.md` (reading guide §4).
Currency throughout: the pair-margin discriminability d' of Theorem D3 —
so W/T ratios compose multiplicatively with the D-phase d'-ratio and feed
the unchanged downstream chain (Lemma B -> Prop C, `dwt_theory.md` §5-6).

Reference keys: [RCA05] Bar-Hillel et al. JMLR 2005; [F90] Fukunaga 1990
ch. 10; [LW04] Ledoit-Wolf JMVA 2004; [C83] Chang JRSS-C 1983; [P07] Paul
Statistica Sinica 2007; [BBP05] Baik-Ben Arous-Peche 2005; [LZZ21]
Loffler-Zhang-Zhou Ann. Stat. 2021; [DGJ18] Donoho-Gavish-Johnstone 2018;
[GD14] Gavish-Donoho 2014; [VW04] Vempala-Wang JCSS 2004; [G22] Galanti et
al. ICLR 2022; [S22] Sorscher et al. PNAS 2022; [B20] Bartlett et al. PNAS
2020; [SC20] Bateni et al. CVPR 2020 (Simple CNAPS, arXiv 1912.03432);
[TC22] Bateni et al. WACV 2022 (Transductive CNAPS); [RM23] Garrido et al.
ICML 2023 (RankMe, arXiv 2210.02885); [J22] Jing et al. ICLR 2022
(dimensional collapse, arXiv 2110.09348). All math plain ASCII.

---

## 1. Setting and the one quantity

Classes with anchors mu_c, shared within-class covariance Sigma (M1; on our
data Sigma is dominated by the 8b structured field — that fact enters only
the conditional clauses). Confusable pair (y,c): delta = mu_y - mu_c. A
linear transform x -> Ax acts on everything covariantly (`dwt_theory.md`
§4a); only the metric M = A'A matters for margins. The pair margin
statistic in the transformed space, along the transformed pair axis
v = A*delta/||A*delta||, has

```
separation   = ||A delta||
noise var    = delta' M Sigma M delta / (delta' M delta)
d'^2(M)      = (delta' M delta)^2 / (delta' M Sigma M delta) .
```

d'^2(I) = (delta'delta)^2 / (delta' Sigma delta) is the raw discriminability.
Truncation is the special case M = P_m (an orthogonal projector, treated as
a degenerate metric). Everything below is a statement about d'^2(M).

---

## 2. Lemma W1 — whitening is the optimal metric, and the pool can fit it

### W1a (deterministic core; exact, gap-free — the D1-grade layer)

**Lemma W1a.** For every pair (y,c) and every PSD metric M:

```
d'^2(M)  <=  delta' Sigma^{-1} delta ,
```

with equality iff M*delta is proportional to Sigma^{-1}*delta — in
particular at M = Sigma^{-1} (within-class whitening), which therefore
attains the maximum FOR EVERY PAIR SIMULTANEOUSLY. Moreover whitening never
hurts: d'^2(Sigma^{-1}) >= d'^2(I), with equality iff delta is an
eigenvector of Sigma.

*Proof.* Set a = Sigma^{1/2} M delta, b = Sigma^{-1/2} delta. Then
delta'M delta = a'b, delta'M Sigma M delta = a'a, delta'Sigma^{-1}delta =
b'b, and d'^2(M) = (a'b)^2/(a'a) <= b'b by Cauchy-Schwarz, with equality
iff a || b. The never-hurts clause is Cauchy-Schwarz again on
(delta'delta)^2 = (b' Sigma^{1/2} Sigma^{1/2} b)^2... i.e.
(delta'delta)^2 <= (delta'Sigma delta)(delta'Sigma^{-1}delta). QED.

Same epistemic grade as D1: triangle-inequality-free, assumption-free
(any Sigma > 0), per-pair, and it explains the phase's JOB in one line —
whitening converts the Chang functional [C83] delta'Sigma^{-1}delta (the
correct discriminant currency) into plain Euclidean geometry, after which
variance ordering IS discriminant ordering. [RCA05]'s three optimality
programs and Fisher/[F90] optimality are the published containers of the
same maximum; W1a is their per-pair, metric-free statement.

### W1b (unlabeled-fit clause: the pool total covariance suffices)

**Lemma W1b (two-class exact).** Let Sigma_T = Sigma + pi_y pi_c delta delta'
be the pair-mixture TOTAL covariance (computable from unlabeled data).
Then by Sherman-Morrison

```
Sigma_T^{-1} delta = Sigma^{-1} delta / (1 + pi_y pi_c * delta'Sigma^{-1}delta)
```

— proportional to Sigma^{-1} delta — so M = Sigma_T^{-1} satisfies W1a's
equality condition: **total-covariance whitening attains the label-oracle
maximum d'^2 = delta'Sigma^{-1}delta exactly, with zero labels.**

**K-class remainder (GW1).** With B = sum_c pi_c (mu_c-mubar)(mu_c-mubar)'
(rank <= K-1), Woodbury gives Sigma_T^{-1}delta = Sigma^{-1}delta - r with
r confined to the K-1-dimensional whitened between-class subspace; the d'
shortfall is controlled by the angle between Sigma^{-1}delta and that
subspace's contamination. [F90]'s identity (S_T^{-1}S_B and S_W^{-1}S_B
share eigenvectors, eigenvalues lam_T = lam_W/(1+lam_W)) is the global
version: the DIRECTIONS are exactly preserved; only a bounded low-rank
rescaling differs. Status: exact statement owed; severity low (the
correction is rank-(K-1) in d = 768 and vanishes in the pair-dominant
geometry). Empirical handle: `wlw_frac_of_oracle` in the diagnostic.

### W1c (estimation clause: pseudo-clusters and n < d; conditional)

The deployed transform estimates Sigma from POOL data via k-means
pseudo-clusters (within-cluster residuals), Ledoit-Wolf regularized
(champion lw_cluster768) or diagonal (deployed engine). Two error terms:

- **Sampling.** [LW04] Thms 3.2-3.4: the shrinkage estimator converges to
  the oracle-optimal linear combination in quadratic loss for d/n -> c,
  INCLUDING c > 1, distribution-free — the clause that makes full-rank
  whitening legal at n_cluster << 768. Remaining glue (GW2a): compose
  quadratic-loss consistency with d'^2(M_hat) continuity (the ratio in §1
  is smooth in M; a first-order perturbation bound suffices).
- **Impurity.** Pseudo-cluster residuals include between-class
  displacement WITHIN impure clusters: Sigma_hat = Sigma + E with E the
  PSD leakage second moment, growing as cluster label-purity falls — the
  SAME homophily dial that gates D3, appearing in the W phase (GW2b:
  quantify d' loss as a function of cluster purity; predicted direction:
  pool whitening lags the label oracle most on low-h data).

### W1c' — one family: every menu member estimates the same metric (2026-08-18)

The unifying reading that W1a licenses and the estimation clause makes
concrete: the entire deployed transform menu consists of REGULARIZED
ESTIMATORS OF THE SINGLE W1a-OPTIMAL METRIC Sigma^{-1}, differing only in
where they sit on a bias-variance dial:

```
estimator                         regularizer                       family member
oracle Sigma_w^{-1}               none (labels)                     the W1a max
lw_cluster768 (champion, airc)    linear shrinkage toward mu*I      soft [LW04]
[SC20] Q_k = lam*Sig_k            hierarchical shrinkage            soft, shot-indexed
   + (1-lam)*Sig_task + beta*I      (lam = n_k/(n_k+1)) + ridge       lam: class <-> task
t128w (champion, separable)       rank-128 projection + diag        blunt (rank cut)
wdiag (deployed engine)           diagonal restriction              blunt (no rotation)
raw / Euclidean prototypes        full shrink to identity           the t=1 end
```

The few-shot literature independently converged on the same family:
[SC20] replaces CNAPS's meta-learned classifier head with exactly this
object — softmax over -(1/2)(x-mu_k)' Q_k^{-1} (x-mu_k) with the
hierarchical-shrinkage Q_k above — and measures the family against its
own degenerate ends at benchmark scale (Meta-Dataset, 8 domains):
Mahalanobis 72.2% vs squared-Euclidean 69.6% vs cosine 68.3%, while
DELETING 788k classifier parameters (+6.1% over the learned head). Their
own justification is the W1a one-liner: Euclidean prototypes "implicitly
assume each cluster is distributed according to a unit normal"; the
Mahalanobis rule is the Gaussian-mixture responsibility (Bregman
divergence of the multivariate normal family). [TC22] then refines Q_k
with UNLABELED examples (soft k-means over the query set) — pool-fit W in
deployed form, minus exchangeability discipline and validity. So the W
stage needs no bespoke defense: it is the few-shot field's own best
practice, factored into an exchangeable preprocessing transform, with
W1a/W1b supplying the optimality and no-labels statements the deployed
versions lack.

Under this frame T is NOT a separate bet: truncation is the BLUNT
regularizer in the same estimation problem (rank cut instead of
shrinkage), justified precisely where [SC20]'s lam blend is justified —
too few samples for the full-rank estimate (T1b's 2m/s term) — and
contraindicated exactly where the cut removes discriminant mass (Chang /
F4). The regime story becomes one sentence: LOW PR = the tail carries
the discriminant, use the soft regularizer (lw768); HIGH PR + starved
labels = the tail is dead weight, the blunt regularizer is cheaper and
its estimation savings win (t128w).

### W1d (interaction with the D phase)

Whitening treats ALL within-class variance as noise — including the 8b
shared field, which is exactly the component qe moves along. The d'
currency is covariant under the affine W map (§4a of `dwt_theory.md`), so
the composition is well-defined; but W changes the RELATIVE weight of
field vs shell directions, hence the measured phi and the D-phase gate
constants are W-dependent. Measured, not modeled (the D-phase instruments
already run in the deployed post-W space); flagged as the W x D
interaction note.

---

## 3. Lemma T1 — truncation trades signal alignment for estimation noise

Stated POST-W1 (whitened metric, Sigma = I), which is the pipeline order;
the un-whitened case is a remark (T1a').

### T1a (deterministic core; exact, gap-free — the D1-grade layer)

**Lemma T1a.** In the whitened metric, for any orthogonal projector P_m of
rank m and every pair:

```
d'_m / d'  =  sqrt(a_m),    a_m = ||P_m delta||^2 / ||delta||^2  in [0,1].
```

*Proof.* Separation along the projected axis = ||P_m delta||; the noise is
isotropic so its sd along ANY unit axis is 1; divide. QED.

Consequence (the honest half the folklore skips): **post-whitening,
population truncation NEVER increases discriminability** — it can only
discard signal, at the alignment rate a_m. Whatever truncation buys, it
does NOT buy it at the population-geometry level. [C83] is the certificate
that a_m can be arbitrarily poor for the top-variance choice of P_m (his
example: the LAST PC carries the separation); [DGJ18] sharpens: no
spectral reweighting rescues what the retained subspace lost.

**Remark T1a' (un-whitened truncation = crude whitening).** Without W,
d'^2(P) = ||P delta||^4 / (delta'P Sigma P delta) CAN exceed d'^2(I) —
precisely when the discarded directions carry much noise variance and
little of delta (down-weighting nuisance to zero = an infinite-contrast
metric). Truncation-helps is therefore a SPECIAL CASE of W1a's optimal
metric, implemented bluntly; this is why t128w ~ champion on separable
data and why PCA-alone already helps there (t128 vs raw), while both are
dominated by the full W1a maximum whenever the signal is not
top-spectrum-aligned.

### T1b (the actual gain: finite-shot prototype estimation; conditional)

Anchors are ESTIMATED: with s shots/class in the whitened space,
mu_hat = mu + eps/sqrt(s), eps isotropic. The estimated pair axis is
v_hat ∝ P_m delta + eta with E||eta||^2 = 2m/s, so (isotropic eta)

```
E[ cos^2(v_hat, v) ] ~= A_m / (A_m + 2m/s),    A_m = ||P_m delta||^2 ,
```

and the EFFECTIVE discriminability of the deployed (estimated-prototype)
margin is

```
d'_eff(m, s)^2  ~=  A_m^2 / (A_m + 2m/s) .            (T1b)
```

Reading (this is the entire mechanism of the PCA win):

- m enters TWICE with opposite signs: A_m rises with m (alignment), the
  penalty 2m/s rises with m (estimation noise). Interior optimum m*.
- If the class-mean subspace sits in the top spikes ([VW04]: between-class
  scatter IS a rank-(K-1) spike of the pool total covariance), then
  A_m ~ ||delta||^2 already at m ~ K << d, and truncating from d = 768 to
  m ~ 128 cuts the penalty ~6-fold at NO alignment cost: at s = 2,
  d'_eff(128)/d'_eff(768) ~ sqrt((A + 768)/(A + 128)) ~ 2 for A ~ 15 —
  the label-starved PCA payoff, VANISHING as s grows (matches: PCA-128
  payoff largest at small cal, "balanced split matters most at small
  cal").
- If A_128 is SMALL (signal below the retained spectrum — aircraft), the
  numerator dies first and no m rescues it: truncation harm not tunable,
  the T-twin of D3's (C2). [G22]'s finite-shot CDNV terms and [S22]'s
  SNR-with-PR formula are the published containers of exactly this
  tradeoff; [LZZ21] is the end-to-end unlabeled version (project on top-K
  SVD of the pool, then cluster: minimax error exp(-d'^2/8) under an SNR
  condition).

Status: conditional (Gaussian isotropic prototype noise, axis-estimation
dominant); constants owed (GT2). The diagnostic measures cos^2(v_hat,v)
directly against the T1b factor.

### T1c (pool-estimated subspace: detectability; conditional, signed)

P_m is estimated from the pool. Under the spiked model, [P07]: a
supercritical spike ell > 1 + sqrt(c) (c = d/n_pool) yields sample-PC
overlap |<v_hat, v>|^2 -> (1 - c/(ell-1)^2)/(1 + c/(ell-1)); a subcritical
spike yields overlap -> 0 — the sample subspace contains asymptotically
NONE of it ([BBP05] names the threshold; [GD14] gives the label-free
2.858*median rule for locating the bulk edge from the pool alone). So the
population a_m must be multiplied by the overlap factor, and class-mean
directions that are subcritical RELATIVE TO THE BULK are invisible to any
pool-spectrum procedure.

**Signed violation (GT1), the T-twin of V1/V2:** our bulk is not white —
it is the 8b structured field. Sample PCs chase the field's variance, not
the class means, so on fine-grained data the effective overlap is LOWER
than the white-bulk formula predicts: a_hat_m <= a_m * overlap, with the
gap growing as field variance dominates. Direction known (truncation
worse than the idealized model), magnitude owed — same epistemic shape as
the D-phase (I) violations.

### T1d (the label-free dial)

a_m needs labels; the deployable proxies are pool-spectrum functionals:
participation ratio PR (precedents: [B20]'s effective rank R_k IS the
tail-spectrum PR; [S22] puts PR natively in a prototype-cosine error
formula) and the crude spike count above the bulk edge. The diagnostic
logs both next to the measured a_m so the proxy's fidelity is itself a
measurement, not an assumption.

The SSL literature supplies both halves of the dial's justification
(2026-08-18):

- **The premise that the spectrum is the right label-free instrument is
  [RM23]'s headline result**: RankMe(Z) = exp(-sum_k p_k log p_k), p_k =
  sigma_k/||sigma||_1 — the spectral-entropy effective rank of the
  embedding matrix, a smooth cousin of our PR — predicts downstream
  accuracy across 110 SSL models x 11 datasets with ZERO labels. Their
  theoretical motivation is a one-liner of the right shape: a linear
  readout cannot increase rank, so embedding rank bounds what any
  downstream linear/prototype classifier can separate (Cover's theorem).
  Their registered caveat is OUR regime map appearing in their data: rank
  is "only a necessary condition", and the one benchmark where
  best-performance breaks rank-monotonicity is STANFORD CARS — the same
  dataset that sits mid-regime in every one of our tables.
- **The premise that SSL spectra HAVE a dead tail to cut is [J22]**:
  contrastive SSL embeddings exhibit dimensional collapse — a set of
  covariance singular values driven to ~zero by augmentation strength
  and implicit regularization — so on collapse-shaped spectra truncation
  discards genuinely empty directions and T1a's alignment cost a_m ~ 1 is
  structural, not lucky. The T bet in one line: **is the tail dead [J22]
  or alive [C83]?** High PR / high RankMe = dead tail (cut is free, T1b's
  savings win); low PR = the discriminant hides in the tail (Chang; cut
  destroys it; use the soft regularizer of W1c'). The pool spectrum
  answers label-free, and the five-dataset table is the measured form of
  exactly this dichotomy.

---

## 4. Measured constants (run 2026-08-16, `src/wt_phase_diagnostics.py`, `output/dwt_theory/wt_phase_diagnostics.json`)

Per-pair d'-ratios vs raw (normalized/deployed currency; nearest-prototype
pairs fixed from raw class means; all transforms pool-fit, k-means 20,
seed 42):

```
dataset        d'raw  wdiag  wlw    t128   t128w | a_128  sqrt  changTail  PR    wlw/oracle  s=2cos2 s=8cos2
cifar100       3.93   x1.00  x1.03  x1.00  x1.00 | 0.82   0.91  0.45       243   0.26        0.32    0.63
miniimagenet   5.66   x1.00  x0.91  x1.17  x1.11 | 0.93   0.96  0.24       255   0.23        0.50    0.77
cifar10        5.45   x1.00  x1.05  x1.09  x1.07 | 0.99   1.00  0.15       119   0.44        0.32    0.63
stanford_cars  2.85   x1.01  x1.87  x0.89  x1.34 | 0.85   0.92  0.79       24    0.33        0.23    0.54
aircraft       1.67   x1.03  x3.13  x0.89  x1.77 | 0.87   0.93  0.83       16    0.35        0.15    0.33
```

W1a bound violations: **0 on all five datasets** (the Cauchy-Schwarz
maximum holds pointwise — instrument-verified exactness, the D1-check
analogue).

Registered predictions, scored:

- (P-W1) "wlw >= raw everywhere": held 4/5 population-wise; **mini x0.91
  fails it** — full-rank LW estimation noise from 20 pseudo-clusters makes
  whitening mildly harmful when raw d' is already 5.7 (frac_improved 0.14).
  Estimation clause W1c is not decorative. "Oracle fraction ordered by h":
  **FAILED** — 0.23-0.44 with cifar100 (h=.81) at 0.26 vs aircraft (h=.26)
  at 0.35. Cluster impurity is NOT the dominant limiter of pool whitening;
  the gap is coarse pseudo-cluster granularity + shrinkage everywhere
  (GW2b needs reframing toward granularity, not just purity).
- (P-T1) harm side: **held quantitatively** — measured t128 ratio 0.89 on
  cars AND aircraft vs sqrt(a_128) = 0.92/0.93 (slightly worse than the
  population identity, the GT1 signed direction). Separable side: measured
  1.00-1.17 EXCEEDS sqrt(a_128) — the T1a' crude-whitening effect
  (discarded dims carried nuisance variance orthogonal to delta), predicted
  in direction. The a_128-low prediction for aircraft **FAILED in an
  instructive way** — see finding F4.
- (P-T1b) ordering: held (cos^2 rises with s everywhere; aircraft lowest
  at fixed s). Quantitative check against A_m/(A_m + 2m/s) requires the
  whitened-metric version of the instrument (raw-space trace replaces m);
  follow-up, not run here.

### Findings (what the table says that the lemmas' first draft did not)

- **F1 — the W phase is the fine-grained lever, and it is the
  OFF-DIAGONAL structure.** Full-rank wlw: x1.87 (cars), x3.13 (aircraft);
  diagonal wdiag: x1.00-1.03 EVERYWHERE. The deployed per-dim inverse-std
  does not move population pair-d' at all; the entire population-geometry
  gain of whitening rides the covariance ROTATION (Sigma^{-1/2}'s
  eigenbasis), not the per-dim rescale. Matches the champion table
  (lw_cluster768 wins aircraft) and gives W1 its sharp empirical content.
- **F2 — on separable data the pipeline's set-size wins are NOT
  population-d' effects.** cifar100: every ratio ~1.00 while the deployed
  pipeline (t128w engine) crushes set sizes empirically. By elimination —
  and by T1b's math — the separable-regime gain is FINITE-SHOT: truncation
  cuts prototype-estimation noise ~d/m-fold (plus NCM-neighborhood
  effects outside this instrument). Consistent with the known "PCA payoff
  largest at small cal" and with T1b's s-scaling.
- **F3 — order matters: W-before-T is the theorem-consistent order.** On
  aircraft, t128w recovers only x1.77 of wlw's x3.13: truncation FIRST
  discards the low-variance tail that whitening would have up-weighted
  (t128 x0.89), and no post-truncation whitening can bring it back
  ([DGJ18]'s no-rescue, order-of-operations form). The champion's
  "aircraft = full-rank W, no truncation" is now T1a + Chang, measured.
- **F4 — the T failure mode is discriminant-mass-in-tail, NOT axis
  misalignment.** a_128 is HIGH everywhere (0.82-0.99, aircraft 0.87): the
  pair axes' ENERGY sits in the top spectrum on every dataset. What
  separates the regimes is the Chang functional: the fraction of
  delta'Sigma^{-1}delta carried BEYOND PC-128 is 0.15-0.45 on separable
  data vs 0.79/0.83 on cars/aircraft. So T1c's story sharpens: truncation
  on fine-grained data does not lose the axis, it loses the axis's
  DISCRIMINANT-DENSE (low-variance) component — the failure is inherited
  from the W-phase need, exactly Chang's 1/lambda weighting. GT1 should be
  restated in Chang-functional terms rather than subspace-overlap terms.
- **F5 — dials.** PR separates the regimes cleanly (243/255/119 vs 24/16 —
  threshold anywhere in ~[30, 100]); the crude MP spike count does NOT
  (233-281 on all datasets — the structured bulk swamps the white-bulk
  edge estimate; dropped as a dial, which is itself GT1 evidence: the
  white-bulk model is wrong on every dataset, worst where PR is low).
- **F6 — headroom.** Pool whitening reaches only 23-44% of the
  label-oracle Mahalanobis bound on every dataset. Either the oracle
  (labeled LW Sigma_w^{-1}) overstates reachable d' (small within-class
  eigenvalues inverted — needs a shrinkage-honest oracle variant), or
  there is a real 2-4x d' improvement unclaimed by finer/purer pool
  clustering. Discriminating experiment: oracle-clustered (label-driven)
  pool whitening at matched shrinkage; if the gap persists, it is the
  bound's optimism; if it closes, W1c's granularity clause is a real
  lever. Follow-up registered.

---

## 5. Remainder ledger (the honest debt, D1-style)

| gap | clause | content | severity |
|-----|--------|---------|----------|
| GW1 | W1b | K-class rank-(K-1) correction: exact bound on the d' shortfall of total-vs-within whitening | low (directions exact by [F90]; bounded low-rank rescaling) |
| GW2a | W1c | [LW04] quadratic-loss -> d'(M_hat) continuity composition | low-medium (standard perturbation) |
| GW2b | W1c | pseudo-cluster impurity: d' loss as a function of cluster purity (the W-phase homophily clause) | medium — the deployable-gate twin of G2 |
| GT1 | T1c | structured (field) bulk: signed overlap degradation beyond the white-bulk [P07] formula; per F4/F5, restate in Chang-functional terms (tail discriminant mass), since a_m stays high and the white-bulk spike count fails on ALL datasets | medium — the T-phase (I)-violation analogue |
| GT2 | T1b | finite-shot constants beyond Gaussian-isotropic; a_m concentration across pairs | low-medium |

Downstream hook (unchanged chain): W/T enter Lemma B and Prop C ONLY
through the same d' currency D3 outputs; the pipeline d'-ratio factorizes
as (W ratio) x (T ratio) x (D ratio) in the composed metric, so the G6 /
contraction-anchor work (Section 10 of `dwt_denoise_theorem.md`) is
agnostic to which phase moved d'.

## 6. Scorecard against the D1 template

Matched: W1a and T1a are deterministic, per-pair, assumption-free (any
Sigma > 0), one-line proofs — the same epistemic grade as D1; each phase's
empirical failure mode is recovered inside the lemma (Chang tail for T,
impurity clause for W) rather than assumed away.

Added beyond the D template: W1b's two-class UNLABELED-FIT EXACTNESS (the
pool total covariance attains the label-oracle maximum via
Sherman-Morrison) — the W-phase has a stronger no-labels story than D
(where selection bias is irreducible).

Owed: the ledger above; plus the composed-pipeline statement (one theorem
multiplying the three phase ratios with their three gates) — the DWT
theorem target, blocked only on the conditional clauses' constants.
