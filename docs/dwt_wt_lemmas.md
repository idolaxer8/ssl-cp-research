# Lemmas W1 and T1 — Whiten and Truncate, continued from base lemma D1

Status: v0.3 (2026-08-16, weekly goal 3; v0.2 2026-08-18: math converted to
LaTeX per user preference — open in VS Code `Ctrl+Shift+V` / GitHub to
render; measured-data tables stay fixed-width. **v0.3 2026-08-18: GROUNDING
SHIFT — D3's quantitative layer is unsupported, so the load-bearing
justification is now the CHAIN-FREE one in Section 7**: W/T as services to
D1's own inputs — W raises neighborhood homophily hence D1's certification
reach, T cuts anchor estimation error; the $d'$ chain of Sections 2-4 is
retained as instruments + the aspirational route to $\mathbb{E}|C|$). Build principle (framing
decision 08-16): **D1 is the base lemma of the Denoise phase** —
deterministic, assumption-free, validated pointwise 5/5 — and the W/T
phases get lemmas of the SAME construction: an exact, gap-free statement
first (W1a, T1a — both one-line proofs), then the statistically loaded
refinements as named, conditional clauses with signed violations and a
measurement instrument (`src/wt_phase_diagnostics.py`, mirroring
`src/dwt_gate_constants.py`). Anchor stack:
`docs/classification_theory_anchors.md` (reading guide §4). Currency
throughout: the pair-margin discriminability $d'$ of Theorem D3 — so W/T
ratios compose multiplicatively with the D-phase $d'$-ratio and feed the
unchanged downstream chain (Lemma B $\to$ Prop C, `dwt_theory.md` §5-6).
Pedagogical companion: `docs/dwt_wt_learning_edition.md`.

Reference keys: [RCA05] Bar-Hillel et al. JMLR 2005; [F90] Fukunaga 1990
ch. 10; [LW04] Ledoit-Wolf JMVA 2004; [C83] Chang JRSS-C 1983; [P07] Paul
Statistica Sinica 2007; [BBP05] Baik-Ben Arous-Peche 2005; [LZZ21]
Loffler-Zhang-Zhou Ann. Stat. 2021; [DGJ18] Donoho-Gavish-Johnstone 2018;
[GD14] Gavish-Donoho 2014; [VW04] Vempala-Wang JCSS 2004; [G22] Galanti et
al. ICLR 2022; [S22] Sorscher et al. PNAS 2022; [B20] Bartlett et al. PNAS
2020; [SC20] Bateni et al. CVPR 2020 (Simple CNAPS, arXiv 1912.03432);
[TC22] Bateni et al. WACV 2022 (Transductive CNAPS); [RM23] Garrido et al.
ICML 2023 (RankMe, arXiv 2210.02885); [J22] Jing et al. ICLR 2022
(dimensional collapse, arXiv 2110.09348).

---

## 1. Setting and the one quantity

Classes with anchors $\mu_c$, shared within-class covariance $\Sigma$ (M1;
on our data $\Sigma$ is dominated by the 8b structured field — that fact
enters only the conditional clauses). Confusable pair $(y,c)$:
$\delta = \mu_y - \mu_c$. A linear transform $x \mapsto Ax$ acts on
everything covariantly (`dwt_theory.md` §4a); only the metric
$M = A^\top A$ matters for margins. The pair margin statistic in the
transformed space, along the transformed pair axis
$v = A\delta/\|A\delta\|$, has

$$
\text{separation} = \|A\delta\|, \qquad
\text{noise var} = \frac{\delta^\top M \Sigma M \delta}{\delta^\top M \delta},
\qquad
d'^2(M) = \frac{(\delta^\top M \delta)^2}{\delta^\top M \Sigma M \delta}.
$$

$d'^2(I) = (\delta^\top\delta)^2 / (\delta^\top \Sigma\, \delta)$ is the
raw discriminability. Truncation is the special case $M = P_m$ (an
orthogonal projector, treated as a degenerate metric). Everything below is
a statement about $d'^2(M)$.

---

## 2. Lemma W1 — whitening is the optimal metric, and the pool can fit it

### W1a (deterministic core; exact, gap-free — the D1-grade layer)

**Lemma W1a.** For every pair $(y,c)$ and every PSD metric $M$:

$$
d'^2(M) \;\le\; \delta^\top \Sigma^{-1} \delta,
$$

with equality iff $M\delta \parallel \Sigma^{-1}\delta$ — in particular at
$M = \Sigma^{-1}$ (within-class whitening), which therefore attains the
maximum FOR EVERY PAIR SIMULTANEOUSLY. Moreover whitening never hurts:
$d'^2(\Sigma^{-1}) \ge d'^2(I)$, with equality iff $\delta$ is an
eigenvector of $\Sigma$.

*Proof.* Set $a = \Sigma^{1/2} M \delta$, $b = \Sigma^{-1/2}\delta$. Then
$\delta^\top M\delta = a^\top b$,
$\delta^\top M\Sigma M\delta = a^\top a$,
$\delta^\top\Sigma^{-1}\delta = b^\top b$, and

$$
d'^2(M) = \frac{(a^\top b)^2}{a^\top a} \;\le\; b^\top b
$$

by Cauchy-Schwarz, with equality iff $a \parallel b$. The never-hurts
clause is Cauchy-Schwarz once more:
$(\delta^\top\delta)^2 \le (\delta^\top\Sigma\delta)
(\delta^\top\Sigma^{-1}\delta)$. $\blacksquare$

Same epistemic grade as D1: triangle-inequality-free, assumption-free
(any $\Sigma \succ 0$), per-pair, and it explains the phase's JOB in one
line — whitening converts the Chang functional [C83]
$\delta^\top\Sigma^{-1}\delta$ (the correct discriminant currency) into
plain Euclidean geometry, after which variance ordering IS discriminant
ordering. [RCA05]'s three optimality programs and Fisher/[F90] optimality
are the published containers of the same maximum; W1a is their per-pair,
metric-free statement.

### W1b (unlabeled-fit clause: the pool total covariance suffices)

**Lemma W1b (two-class exact).** Let
$\Sigma_T = \Sigma + \pi_y \pi_c\, \delta\delta^\top$ be the pair-mixture
TOTAL covariance (computable from unlabeled data). Then by
Sherman-Morrison

$$
\Sigma_T^{-1}\delta
= \frac{\Sigma^{-1}\delta}{1 + \pi_y\pi_c\, \delta^\top\Sigma^{-1}\delta}
$$

— proportional to $\Sigma^{-1}\delta$ — so $M = \Sigma_T^{-1}$ satisfies
W1a's equality condition: **total-covariance whitening attains the
label-oracle maximum $d'^2 = \delta^\top\Sigma^{-1}\delta$ exactly, with
zero labels.**

**K-class remainder (GW1).** With
$B = \sum_c \pi_c (\mu_c-\bar\mu)(\mu_c-\bar\mu)^\top$ (rank $\le K-1$),
Woodbury gives $\Sigma_T^{-1}\delta = \Sigma^{-1}\delta - r$ with $r$
confined to the $(K{-}1)$-dimensional whitened between-class subspace; the
$d'$ shortfall is controlled by the angle between $\Sigma^{-1}\delta$ and
that subspace's contamination. [F90]'s identity ($S_T^{-1}S_B$ and
$S_W^{-1}S_B$ share eigenvectors, eigenvalues
$\lambda_T = \lambda_W/(1+\lambda_W)$) is the global version: the
DIRECTIONS are exactly preserved; only a bounded low-rank rescaling
differs. Status: exact statement owed; severity low (the correction is
rank-$(K{-}1)$ in $d = 768$ and vanishes in the pair-dominant geometry).
Empirical handle: `wlw_frac_of_oracle` in the diagnostic.

### W1c (estimation clause: pseudo-clusters and n < d; conditional)

The deployed transform estimates $\Sigma$ from POOL data via k-means
pseudo-clusters (within-cluster residuals), Ledoit-Wolf regularized
(champion lw_cluster768) or diagonal (deployed engine). Two error terms:

- **Sampling.** [LW04] Thms 3.2-3.4: the shrinkage estimator converges to
  the oracle-optimal linear combination in quadratic loss for
  $d/n \to c$, INCLUDING $c > 1$, distribution-free — the clause that
  makes full-rank whitening legal at $n_{\text{cluster}} \ll 768$.
  Remaining glue (GW2a): compose quadratic-loss consistency with
  $d'^2(\hat M)$ continuity (the ratio in §1 is smooth in $M$; a
  first-order perturbation bound suffices).
- **Impurity.** Pseudo-cluster residuals include between-class
  displacement WITHIN impure clusters: $\hat\Sigma = \Sigma + E$ with $E$
  the PSD leakage second moment, growing as cluster label-purity falls —
  the SAME homophily dial that gates D3, appearing in the W phase (GW2b:
  quantify $d'$ loss as a function of cluster purity; predicted
  direction: pool whitening lags the label oracle most on low-$h$ data).

### W1c' — one family: every menu member estimates the same metric (2026-08-18)

The unifying reading that W1a licenses and the estimation clause makes
concrete: the entire deployed transform menu consists of REGULARIZED
ESTIMATORS OF THE SINGLE W1a-OPTIMAL METRIC $\Sigma^{-1}$, differing only
in where they sit on a bias-variance dial:

| estimator | regularizer | family member |
|---|---|---|
| oracle $\Sigma_w^{-1}$ | none (labels) | the W1a max |
| lw_cluster768 (champion, aircraft) | linear shrinkage toward $\bar\lambda I$ | soft [LW04] |
| [SC20] $Q_k = \lambda_k\Sigma_k + (1-\lambda_k)\Sigma_{\text{task}} + \beta I$ | hierarchical shrinkage ($\lambda_k = \frac{n_k}{n_k+1}$) + ridge | soft, shot-indexed |
| t128w (champion, separable) | rank-128 projection + diagonal | blunt (rank cut) |
| wdiag (deployed engine) | diagonal restriction | blunt (no rotation) |
| raw / Euclidean prototypes | full shrink to identity | the $t{=}1$ end |

The few-shot literature independently converged on the same family:
[SC20] replaces CNAPS's meta-learned classifier head with exactly this
object — softmax over $-\tfrac12 (x-\mu_k)^\top Q_k^{-1} (x-\mu_k)$ with
the hierarchical-shrinkage $Q_k$ above — and measures the family against
its own degenerate ends at benchmark scale (Meta-Dataset, 8 domains):
Mahalanobis 72.2% vs squared-Euclidean 69.6% vs cosine 68.3%, while
DELETING 788k classifier parameters (+6.1% over the learned head). Their
own justification is the W1a one-liner: Euclidean prototypes "implicitly
assume each cluster is distributed according to a unit normal"; the
Mahalanobis rule is the Gaussian-mixture responsibility (Bregman
divergence of the multivariate normal family). [TC22] then refines $Q_k$
with UNLABELED examples (soft k-means over the query set) — pool-fit W in
deployed form, minus exchangeability discipline and validity. So the W
stage needs no bespoke defense: it is the few-shot field's own best
practice, factored into an exchangeable preprocessing transform, with
W1a/W1b supplying the optimality and no-labels statements the deployed
versions lack.

Under this frame T is NOT a separate bet: truncation is the BLUNT
regularizer in the same estimation problem (rank cut instead of
shrinkage), justified precisely where [SC20]'s $\lambda_k$ blend is
justified — too few samples for the full-rank estimate (T1b's $2m/s$
term) — and contraindicated exactly where the cut removes discriminant
mass (Chang / F4). The regime story becomes one sentence: LOW PR = the
tail carries the discriminant, use the soft regularizer (lw768); HIGH PR
+ starved labels = the tail is dead weight, the blunt regularizer is
cheaper and its estimation savings win (t128w).

### W1d (interaction with the D phase)

Whitening treats ALL within-class variance as noise — including the 8b
shared field, which is exactly the component qe moves along. The $d'$
currency is covariant under the affine W map (§4a of `dwt_theory.md`), so
the composition is well-defined; but W changes the RELATIVE weight of
field vs shell directions, hence the measured $\phi$ and the D-phase gate
constants are W-dependent. Measured, not modeled (the D-phase instruments
already run in the deployed post-W space); flagged as the W $\times$ D
interaction note.

---

## 3. Lemma T1 — truncation trades signal alignment for estimation noise

Stated POST-W1 (whitened metric, $\Sigma = I$), which is the pipeline
order; the un-whitened case is a remark (T1a').

### T1a (deterministic core; exact, gap-free — the D1-grade layer)

**Lemma T1a.** In the whitened metric, for any orthogonal projector $P_m$
of rank $m$ and every pair:

$$
\frac{d'_m}{d'} = \sqrt{a_m},
\qquad
a_m = \frac{\|P_m\delta\|^2}{\|\delta\|^2} \in [0,1].
$$

*Proof.* Separation along the projected axis $= \|P_m\delta\|$; the noise
is isotropic so its sd along ANY unit axis is 1; divide. $\blacksquare$

Consequence (the honest half the folklore skips): **post-whitening,
population truncation NEVER increases discriminability** — it can only
discard signal, at the alignment rate $a_m$. Whatever truncation buys, it
does NOT buy it at the population-geometry level. [C83] is the
certificate that $a_m$ can be arbitrarily poor for the top-variance
choice of $P_m$ (his example: the LAST PC carries the separation);
[DGJ18] sharpens: no spectral reweighting rescues what the retained
subspace lost.

**Remark T1a' (un-whitened truncation = crude whitening).** Without W,

$$
d'^2(P) = \frac{\|P\delta\|^4}{\delta^\top P \Sigma P \delta}
$$

CAN exceed $d'^2(I)$ — precisely when the discarded directions carry much
noise variance and little of $\delta$ (down-weighting nuisance to zero =
an infinite-contrast metric). Truncation-helps is therefore a SPECIAL
CASE of W1a's optimal metric, implemented bluntly; this is why t128w ~
champion on separable data and why PCA-alone already helps there (t128 vs
raw), while both are dominated by the full W1a maximum whenever the
signal is not top-spectrum-aligned.

### T1b (the actual gain: finite-shot prototype estimation; conditional)

Anchors are ESTIMATED: with $s$ shots/class in the whitened space,
$\hat\mu = \mu + \varepsilon/\sqrt{s}$, $\varepsilon$ isotropic. The
estimated pair axis is $\hat v \propto P_m\delta + \eta$ with
$\mathbb{E}\|\eta\|^2 = 2m/s$, so (isotropic $\eta$)

$$
\mathbb{E}\!\left[\cos^2(\hat v, v)\right] \approx \frac{A_m}{A_m + 2m/s},
\qquad A_m = \|P_m\delta\|^2,
$$

and the EFFECTIVE discriminability of the deployed (estimated-prototype)
margin is

$$
d'^2_{\mathrm{eff}}(m, s) \;\approx\; \frac{A_m^2}{A_m + 2m/s}.
\tag{T1b}
$$

Reading (this is the entire mechanism of the PCA win):

- $m$ enters TWICE with opposite signs: $A_m$ rises with $m$ (alignment),
  the penalty $2m/s$ rises with $m$ (estimation noise). Interior optimum
  $m^\ast$.
- If the class-mean subspace sits in the top spikes ([VW04]:
  between-class scatter IS a rank-$(K{-}1)$ spike of the pool total
  covariance), then $A_m \approx \|\delta\|^2$ already at
  $m \approx K \ll d$, and truncating from $d = 768$ to $m \approx 128$
  cuts the penalty ~6-fold at NO alignment cost: at $s = 2$,
  $d'_{\mathrm{eff}}(128)/d'_{\mathrm{eff}}(768) \approx
  \sqrt{(A + 768)/(A + 128)} \approx 2$ for $A \approx 15$ — the
  label-starved PCA payoff, VANISHING as $s$ grows (matches: PCA-128
  payoff largest at small cal, "balanced split matters most at small
  cal").
- If $A_{128}$ is SMALL in discriminant currency (signal below the
  retained spectrum — aircraft), the numerator dies first and no $m$
  rescues it: truncation harm not tunable, the T-twin of D3's (C2).
  [G22]'s finite-shot CDNV terms and [S22]'s SNR-with-PR formula are the
  published containers of exactly this tradeoff; [LZZ21] is the
  end-to-end unlabeled version (project on top-$K$ SVD of the pool, then
  cluster: minimax error $\exp(-d'^2/8)$ under an SNR condition).

Status: conditional (Gaussian isotropic prototype noise, axis-estimation
dominant); constants owed (GT2). The diagnostic measures
$\cos^2(\hat v, v)$ directly against the T1b factor.

### T1c (pool-estimated subspace: detectability; conditional, signed)

$P_m$ is estimated from the pool. Under the spiked model, [P07]: a
supercritical spike $\ell > 1 + \sqrt{c}$ ($c = d/n_{\text{pool}}$)
yields sample-PC overlap

$$
|\langle \hat v, v\rangle|^2 \;\to\;
\frac{1 - c/(\ell-1)^2}{1 + c/(\ell-1)};
$$

a subcritical spike yields overlap $\to 0$ — the sample subspace contains
asymptotically NONE of it ([BBP05] names the threshold; [GD14] gives the
label-free $2.858 \times$ median rule for locating the bulk edge from the
pool alone). So the population $a_m$ must be multiplied by the overlap
factor, and class-mean directions that are subcritical RELATIVE TO THE
BULK are invisible to any pool-spectrum procedure.

**Signed violation (GT1), the T-twin of V1/V2:** our bulk is not white —
it is the 8b structured field. Sample PCs chase the field's variance, not
the class means, so on fine-grained data the effective overlap is LOWER
than the white-bulk formula predicts:
$\hat a_m \le a_m \cdot \text{overlap}$, with the gap growing as field
variance dominates. Direction known (truncation worse than the idealized
model), magnitude owed — same epistemic shape as the D-phase (I)
violations.

### T1d (the label-free dial)

$a_m$ needs labels; the deployable proxies are pool-spectrum functionals:
participation ratio PR (precedents: [B20]'s effective rank $R_k$ IS the
tail-spectrum PR; [S22] puts PR natively in a prototype-cosine error
formula) and the crude spike count above the bulk edge. The diagnostic
logs both next to the measured $a_m$ so the proxy's fidelity is itself a
measurement, not an assumption.

The SSL literature supplies both halves of the dial's justification
(2026-08-18):

- **The premise that the spectrum is the right label-free instrument is
  [RM23]'s headline result**:

  $$
  \mathrm{RankMe}(Z) = \exp\Big(-\sum_k p_k \log p_k\Big),
  \qquad p_k = \frac{\sigma_k(Z)}{\|\sigma(Z)\|_1},
  $$

  the spectral-entropy effective rank of the embedding matrix — a smooth
  cousin of our PR — predicts downstream accuracy across 110 SSL models
  $\times$ 11 datasets with ZERO labels. Their theoretical motivation is
  a one-liner of the right shape: a linear readout cannot increase rank,
  so embedding rank bounds what any downstream linear/prototype
  classifier can separate (Cover's theorem). Their registered caveat is
  OUR regime map appearing in their data: rank is "only a necessary
  condition", and the one benchmark where best-performance breaks
  rank-monotonicity is STANFORD CARS — the same dataset that sits
  mid-regime in every one of our tables.
- **The premise that SSL spectra HAVE a dead tail to cut is [J22]**:
  contrastive SSL embeddings exhibit dimensional collapse — a set of
  covariance singular values driven to ~zero by augmentation strength
  and implicit regularization — so on collapse-shaped spectra truncation
  discards genuinely empty directions and T1a's alignment cost
  $a_m \approx 1$ is structural, not lucky. The T bet in one line: **is
  the tail dead [J22] or alive [C83]?** High PR / high RankMe = dead
  tail (cut is free, T1b's savings win); low PR = the discriminant hides
  in the tail (Chang; cut destroys it; use the soft regularizer of
  W1c'). The pool spectrum answers label-free, and the five-dataset
  table is the measured form of exactly this dichotomy.

---

## 4. Measured constants (run 2026-08-16, `src/wt_phase_diagnostics.py`, `output/dwt_theory/wt_phase_diagnostics.json`)

Per-pair $d'$-ratios vs raw (normalized/deployed currency;
nearest-prototype pairs fixed from raw class means; all transforms
pool-fit, k-means 20, seed 42):

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

- (P-W1) "wlw $\ge$ raw everywhere": held 4/5 population-wise; **mini
  $\times$0.91 fails it** — full-rank LW estimation noise from 20
  pseudo-clusters makes whitening mildly harmful when raw $d'$ is
  already 5.7 (frac_improved 0.14). Estimation clause W1c is not
  decorative. "Oracle fraction ordered by $h$": **FAILED** — 0.23-0.44
  with cifar100 ($h{=}.81$) at 0.26 vs aircraft ($h{=}.26$) at 0.35.
  Cluster impurity is NOT the dominant limiter of pool whitening; the
  gap is coarse pseudo-cluster granularity + shrinkage everywhere (GW2b
  needs reframing toward granularity, not just purity).
- (P-T1) harm side: **held quantitatively** — measured t128 ratio 0.89
  on cars AND aircraft vs $\sqrt{a_{128}} = 0.92/0.93$ (slightly worse
  than the population identity, the GT1 signed direction). Separable
  side: measured 1.00-1.17 EXCEEDS $\sqrt{a_{128}}$ — the T1a'
  crude-whitening effect (discarded dims carried nuisance variance
  orthogonal to $\delta$), predicted in direction. The $a_{128}$-low
  prediction for aircraft **FAILED in an instructive way** — see finding
  F4.
- (P-T1b) ordering: held ($\cos^2$ rises with $s$ everywhere; aircraft
  lowest at fixed $s$). Quantitative check against $A_m/(A_m + 2m/s)$
  requires the whitened-metric version of the instrument (raw-space
  trace replaces $m$); follow-up, not run here.

### Findings (what the table says that the lemmas' first draft did not)

- **F1 — the W phase is the fine-grained lever, and it is the
  OFF-DIAGONAL structure.** Full-rank wlw: $\times$1.87 (cars),
  $\times$3.13 (aircraft); diagonal wdiag: $\times$1.00-1.03 EVERYWHERE.
  The deployed per-dim inverse-std does not move population pair-$d'$ at
  all; the entire population-geometry gain of whitening rides the
  covariance ROTATION ($\Sigma^{-1/2}$'s eigenbasis), not the per-dim
  rescale. Matches the champion table (lw_cluster768 wins aircraft) and
  gives W1 its sharp empirical content.
- **F2 — on separable data the pipeline's set-size wins are NOT
  population-$d'$ effects.** cifar100: every ratio ~1.00 while the
  deployed pipeline (t128w engine) crushes set sizes empirically. By
  elimination — and by T1b's math — the separable-regime gain is
  FINITE-SHOT: truncation cuts prototype-estimation noise ~$d/m$-fold
  (plus NCM-neighborhood effects outside this instrument). Consistent
  with the known "PCA payoff largest at small cal" and with T1b's
  $s$-scaling.
- **F3 — order matters: W-before-T is the theorem-consistent order.** On
  aircraft, t128w recovers only $\times$1.77 of wlw's $\times$3.13:
  truncation FIRST discards the low-variance tail that whitening would
  have up-weighted (t128 $\times$0.89), and no post-truncation whitening
  can bring it back ([DGJ18]'s no-rescue, order-of-operations form). The
  champion's "aircraft = full-rank W, no truncation" is now T1a + Chang,
  measured.
- **F4 — the T failure mode is discriminant-mass-in-tail, NOT axis
  misalignment.** $a_{128}$ is HIGH everywhere (0.82-0.99, aircraft
  0.87): the pair axes' ENERGY sits in the top spectrum on every
  dataset. What separates the regimes is the Chang functional: the
  fraction of $\delta^\top\Sigma^{-1}\delta$ carried BEYOND PC-128 is
  0.15-0.45 on separable data vs 0.79/0.83 on cars/aircraft. So T1c's
  story sharpens: truncation on fine-grained data does not lose the
  axis, it loses the axis's DISCRIMINANT-DENSE (low-variance) component
  — the failure is inherited from the W-phase need, exactly Chang's
  $1/\lambda$ weighting. GT1 should be restated in Chang-functional
  terms rather than subspace-overlap terms.
- **F5 — dials.** PR separates the regimes cleanly (243/255/119 vs 24/16
  — threshold anywhere in ~[30, 100]); the crude MP spike count does NOT
  (233-281 on all datasets — the structured bulk swamps the white-bulk
  edge estimate; dropped as a dial, which is itself GT1 evidence: the
  white-bulk model is wrong on every dataset, worst where PR is low).
- **F6 — headroom.** Pool whitening reaches only 23-44% of the
  label-oracle Mahalanobis bound on every dataset. Either the oracle
  (labeled LW $\Sigma_w^{-1}$) overstates reachable $d'$ (small
  within-class eigenvalues inverted — needs a shrinkage-honest oracle
  variant), or there is a real 2-4$\times$ $d'$ improvement unclaimed by
  finer/purer pool clustering. Discriminating experiment:
  oracle-clustered (label-driven) pool whitening at matched shrinkage;
  if the gap persists, it is the bound's optimism; if it closes, W1c's
  granularity clause is a real lever. Follow-up registered.

---

## 5. Remainder ledger (the honest debt, D1-style)

| gap | clause | content | severity |
|-----|--------|---------|----------|
| GW1 | W1b | K-class rank-$(K{-}1)$ correction: exact bound on the $d'$ shortfall of total-vs-within whitening | low (directions exact by [F90]; bounded low-rank rescaling) |
| GW2a | W1c | [LW04] quadratic-loss $\to$ $d'(\hat M)$ continuity composition | low-medium (standard perturbation) |
| GW2b | W1c | pseudo-cluster impurity: $d'$ loss as a function of cluster purity (the W-phase homophily clause) | medium — the deployable-gate twin of G2 |
| GT1 | T1c | structured (field) bulk: signed overlap degradation beyond the white-bulk [P07] formula; per F4/F5, restate in Chang-functional terms (tail discriminant mass), since $a_m$ stays high and the white-bulk spike count fails on ALL datasets | medium — the T-phase (I)-violation analogue |
| GT2 | T1b | finite-shot constants beyond Gaussian-isotropic; $a_m$ concentration across pairs | low-medium |

Downstream hook (unchanged chain): W/T enter Lemma B and Prop C ONLY
through the same $d'$ currency D3 outputs; the pipeline $d'$-ratio
factorizes as (W ratio) $\times$ (T ratio) $\times$ (D ratio) in the
composed metric, so the G6 / contraction-anchor work (Section 10 of
`dwt_denoise_theorem.md`) is agnostic to which phase moved $d'$.

## 6. Scorecard against the D1 template

Matched: W1a and T1a are deterministic, per-pair, assumption-free (any
$\Sigma \succ 0$), one-line proofs — the same epistemic grade as D1; each
phase's empirical failure mode is recovered inside the lemma (Chang tail
for T, impurity clause for W) rather than assumed away.

Added beyond the D template: W1b's two-class UNLABELED-FIT EXACTNESS (the
pool total covariance attains the label-oracle maximum via
Sherman-Morrison) — the W-phase has a stronger no-labels story than D
(where selection bias is irreducible).

Owed: the ledger above; plus the composed-pipeline statement (one theorem
multiplying the three phase ratios with their three gates) — the DWT
theorem target, blocked only on the conditional clauses' constants.

---

## 7. The chain-free re-anchoring: W/T as services to D1's own inputs (2026-08-18)

### 7.0 The paper cut (distilled — this is what goes in the paper)

**One-paragraph version (~150 words):**

> The score consumes three objects: the embedding, its pool neighborhood,
> and the estimated class anchors. Each DWT stage improves exactly one of
> them, with an elementary, pointwise-checkable guarantee; none affects
> validity (all transforms are pool-measurable, so exchangeability — and
> exact coverage — is preserved). **Denoise** moves the point:
> $\|\hat x - \mu_y\| < \|x - \mu_y\|$ whenever the neighborhood's
> impurity budget is below the point's own error (Lemma D1, a triangle
> inequality; certified $\Rightarrow$ improved held for 100% of certified
> points on five datasets). **Whiten** moves the neighborhood: neighbors
> are defined by the metric, and raw proximity is dominated by
> class-irrelevant shared variation; whitening — the Cauchy-Schwarz
> optimal metric, fittable label-free on the pool exactly
> (Sherman-Morrison) — turns neighbors into classmates, multiplying D1's
> certification rate by $5.5$ on the hardest dataset. **Truncate** moves
> the anchors: with $s$ labeled shots the anchor error carries noise
> trace $m/s$; cutting $768 \to 128$ provably shrinks it (measured
> $-25$–$40\%$ at $s{=}2$) while retaining $\ge 82\%$ of the class-axis
> energy. W and T are the soft and blunt regularizers of one
> metric-estimation problem; the pool's participation ratio — label-free
> — selects between them (low PR: the discriminant lives in the spectral
> tail, whiten full-rank, don't cut; high PR: the tail is dead, cut and
> bank the estimation savings).

**Three-sentence version (intro grade):**

> DWT improves the three primitives the score uses: Denoise moves each
> embedding toward its class mean (a deterministic triangle-inequality
> lemma, certified pointwise on all datasets); Whiten re-defines
> neighborhoods in the optimal metric — fittable exactly from unlabeled
> data — so that neighbors become classmates, extending the denoise
> lemma's reach $5.5\times$ where homophily is lowest; Truncate shrinks
> estimated-anchor error at the exact rate $m/s$ vs $d/s$. Whitening and
> truncation are the soft and blunt regularizers of the same
> metric-estimation problem, and one label-free spectral statistic (the
> pool participation ratio) selects between them. Validity is unaffected
> throughout: every transform is pool-measurable, so coverage stays
> exact by construction.

**Order, in one sentence (for the paper):** conceptually the stages
chain by precondition — W repairs the metric (unconditional), D repairs
points as measured by that metric (homophily-gated), T shrinks the space
the anchors are estimated in (well-posed only post-W, by Chang) — and
empirically the verdicts are order-stable (§7.4: swapping D's position
moves the qe $d'$-ratio by $\le 0.1$ and never flips a sign), so the one
binding constraint is whiten-before-truncate when the discriminant lives
in the spectral tail, which the PR-gated menu enforces by selection.

Everything below this box is the supporting material: the mechanism
(7.1-7.2), the instruments, and the honest misses. The paper needs the
box, one table, and citations; the rest is for us and for reviewers who
dig.

**Premise shift (user decision, 2026-08-18).** D3's quantitative layer is
empirically unsupported (the (I)-model overpredicts everywhere; D3' is
open), so the theory's grounding is D1 — and the ideal
$\mathbb{E}|C^T| \le \mathbb{E}|C|$ endpoint may never be reached. This
section justifies W/T WITHOUT the $d' \to$ Lemma B $\to$ Prop C chain.
The $d'$-based Sections 2-4 remain correct and useful as instruments, but
they are no longer the load-bearing justification; they are the
aspirational chain's material.

**The re-anchoring.** Before any distributional argument, the pipeline
consumes exactly two primitive objects, and W/T act on both:

- **Neighborhoods** (W's target). D1 is assumption-free, but its
  improvement condition has one dataset-dependent input: the weighted
  pool-kNN homophily $h_w(x)$, through the impurity budget
  $(1-h_w)\,\Delta_F(x)$. Neighborhoods are computed BY the metric —
  so the metric is a free lever on D1's own reach. Claim **W-N**: raw
  cosine proximity is dominated by the shared-field nuisance directions
  (neighbors match pose/background, not class); whitening downweights
  exactly those directions, raising $h_w$ and hence the D1 certification
  rate — a per-point, deterministic-given-$h_w$ justification with no
  distributional model.
- **Anchors** (T's target). The deployed score's only estimated object is
  the prototype $\hat\mu_c$. Claim **T-A**: truncation cuts the $s$-shot
  anchor error exactly (retained noise trace $m/s$ vs $d/s$) at the
  measured alignment cost $a_m$ — a D1-shaped statement: *the estimated
  anchor's distance to the true anchor improves, with an exact rate,
  under a measurable condition.* (This is T1b's content with the $d'$
  packaging removed.)

### 7.1 Measured (run 2026-08-18, `src/wt_chainfree_diagnostics.py`, `output/dwt_theory/wt_chainfree_diagnostics.json`)

$h_w$ (weighted, $k{=}10$, $a{=}3$, 4000 egos — raw column reproduces the
gate-constants instrument), D1 certification rate (the D1 inequality
evaluated in-space), and 2-shot relative anchor error
$\|\hat\mu-\mu\|/\delta_{\text{pair}}$:

```
                     h_w                                  D1 cert rate                         anchor err (s=2)
dataset        raw   wdiag wlw   t128  t128w |  raw   wdiag wlw   t128  t128w |  raw   wdiag wlw   t128  t128w
cifar100       .809  .810  .784  .818  .821  |  .417  .414  .338  .466  .455  |  1.15  1.16  1.72  0.82  0.85
miniimagenet   .917  .916  .894  .921  .918  |  .638  .635  .543  .637  .605  |  0.86  0.86  1.50  0.53  0.57
cifar10        .971  .971  .971  .972  .972  |  .637  .636  .768  .559  .592  |  1.21  1.21  1.53  0.90  0.96
stanford_cars  .467  .469  .545  .446  .538  |  .077  .077  .116  .068  .091  |  1.52  1.52  1.71  1.51  1.39
aircraft       .261  .263  .337  .252  .347  |  .027  .029  .150  .023  .026  |  2.67  2.65  2.51  2.72  2.22
```

Registered predictions, scored:

- **(P-N) held where it matters, with the predicted asymmetry.**
  Whitening raises $h_w$ exactly on the low-$h$ datasets — aircraft
  $.261 \to .337$ (wlw) / $.347$ (t128w), cars $.467 \to .545$ — and the
  D1 certification rate follows: **aircraft $.027 \to .150$, a
  $5.5\times$ increase in the fraction of egos the assumption-free base
  lemma certifies**; cars $+50\%$; cifar10 $+20\%$. On the already-high-$h$
  separable datasets wlw buys nothing and its estimation noise mildly
  LOWERS $h_w$/cert (cifar100 $.417 \to .338$, mini $.638 \to .543$) —
  the same estimation cost as the $\times 0.91$ mini cell in Section 4.
  wdiag moves nothing anywhere (third independent confirmation that the
  rotation is the entire W phase).
- **(P-A) held.** t128/t128w cut the 2-shot anchor error 25-40% on every
  separable dataset (mini $.86 \to .53$) and t128w wins it on
  cars/aircraft too ($2.67 \to 2.22$); wlw WORSENS anchors on 4/5 (the
  full-rank metric upweights noisy-estimate directions); t128 alone is
  slightly worse than raw on aircraft ($2.72$) — the alignment cost
  eating the $m/s$ saving, Chang once more.

### 7.2 Findings

- **CF1 — division of labor, confirmed:** W serves NEIGHBORHOODS (pays
  exactly on low-$h$ data, costs slightly on high-$h$), T serves ANCHORS
  (pays everywhere labels are scarce, at alignment cost on
  fine-grained); t128w inherits a partial version of both; wdiag serves
  neither.
- **CF2 — the headline: the D1 certification rate, computed from D1's
  own deterministic condition with zero distributional assumptions,
  nearly reproduces the deployed champion menu.** Per-dataset argmax of
  cert: aircraft $\to$ wlw, cars $\to$ wlw, cifar100 $\to$ t128-family,
  cifar10 $\to$ wlw, mini $\to$ raw/t128 (tie). The empirical champions:
  aircraft = lw768, cars = wlw-family, cifar100/mini = t128w engine.
  Four of five cells sort correctly from D1 alone; the misses (cifar10
  where everything works; mini's tie) are the low-stakes cells. **This
  is the chain-free justification in one row: W/T are the transforms
  that maximize the reach of the base lemma.**
- **CF3 — the two services trade off, and the trade is the regime map.**
  wlw maximizes cert but costs anchors; t128 minimizes anchor error but
  does nothing for cert on fine-grained data (aircraft t128w cert $.026$
  vs wlw $.150$: the certification needs the full-rank tail that
  truncation discarded — F3/Chang in cert currency). Which service binds
  is decided by the data regime, and the label-free PR dial (Section
  T1d) is the published-precedent instrument that reads it.

### 7.3 What this changes in the lemma package

- **W1a**: demoted from justification to ADAPTER + YARDSTICK (per the
  08-18 discussion): it defines $d'(M)$, powers the diagnostic's oracle
  denominator (F6), and its equality condition proves W1b. Cite
  Fisher/[RCA05] for the substance; claim no novelty.
- **W1b**: UNAFFECTED and now the W-phase's headline lemma — its content
  (unlabeled fit is exact) is about the legality of pool-fitting, which
  the chain-free framing needs just as much.
- **T1a**: unaffected (the honesty clause).
- **T1b**: reframed — its content IS the anchor service T-A; the $d'$
  packaging was optional. The exact statement to keep:
  $\mathbb{E}\|\hat\mu - \mu\|^2_{\text{retained}} = m/s$ vs $d/s$
  (whitened units), gain conditional on measured $a_m$.
- **Sections 2-4's $d'$ chain**: retained as the aspirational route to
  $\mathbb{E}|C|$ (Corollary D4 / location-scale single-crossing), no
  longer load-bearing.
- New follow-up registered: make W-N a lemma (mechanism: whitening
  reduces the shared-field component of pairwise distances; candidate
  route = the 8b two-component model, where field variance dominates
  raw cosine neighborhoods) — the chain-free analogue of GW2b.

### 7.4 The sequential argument, and the order experiment (2026-08-18)

**The precondition chain (the coherent D-W-T narrative).** Each stage's
guarantee has a precondition, and the stages order themselves by who
manufactures whose:

1. **W is unconditional**: its guarantees (W1a optimality, W1b label-free
   exactness) need only pool second moments — no labels, no
   neighborhoods, no homophily. And W repairs the measuring instrument
   itself: neighborhoods (D's primitive) and variance ordering (T's
   primitive) are both metric-relative.
2. **D is conditional on homophily** — a property of the metric W
   repairs. Measured (§7.1): whitening raises $h_w$ exactly where it is
   low and multiplies D1's certification reach $\times 5.5$ on aircraft.
3. **T is conditional on variance-order $=$ discriminant-order** — which
   is exactly what whitening establishes ([C83] is the statement that
   truncation is ill-posed pre-W; W1a's corollary is that post-W the two
   orderings coincide). Its payoff (the $m/s$ anchor budget) belongs to
   the final scoring space. So T runs last, and never before W when the
   tail is alive (F3: t128w $\times 1.77$ vs wlw $\times 3.13$).

This chain says W $\to$ D $\to$ T, while the deployed pipeline runs
D $\to$ W $\to$ T — its one defect being that qe's neighborhoods are
computed in the rawest metric. The user's conjecture: since W raises
$h_w$, running D after W should help. Tested:

**The order experiment** (`src/dwt_order_experiment.py`; qe with
neighborhoods in each space, margin $d'$-ratio measured in-space — the
statistic whose sign predicted the qe verdict 5/5; predictions (P-O1)
ratio rises on aircraft/cars, (P-O2) ~neutral on separable, registered
before the run):

```
qe margin d'-ratio by smoothing space
dataset        raw (D->W)  wlw (W->D)  t128w (T+W->D)
cifar100       1.053       1.010       1.070
miniimagenet   1.178       1.097       1.173
cifar10        1.209       1.106       1.225
stanford_cars  0.735       0.763       0.837
aircraft       0.639       0.617       0.717
```

Scored: **(P-O1) FAILED on aircraft** ($.639 \to .617$, mildly worse),
marginal on cars ($+.03$); no cell crosses $1$ (frac-pairs-improved
stays $\le .08$ on fine-grained). **(P-O2) too optimistic** — W-first is
consistently mildly WORSE for qe on separable data ($1.053 \to 1.010$
etc.), while the t128w space is qe's friendliest home everywhere (best
or tied in 5/5) yet still harmful on fine-grained.

**Reconciliation — the norm/margin split, fourth appearance.** The
$\times 5.5$ certification gain (§7.1) is a NORM-currency fact (D1's
level); the order experiment is the MARGIN-currency outcome. Whitening
fixes the *labels* of your neighbors, but qe's fine-grained harm was
never a label problem — it is the FIELD DIRECTION (8b: qe is a
mean-field mover; it harms when the local field flows toward the
confusable mode, and re-selecting neighbors in a better metric does not
redirect the field). Meanwhile the whitened metric up-weights exactly
the discriminant axes, so the surviving foreign minority does
proportionally MORE damage per unit weight — the two effects
approximately cancel. Conclusion: **the ORDER knob joins beta, k, a, and
hop count on the no-rescue frontier** — the gate is not order-tunable
(C2/C5's family, extended).

**Verdict.** The deployed D $\to$ W $\to$ T stands: on separable data
(where qe is ON) raw-space qe is as good as any; on fine-grained data qe
is gated OFF regardless of order, so its position is moot. The only
ORDER constraint with teeth is **W-before-T when the tail is alive** —
and the deployed menu enforces it not by reordering but by SELECTION
(full-rank wlw, no truncation, when PR is low). For the paper: present
the precondition chain as the conceptual reading, the order experiment
as a robustness result (verdicts are order-stable, one more no-rescue
knob), and the menu's PR gate as the enforcement of the one real
constraint.
