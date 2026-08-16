# Classification-theory anchors for the DWT chain (non-CP literature sweep)

Weekly goal 2 (week 08-17, `docs/weekly_summary.md`). Date: 2026-08-16.
Three-vein sweep of GENERAL classification/statistics theory — deliberately
not conformal-prediction papers — mapped onto the lemma slots of
`docs/dwt_theory.md` + `docs/dwt_denoise_theorem.md`. Feeds `literature.md`
after PDF-level verification. All math plain ASCII.

Slots:

```
[D]   Denoise phase (base lemma D1; refinements D2/D3; G2 remainder)
[W]   Whiten lemma W1 (to be written)
[T]   Truncate lemma T1 (to be written)
[P]   prototype-score justification (why cosine-to-class-mean is the right score)
[E]   softmax-vs-cosine equivalence (thread-A / goal-1 theory support)
[M]   margin -> dominance bridge (G6)
[Q]   quantile-Lipschitz step
[phi] two-component noise model / shared-field variance law (G2)
[V]   selection-bias remainder V1/V2 (G2)
[OH]  one-hop optimality (C4)
[PR]  participation-ratio dial precedent
```

Verification status: all entries below were located by web sweep on
2026-08-16; entries marked **[v?]** need the PDF checked before they enter
`literature.md` or a paper draft (recent/obscure items and any whose venue
line the sweep could not pin down). Everything else was corroborated by at
least an abstract page.

---

## 1. The five unlocks (headline synthesis)

1. **[phi] The 8b variance law is a classical identity, not new math.** Kish
   design effect (Kish 1965; weighted deff literature): for k observations
   with equicorrelation rho, Var(mean) = sigma^2 * (rho + (1-rho)/k) —
   EXACTLY our corrected law N^2 = phi + (1-phi)/k with phi = intraclass
   noise correlation, and the weighted version already uses our
   k_eff = W^2/sum w_u^2 (Kish's own effective n). G2's variance half needs
   citation + adaptation, not invention. Backup at asymptotic strength:
   Andrews (Econometrica 2005) — averaging cannot kill a common factor.
2. **[P][M] Our d' already lives inside published frozen-SSL bounds.**
   Galanti-Gyorgy-Hutter (ICLR 2022): NCC/prototype error <= O(CDNV)
   ~ 1/(2 d'^2), distribution-free, for FROZEN features and unseen classes;
   upgraded by directional CDNV (arXiv 2603.03530 **[v?]**) measured along
   the pair axis — the exact sigma_v^2/Delta_pair^2 object of D3. The
   prototype score and the d' currency are both published-theorem objects.
3. **[M] The most promising G6 route: location-scale single-crossing.**
   Hanoch-Levy (1969)/Meyer (1987): within a location-scale family, mean-up
   + spread-down implies the two CDFs cross exactly ONCE and dominance holds
   on the whole tail beyond the crossing — no Gaussianity, only shape
   invariance. And our own 8b finding STRENGTHENS the premise: with the
   margin map g(x_hat) = (1-beta) g(x) + beta g(nu) and corr(g, g(nu)) ~
   0.85-0.90 (phi ~ 0.8-0.96), the map is near-AFFINE in g — an affine map
   is a location-scale change of the same shape, with the iid shell
   perturbing the shape only at order (1-phi). Pairs with the
   contraction-anchor device (Corollary D4 target,
   `dwt_denoise_theorem.md` Section 10): D4 for the pointwise route,
   single-crossing for the distributional route; Marshall-Olkin/MPM
   (Lanckriet et al., JMLR 2002) as the assumption-free worst-case envelope
   (tail mass <= 1/(1+d'^2) from two moments alone).
4. **[T][W] W1/T1 have a ready-made theorem stack.** T1: Paul (2007) is both
   halves in one theorem — supercritical spikes give the explicit
   eigenvector-overlap multiplier on d'^2, subcritical spikes give
   asymptotic ORTHOGONALITY (the aircraft failure as a theorem, sharpened
   by DGJ 2018: NO spectral rule rescues subcritical signal = "harm not
   tunable" for truncation); Chang (1983) gives the population-level
   failure functional (discriminant contribution (gamma_j' delta)^2 /
   lambda_j — the 1/lambda weighting FAVORS low-variance directions);
   Loffler-Zhang-Zhou (2021) is the end-to-end unlabeled-projection ->
   cluster bound in exp(-d'^2/8) currency. W1: RCA (Bar-Hillel et al.,
   JMLR 2005) gives three closed-form optimality characterizations of
   within-class whitening; the Fukunaga total-vs-within identity
   (S_T = S_W + S_B; same discriminant eigenvectors, monotone eigenvalue
   map) converts "pool whitening approximates within-class whitening" into
   an EXACT statement + rank-(K-1) correction — the cleanest unlabeled-fit
   justification in the sweep; Ledoit-Wolf (2004) licenses full-rank
   whitening at n < d distribution-free.
5. **[E] Goal-1 (softmax ablation) has a theory prediction before the
   experiment runs.** Zhu et al. (NeurIPS 2021, unconstrained-features
   model): at the global optimum, self-duality W ∝ M makes the softmax rule
   LITERALLY the cosine-prototype rule; Hui-Belkin-Nakkiran (2022) shows
   collapse does not transfer off the training set — so the prediction is
   softmax ~= plain -cos(mu_c, x) where measured collapse/homophily is high
   (cifar10/mini), with the gap growing as geometry departs from collapse
   (aircraft). Thread A's ablation is a direct test of this; if confirmed,
   the theory covers the deployed score with a quantified equivalence gap.

---

## 2. Ranked anchors by slot

Format: citation | statement (compressed) | transplant verdict.

### [D]/[phi]/[V] — Denoise: G2 remainder ingredients

- **Kish design effect** (Kish, Survey Sampling 1965; weighted-deff
  literature) | Var(weighted mean) under equicorrelation rho =
  sigma^2 (rho + (1-rho)/k_eff), k_eff = W^2/sum w^2 | **YES** — pure
  second-moment algebra; the modeling step (noise = cluster effect + iid)
  IS the two-component model G2 wants. THE [phi] anchor.
- **Frosio & Kautz** (arXiv 1711.07568; IEEE TIP 2019) | NLM "noise-to-noise
  matching": proximity-selection by a noisy reference tilts selected noises
  toward the reference's own noise; corrected rule selects the
  E||r-q||^2 = 2 d sigma^2 shell instead of the minimum | **PARTIAL-YES** —
  V1 verbatim + a constructive corrected-selection device; iid-Gaussian
  noise model covers the shell component of Cov(e,nu), not the field.
- **Batson & Royer, Noise2Self** (ICML 2019, arXiv 1901.11365) |
  J-invariance: any denoiser that uses the ego's own coordinates gains
  apparent loss by copying its noise; the gap term is exactly the
  cross-covariance | **YES as identity/naming** — our kNN mean excludes the
  ego vector but SELECTS with it, hence not J-invariant; the 8b-measured
  2 beta (1-beta) Cov(e_v, nu_v)/sigma_v^2 = 0.77-0.91 term is the
  Noise2Self gap. No rates.
- **Shen et al.** (arXiv 2506.18761, 2025) **[v?]** | local averaging under
  high ambient noise concentrates near the manifold; selection by noisy
  centers handled rigorously | **PARTIAL** — reframing: phi = ON-MANIFOLD
  fraction of within-class variance (field = manifold variation, shell =
  ambient noise). Candidate for the cleanest formal shape of G2's model.
- **Andrews** (Econometrica 2005) | common shocks survive averaging;
  innocuous-vs-fatal dichotomy for factor-design correlation | **PARTIAL**
  — the asymptotic backup for the phi floor; maps to field-toward-own-mode
  vs field-toward-confusable split.
- **Lee et al., CFH** (ICML 2024, arXiv 2402.04621) | class-controlled
  feature homophily: benefit of graph convolution is mediated by residual
  (class-removed) feature dependence between neighbors | **PARTIAL** — the
  graph-learning literature's own discovery of the shared field; CFH
  protocol = a phi-instrument candidate (label-free-ish surrogate).

### [OH] — one-hop optimality / operator corrections

- **Keriven** (NeurIPS 2022, arXiv 2205.12156) | iterated mean aggregation:
  variance reduction saturates while signal decays geometrically; a finite
  small number of steps provably helps, then harms | **PARTIAL** —
  exogenous graph; the per-hop bookkeeping is C4's rigorous template.
- **Wang-Baranwal-Fountoulakis** (NeurIPS 2024, arXiv 2405.13987) |
  removing the principal (degree/hub) eigenvector makes EACH convolution
  round help up to saturation | **PARTIAL** — theory twin of our confirmed
  hub-debias gamma~1 upgrade; sharpens C4 into a dichotomy (one hop optimal
  for the UNCORRECTED operator only).
- **Baranwal-Fountoulakis-Jagannath** (ICML 2021, arXiv 2102.06966; ICLR
  2023 arXiv 2204.09297) | CSBM: convolution shrinks noise sd by sqrt(deg),
  means by the homophily factor | **PARTIAL** — polished template for D3,
  but assumes exactly (I) (graph independent of feature noise); anchors the
  (I)-model side, NOT the correction — cite as the model 8b falsified for
  self-built graphs.
- **Green-Balakrishnan-Tibshirani** (AISTATS 2021, arXiv 2106.01529) |
  Laplacian/Tikhonov smoothing on data-built neighborhood graphs is minimax
  over Sobolev, manifold-adaptive; alpha-QE = one step on this objective |
  **PARTIAL** — the only minimax pedigree for the QE objective; errors on
  responses only.

### [W] — Whiten lemma W1

- **Bar-Hillel et al., RCA** (JMLR 2005) | within-class (chunklet)
  whitening = closed-form optimum of three programs (max mutual info;
  min within-distance under log-det budget; ML under shared Sigma_w);
  chunklets need groupings, not labels | **YES** — primary W1 anchor; the
  W1 work item = perturbation statement replacing chunklets by k-means
  pseudo-clusters (impurity term controlled by the SAME homophily h as D3
  — one dial, both lemmas).
- **Fukunaga 1990 ch. 10 identity** (+ WCCN, Hatch et al. 2006) |
  S_T = S_W + S_B; S_T^{-1} S_B and S_W^{-1} S_B share eigenvectors with
  monotone eigenvalue map lam_T = lam_W/(1+lam_W); Bayes rule after
  Sigma_w-whitening is the nearest-prototype rule (Fisher) | **YES** — the
  unlabeled-fit unlock: pool TOTAL-covariance whitening provably yields the
  within-class discriminant directions exactly, up to a bounded rank-(K-1)
  rescaling. Converts our approximation claim into identity + correction.
- **Ledoit-Wolf** (JMVA 2004) | linear shrinkage toward mu*I converges to
  the oracle combination in quadratic loss for d/n -> c INCLUDING c > 1;
  well-conditioned, distribution-free | **YES** — licenses champion
  lw_cluster768 at n_c << d; remaining step: compose quadratic-loss
  guarantee with Sigma^{-1/2} continuity for the whitened d'.
- **Jegou-Chum** (ECCV 2012) + **Mu et al. all-but-the-top** (ICLR 2018,
  arXiv 1702.01417) | for cosine scores, top-variance directions are
  bursty/frequency nuisance; whitening is the fix | **NO as theorem,
  YES as motivation** — the sweep confirms the cosine-specific whitening
  slot has no rigorous representative (open niche, matches the alpha-QE
  zero-theory finding).

### [T] — Truncate lemma T1

- **Paul** (Statistica Sinica 2007) | spiked model, d/n -> c: supercritical
  spikes (ell > 1+sqrt(c)) give sample-eigenvector overlap
  (1 - c/(ell-1)^2)/(1 + c/(ell-1)); subcritical spikes are absorbed and
  the sample PC is asymptotically ORTHOGONAL to truth | **PARTIAL** — both
  T1 halves in one theorem; c = 768/5000 ~ 0.15 is honestly in-regime; read
  AFTER whitening (isotropic-bulk assumption), which matches the pipeline
  order — T1 should be stated post-W1.
- **BBP** (Ann. Prob. 2005, math/0403022) | the detectability threshold
  ell = 1 + sqrt(c) itself | **PARTIAL** — names the constant; subcritical
  class-signal is INVISIBLE to any pool-spectrum procedure (so pool-fit
  truncation is not at fault on aircraft; only full-rank W can keep it).
- **Chang** (JRSS-C 1983) + Jolliffe Sec 9.1 | discriminant contribution of
  PC j = (gamma_j' delta)^2 / lambda_j: eigenvalue order bears NO relation
  to discriminant order; low-variance PCs can dominate | **YES** —
  population-level, assumption-light; simultaneously the T1 failure
  certificate and the W1 motivation (whitening makes variance order =
  discriminant order).
- **Loffler-Zhang-Zhou** (Ann. Stat. 2021, arXiv 1911.00538) | unlabeled
  top-K SVD projection + k-means achieves minimax
  E[err] <= exp(-(1+o(1)) Delta^2 / (8 sigma^2)) | **YES** — closest
  existing "pool-fit PCA-K then prototype classification with explicit SNR
  condition"; isotropic-within is the known idealization.
- **Donoho-Gavish-Johnstone** (Ann. Stat. 2018, arXiv 1311.0851) +
  **Gavish-Donoho 4/sqrt(3)** (IEEE-IT 2014) | optimal spectral shrinkers
  collapse subcritical eigenvalues — no spectral rule rescues sub-BBP
  signal; data-driven label-free hard threshold 2.858 * median SV |
  **PARTIAL** — strongest form of "truncation harm not tunable"; the
  data-driven threshold is a pool-statistic cousin of the PR dial.
- **Vempala-Wang** (JCSS 2004) | rank-K SVD subspace of unlabeled mixture
  data contains the span of component means (spherical within) | **PARTIAL**
  — why pool PCA finds class means without labels; its spherical-within
  clause failing IS the aircraft regime. Cite for both directions.

### [P]/[E] — prototype score and softmax equivalence

- **Galanti-Gyorgy-Hutter** (ICLR 2022, arXiv 2112.15121) | NCC error on
  frozen features <= O(CDNV), CDNV ~ 1/(2 d'^2), distribution-free, unseen
  classes, finite-shot terms | **YES** — primary [P]; gap: full-trace
  variance, not axis-projected.
- **Directional CDNV** (arXiv 2603.03530, 2026) **[v?]** | CDNV along the
  decision axis collapses in SSL even when isotropic CDNV does not;
  non-asymptotic prototype bounds with directional leading term | **YES if
  verified** — the axis-projected upgrade = our sigma_v object exactly.
- **Zhu et al.** (NeurIPS 2021, arXiv 2105.02375) | UFM global optima
  satisfy NC1-NC3; self-duality W ∝ M => softmax rule = cosine-prototype
  rule at the optimum | **PARTIAL** — [E] primary; our features are NOT at
  that optimum (aircraft = anti-collapse), so use as "rules coincide in the
  collapse limit; the gap is measurable distance-from-collapse".
- **Papyan-Han-Donoho** (PNAS 2020) | NC1-NC4; NC4 = trained softmax
  decisions -> nearest-class-center | **PARTIAL** — canonical geometry
  citation; train-set phenomenon (see next).
- **Hui-Belkin-Nakkiran** (arXiv 2202.08384) | NC1 largely fails to
  transfer to test data; collapse = optimization phenomenon | **YES as
  caveat** — the published license for measuring (h_w, d', sigma_v) instead
  of assuming collapse; explains aircraft under a "collapsed" backbone.
- **SCL prototype result** (arXiv 2605.20302, 2026) **[v?]** | supervised-
  contrastive loss implicitly fits the class-mean cosine classifier
  throughout training; linear probe redundant | **PARTIAL-YES if verified**
  — strongest [P] for contrastive-family encoders; DINOv2's loss is not
  SCL (analogy, not theorem).
- **Sorscher-Ganguli-Sompolinsky** (PNAS 2022) | prototype few-shot error
  ~= H(SNR) with SNR built from pair distance, radii, PARTICIPATION RATIO,
  and signal-noise overlap, validated on frozen deep embeddings |
  **PARTIAL** — [P] + [PR]: PR natively inside a prototype-cosine error
  formula; asymptotic (CLT), not a bound. Precedent for the norm-vs-margin
  lesson (only axis-projected variance hurts).

### [M]/[Q] — margin -> dominance bridge (G6) and quantile step

- **Location-scale single-crossing** (Hanoch-Levy 1969; Meyer 1987; Wong
  2006) | within a location-scale family, mean-up + spread-down => exactly
  one CDF crossing => first-order dominance on the tail side | **PARTIAL,
  most promising** — needed assumption = shape invariance under smoothing;
  8b's near-affine margin map (corr ~ 0.85-0.90) makes it near-exact with
  an O(1-phi) shape perturbation. THE G6 distributional route.
- **Marshall-Olkin one-sided Chebyshev via MPM** (Lanckriet et al., JMLR
  2002) | sup over all distributions with given (mu, Sigma) of
  P[a'x >= b] = 1/(1+delta^2), delta = standardized margin | **PARTIAL** —
  envelope dominance (worst-case, not realized CDFs); assumption-free
  one-sided lemma: d' up => certified overlap bound down. Complements D4.
- **Koltchinskii-Panchenko** (Ann. Stat. 2002) + **Bartlett-Foster-
  Telgarsky** (NeurIPS 2017) | error bounded by margin-CDF functional,
  monotone under CDF dominance; margins scale as L*delta under Lipschitz
  maps | **PARTIAL** — formalism citation (fixed score => complexity terms
  vanish benignly); [Q] composition template.
- **Bobkov-Ledoux** (Memoirs AMS 2019) + Lipschitz pushforward | 1-D
  W_inf(F,G) = sup-quantile gap; s L-Lipschitz and pointwise map with
  sup||T(x)-x|| <= delta => every score quantile moves <= L*delta, explicit
  coupling | **YES** — [Q] primary; assumption-free; prototype-cosine is
  2-Lipschitz on the sphere; ||T_D(x)-x|| <= beta||nu-x|| is measurable
  per-point. The cleanest published form of Lemma B (5a) -> quantile step.

### Baseline vocabulary (kNN classification proper)

- **Chaudhuri-Dasgupta** (NeurIPS 2014) | kNN risk via effective-boundary
  mass | **PARTIAL** — vocabulary: the sub-h* harm region = effective
  boundary mass; consonant with D3 scale-freeness.
- **Cannings-Berrett-Samworth** (Ann. Stat. 2020) | local k(x) from
  pool-estimated density attains minimax; semi-supervised version |
  **PARTIAL** — theorem-grade precedent for a label-free LOCAL gate
  (their density-tuned k(x) ~ our h_w(x)-gated qe).

---

## 3. Best anchor per slot (the table for literature.md)

```
slot                       primary anchor                       backup
[phi] variance law         Kish design effect (1965)            Andrews 2005; CFH 2024
[V] selection remainder    Frosio-Kautz 2017/2019               Noise2Self 2019 (identity); Shen 2025 [v?]
[OH] one hop               Keriven 2022                         Wang-Baranwal 2024 (corrected op)
[W] whitening              RCA 2005 + Fukunaga S_T=S_W+S_B      Ledoit-Wolf 2004 (n<d clause)
[T] truncation gain        Paul 2007 (supercritical) + LZZ 2021 DGJ 2018 (optimal shrinkage)
[T] truncation failure     Chang 1983 + Paul 2007 (subcritical) BBP 2005 (threshold); DGJ (not tunable)
[P] prototype score        Galanti CDNV 2022 (+directional [v?]) Sorscher PNAS 2022; SCL [v?]
[E] softmax-vs-cosine      Zhu 2021 (UFM self-duality)          PHD 2020 NC4; Hui 2022 (caveat)
[M] G6 bridge              location-scale single-crossing       Marshall-Olkin/MPM; D4 device (Sec 10)
[Q] quantile step          Bobkov-Ledoux + Lipschitz pushforward KP 2002 formalism
[PR] dial precedent        Bartlett 2020 R_k; Sorscher 2022     Green-Romanov 2025 (PCR)
```

## 4. Follow-ups this sweep creates

1. **Verify the [v?] items** (directional CDNV 2603.03530; SCL 2605.20302;
   Shen 2506.18761) at PDF level before any of them enters a draft.
2. **W1 concrete plan**: state W1 = RCA optimality + Fukunaga identity
   (pool-total whitening = within-class directions + rank-(K-1) term) +
   LW clause for n < d; the pseudo-cluster impurity perturbation reuses the
   D3 homophily dial. T1 = Chang functional + Paul two-branch, stated
   post-W1; PR enters via Bartlett R_k. (Goal-3 work item.)
3. **G6 route decision**: D4 contraction-anchor (pointwise) vs
   location-scale single-crossing (distributional) — draft both one page
   each, pick by which assumption is cheaper to validate empirically
   (anchor slack tau vs shape invariance).
4. **Goal-1 prediction registered**: softmax ~= -cos gap should grow with
   distance from collapse (Zhu/Hui) — hand to thread A as the thing to
   measure, not just a win/lose ablation.
5. **Open niche confirmed twice**: no rigorous whitening-for-cosine theorem
   exists (Jegou-Chum slot), matching the earlier "alpha-QE lit has ZERO
   theory" finding — both are publishable-gap statements for the paper.
