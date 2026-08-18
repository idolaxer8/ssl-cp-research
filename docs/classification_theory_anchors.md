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
- **Simple CNAPS** (Bateni et al., CVPR 2020, arXiv 1912.03432) [added
  08-18] | replaces the meta-learned classifier head with a parameter-free
  Mahalanobis prototype rule; per-class covariance = shrinkage blend
  lambda_k = n_k/(n_k+1) of class-level and task-level estimates; up to
  +6.1% on Meta-Dataset with 9.2% fewer params | **YES as deployed
  precedent** — the few-shot literature's own discovery that
  covariance-corrected (= whitened) prototype metrics on frozen features
  are load-bearing, with the shot-indexed shrinkage = our W1c clause in
  deployed form; its Bayes-rule reading (Mahalanobis = Gaussian
  class-conditional log-likelihood) is the one-line W justification.
- **Transductive CNAPS** (Bateni et al., WACV 2022) [added 08-18] |
  refines means AND covariances with unlabeled query examples via soft
  k-means | **YES as precedent** — unlabeled-data-improved covariance
  estimation = our pool-fit W, minus exchangeability discipline and
  validity; positions our contribution precisely.
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

## 4. Reading guide — the D1-first build order

How this sweep relates to the theory build (framing decision 08-16): **D1 is
the base lemma of the Denoise phase and needs no anchor** — it is
assumption-free and empirically validated pointwise 5/5. Every anchor below
serves a layer built ON TOP of D1: the G2 refinement layer of Denoise, then
W1/T1 (continued from D1 with the same deterministic-mirror-first
structure), then the pipeline-level bridge from margins to downstream
classification error / CP set size. Read in build order; the STARRED items
are the must-reads.

### Stage 1 — Denoise refinement above D1 (G2: variance law + selection bias)

1. ***Kish design effect** (short — one identity): read the deff definition
   and the WEIGHTED deff / n_eff = W^2/sum w^2 subsections (PracTools
   vignette is the fastest modern statement). Goal: cite
   Var = sigma^2*(rho + (1-rho)/k_eff) as the phi law with phi =
   intraclass correlation of neighbor noise.
2. ***Frosio & Kautz 1711.07568**: the sections deriving the expected
   noisy-pair offset E||r-q||^2 = 2*d*sigma^2 and the SNN selection rule
   built on it. Goal: V1 as their "noise-to-noise matching", plus the
   corrected-selection device for the remainder lemma's constructive half.
3. **Noise2Self 1901.11365**: Section 2 (J-invariance definition and the
   self-supervised loss decomposition identity). Goal: the exact identity
   naming our measured 2*beta*(1-beta)*Cov(e_v, nu_v) term as the
   non-J-invariance gap.
4. **Keriven 2205.12156**: the mean/covariance per-step recursions and the
   finite-optimal-steps theorems (middle sections). Goal: one-hop
   optimality (C4) with rigor; his variance-saturation bookkeeping is the
   exogenous-graph version of our field floor.
5. Optional: Baranwal 2102.06966 main separability theorems — read as the
   polished (I)-MODEL template that 8b falsified for self-built graphs
   (cite as contrast, not support); Shen 2506.18761 [v?] main theorem for
   the phi = on-manifold-fraction reframing.

### Stage 2 — W1 (read before T1: T1 is stated post-whitening)

6. ***RCA, Bar-Hillel et al. JMLR 2005**: the three optimality
   derivations (information-theoretic program, log-det-constrained
   within-distance minimization, ML under shared Sigma_w — the middle
   sections of the paper), plus the chunklet-estimation discussion. Goal:
   W1's optimality core; our k-means pseudo-clusters replace chunklets via
   a perturbation term gated by the SAME h dial as D3.
7. ***Fukunaga 1990, ch. 10**: the S_T = S_W + S_B identity and the
   equivalence of the S_T^{-1}S_B and S_W^{-1}S_B eigenproblems (monotone
   eigenvalue map). Goal: the unlabeled-fit unlock — pool TOTAL-covariance
   whitening yields within-class discriminant directions EXACTLY, up to a
   rank-(K-1) rescaling. WCCN (Hatch 2006) is the modern echo.
8. **Ledoit-Wolf JMVA 2004**: Theorems 3.2-3.4 (oracle-optimal shrinkage,
   d/n -> c including c > 1, distribution-free). Goal: the n < d clause
   that makes lw_cluster768 legal; the remaining glue is Sigma^{-1/2}
   continuity.

### Stage 3 — T1 (post-W1 metric)

9. ***Chang JRSS-C 1983** (short): the whole paper — the Mahalanobis
   decomposition Delta^2 = sum_j (gamma_j' delta)^2 / lambda_j and the
   counterexample where the LAST PC carries the separation. Goal: T1's
   failure mode as a population-level theorem; simultaneously the W1
   motivation (whitening makes variance order = discriminant order).
10. ***Paul, Statistica Sinica 2007**: the a.s. limit theorems for spiked
    sample eigenvalues and eigenvectors — especially the eigenvector
    overlap formula (1 - c/(ell-1)^2)/(1 + c/(ell-1)) and the subcritical
    orthogonality branch. Goal: BOTH T1 branches in one theorem — the
    supercritical overlap = the d'^2 multiplier under truncation, the
    subcritical branch = the aircraft failure. BBP math/0403022 only for
    naming the 1+sqrt(c) threshold.
11. ***Loffler-Zhang-Zhou 1911.00538**: the main misclustering theorem
    E[err] <= exp(-(1+o(1))*Delta^2/(8*sigma^2)) and its SNR condition.
    Goal: the end-to-end "unlabeled top-K projection then cluster" bound —
    the closest existing pool-fit-PCA-then-prototype statement, in exactly
    the d' currency.
12. Optional: DGJ 1311.0851 (no spectral shrinker rescues sub-BBP signal =
    truncation harm not tunable) + Gavish-Donoho 1305.5870 (the label-free
    2.858*median threshold — pool-statistic cousin of the PR dial).

### Stage 4 — pipeline -> downstream classification / set size (the bridge)

13. ***Galanti-Gyorgy-Hutter 2112.15121**: the CDNV definition and the
    NCC/prototype error bound theorems (frozen features, unseen classes,
    finite-shot terms). Goal: downstream-classification justification of
    the prototype score with error <= ~1/(2*d'^2) — the published
    container our d' already lives in. Then the directional-CDNV upgrade
    (2603.03530 [v?]) IF it verifies.
14. **Location-scale single-crossing** (Hanoch-Levy 1969 / Meyer 1987 /
    Wong 2006): the single-crossing theorem for mean-up + spread-down
    within a location-scale family. Goal: the G6 distributional route;
    8b's phi ~ 0.8-0.96 makes the margin map near-affine, so the
    shape-invariance premise is nearly exact (O(1-phi) perturbation).
    Compare against Corollary D4 (dwt_denoise_theorem.md Sec 10) and pick
    the cheaper-to-validate assumption.
15. **Zhu 2105.02375**: the UFM global-optimality theorem and the
    self-duality (W ∝ M) consequence; pair with Hui 2202.08384 (collapse
    does not transfer off the train set). Goal: the goal-1 prediction —
    softmax = cosine-prototype AT collapse, gap grows with
    distance-from-collapse; hand to thread A as a measured quantity.
16. **Bobkov-Ledoux Memoirs 2019** (dip in, don't read cover to cover):
    the 1-D quantile representation of Wasserstein distances (early
    chapters). Goal: the assumption-free quantile step — s L-Lipschitz +
    sup||T(x)-x|| <= delta => every score quantile moves <= L*delta, with
    the explicit coupling from our deterministic pointwise smoother.

Minimal path if time-boxed (5 papers): Kish -> Fukunaga ch. 10 + RCA ->
Chang -> Paul -> Galanti. That covers the phi law, W1's two pillars, T1's
two branches, and the downstream bridge.

## 5. Links

Verified-by-abstract during the sweep unless marked [v?] (verify PDF) or
[link?] (link not pinned — locate via the stated venue).

```
Kish deff (modern statement)   https://cran.r-project.org/web/packages/PracTools/vignettes/Design-effects.html
  weighted-deff extension      https://pmc.ncbi.nlm.nih.gov/articles/PMC10426793/
Frosio-Kautz NLM selection     https://arxiv.org/abs/1711.07568
Noise2Self                     https://arxiv.org/abs/1901.11365
Keriven oversmoothing          https://arxiv.org/abs/2205.12156
Baranwal et al. CSBM conv      https://arxiv.org/abs/2102.06966   (+ multilayer https://arxiv.org/abs/2204.09297)
Wang-Baranwal corrected conv   https://arxiv.org/abs/2405.13987
Green et al. Laplacian minimax https://arxiv.org/abs/2106.01529
Andrews common shocks          https://papers.ssrn.com/sol3/papers.cfm?abstract_id=420563
Lee et al. CFH                 https://arxiv.org/abs/2402.04621
Shen et al. local averaging    https://arxiv.org/abs/2506.18761   [v?]
Hein-Maier manifold denoising  http://papers.neurips.cc/paper/2997-manifold-denoising.pdf
RCA (Bar-Hillel et al.)        https://jmlr.org/papers/v6/bar-hillel05a.html
WCCN (Hatch et al.)            https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.html
Ledoit-Wolf 2004               https://www.sciencedirect.com/science/article/pii/S0047259X03000964
Fukunaga 1990 ch.10            (book; no link)
Paul 2007                      https://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n418.pdf
BBP 2005                       https://arxiv.org/abs/math/0403022
Chang 1983                     [link?] JRSS-C 32(3):267-275 — locate via JSTOR; sweep link was wrong
Loffler-Zhang-Zhou             https://arxiv.org/abs/1911.00538
Donoho-Gavish-Johnstone 2018   https://arxiv.org/abs/1311.0851
Gavish-Donoho 4/sqrt(3)        https://arxiv.org/abs/1305.5870
Vempala-Wang 2004              https://www.sciencedirect.com/science/article/pii/S0022000003001806
Bartlett benign overfitting    https://www.pnas.org/doi/10.1073/pnas.1907378117
Green-Romanov PCR              https://arxiv.org/abs/2405.11676
Simple CNAPS                   https://arxiv.org/abs/1912.03432
Transductive CNAPS             https://openaccess.thecvf.com/content/WACV2022/papers/Bateni_Enhancing_Few-Shot_Image_Classification_With_Unlabelled_Examples_WACV_2022_paper.pdf
Jegou-Chum 2012                https://hal.science/hal-00722622
Mu et al. all-but-the-top      https://arxiv.org/abs/1702.01417
Galanti CDNV                   https://arxiv.org/abs/2112.15121  (+ few-shot https://arxiv.org/abs/2212.12532)
Directional CDNV               https://arxiv.org/abs/2603.03530   [v?]
Zhu et al. UFM                 https://arxiv.org/abs/2105.02375
Papyan-Han-Donoho NC           https://arxiv.org/abs/2008.08186
Hui-Belkin-Nakkiran            https://arxiv.org/abs/2202.08384
SCL prototype (NC by design)   https://arxiv.org/abs/2605.20302   [v?]
Sorscher et al. PNAS 2022      https://www.pnas.org/doi/10.1073/pnas.2200800119
Koltchinskii-Panchenko         https://projecteuclid.org/journals/annals-of-statistics/volume-30/issue-1/Empirical-Margin-Distributions-and-Bounding-the-Generalization--Error-of/10.1214/aos/1015362183.full
Bartlett-Foster-Telgarsky      https://arxiv.org/abs/1706.08498
MPM (Lanckriet et al. 2002)    http://eceweb.ucsd.edu/~gert/papers/CSD-02-1218.pdf
Location-scale SD (Wong 2006)  https://www.hindawi.com/journals/ads/2006/082049/abs/
  between 1st/2nd order SD     https://pubsonline.informs.org/doi/10.1287/mnsc.2016.2486
Bobkov-Ledoux Memoirs          https://par.nsf.gov/biblio/10147991-one-dimensional-empirical-measures-order-statistics-kantorovich-transport-distances
Chaudhuri-Dasgupta kNN rates   https://arxiv.org/abs/1407.0067
Cannings et al. local kNN      https://arxiv.org/abs/1704.00642
```

## 6. Follow-ups this sweep creates

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
