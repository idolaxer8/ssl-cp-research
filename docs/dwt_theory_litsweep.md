# DWT theory — literature base (sweep 2026-08-10)

Research base for the week-08-10 main goal (see weekly_summary.md): a
DAPS-Theorem-2 analogue justifying the DWT pipeline (Denoise -> Whiten ->
Truncate, all pool-fit) where the intervention is on the REPRESENTATION, not
the score. Compiled from the Feature CP deep read + a 4-bucket web sweep.
All math plain ASCII per repo convention.

## The template — DAPS (Zargarbashi, Antonelli, Bojchevski, ICML 2023)

"Conformal Prediction Sets for Graph Neural Networks", PMLR v202.

- Diffusion: s_hat(v,y) = (1-lam)*s(v,y) + (lam/|N_v|) * sum_{u in N_v} s(u,y).
- Prop 2 (exchangeability): permutation-equivariant transforms of an
  exchangeable score matrix stay exchangeable (H_hat = (1-lam)H + lam D^-1 A H).
- **Theorem 2**: pi_i = model approx of true conditional p_i, diffused
  pi_hat_i = (1-lam)*pi_i + (lam/|N_i|)*sum_{j in N_i} pi_j. If the graph
  satisfies A_ij = 1 iff ||p_i - p_j||_TV <= Delta, then diffusion improves
  the error eps_i = ||pi_i - p_i|| (i.e. ||pi_hat_i - p_i|| < eps_i) if
  (1/|N_i|) * sum_{j in N_i} eps_j + Delta < eps_i.
- Fine print that licenses our generalization: footnote 5 (theorem is in
  probability space, score version "easy but cumbersome") and Sec 5.3 (kNN
  bridge: for a kNN feature graph and lam=1, diffusion = kNN label average,
  minimax-consistent via Bahri & Jiang 2021; "we leave it for future work to
  theoretically characterize this setting"). Our theorem fills that gap one
  level earlier — on the representation.

## Bucket 1 — CP in feature/representation space

**1.1 Teng, Wen, Zhang, Bengio, Gao, Yuan — "Predictive Inference with
Feature Conformal Prediction", ICLR 2023, arXiv 2210.00173.** THE paper the
instructor recalled; deep-read 08-10 (PDF in session notes).
- Score in feature space via surrogate features:
  s(X,Y, g o f) = inf_{v: g(v)=Y} ||v - f(X)|| (computed by gradient descent,
  Alg 2); band mapped back by LiRPA relaxation (Band Estimation) or
  membership testing (Band Detection).
- Thm 5: coverage >= 1-alpha under exchangeability alone (feature-space
  scoring costs nothing in validity).
- **Thm 6 (efficiency)**: with H the feature->output length operator, Holder
  |H(v,X)-H(u,X)| <= L|v-u|^a, Feature CP gives strictly shorter expected
  bands, E[H(Q_{1-alpha}(V^f_cal), X')] < Q_{1-alpha}(V^o_cal), under the
  three "cubic conditions":
  1. Length preservation: E Q_{1-alpha}(H(V^f_D, D)) < E Q_{1-alpha}(V^o_D) + eps.
  2. Expansion (the core): L * E M|Q_{1-alpha}(V^f_D) - V^f_D|^a
     < E M[Q_{1-alpha}(H(V^f_D,D)) - H(V^f_D,D)] - eps - 2max{L,1}(c/sqrt(n))^min{a,1}
     — mean deviation of scores from their quantile is smaller in feature
     space; "the quantile step costs less" where scores concentrate.
  3. Quantile stability: E|Q_{1-alpha}(V_D) - Q_{1-alpha}(V_cal)| <= c/sqrt(n)
     in both spaces.
- Supporting: Thm 9 (band-length variance lower bound: nonlinear head =>
  genuinely per-sample adaptive bands), Thm 15 (O(1/n) width convergence).
- They VALIDATE the cubic conditions empirically (Table 5: mean
  |score - quantile| per space; Fig 7: "cubic metric" tracks length) — a
  diagnostic to mirror pre/post DWT.
- Verdict for us: proof ARCHITECTURE transfers (quantile-cost comparison +
  stability), statement does not (regression bands; feature-vs-output space
  of ONE model with a decoder head; ours = one distance-score family on TWO
  representations, classification; no head).

**1.2 FFCP — Tang, Wang, Wen, Teng, arXiv 2412.00653 (NeurIPS 2025).**
Taylor-collapsed FCP: s_ff = |Y - f(X)| / ||grad g(v_hat)||. Thm 5: shorter
than vanilla under "square conditions" (cubic minus length preservation).
Evidence the line is extendable; no new technique for us.

**1.3 PLCP — Kiyani, Pappas, Hassani, ICML 2024, arXiv 2404.17487.**
Learned soft partition h: X -> m groups + per-group thresholds;
MSCE(C_inf) <= 4L*sqrt(var(q_{1-alpha}(X))/(m-1)) + finite-sample version.
"Feature" = conditioning structure, not representation for a distance score.
Peripheral.

**1.4 Contrastive Conformal Sets (arXiv 2603.26261)**: learnable hyper-balls
in contrastive space; NO theorem linking representation quality to size.
**1.5 COMPASS (arXiv 2509.22240)**: FCP machinery for segmentation metrics;
perturbs latents along dominant (PCA-like) directions; inherits Teng theory.

**Bottom line: nobody has theory for "pool-fit representation transform +
distance-based conformal score". The niche is open.**

## Bucket 2 — expected set size as a functional of the score distribution

**2.1 Dhillon, Deligiannidis, Rainforth — AISTATS 2024, arXiv 2306.07254.**
- **Thm 1 (exact identity)**: split CP, n iid cal scores,
  E|C_alpha(X_{n+1})| = integral_r P_B(n, F_R(r))(n_alpha) * #_R(r) dr,
  F_R = true-label score CDF, P_B(n,p)(k) = Binomial CDF,
  n_alpha = ceil((1-alpha)(n+1)) - 1, #_R(r) = label-multiplicity factor.
- **Comparison license (Sec 4.1)**: holding #_R fixed, first-order stochastic
  dominance of true-label score CDFs => ordering of expected set sizes, at
  ANY fixed n and alpha, no asymptotics. This is the Step-C conversion lemma:
  "DWT shifts score distributions favorably" => "smaller expected sets".
- Assumptions: iid cal scores (our split arm directly; FCP arm needs the
  transductive analogue — open but same structure).

**2.2 Sadinle, Lei, Wasserman — JASA 2019, arXiv 1609.00451.**
- Thm 1: oracle least-ambiguous sets = threshold p(y|x) at its alpha-quantile.
- **Thm 14 (plug-in)**: sup_x |p_hat - p| <= eps_n w.h.p. + margin condition
  near the threshold (c1|s|^gamma <= |G_y(t_y+s)-G_y(t_y)| <= c2|s|^gamma)
  => P(H_hat symdiff H*) <= c*(eps_n^gamma + K*sqrt(log n / n)).
  Quantitative "closer to p(y|x) => sets closer to smallest valid". Our
  prototype-cosine softmax is literally a plug-in p_hat.

**2.3 Romano, Sesia, Candes — APS, NeurIPS 2020, arXiv 2006.02544.** Oracle
APS = smallest sets with CONDITIONAL coverage (exact, randomized). Cite for
adaptivity; Sadinle for marginal-size optimality. No quantitative plug-in
bound.

**2.4 Conrad, Isaev, Belomestny, Moulines, Samsonov — arXiv 2608.06206
(2026), localized CP finite-sample guarantees.**
- **Thm 6 (learned score)**: | |C_hat(x)| - |C_oracle(x)| | <~
  L_len(x) * [ (A_cal + eps_inf)/kappa + Delta_{S*}(x) ], with eps_inf =
  uniform score-estimation error, Delta_{S*}(x) = sup_y |S_hat - S*| —
  **set length is within LINEAR-in-score-error of oracle length at fixed
  coverage**. The ready-made Step-B->C converter. Assumptions: Holder
  conditional score CDF, density minorization (kappa).

**2.5 Behboodi, Correia, Massoli, Louizos — arXiv 2509.04631, transductive
efficiency bounds.**
- Thm 3.5/3.6: n*gamma_n ~ n*H(Y|X) + sqrt(n)*sigma*Qinv(alpha) (gamma_n =
  (1/n) log E|set|), oracle-achievable.
- **Thm 3.7**: scoring with Q instead of true P inflates the bound by
  exactly n*E[KL(P || Q)] — inefficiency LINEAR in the KL misspecification.
  TRANSDUCTIVE: best structural match for the FCP arm.

**2.6 Correia et al., NeurIPS 2024, arXiv 2405.02140**: H(Y|X) <=
f(inefficiency) bounds (wrong direction for us; foundation of 2.5).

**2.7 Denis & Hebiri, JMLR 2017, arXiv 1608.08783**: oracle sets at fixed
expected size beta; **Thm 2: thresholds estimated on an UNLABELED pool of
size N cost excess risk ... + C'*K/sqrt(N)** — classical precedent for
"pool-fit components cost only O(1/sqrt(N_pool))".

**2.8 Luo & Zhou, arXiv 2407.10230**: weighted averaging over d score
FUNCTIONS; efficiency <= best-in-class + O(sqrt(d log|I|/|I|)). Same proof
pattern as "fit on independent data, pay concentration, keep validity".

## Bucket 3 — feature-space smoothing provably denoises under homophily

**3.1 Baranwal, Fountoulakis, Jagannath — ICML 2021, arXiv 2102.06966
(+ 2023 multi-layer follow-up). THE Step-A anchor.**
- CSBM (2-class SBM, intra p / inter q, Gaussian features, means mu/nu,
  var sigma^2, expected degree D): graph convolution Xtilde = D^-1 A X
  shrinks within-class noise sigma -> ~sigma/sqrt(D); the linear-separability
  threshold on ||mu - nu|| improves by ~1/sqrt(D). Assumptions: degree
  omega(log^2 n), homophily gap Gamma = (p-q)/(p+q) bounded from 0.
- Our transplant (pool-kNN graph, qe weights): for
  x_hat = (1-lam)x + lam*mean_k(pool-kNN), mixture model with kNN homophily
  h and inter-class mean shift b gives within-class variance
  ((1-lam)^2 + lam^2/k)*sigma^2 + lam^2*(1-h)^2*b^2-type bias — a
  bias-variance crossover that PREDICTS the empirical h >= ~0.7 gate.

**3.2 NT & Maehara — arXiv 1905.09550.** X = Xbar + Z, Xbar of graph
frequency <= eps; **Lemma 5**: ||Xbar - A_rw^k X||_D <=
sqrt(k*eps)*||Xbar||_D + O(sqrt(log(1/delta))*R(2k))*E||Z||_D — explicit
bias (grows with smoothing) + variance (shrinks with smoothing) split;
Cor 6: optimal #passes; Thm 7-8: downstream model on smoothed features
matches the model on TRUE features up to Otilde(sqrt(eps))*noise. The SHAPE
of our D-stage inequality, with "low graph frequency" = homophily.

**3.3 Li, Han, Wu — AAAI 2018, arXiv 1801.07606.** GCN = Laplacian
smoothing + over-smoothing theorem (iterated smoothing kills all class
info). NO quantitative gain bound; cite for mechanism naming + the lam<1 /
one-shot warning.

**3.4 Ma et al. — CIKM 2021, arXiv 2010.01777.** Aggregation ~ solving
min_F ||F - X||^2 + c*tr(F' L F): qe = one gradient/Jacobi step of MAP
denoising under a cluster prior on the pool-kNN Laplacian. Variational
framing only, no bound.

**3.5 Bahri & Jiang — ICML 2021, arXiv 2102.05140.** **Thm 1**: kNN label
average eta_k has sup_x |eta_k(x) - eta(x)| <= C*((k/n)^{alpha/D} + 1/sqrt(k))
under Holder-alpha eta, density bounded below; optimal k gives the minimax
uniform rate. The population target of pool-kNN smoothing; the exact bridge
DAPS Sec 5.3 already invokes.

## Bucket 4 — post-DAPS graph/similarity CP (2023-2026): the gap is open

- **SNAPS (NeurIPS 2024, arXiv 2405.14303)**: adds feature-similarity kNN +
  graph neighbors to score aggregation. Prop 1 = exchangeability. **Prop 2 =
  its ONLY efficiency statement: if ALL aggregated nodes share the ego's
  true label then E|C_SNAPS| <= E|C_APS| — a perfect-homophily (h=1,
  Delta=0) corner of DAPS Thm 2, no impurity tolerance, no error budget.**
  The feature-similarity analogue of DAPS's Delta condition does not exist
  in the literature.
- CF-GNN (NeurIPS 2023, arXiv 2305.14535): learned topology-aware score
  corrector, size-loss trained — no efficiency theorem.
- SemiCP (arXiv 2505.21147): pool augments CALIBRATION via NNM pseudo-scores;
  coverage gap O(1/sqrt(N)) + unbounded matching-error term; no size theorem.
  The competing pool use (augment cal) vs ours (transform representation).
- Maneriker et al. (TMLR 2025, arXiv 2409.18332): graph-CP benchmarking +
  transductive evaluation notes — protocol citations only.
- GraphLCP (arXiv 2605.08074), Conformal Inductive GNNs (ICLR 2024):
  coverage-only.

## W and T stage backers (flag list)

- **Ledoit & Wolf 2004** (J. Multivar. Anal.): shrinkage covariance =
  Frobenius-risk-optimal in the class rho1*I + rho2*S, well-conditioned —
  backs LW whitening.
- **Bar-Hillel et al. 2005 (RCA, JMLR)**: within-chunklet whitening is the
  ML-optimal Mahalanobis metric under a Gaussian model and maximizes mutual
  information under distance constraints — cluster-conditional pool
  whitening = unsupervised RCA. The citable "whitening provably helps".
- LMNN (Weinberger & Saul 2009): influential, but no clean error theorem —
  metric-learning theory for kNN is thin; RCA is the anchor.
- **alpha-QE (Radenovic et al., TPAMI 2019; Gordo et al. 2020)**: ZERO
  theory — D-stage theory must come from Bucket 3, not retrieval lit.
- **Spiked covariance**: Gavish & Donoho 2014 (optimal SV hard threshold
  4/sqrt(3)); Donoho, Gavish, Johnstone 2018 (below the BBP transition,
  sample eigendirections carry no signal — discarding is optimal); uniform
  PCA-denoising bound arXiv 2306.12690. T stage = textbook-optimal denoiser
  under rank-r spike + isotropic noise.

## Ranked anchors (directness for our proof)

1. DAPS Thm 2 — the template; its own footnote 5 + Sec 5.3 invite exactly
   our generalization.
2. Baranwal et al. 2021 — Step A: feature averaging provably shrinks
   within-class noise under a cluster model; bias = (1-h) term predicts the
   0.7 gate.
3. Conrad et al. 2026 Thm 6 — Step C: set length within linear-in-score-
   error of oracle length at fixed coverage.
4. Dhillon et al. 2024 Thm 1 + dominance corollary — Step C alternative:
   exact E|C| identity; stochastic dominance of score CDFs => size ordering,
   finite-n, assumption-light.
5. Sadinle et al. 2019 Thm 1+14 — plug-in scores: sets within
   c*(eps^gamma + K*sqrt(log n/n)) of oracle least-ambiguous sets.

Honorable mentions: NT & Maehara Lemma 5 (D-stage inequality shape),
Behboodi Thm 3.7 (transductive KL-linear penalty — FCP arm), Bahri & Jiang
Thm 1 (kNN bridge), SNAPS Prop 2 (documents the gap), Denis & Hebiri Thm 2
(K/sqrt(N_pool) pool price), Feature CP Thm 6 (quantile-cost architecture +
empirical condition-validation protocol).

## Structure of the DWT map (linearity audit, 08-10; from code)

Is DWT linear? No — but it decomposes as (nonlinear conditionally-convex
smoother) then (one global affine map), and each half is usable as-is.

- **W + T = ONE global affine map.** `UnlabeledTransform.transform` = PCA
  projection then elementwise multiply by a FIXED inv_std vector:
  T_WT(x) = A x + b, A = D_w P'. Clusters enter only the FIT of the diagonal
  (pooled within-cluster residual variance); application is cluster-free.
  Exact consequences for the proof: prototypes transform covariantly
  (empirical mean commutes with affine), Gaussian mixtures closed under the
  map (noise covariance -> A Sigma A', where spiked-cov/RCA anchors plug
  in), whitened distance = fixed quadratic metric.
- **D (qe) is nonlinear** four ways: input L2-norm, kNN selection vs pool
  (piecewise / k-th-order Voronoi), cos^alpha weights (alpha=3, x-dependent
  within a cell), output L2-norm. NOT fatal: DAPS Thm 2 never uses
  linearity of the map — it bounds a CONVEX combination with fixed
  coefficients by triangle inequality. Same move works conditionally
  (pool independent of cal/test => condition on realized neighbors+weights):
  ||x_hat - mu_y|| <= (1-lam)||x - mu_y|| + lam*[avg same-class neighbor
  error + (1-h(k))*Delta_mu + r_k(x)], i.e. the nonlinearity converts into
  exactly two extra terms — the impurity bias (1-h(k))*Delta_mu (the DAPS
  Delta analogue, WANTED: derives the gate) and the selection-locality bias
  r_k ~ (k/N_pool)^{1/d_eff} (neighbors are the closest points, noise tilts
  toward x; vanishes with pool size at the Bahri-Jiang rate; needs its OWN
  lemma — the term a naive linear-smoother analysis silently drops).
  alpha>0 weights stay nonnegative => convexity intact; effect = replace k
  by k_eff = (sum w)^2 / sum w^2 in the variance term. Sphere
  normalizations free: embeddings pre-L2-normed + both scores
  scale-invariant.
- **Code alignment**: the `qe_beta` knob (T(x) = (1-b)x + b*neighbor-mean)
  with alpha=0 is LITERALLY DAPS Eq. 2 in representation space — state the
  theorem for the beta-form, champion classic-alphaQE inherits via k_eff.
- Linearity irrelevant for: exchangeability (needs only fixed-pool-function
  applied pointwise) and Step C (Dhillon/Conrad accept any measurable
  transform).

## Assembled proof skeleton (target for dwt_theory.md)

(i) Model embeddings as cluster signal + noise; pool-kNN homophily h plays
DAPS's Delta role. (ii) Step A: qe shrinks within-class variance by
~lam^2/k at bias cost ~lam*(1-h)*b (Baranwal-style computation on the kNN
graph); W and T normalize and drop noise dims (RCA + spiked-model
optimality). (iii) Step B: representation gain -> true-class score CDF
shifts left / false-class right (prototype-margin increase; Lipschitz
transfer, geodesic-ratio denominator needs care). (iv) Step C: close with
Dhillon dominance (=> smaller E|C|) or Conrad/Sadinle (error-linear distance
to oracle size). Resulting statement: DWT improves expected set size at
fixed coverage whenever variance reduction exceeds the (1-h) contamination
penalty — the empirical homophily gate, derived.
