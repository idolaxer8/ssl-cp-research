# DWT efficiency theory — proof draft v0.1 (2026-08-10)

Goal: a DAPS-Theorem-2-style justification [Z23, Thm 2] that the DWT
representation map (Denoise -> Whiten -> Truncate, all pool-fit) yields
smaller conformal prediction sets at fixed coverage, with an explicit
homophily condition. This is a DRAFT: every step is either (a) proved, (b)
cited to a specific external theorem, or (c) flagged as a numbered GAP
(G1-G10, ledger in Section 8). Companion literature base with full
citations: `docs/dwt_theory_litsweep.md`. All math plain ASCII.

Reference keys: [Z23] Zargarbashi-Antonelli-Bojchevski ICML 2023 (DAPS);
[T23] Teng et al. ICLR 2023 (Feature CP, arXiv 2210.00173); [D24] Dhillon-
Deligiannidis-Rainforth AISTATS 2024 (arXiv 2306.07254); [S19] Sadinle-Lei-
Wasserman JASA 2019; [C26] Conrad et al. arXiv 2608.06206; [B25] Behboodi
et al. arXiv 2509.04631; [BFJ21] Baranwal-Fountoulakis-Jagannath ICML 2021;
[NM19] NT-Maehara arXiv 1905.09550; [BJ21] Bahri-Jiang ICML 2021; [A55]
Anderson, Proc. AMS 1955; [ACMP16] Arias-Castro-Mason-Pelletier JMLR 2016
(mean-shift); [FH75] Fukunaga-Hostetler IEEE-IT 1975; [GD14] Gavish-Donoho
2014; [DGJ18] Donoho-Gavish-Johnstone Ann.Stat. 2018; [LW04] Ledoit-Wolf
2004; [BH05] Bar-Hillel et al. JMLR 2005 (RCA); [DH17] Denis-Hebiri JMLR
2017; [V05] Vovk et al. 2005; [Tib19] Tibshirani et al. NeurIPS 2019;
[R20] Romano-Sesia-Candes NeurIPS 2020 (APS).

---

## 0. The claim, the picture, and the ladder from intuition to proof

**0.1 The claim in one sentence.** Under the SAME nonconformity scoring
s(x,c), replacing the natural representation x by T(x) (DWT) changes the
score distributions so that taking the calibration quantile on
{s(T(x_i), y_i)} yields tighter prediction sets at the same (exact)
coverage — provided pool-kNN homophily is high.

**0.2 The empirical picture.** `src/dwt_score_histograms.py` (this branch)
plots, for the prototype-cosine score s = -cos(z, m_c), the test-set
distribution of TRUE-class scores s(x,y) vs FALSE-class scores s(x,c),
c != y, with the split-CP quantile q_hat marked — raw arm vs DWT arm,
cal=400 balanced, alpha=0.1, single seed. Figure:
`output/dwt_histograms/dwt_score_histograms.png` (+ stats JSON). Numbers
from the 2026-08-10 run:

```
dataset   h(k=10)  arm   coverage  avg|C|   false-leak   q_hat
cifar100   0.79    raw    0.916     3.59      2.68       -0.243
cifar100   0.79    DWT    0.903     1.76      0.86       -0.330
aircraft   0.30    raw    0.919    26.44     25.52       -0.238
aircraft   0.30    DWT    0.919    22.34     21.42       -0.205
```

What to see, panel by panel:
- cifar100/raw: the true-class (green) hump overlaps the left flank of the
  false-class (red) peak; q_hat must sit inside that flank, so ~2.7 false
  classes per test point leak below it.
- cifar100/DWT: the green mass moves far left (points now hug their own
  prototypes) AND the red peak tightens around 0 (wrong-class angles
  concentrate near orthogonal); q_hat moves left; overlap — and with it
  the leak — collapses by ~3x. Coverage unchanged (~0.90 by construction,
  Prop 1). BOTH histogram moves predicted by Lemma B are visible.
- aircraft (h ~= 0.30, the gate-OFF regime): green and red stay
  interleaved in BOTH arms; q_hat barely moves; the leak stays >21 of
  K=100. The mechanism has almost nothing to harvest at low homophily —
  the scope limit is visible in the score space itself, exactly where
  (M2) fails. (Note: in this simplified single-seed run DWT is mildly
  helpful on aircraft, not harmful; the champion-config result is
  neutral-to-harmful. The claim the panel supports is "no separation to
  harvest", not "harm".)

**0.2b The CDF / G(r) overlays (the Prop-C hypotheses, directly).**
Second figure `output/dwt_histograms/dwt_cdf_G_overlays.png`: per dataset,
the true-class score CDF F_true (condition (i): DWT curve ABOVE raw near
the quantiles) and the mean false-count G(r) (condition (ii): DWT curve
BELOW raw there), zoomed on the quantile region, both arms' q_hat marked.
Findings (window fractions in the JSON; note the window is deliberately
wide, so the fractions are conservative — the binding region is at/left of
the arms' q_hats):

- cifar100 (h=.79): BOTH hypotheses hold where they bind. (i) holds on 79%
  of the window — all failures sit right of q_raw, outside the binding
  region; the DWT CDF reaches the 90% level 0.09 earlier (q_hat -0.24 ->
  -0.33). (ii) holds throughout the decision region (G curves cross only
  far left among near-zero values); G(q_hat): 2.68 -> 0.86. Prop C's
  premises are empirically TRUE on-gate, and the conclusion (sets 3.59 ->
  1.76) follows as the theorem says.
- aircraft (h=.30): **condition (i) FAILS almost everywhere (DWT CDF above
  raw on only 1% of the window): the true-class scores get stochastically
  WORSE** — the impurity drag of Lemma A at low homophily, visible as
  q_hat moving right (-0.238 -> -0.205). Meanwhile (ii) holds on 100%:
  false-class mass is pushed right anyway (the affine W/T metric
  normalization does not need homophily), which is why this run still
  shows a mild size reduction (26.4 -> 22.3) despite the gate being off.

Refinement this buys the theory: **the homophily gate governs condition
(i) specifically** — the D stage's improvement condition (Lemma A) is what
breaks at low h(k), while condition (ii) (false side, driven mostly by
W/T) is regime-robust. The gate is not "DWT works/fails" but "the
true-class side of the overlap improves/degrades". (Single seed; the
multi-seed/multi-dataset battery is Section 9 item 4.)

The whole formal draft below is an annotation of this picture. The proof
is a five-stage ladder. Each stage has exactly ONE headline to prove and
feeds the next; this map states only that headline and why the stage is
needed — the elaborate proof lives in the cited section. Read bottom-up as
mechanism (Rung 4 -> 1), top-down as logic (Rung 1 needs 2 needs ...).

**Rung 1 — the size identity.** Proved: Section 2, eq. 2.1.

```
HEADLINE:  E|C| = (1-alpha) + O(1/(n+1)) + E[G(q_hat)]
           where G(r) = mean # false classes scoring <= r.
```

WHY: validity pins the first two terms for BOTH arms, so they cancel in
any comparison. Set size is therefore ENTIRELY the false-class overlap
E[G(q_hat)] — "tighter sets" = "less red mass left of q_hat". This is the
only quantity every later stage aims at.

**Rung 2 — the comparison principle.** Proved: Proposition C, Section 6.

```
HEADLINE:  IF (a) F^T_true >= F_true on R_alpha  (true tail pulled left, so q_hat^T <= q_hat)
           AND (b) G^T <= G on R_alpha           (false mass pushed right)
           THEN E|C^T| <= E|C| (+ O terms).
```

WHY: it is the only place set SIZE is actually compared — it converts two
one-sided histogram MOVES into the size inequality (order-statistic law +
monotone coupling, [D24]). Rungs 3-4 exist solely to supply premises (a)
and (b); the figure's cifar100 row is (a)+(b) seen empirically.

**Rung 3 — why both histograms move at once.** Proved: Lemma B, Section 5
(given Rung 4).

```
HEADLINE:  s(x,c) - s(x,y) = <z_bar, m_y - m_c> = [fixed class-separation] + [noise projection]
           with T fixing the separation term and shrinking the noise projection.
```

WHY: it reduces premises (a)+(b) to a SINGLE fact about representation
noise — one geometric event (noise shrinks) moves green left AND red right
together, because the same error sits in both scores with opposite sign.

**Rung 4 — why the noise shrinks.** Proved: Lemma A, Section 3 (+ W/T,
Section 4). **UPGRADED 2026-08-12 to a theorem package with a measured
gate: `docs/dwt_denoise_theorem.md`** (D1 = deterministic DAPS-2 mirror,
gap-free; D2 = norm-level expectation — and the negative result that the
norm level has NO homophily gate; D3 = margin-level, the operative one).

```
HEADLINE (D3):  d'_smoothed/d'_raw = [1 - 2*beta*kappa*(1-h)] / rho(beta,k_eff),
                improvement iff h > h* = 1 - (1-rho)/(2*beta*kappa)  — scale-free.
```

WHY: qe averages each point with its k pool neighbors — same-class mass
cancels noise (rho ~ 1/sqrt(k_eff+1) on the margin axis), wrong-class mass
drags the class means toward each other by beta*kappa*(1-h) per side. Both
effects are multiplicative, so sigma and Delta CANCEL: one gate for all
datasets (the sweep's "one law"), first-order in beta on both sides (so
"harm not tunable" below the gate), and one hop optimal (drift compounds,
variance is already at its floor). Measured on all five datasets with zero
fitted constants (dwt_denoise_theorem.md §6): h* ~ 0.27-0.40, every tested
dataset on the predicted side; the folklore "~0.7" break-even was an
interpolation across the h in (0.46, 0.80) data gap using the SNAPS SCORE
correction — a different operator. W/T (Section 4) then re-expresses the
contraction in the metric the cosine score reads (RCA + spiked covariance,
4b). Supplies Rung 3's "noise shrinks".

**Rung 5 — assemble, and the two IOUs.** Sections 7-8. Rungs 1-2 proved;
Rung 3 proved given Rung 4; Rung 4 proved up to two medium gaps — G2
(neighbors are selected using the very noisy features being averaged;
route: mean-shift small-ball expansion) and G6 (Rung 2 needs right-tail
DISTRIBUTIONAL dominance, but Rungs 3-4 deliver only the moment
statements; route: Anderson/conditional Gaussianity, or name-the-condition
and validate). Both match where DAPS and Feature-CP [T23] respectively
stopped; the histogram protocol of 0.2 IS G6's validation instrument (the
[T23] Table-5 precedent).

---

## 1. Setting and notation

Data. Classes c in {1..K} with unit-norm means mu_c in R^d. Embeddings

```
x = mu_y + eps,   E[eps | y] = 0,   Cov(eps | y) = Sigma,  eps sub-Gaussian(sigma^2)   (M1)
```

(x renormalized to the sphere; see Remark 1.1). Unlabeled pool
U = {u_1..u_N} iid from the same mixture, INDEPENDENT of everything else.
Calibration {(x_i, y_i)}_{i=1..n} and test (x_{n+1}, y_{n+1}) exchangeable
draws from P (iid for the split arm; Section 6 treats the FCP arm
separately). Class-separation scale: Delta_min = min_{c != c'}
||mu_c - mu_c'||; Delta_max the max.

The map. T = T_WT o T_D with, per the implementations
(`exchangeable_features.py`, both worktrees):

```
T_D(x)  = Pi_S[ (1-b)*x + b * sum_{i in NN_k(x)} w_i(x) u_i / W(x) ],
          w_i(x) = max(cos(x,u_i),0)^a,  W = sum w_i,  Pi_S = L2-normalize
T_WT(z) = A z + a0,   A = D_w P'  (pool PCA projection P', fixed diagonal
          whitening D_w from pooled within-k-means-cluster pool variance)
```

T_D is the qe_beta form; the deployed classic alpha-QE (self-weight 1)
is recovered with b = W/(1+W) pointwise, and enters all bounds through
the effective neighbor count k_eff(x) = W^2 / sum w_i^2 in place of k
(Lemma A covers both). T is a deterministic function of the pool only.

Scores. Primary: the prototype-cosine nonconformity score

```
s(x, c) = -cos( T(x), m_c ),   m_c = prototype of class c in T-space
```

(v0.1 uses population prototypes m_c = A T_D-image of mu_c; finite-shot
prototypes are Remark 5.3 / G10; the softmax-LAC wrapper is a strictly
monotone reparametrization at fixed temperature, so all stochastic-order
statements pass through it unchanged). The geodesic top-k ratio NCM is NOT
covered by this draft — see G7.

Prediction sets. Split CP: q_hat = the ceil((1-alpha)(n+1))-th smallest of
{s(x_i, y_i)}, C(x) = {c : s(x, c) <= q_hat} [V05].

Homophily. h(k) = expected weight-fraction of the k pool neighbors of a
point of class y that are themselves class y (the kNN label homophily we
log as `purity`). Population analogue h_pop(x) = posterior mass of class y
in the kNN ball B(x, r_k(x)); the two are linked by G3.

Remark 1.1 (sphere normalizations are free). Embeddings arrive
L2-normalized and the cosine score is scale-invariant, so the input
normalization is the identity on the data manifold and the output
normalization Pi_S is invisible to s. All statements are made for the
un-normalized convex combination; no gap here.

---

## 2. Validity (complete — no gaps)

**Proposition 1 (exact coverage, both arms).** T is a fixed measurable
function of the pool U, applied pointwise to cal/test. Conditionally on U,
{(T(x_i), y_i)}_{i=1..n+1} is exchangeable whenever {(x_i, y_i)} is
(a deterministic pointwise map preserves exchangeability). Hence split-CP
coverage P(y_{n+1} in C(x_{n+1})) in [1-alpha, 1-alpha + 1/(n+1)) [V05;
Tib19 Lemma 1], and the FCP wrapping stays exact because s(.|bag) remains
a symmetric function of the calibration+test bag (the pool is external to
the bag). This is the same argument as [Z23, Prop 1-2] — permutation
equivariance — one level down (representation instead of scores), and it
is the property our qe implementation was already validated against
(exchangeability oracle, pool-repr-menu round 1). PROVED.

Consequence used later (Section 6): the true-class contribution to E|C| is
PINNED by the guarantee itself,

```
E|C| = P(s(X,y) <= q_hat) + E[ #{c != y : s(X,c) <= q_hat} ]
     = (1-alpha) + O(1/(n+1)) + E[ G(q_hat) ]                    (2.1)
```

where G(r) = E[#{c != y_{n+1} : s(x_{n+1}, c) <= r}] is the mean
false-class count function. **DWT can only act — and only needs to act —
on the false-class overlap term E[G(q_hat)].** Efficiency is entirely a
statement about (i) where the true-class quantile q_hat sits and (ii) how
much false-class score mass lies below it.

---

## 3. Lemma A — the D stage contracts representation error (DAPS-2 analogue)

> **UPGRADE (2026-08-12): this section is superseded by
> `docs/dwt_denoise_theorem.md`**, which restates it as Theorem D1
> (deterministic per-point mirror of [Z23, Thm 2], gap-free), Theorem D2
> (the expectation identity below made exact, with the finding that the
> norm level cannot produce the homophily gate), and Theorem D3 (the
> margin-level scale-free gate h* = 1 - (1-rho)/(2*beta*kappa), measured
> against all five datasets in `src/dwt_gate_constants.py`). The text
> below is kept as the original derivation context; G2/G3 discussions
> remain current.

Write e = x - mu_y, and for the smoothed point e_hat = T_D(x) - mu_y
(pre-normalization form). Condition on x and on the pool; the neighbor set
and weights are then fixed, and T_D(x) is a convex combination — the same
object as [Z23, Eq. 2], with pool vectors in place of neighbor scores.

**Lemma A (conditional error budget).** With S = same-class neighbors,
F = wrong-class neighbors (weight fractions h_w and 1-h_w),

```
e_hat = (1-b) e + b * [ sum_{i in S} (w_i/W) eps_i                (noise avg)
                      + sum_{i in F} (w_i/W) (mu_{c_i} - mu_y)    (impurity)
                      + sum_{i in F} (w_i/W) eps_i ]              (wrong noise)
```

and hence, by the triangle inequality (the exact move of [Z23, Thm 2]):

```
||e_hat|| <= (1-b) ||e|| + b [ ||avg same-class noise|| + (1-h_w) Delta_max + (1-h_w)^{1/2} * noise term ]   (3.1)
```

In expectation, IF the selected noises {eps_i, i in S} behaved as
independent draws (see G2), E||avg same-class noise||^2 ~ sigma_eff^2 /
k_eff, giving the improvement condition

```
E||e_hat||^2 < E||e||^2   whenever
b * [ trace(Sigma)*(1 - (1-b) - b/k_eff) ] > b * [ (1-h(k)) * Delta_max ]^2-order terms,
```

informally: **variance removed by averaging > impurity bias added**, which
holds iff h(k) is large — the representation-space analogue of DAPS's
"(1/|N_i|) sum eps_j + Delta < eps_i", with (1-h(k))*Delta_max playing the
role of DAPS's Delta. This is the inequality that, once quantified, should
DERIVE the empirical gate h(k) >= ~0.7 (SNAPS/qe regime map).

Status of each ingredient:

- Convex-combination decomposition and (3.1): PROVED (elementary; identical
  in form to [Z23, Thm 2] proof).
- Variance contraction 1/k_eff for neighborhood averaging under a cluster
  model: [BFJ21] prove exactly this for graph convolution under CSBM
  (within-class noise sigma -> sigma/sqrt(deg), separability threshold
  improves by sqrt(deg)); [NM19, Lemma 5] give the same bias-variance split
  for k smoothing passes. Both assume the averaging GRAPH is independent of
  the feature noise. **Our kNN graph is built FROM the features — see G2.**
- **G2 (the main gap of Lemma A): selection-noise correlation.** Neighbors
  are selected BECAUSE they are close to x, so (i) selected same-class
  noises are tilted toward e (positively correlated with x's own error),
  weakening the effective averaging, and (ii) the neighbor mean estimates
  the LOCAL mean around x, not mu_y. Standard kNN regression bounds
  ([BJ21, Thm 1]: sup-error <= C[(k/N)^{a/D} + 1/sqrt(k)]) do NOT apply
  verbatim: they assume response noise independent of the design, and our
  pool vector is both design and response. The correct object is the
  mean-shift vector [FH75]: for small kNN radius r_k,
  `neighbor-mean(x) - x ~= (r_k^2/(D+2)) * grad log f(x) + O_P(1/sqrt(k))`
  ([ACMP16] make this rigorous for mean-shift iterations), and under (M1)
  with separated components, grad log f(x) ~= -Sigma^{-1} e + (posterior
  contamination), so one qe step is a score-ascent/empirical-Bayes
  (Tweedie-type) denoising step of size ~ b*r_k^2: it contracts e
  MULTIPLICATIVELY and the "locality bias toward x" is not a new error —
  it is retained original noise, i.e. an effective shrinkage of b, never a
  sign flip. WHAT IS MISSING: a self-contained lemma making the small-ball
  expansion + fluctuation control rigorous for a single weighted-kNN
  average at our (N ~ 1e4, d_eff moderate) scales, with explicit constants.
  Resolution route: adapt [ACMP16] Lemma-level results; assume a bounded
  density ratio on the kNN ball. Severity: medium — the mechanism survives
  (both views agree qualitatively), but constants and the exact gate
  threshold depend on it.
- **G3: h(k) (empirical kNN label homophily) vs h_pop (posterior purity of
  the kNN ball).** The lemma is stated with h_pop; the gate we measure and
  deploy is the empirical h(k). E[h(k)] = E[h_pop] by exchangeability of
  pool draws, but concentration of h(k) per class/region (it varies —
  that is exactly the aircraft/cars failure) needs to be assumed or
  measured. Mild: we USE h(k) as a measured diagnostic, so the theorem can
  simply be stated conditionally on h(k).
- **G1 (modeling): (M1) mixture with common sub-Gaussian noise** for
  DINOv2 embeddings is an idealization: real class-conditionals are
  anisotropic (whitening exists because they are!) and possibly
  multi-modal. Checkable in spirit (per-class covariance spectra);
  the draft keeps Sigma shared across classes (homoscedastic) — the RCA
  optimality in Section 4 needs exactly that, so the assumptions are at
  least mutually consistent.

---

## 4. Lemma W/T — the affine stage: exact bookkeeping + cited optimality

The application-time map is a single global affine z -> A z + a0 (verified
in code: PCA projection then a FIXED diagonal scale; k-means clusters
enter only the fit of D_w). Two kinds of statements:

**(4a) Exact, assumption-free (PROVED):**
- Empirical means commute with affine maps, so class prototypes transform
  covariantly: prototype(A z + a0 over class) = A*prototype + a0. Step-B
  bookkeeping for the prototype score through W/T is exact.
- (M1) is closed under the map: noise covariance Sigma -> A Sigma A'. The
  entire W/T effect on the model is one covariance computation.
- Whitened Euclidean distance = a fixed quadratic (Mahalanobis-type)
  metric on the original space: W/T is a metric change, nothing else.

**(4b) That the metric change HELPS (cited, with surrogate gaps):**
- Whitening by the pooled WITHIN-class covariance is the optimal
  Mahalanobis metric under a homoscedastic Gaussian model — RCA [BH05]
  (ML-optimal and the information-theoretic optimum under distance
  constraints). Shrinkage of the covariance estimate: [LW04].
- PCA truncation: under a spiked-covariance model (signal rank r spikes +
  isotropic noise), discarding sub-BBP components is optimal denoising
  [GD14; DGJ18]. Assumption (M3): the class-mean subspace lies in the
  pool's leading spikes.
- **G4 (two surrogate gaps, both label-free-checkable):** (i) the code
  whitens DIAGONALLY in the pool-PCA basis, not by the full within-class
  covariance — equals RCA only if the within-class covariance is
  approximately diagonal in that basis (measure: off-diagonal energy of
  the within-class covariance in the PCA basis); (ii) "within-cluster"
  (k-means pseudo-classes on the pool) stands in for "within-class" —
  equals RCA only if clusters align with classes (measure: compare pooled
  within-cluster vs within-class covariances on a labeled probe).
- **G5 (known scope limit, and the theory PREDICTS it):** when (M3) fails
  — class signal in the low-variance tail, pool participation ratio small
  — truncation removes signal, and no efficiency claim is made. This is
  the aircraft inversion (PCA no-op, full-rank LW-whitening wins,
  PR = 16 vs ~240): the empirical regime rule is the checkable form of
  (M3). The theorem self-limits exactly where the data said it should.

---

## 5. Lemma B — score transfer: representation gain -> score-distribution gain

For the cosine-prototype score, with unit prototypes and z = T(x):

```
s(x, c) = -<z_bar, m_c>,   z_bar = z/||z||
s(x, c) - s(x, y) = <z_bar, m_y - m_c>                       (score margin)
```

**(5a) Lipschitz transfer (PROVED, constant 1):** |s(x,c) - s'(x,c)| <=
||z_bar - z_bar'|| for unit prototypes — score perturbations are bounded
by representation perturbations; no blow-up anywhere. (The softmax-LAC
wrapper is 1/(4*temp)-Lipschitz and strictly monotone; harmless.)

**(5b) Mean margin grows under contraction (PROVED under (M1)+(M4)):**
E[s(x,c) - s(x,y)] = <mu-terms> + E<e_hat-projection>; contraction of
e_hat leaves the mean-separation term (a fixed function of the whitened
class means) unchanged and shrinks the noise projections; under the
geometric condition (M4) "prototype directions not degenerate"
(min_{c!=y} <m_y, m_y - m_c> >= g0 > 0, i.e. classes are separated in the
whitened metric), the standardized margin (mean gap / noise s.d. of the
relevant 1-D projections) strictly increases whenever Lemma A's
improvement condition holds. Bias caveat: the impurity term of Lemma A
shifts the mean toward wrong-class prototypes; the SAME improvement
condition (variance removed > impurity added) is what keeps the
standardized margin growing — the homophily gate appears here a second
time, consistently.

**(5c) From moments to stochastic order — G6 (real gap, with a route):**
Section 6 needs distributional statements, not moments:

```
(i)  true-class scores under T stochastically smaller ON THE UPPER-QUANTILE
     REGION R_alpha (where q_hat lives),
(ii) mean false-count G^T(r) <= G(r) for r in R_alpha.
```

Moment improvements do NOT imply full-range first-order dominance (two
Gaussians with different variances have crossing CDFs — dominance FAILS in
the far tail on the wrong side). The rescue is that we only need dominance
on R_alpha, the right-tail region of the true-class score distribution:
- Under conditional Gaussianity of e_hat (plausible post-averaging;
  assumption (M5)), the true-class score is a 1-D Gaussian functional;
  mean-down + variance-down implies CDF dominance ABOVE the crossing
  point, and R_alpha sits above it for the alphas we use (alpha ~ 0.1,
  quantile above the mean) provided the bias shift is small — again the
  improvement condition. For norm-type scores the multivariate version is
  Anderson's inequality [A55]: Loewner-smaller centered Gaussian
  covariance => stochastically smaller norms, globally.
- WHAT IS MISSING: (a) conditional Gaussianity or an
  Anderson-type/unimodal-symmetry assumption on e_hat (the impurity term
  makes it non-centered — needs a noncentral variant or an explicit bias
  budget); (b) locating the CDF crossing point relative to R_alpha with
  constants. Severity: medium. This is the exact analogue of [T23]'s
  "expansion" cubic condition — they also could not derive it from first
  principles and VALIDATED it empirically (their Table 5). Fallback of the
  same epistemic grade as the published precedent: state (i)+(ii) as the
  named condition "score-margin dominance on R_alpha", validate
  empirically pre/post DWT (Section 9), and present the Gaussian case as
  a proved special case.

**G7 (out of scope of v0.1): the geodesic top-k RATIO NCM.** The ratio is
invariant to global metric scaling, so any "all distances shrink" argument
says nothing; only NOISE-relative-to-MARGIN improvements act, and near-zero
denominators break Lipschitz control. The prototype NCM is the theorem's
object; the champion asym ratio NCM inherits only heuristically for now.
(Empirical support that this is not vacuous: prototype_softmax is
our best or near-best NCM on balanced splits at every cal.)

---

## 6. Proposition C — score dominance -> smaller expected sets (assembled)

**Proposition C.** Fix n and alpha. Suppose (i) and (ii) of (5c) hold on a
region R_alpha, together with (iii) quantile concentration: both pipelines'
q_hat lie in R_alpha with probability >= 1-delta_n (a DKW/[T23]-condition-3
style requirement, cost delta_n * K in set size). Then

```
E|C^T| <= E|C| + K*delta_n + O(1/(n+1)).
```

Proof (complete given (i)-(iii)). By (2.1) the true-class terms agree to
O(1/(n+1)) (both pinned by coverage — Proposition 1). For the false-class
terms: q_hat is the m-th order statistic (m = ceil((1-alpha)(n+1))) of the
cal true-class scores, whose law is P(q_hat <= t) = P(Binom(n, F(t)) >= m)
— a pointwise-monotone functional of the true-class score CDF F. By (i),
F^T >= F on R_alpha, hence q_hat^T is stochastically <= q_hat restricted
to R_alpha (up to the delta_n escape event); choose the monotone coupling
so q_hat^T <= q_hat a.s. on R_alpha. Then, since G^T is nondecreasing and
by (ii) G^T <= G on R_alpha:

```
E[G^T(q_hat^T)] <= E[G^T(q_hat)] <= E[G(q_hat)]     (on the 1-delta_n event)
```

and the escape event costs at most K*delta_n. QED.

Notes and references:
- The identity behind (2.1) and the order-statistic law is the
  finite-sample expected-size machinery of [D24, Thm 1] (their Binomial-
  CDF integral is exactly P(Binom(n,F(r)) >= m) integrated against the
  label-count factor); our (ii) is the two-sided extension of their
  Section 4.1 one-sided comparison license. The restriction to R_alpha and
  condition (iii) are the analogue of [T23]'s quantile-stability cubic
  condition.
- Alternative converters if dominance proves too strong: [C26, Thm 6]
  bounds | |C| - |C_oracle| | LINEARLY in uniform score-estimation error
  (needs Holder conditional-CDF + density minorization); [S19, Thm 14]
  bounds the symmetric difference to the oracle least-ambiguous sets by
  the plug-in error (needs sup-norm consistency of p_hat + margin
  condition — **G10: our prototype-softmax is a model, not a consistent
  estimator of p(y|x), so [S19] applies only under an additional
  well-specification assumption**). Both routes trade the dominance gap G6
  for stronger regularity assumptions; kept as fallbacks.
- **G8: the FCP arm.** Everything above is split-CP with iid calibration
  scores. Our deployed default wraps DWT in exact full CP. The identity
  (2.1) survives conditionally on the bag (coverage pinning is
  bag-conditional), but the order-statistic law and [D24] need the
  transductive re-derivation; [B25, Thm 3.5-3.7] provide the transductive
  size machinery (and Thm 3.7 gives the KL-linear misspecification
  penalty as an alternative statement: DWT reduces E[KL(p || p_hat)] =>
  the size exponent shrinks). Empirical support that the port is a
  formality, not a surprise: fullcp <= split for the qe/SNAPS pipeline
  (stage-3 result, snaps-pool memory). Severity: medium-low, but it IS
  the arm we headline.
- **G9: the balanced-split arm.** Balanced cal is a label-dependent split:
  cal scores are NOT iid draws from the marginal score law, so
  Proposition C applies verbatim only to the random-split arm (which is
  our exact-validity arm anyway). The balanced arm over-covers by +1-3pp
  (known) and the size comparison there is conjectured to inherit the
  same ordering; not proved. Mild: the paper's exactness claims already
  ride the random arm.

---

## 7. Assembled target theorem (v0.1 statement)

**Theorem (draft).** Assume (M1) homoscedastic sub-Gaussian mixture on the
sphere; (M2) homophily: the measured kNN homophily h(k) satisfies the
Lemma-A improvement condition (variance removed at k_eff exceeds the
(1-h(k))*Delta_max impurity budget plus the G2 locality remainder); (M3)
spiked pool covariance with class-mean subspace in the retained spikes;
(M4) whitened class separation g0 > 0; (M5) conditional Gaussianity of the
smoothed error (or: the named "score-margin dominance on R_alpha"
condition, validated empirically); (iii) quantile concentration delta_n.
Then for the prototype-cosine score and split CP on a random split:

```
coverage:  P(y in C^T(x)) in [1-alpha, 1-alpha + 1/(n+1))      (exact; Prop 1)
size:      E|C^T| <= E|C| + K*delta_n + O(1/(n+1)),            (Prop C)
```

with strict size improvement whenever the Lemma-A inequality is strict.
Interpretation: DWT improves expected set size at fixed exact coverage
precisely when pool-kNN homophily is high enough that averaging removes
more noise than wrong-class contamination injects. The derived, measured
form of the gate is Theorem D3 (`dwt_denoise_theorem.md`): h > h* =
1 - (1-rho)/(2*beta*kappa) ~ 0.27-0.40 on our data — LOWER than the
folklore ~0.7, which came from the SNAPS score correction across a data
gap; stanford_cars (h_w = 0.46) is the registered discriminating
prediction (D3 says qe gains there).

Dependency graph: (M1) -> Lemma A [G1,G2,G3] -> Lemma W/T [G4,G5] ->
Lemma B [G6,G7] -> Prop C [G8,G9] -> Theorem. Proposition 1 (validity)
has NO gaps and stands alone.

---

## 8. Gap ledger

| # | Where | What is missing | Severity | Route |
|---|-------|-----------------|----------|-------|
| G1 | (M1) | mixture/homoscedastic sub-Gaussian model for DINOv2 embeddings is an idealization | modeling | report per-class covariance spectra; state theorem as model-relative |
| G2 | Lemma A | selection-noise correlation: kNN graph built from the very features being averaged; no off-the-shelf theorem (kNN-regression bounds assume design/response independence) | MEDIUM (main math gap) | mean-shift small-ball expansion [FH75; ACMP16] + bounded density ratio on the kNN ball; own lemma. Both violation directions now SIGNED (dwt_denoise_theorem.md §8): (V1) same-class tilt = retained own noise = effective beta shrink; (V2) foreign selection aligned with own error = kappa inflation; both push h* UP, so the (I)-model h* is a floor |
| G3 | Lemma A | empirical h(k) vs population posterior purity h_pop | mild | state theorem conditionally on measured h(k) |
| G4 | Lemma W/T | diagonal-in-PCA whitening != full RCA; k-means clusters != classes | mild, checkable | two label-free diagnostics (off-diagonal energy; cluster/class covariance match) |
| G5 | (M3) | spiked model fails on low-PR data (aircraft) | none (scope, predicted) | PR rule = checkable form of (M3); theorem excludes that regime by assumption |
| G6 | Lemma B->C | moments -> stochastic dominance on R_alpha; noncentral (impurity-biased) case | MEDIUM | Gaussian case via [A55] + crossing-point control; else name the condition and validate empirically (same epistemic grade as [T23] cubic conditions). ALT ROUTE (2026-08-16): contraction-anchor device from SNAPS Prop 2 autopsy — replaces dominance with a first-moment anchor condition (A) + explicit Chebyshev remainder; see `dwt_denoise_theorem.md` Section 10 (Corollary D4 target shape) |
| G7 | scores | geodesic top-k RATIO NCM not covered (scale-invariance neuters naive arguments; denominator Lipschitz failure) | medium (champion NCM!) | v0.2: analyze noise-to-margin functional; or headline prototype NCM (empirically near-best) |
| G8 | Prop C | transductive/FCP version of the size identity | medium-low | bag-conditional re-derivation; [B25] machinery; empirical fullcp<=split backs it |
| G9 | splits | balanced-cal arm: scores not iid | mild | claim only random arm; balanced arm conservative + conjectured |
| G10 | fallback route | [S19] plug-in bound needs p_hat consistency our prototype-softmax does not have; finite-shot prototype noise ignored in v0.1 | mild | well-specification assumption if that route is used; prototype noise contracts under the same Lemma A |

Honest summary: the chain is COMPLETE at the level of "proved or cited"
except for two medium mathematical gaps — G2 (selection-noise correlation
in the denoising lemma) and G6 (right-tail stochastic dominance) — plus
the scope decisions G7 (ratio NCM) and G8 (FCP arm). G2 and G6 are exactly
where DAPS and Feature CP respectively stopped short as well: [Z23] proved
the fixed-graph version and left the kNN-graph case explicitly to future
work (their Sec 5.3), and [T23] assumed their expansion condition and
validated it empirically. Matching that precedent, v0.1 is publishable-
shaped with G6-as-named-condition + Section 9 diagnostics; closing G2
outright would exceed the prior art.

---

## 9. Empirical diagnostics to attach (the [T23] Table-5 protocol)

Each checkable assumption gets a measured pre/post-DWT panel:
1. (M2): h(k)/k_eff per dataset (already logged as `purity`).
2. (M3): pool spectrum + participation ratio; class-mean energy in
   retained spikes (needs labels once, offline probe).
3. (M4): whitened prototype separation g0.
4. (5c)(i): true-class score CDFs pre/post DWT on R_alpha (the dominance
   picture); (5c)(ii): mean false-count G(r) curves on R_alpha.
   **STARTED 08-10**: `src/dwt_score_histograms.py` + Section 0.2 figure
   (cifar100 both moves visible, leak 2.68 -> 0.86; aircraft flat) —
   extend to CDF/G(r) overlays on R_alpha, more seeds/datasets.
5. (iii): dispersion of q_hat across trials (quantile stability), and the
   [T23] cubic metric analogue: mean |score - q_hat| pre/post DWT.
6. G4: off-diagonal energy of within-class covariance in the pool-PCA
   basis; within-cluster vs within-class covariance distance.
7. **DONE 08-12 — Theorem D3 gate constants** (`src/dwt_gate_constants.py`,
   `output/dwt_theory/gate_constants.json`): (h_w, beta_hat, k_eff, kappa)
   -> h* and predicted d'-ratio per dataset, zero fitted knobs. All tested
   datasets land on the predicted side (cifar100/mini/cifar10 gain,
   aircraft harm); **OPEN: the cars discriminating run** — qe arm on
   stanford_cars in the champion pipeline (D3 predicts d'-ratio 1.52 =
   gain at h_w = 0.46; the folklore 0.7 gate predicts harm).
Expected picture per the regime map: panels 1-5 favorable on
cifar100/mini/cifar10, condition (M2) or (M3) visibly violated on
aircraft/cars — the theorem's assumptions should FAIL exactly where the
method empirically fails; that alignment is itself a result. Panel 7
sharpens this: on the D stage the aircraft failure is predicted
QUANTITATIVELY, and cars is predicted to be a (mild) success, not a
failure — the two low-homophily datasets are no longer one regime.
