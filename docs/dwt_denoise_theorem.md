# Theorem D — pool-kNN smoothing contracts class geometry: the DAPS-Theorem-2 analogue in representation space

Status: v1.0 (2026-08-12). This document upgrades Section 3 (Lemma A) of
`docs/dwt_theory.md` into a self-contained theorem package: a deterministic
per-point theorem (D1) that mirrors DAPS Theorem 2 line by line, a
second-moment theorem (D2) that shows why the norm level is the WRONG level,
and the operative margin-level theorem (D3) whose improvement condition is a
scale-free homophily gate with no fitted constants — measured against all
five datasets in Section 6. Companion literature base:
`docs/dwt_theory_litsweep.md`. All math plain ASCII.

Reference keys as in `dwt_theory.md`: [Z23] Zargarbashi-Antonelli-Bojchevski
ICML 2023 (DAPS); [T23] Teng et al. ICLR 2023 (Feature CP); [D24] Dhillon et
al. AISTATS 2024; [FH75] Fukunaga-Hostetler 1975; [ACMP16] Arias-Castro et
al. JMLR 2016; [BJ21] Bahri-Jiang ICML 2021; [V05] Vovk et al. 2005.

---

## 0. The template: what DAPS Theorem 2 actually says

[Z23] smooth conformal scores over a given graph,
`s_hat(v,y) = (1-lam)*s(v,y) + (lam/|N_v|) * sum_{u in N_v} s(u,y)`,
and justify it with a theorem stated one level up, in probability space:

> **DAPS Theorem 2.** Let pi_i be the model's approximation of the
> ground-truth conditional probability vector p_i, and let the diffused
> distribution be pi_hat_i = (1-lam)*pi_i + (lam/|N_i|) * sum_{j in N_i} pi_j.
> Assume the graph is constructed so that A_ij = 1 iff ||p_i - p_j|| <= Delta
> (total variation). Then diffusion improves the approximation error
> eps_i = ||pi_i - p_i||, i.e. ||pi_hat_i - p_i|| < eps_i, if
>
> ```
> (1/|N_i|) * sum_{j in N_i} eps_j  +  Delta  <  eps_i .        (DAPS-2)
> ```

Their proof is four lines of triangle inequality: split
pi_hat_i - p_i into (1-lam)(pi_i - p_i) + lam*(mean_j pi_j - p_i), bound each
neighbor term by ||p_i - p_j|| + ||p_j - pi_j|| <= Delta + eps_j, divide by
lam. Two things carry the theorem: the CONVEX-COMBINATION structure of the
smoother, and the HOMOPHILY RADIUS Delta baked into the graph construction.
Validity is handled separately and for free (their Prop 2: permutation-
equivariant transforms preserve exchangeability).

Two open edges in [Z23] that our setting must cross:

1. Their footnote 5: the theorem lives in probability space while the method
   diffuses scores ("easy but cumbersome" to port). We prove at the exact
   object our method transforms — the representation.
2. Their Section 5.3: when the graph is a kNN graph BUILT FROM the features
   themselves, they explicitly "leave it for future work". Our neighborhoods
   are pool-kNN in embedding space — we must face this (Section 8, (I)/G2).

---

## 1. The dictionary

| DAPS (given graph, probability space)        | Ours (pool kNN, representation space) |
|----------------------------------------------|----------------------------------------|
| node i                                       | cal/test point x |
| model prob. vector pi_i                      | embedding x (the SSL representation) |
| ground truth p_i                             | class anchor mu_y |
| error eps_i = ||pi_i - p_i||                 | representation error eps = ||x - mu_y|| |
| given graph neighbors N_i                    | pool kNN N_k(x), cosine metric |
| uniform weights 1/|N_i|                      | alpha-QE weights w_u/W; |N_i| -> k_eff |
| diffusion (1-lam)*pi_i + lam*mean            | qe: (1-beta)*x + beta*nu(x) |
| homophily radius Delta (assumed of the graph)| impurity budget (1-h_w(x)) * Delta_F(x) (measured) |
| condition: mean_j eps_j + Delta < eps_i      | condition (D1) below |
| Prop 2: exchangeability preserved            | pool-measurable pointwise map preserves exchangeability (Prop 1, `dwt_theory.md` §2) |

The structural difference that reshapes the theorem: DAPS's ground-truth
objects p_i live in a continuum, so "neighbors have close ground truth" is a
radius assumption. Our ground-truth objects are K DISCRETE anchors, so a
neighbor's ground truth is either the ego's own anchor (displacement 0) or a
wrong anchor (displacement >= Delta_min). DAPS's single radius Delta
therefore splits into a MEASURED two-part budget: the wrong-class weight
fraction (1 - h_w) times the wrong-anchor displacement. Homophily moves from
an assumption about a given graph to a measured property of a constructed
one — which is what makes the gate deployable (and what raises G2).

---

## 2. Setting

Classes c in {1..K} with anchors mu_c in R^d; labeled point x = mu_y + e;
pool point u = mu_{c(u)} + xi_u, pool U independent of cal/test. Noise model
(M1): all noises zero-mean, covariance Sigma (shared), independent across
points; sigma^2 = trace(Sigma) (full noise energy), sigma_v^2 = v'Sigma v
(noise energy along a unit direction v). Anchor separations Delta_min <=
||mu_c - mu_c'|| <= Delta_max.

The deployed map (verbatim from `qe_smooth` / `exchangeable_features._qe_smooth`,
k = 10, a = 3):

```
T_D(x) = L2norm( x + sum_{u in N_k(x)} w_u(x) * u ),
w_u(x) = max(cos(x,u), 0)^a,   W(x) = sum_u w_u(x).
```

Convex-combination form (self-weight 1 makes beta self-tuning):

```
T_D(x) (pre-norm)  =  (1 - beta) * x + beta * nu(x),
beta = W/(1+W),   nu(x) = sum_{u in N_k(x)} (w_u/W) * u .
```

Per-neighborhood quantities (all measurable given pool labels):

```
k_eff = W^2 / sum_u w_u^2            Kish effective neighbor count, in [1, k]
h_w   = sum_{u: c(u)=y} w_u / W      weighted label homophily (logged: `purity`)
dbar  = sum_{u: c(u)!=y} (w_u/W) * (mu_{c(u)} - mu_y)     foreign drift vector
Delta_F(x) = max_{u foreign} ||mu_{c(u)} - mu_y||         local anchor spread
```

Note ||dbar|| <= (1 - h_w) * Delta_F(x).

Normalization is free: embeddings arrive L2-normalized and the
prototype-cosine score is scale-invariant, so the outer L2norm is invisible
to the score; all statements are for the pre-norm convex combination
(Remark 1.1 of `dwt_theory.md`). The W/T stage after T_D is one global affine
z -> Az + a0 under which anchors, noises, and all quantities below transform
covariantly (`dwt_theory.md` §4a); read Sections 3-5 in the whitened
truncated metric, where W/T's job (RCA/spiked-covariance, §4b) is to make
Sigma approximately isotropic so sigma_v is uniform across directions.

---

## 3. Theorem D1 — the deterministic mirror (per-point, gap-free)

**Theorem D1.** Fix any x with label y, any beta in (0,1], any nonnegative
weights {w_u}. Write eps = ||x - mu_y||, r_u = ||u - mu_{c(u)}|| (the
neighbor's own error to its OWN anchor), and x_hat = (1-beta)x + beta*nu(x).
Then

```
||x_hat - mu_y||  <=  (1-beta)*eps + beta * [ sum_u (w_u/W) r_u  +  (1-h_w) * Delta_F(x) ]
```

and consequently smoothing strictly improves the representation error,
||x_hat - mu_y|| < eps, whenever

```
sum_u (w_u/W) * r_u  +  (1 - h_w(x)) * Delta_F(x)  <  eps .        (D1)
```

*Proof.* Write mu_y = (1-beta)mu_y + beta*mu_y, so

```
x_hat - mu_y = (1-beta)(x - mu_y) + beta*(nu - mu_y),
nu - mu_y    = sum_u (w_u/W) (u - mu_y).
```

For a same-class neighbor, u - mu_y = xi_u, so ||u - mu_y|| = r_u. For a
foreign neighbor, ||u - mu_y|| <= ||mu_{c(u)} - mu_y|| + ||xi_u||
<= Delta_F(x) + r_u. Weighted triangle inequality over the neighborhood:

```
||nu - mu_y|| <= sum_u (w_u/W) r_u + sum_{u foreign} (w_u/W) ||mu_{c(u)} - mu_y||
             <= sum_u (w_u/W) r_u + (1-h_w) * Delta_F(x),
```

then one more triangle inequality and division by beta, exactly eqs.
(7)-(15) of the [Z23] proof. QED.

This IS DAPS Theorem 2 with two substitutions — neighbor scores -> pool
vectors, graph radius Delta -> measured impurity budget (1-h_w)*Delta_F.
Same strengths (fully deterministic, conditional on realized errors, no
model assumptions at all) and the same two limits, which [Z23] inherit too:

- (L1) The condition is per-point and involves the realized errors, which
  are unobservable; it certifies that smoothing repairs points NOISIER than
  their neighborhood average plus the impurity budget — tail repair, not the
  distribution-wide shift we measure (`output/dwt_histograms/`).
- (L2) The triangle inequality treats same-class neighbor noise as
  adversarially aligned mass: sum (w/W) r_u concentrates near E[r] ~ eps-typical,
  so (D1) can never certify improvement for a typical point. But noise is
  not adversarial — it is zero-mean, and VECTOR AVERAGING CANCELS it. That
  cancellation is invisible to any triangle-inequality argument. Fixing (L2)
  is Theorem D2; DAPS stops at the template.
  **[Post-cars note, 2026-08-13 (§8b): this criticism aged badly. The
  measured same-class neighbor displacements are ~0.85-0.90 correlated
  with the ego's own displacement (shared local field), so the
  cancellation D2 "fixes in" barely exists — D1's adversarial-alignment
  accounting of the same-class term is empirically near-exact, and it was
  D2's optimism that the data rejected. D1 itself is untouched: it is
  assumption-free, and the repair it certifies (above-typical-noise egos
  pulled toward the neighborhood) is exactly the iid-shell trim that
  survives in the revised mean-field-mover mechanism.]**
- **Empirical check (2026-08-16, `src/d1_empirical_check.py`,
  `output/dwt_theory/d1_empirical_check.json`; figure
  `docs/figs/dwt_learning_d1_check.png`):** bound holds for 100% of egos
  on all five datasets; certified => improved holds pointwise (100% of
  certified egos improved, zero exceptions); certification rate is
  monotone in homophily (63.5/64.7/40.9/7.6/2.4% on
  c10/mini/c100/cars/aircraft) — an assumption-free echo of the regime
  map; realized NORM improvement is ~100% everywhere INCLUDING aircraft
  (where qe harms CP) — the one-number proof that the norm level is the
  wrong currency (D2's lesson); median relative slack 0.28-0.53,
  tightest on high-h datasets, consistent with §8b field alignment.

---

## 4. Theorem D2 — the norm level in expectation, and why it is the wrong level

Assumption (I) (selection independence — the idealization; its violations
and their signed directions are Section 8): conditional on the realized
neighborhood COMPOSITION C(x) = ({w_u}, {c(u)}, beta) of N_k(x), the noises
e and {xi_u} retain mean 0, covariance Sigma, and mutual uncorrelatedness.

**Theorem D2.** Under (M1) + (I), with eps_hat = ||x_hat - mu_y||:

```
E[ eps_hat^2 | C ] = (1-beta)^2 * sigma^2 + beta^2 * [ sigma^2/k_eff + ||dbar||^2 ] .   (D2)
```

Writing R = 1/k_eff + ||dbar||^2/sigma^2:
E[eps_hat^2 | C] < sigma^2 iff (1-beta)^2 + beta^2 R < 1 iff 0 < beta < 2/(1+R);
the optimum is beta* = 1/(1+R) with contraction factor R/(1+R) < 1.

*Proof.* Expand the square of x_hat - mu_y = (1-beta)e + beta(sum_u (w_u/W)xi_u + dbar).
Under (I) the cross terms vanish (e independent of neighbor noises; dbar is
fixed by C; all noises mean-zero), E||(1-beta)e||^2 = (1-beta)^2 sigma^2, and
E||sum (w_u/W) xi_u||^2 = sum (w_u/W)^2 sigma^2 = sigma^2/k_eff (definition
of the Kish count). The beta-range claim is the quadratic
(1-beta)^2 + beta^2 R < 1 solved for beta. QED.

Two readings, one lesson:

- The averaging mechanism is now visible: the neighbor-noise term enters at
  sigma^2/k_eff, not sigma^2 — the cancellation (L2) said the triangle
  inequality cannot see. This is [BFJ21]'s graph-convolution variance
  contraction and [NM19] Lemma 5's bias-variance split, transplanted to a
  weighted kNN pool average.
- **But the norm level has NO homophily gate.** R is finite always, so SOME
  beta > 0 improves E[eps_hat^2] regardless of h_w; even at the deployed
  beta_hat = W/(1+W) the condition R < 1 + 2/W is weak (impurity must
  approach ||dbar|| ~ sigma to fail it). Empirically qe HARMS at h ~ 0.26
  (aircraft) at every tested setting — "harm not tunable". So a theorem at
  the norm level cannot explain the observed regime map: representation
  error in the norm sense is simply not the quantity CP set size feels. The
  bias dbar is small in norm but it is not random — it points AT the
  confusable anchors. Norm intuition ("denoising helps") misleads here;
  the damage lives in specific directions. Hence D3.

---

## 5. Theorem D3 — the margin level: a scale-free homophily gate

The quantity CP efficiency feels is the SCORE MARGIN (Lemma B,
`dwt_theory.md` §5): for the prototype-cosine score,
s(x,c) - s(x,y) = <z_bar, m_y - m_c> — a 1-D projection of the
representation onto a class-pair axis. So project the smoothing identity
onto that axis.

Objects. A confusable pair (y,c): unit axis v = (mu_y - mu_c)/Delta_pair,
Delta_pair = ||mu_y - mu_c||. Linear margin statistic g(x) = <x, v>.
Class-y egos x_y = mu_y + e_y and class-c egos x_c = mu_c + e_c are smoothed
by the same map, with compositions C_y, C_c (homophilies h_y, h_c; common
beta and k_eff for symmetry of the statement — the general form carries four
constants). Realized pair drifts (nonneg. in the confusable geometry):

```
D_y = sum_{u in F(x_y)} (w_u/W) * <mu_y - mu_{c(u)}, v>     (drift of y-egos toward c)
D_c = sum_{u in F(x_c)} (w_u/W) * <mu_{c(u)} - mu_c, v>     (drift of c-egos toward y)
```

Drift concentration kappa = (D_y + D_c) / [ ((1-h_y) + (1-h_c)) * Delta_pair ]:
kappa = 1 iff every foreign neighbor is the partner class in full alignment;
kappa < 1 when impurity spreads over off-axis classes; kappa > 1 is possible
(far anchors projecting onto v, and — outside (I) — selection alignment, §8).

**Theorem D3.** Under (M1) + (I), the smoothed margin statistics satisfy

```
E[g(x_hat_y)] - E[g(x_hat_c)] = Delta_pair - beta*(D_y + D_c)
Var[g(x_hat)]                 = rho(beta, k_eff)^2 * sigma_v^2,
rho(beta, k_eff)              = sqrt( (1-beta)^2 + beta^2/k_eff )
```

so the standardized pair margin (discriminability) changes by exactly

```
d'_smoothed / d'_raw = [ 1 - beta*kappa*((1-h_y)+(1-h_c)) ] / rho(beta, k_eff) ,
```

and (symmetric case h_y = h_c = h) strictly improves iff

```
h  >  h*  :=  1 - (1 - rho(beta, k_eff)) / (2*beta*kappa) .        (D3)
```

*Proof.* Mean: E[nu | C] = sum (w_u/W) mu_{c(u)} = mu_y + dbar, so
E[g(x_hat_y)|C_y] = <mu_y, v> - beta*D_y with D_y = -<dbar_y, v>, and
symmetrically for c. Subtract. Variance: g is linear, so
Var[g(x_hat)|C] = (1-beta)^2 v'Sigma v + beta^2 sum (w_u/W)^2 v'Sigma v =
rho^2 sigma_v^2, with the cross term zero under (I). Divide the standardized
margins; set the ratio > 1 and solve for h. QED.

The numerator is the SIGNAL SHRINK (mean separation multiplier), the
denominator the NOISE SHRINK. Smoothing always shrinks both; it helps iff
the signal shrinks less. Everything else cancels — and that cancellation is
the theorem's main structural content:

**(C1) Scale-freeness — the "one law".** sigma_v and Delta_pair cancel: h*
depends only on (beta, k_eff, kappa) — two machine-measurable knobs and one
order-1 geometry factor — not on the dataset's noise scale or class
separation. A single homophily threshold should therefore govern ALL
datasets, which is exactly the empirical finding of the eta x k sweep ("one
law: k-opt where hom(k) >= ~0.7... harm not tunable"). No norm-level theory
can produce this: its condition (D2) depends on sigma/Delta.

**(C2) No small-beta rescue — harm is not tunable below the gate.** As
beta -> 0, rho(beta) = 1 - beta + O(beta^2), so h*(beta -> 0) = 1 - 1/(2*kappa)
(= 0.5 at kappa = 1, ~0.2 at the measured kappa ~ 0.6). At margin level the
signal damage and the noise reduction are BOTH first order in beta, so the
gate persists at every beta: below h*(0) no down-weighting helps, matching
"harm not tunable" and "aircraft champion = NO qe". Contrast D2, where
beta -> 0 always helps — the margin/norm asymmetry in one line. The gate is
monotone in beta: h*(beta) rises from 1 - 1/(2*kappa) toward
1 - (1 - 1/sqrt(k_eff))/(2*kappa) at beta = 1, so MORE smoothing needs MORE
homophily — the correct direction for a gated knob.

**(C3) Classic QE's implicit beta.** Self-weight-1 QE with uniform weights
(a = 0) gives beta_hat = k/(k+1) — exactly the variance-optimal beta
(argmin of rho^2 is beta = k_eff/(k_eff+1); the ego is just one more
neighbor), where rho = 1/sqrt(k_eff+1), the textbook averaging rate. With
a > 0, beta_hat = W/(1+W) rises with neighborhood SIMILARITY: W = sum cos^a
is largest exactly on fine-grained data whose neighborhoods are near-
duplicates but impure (measured: beta_hat = 0.86 on aircraft vs 0.62 on
cifar100). The self-tuning knob adapts in the WRONG direction across our
regime map — smoothing hardest where homophily is lowest — which sharpens,
rather than softens, the need for the gate.

**(C4) One hop is optimal.** After one hop the margin-noise floor is already
rho ~ 1/sqrt(k_eff+1); iterating the map multiplies the mean-separation
factor [1 - 2*beta*kappa*(1-h)] AGAIN each hop (the drift is a systematic
field, it accumulates, it never averages out), while additional variance
reduction is marginal (2-hop neighborhoods overlap, so the smoothed
neighbor vectors are positively correlated and the effective count grows
sublinearly). Hence d' strictly falls at hop 2 except at h ~ 1, where
there is nothing left to harvest. This is the "recip/2-hop/znorm KILLED —
one hop = OPTIMUM" pilot result, now a corollary.

**(C5) The (a,k) knobs move on one frontier.** Raising a concentrates
weight (h_w up through purer near neighbors, k_eff down through fewer
effective ones); raising k does the reverse. Both slide along the same
(h, k_eff) tradeoff inside (D3), which is why the eta x k x weighting sweep
found the incumbent near-optimal and no setting that crosses the gate.

**(C6) Both histogram moves are one event.** d' is the single quantity
whose improvement is "green hump left AND red peak right" simultaneously
(the §0.2 figure of `dwt_theory.md`); its numerator failing first on the
true-class side at low h is precisely the CDF-overlay finding that the gate
localizes to condition (i).

---

## 6. Measured constants: the theorem against five datasets (no fitted knobs)

`src/dwt_gate_constants.py` (this branch) measures every constant in (D3)
in the raw space where qe acts, on the real embeddings and their pool
carve-outs (k = 10, a = 3, 4000 egos, anchors = labeled class means;
kappa one-sided per-ego, ratio-of-means; `output/dwt_theory/gate_constants.json`,
run 2026-08-12):

```
dataset        h_w    beta_hat  k_eff  kappa  rho    h*     d'ratio  axis SNR
cifar100       0.809  0.615     9.38   0.629  0.434  0.269  1.96     3.63
miniimagenet   0.919  0.693     9.30   0.717  0.382  0.378  2.41     4.22
cifar10        0.969  0.644     9.58   0.755  0.413  0.396  2.35     4.81
stanford_cars  0.461  0.827     9.83   0.585  0.315  0.292  1.52     2.41
aircraft       0.258  0.862     9.87   0.615  0.307  0.346  0.70     1.23
```

(axis SNR = Delta_pair/sigma_v, the nearest-pair separation in noise units.)

Reading:

- **Every dataset where qe has been tested lands on the predicted side.**
  cifar100/mini/cifar10 (h_w = 0.81/0.92/0.97, all >> h*): predicted
  d'-ratios 2.0-2.4, observed: qe = first non-{PCA,whitening} menu member,
  new bests (pool-repr-menu round 1). aircraft (h_w = 0.258 < h* = 0.346):
  predicted d'-ratio 0.70 — margin DEGRADATION; observed: champion needs qe
  OFF, harm not tunable. The measured h_w reproduce the logged purity map
  (.80/.92/.26/.46) — the instrument is consistent with the deployed
  diagnostic.
- **The gate sits at h* ~ 0.27-0.40, NOT at the folklore ~0.7.** The
  empirical "break-even ~0.7" was interpolated across a DATA GAP: the
  regime map had gain at h = 0.8 and harm at h = 0.45 for the SNAPS SCORE
  correction — a different operator (score-space, per-class column, no
  variance-optimal self-weight). For the representation smoother, D3 puts
  the break-even far lower.
- **stanford_cars (h_w = 0.461) was the discriminating cell — OUTCOME
  (2026-08-13, `src/cars_qe_gate_experiment.py`, `output/cars_qe_gate/`):
  qe HARMS.** Champion pipeline, paired splits, 20 trials, balanced
  2/4/8 shots/class (cal 392/784/1568; cal=200 is degenerate at K=196 —
  1 shot/class collapses the LOO prototype): wt -> qe_wt set size
  57.1 -> 62.8 / 21.9 -> 28.1 / 13.3 -> 18.6 (+11/+28/+40%), coverage on
  target in every arm (validity free, as §7 promises). The registered
  escape hatch fired exactly as written: measured margin d'-ratio 0.735
  (1% of 196 pairs improved) vs the (I)-model 1.52 — the (V1)/(V2)
  selection effects are first-order at mid-homophily and cross the sign
  boundary there. D3's margin->size link HELD (d' fell, sets grew); what
  failed is the (I)-idealized prediction of d' from composition constants.
  Companion measurement on all five datasets (`src/measure_dprime_all.py`,
  `output/cars_qe_gate/dprime_predicted_vs_measured.json`): measured
  ratios 1.05/1.18/1.21/0.74/0.64 (cifar100/mini/cifar10/cars/aircraft)
  vs predicted 1.96/2.41/2.35/1.52/0.70 — the (I)-model overpredicts
  EVERYWHERE (the damping is universal, not cars-specific), the measured
  ratio is monotone in h_w, and its sign predicts the qe verdict 5/5.
  Net: the empirical gate lies in h_w in (0.46, 0.81); h* is a floor
  only; the quantitative gate needs the G2 remainder lemma (V1/V2
  damping), not more data cells. Deployment rule unchanged in practice:
  qe stays OFF below h ~ 0.8 until G2 is quantitative.
- The axis-SNR column quantifies the norm-vs-margin lesson: aircraft's
  nearest pairs sit 1.2 noise-sd apart along the pair axis (vs 3.6-4.8 on
  the gate-ON datasets) — the bias dbar competes with sigma_v, not with
  sigma, and (1-h)*Delta_pair ~ sigma_v is reachable exactly there.

---

## 7. Validity is free, exactly as in the template

T_D is a fixed measurable function of the pool applied pointwise to
cal/test; conditional on the pool, exchangeability of {(T(x_i), y_i)} is
preserved, so split CP keeps coverage in [1-alpha, 1-alpha + 1/(n+1)) and
the FCP wrapping stays exact (Prop 1, `dwt_theory.md` §2 — proved, no gaps;
validated by the exchangeability oracle in the qe implementation). This is
DAPS Proposition 2 one level down. As in [Z23], the division of labor is
total: Theorems D1-D3 and all their assumptions touch SIZE only, never
coverage. A wrong gate costs efficiency, never validity.

---

## 8. What is assumed where: (I), its two signed violations, and the ledger

Everything above (D1) is unconditional. D2-D3 assume (M1) + (I). (I) fails
in exactly two ways, both with a KNOWN SIGN — this is gap G2 of
`dwt_theory.md`, now with directions:

- **(V1) Same-class selection tilt.** Neighbors are selected because they
  are close to x, so selected same-class noises are tilted toward e: part
  of nu is RETAINED OWN NOISE rather than fresh noise. Mean-shift view
  ([FH75]; rigorous small-ball expansions in [ACMP16]): the kNN mean
  estimates the local mean, so the tilt acts as an effective shrinkage of
  beta — LESS denoising than D2/D3 claim, never a sign flip. Direction:
  d'-ratio overestimated toward 1 from the gain side (gains shrink), gate
  position pushed UP.
- **(V2) Foreign-class selection alignment.** Wrong-class neighbors are
  selected precisely when e points toward their anchor, so the drift dbar
  is conditionally ALIGNED with the ego's own error: E[<e, dbar> | C] > 0
  and the effective kappa exceeds its random-composition value. Direction:
  harm at low h is WORSE than the (I) model predicts; gate position pushed
  UP. (V2) also explains why hub-debiased selection (gamma ~ 1, the
  qe-upgrade pilots' one confirmed upgrade) helps: hubness corrections act
  on the selection distribution, i.e. directly on the (V1)/(V2) tilt.
- Both violations push h* UP: the measured h* ~ 0.3-0.4 is a FLOOR. The
  cars run (Section 6, 2026-08-13) CONFIRMED the floor reading: harm at
  h_w = 0.46 despite the (I)-model predicting gain, with the measured
  d'-ratio (0.735 vs predicted 1.52) exhibiting the damping directly. The
  five-dataset predicted-vs-measured comparison bounds the empirical gate
  in h_w in (0.46, 0.81) and shows the damping is universal (every
  measured ratio sits below its prediction; the measured sign still calls
  the verdict 5/5) — the G2 remainder lemma is now the single blocker for
  a quantitative gate.

### 8b. The overprediction localized (2026-08-13): there is no averaging dividend — the noise is a shared local field

`src/dprime_overprediction_diagnostic.py` +
`output/cars_qe_gate/dprime_overprediction.{json,png}` decompose the
measured ratio into D3's own two factors, ratio = S/N (signal shrink over
noise shrink), per dataset, plus the exact empirical decomposition of the
denominator

```
N^2 = (1-beta)^2 + beta^2 * Var(nu_v)/sigma_v^2
      + 2*beta*(1-beta) * Cov(e_v, nu_v)/sigma_v^2
```

(under (I): Var(nu_v)/sigma_v^2 = 1/k_eff ~ 0.10, Cov = 0). Sanity: the
reconstructed N^2 matches the directly measured N^2 on all five datasets,
S/N reproduces the measured full ratios, and the per-ego h_w reproduces
the Section-6 gate constants. Measured (c10/mini/c100/cars/aircraft):

```
S:  model 0.97/0.92/0.85/0.48/0.21   measured 1.05/1.04/1.00/0.69/0.60
N:  model 0.41/0.38/0.43/0.32/0.31   measured 0.90/0.93/0.98/0.98/0.96
Var(nu_v)/sigma_v^2:  model ~0.10    measured 0.76/0.80/1.02/1.02/0.96
Cov(e_v,nu_v)/sigma_v^2:  model 0    measured 0.77/0.77/0.89/0.91/0.91
```

**Finding 1 — the denominator carries the overprediction, and the
mechanism is mean-shift, not sampling noise.** The promised sqrt(k)
dividend does not exist: N ~ 0.90-0.98 everywhere instead of rho ~ 0.31-
0.43. The neighbor mean nu has the variance of a SINGLE point (not 1/k of
it) and is ~0.85-0.90 correlated with the ego's own error: nu(x) is the
local density mean at x ([FH75]), and the within-class displacement e is
dominated by a smooth structured field (sub-cluster/pose/viewpoint
geometry) that the ego and its pool neighbors SHARE. Averaging cancels
only the thin iid shell around that field. Writing sigma_v^2 =
phi*Var(field) + (1-phi)*Var(iid), the corrected noise law is N^2 = phi +
(1-phi)*(1-beta)^2, and inverting the measured N^2 gives phi ~ 0.78-0.96
across the five datasets — the within-class variance along confusable
axes is ~80-95% shared structure. This is (V1) made precise and
quantified: "retained own noise" is in fact the entire local field.

**Finding 2 — the numerator damage is ALSO overpredicted, in the helpful
direction.** S_meas > S_pred everywhere; on the three gain datasets the
mean separation does not shrink at all (S ~ 1.00-1.05 — mild
mode-sharpening: pool neighbors pull egos toward class-conditional
density modes, which are better separated than means). The composition
kappa overcounts damage because part of the foreign "drift" is itself
absorbed into the shared-field move. On cars/aircraft the residual
signal damage (S = 0.69/0.60) is real and, with no noise dividend to pay
for it, is the entire mechanism of harm. Made quantitative via the
effective drift concentration kappa_2 = (1 - S_meas)/(2*beta*(1-h))
(the kappa that would reproduce the measured S in the D3 numerator):

```
                 kappa_2 (from S_meas)   kappa (composition, (I)-model)
cifar100         0.02                    0.63
miniimagenet    -0.37                    0.72
cifar10         -1.35                    0.76
stanford_cars    0.35                    0.59
aircraft         0.31                    0.62
```

kappa_2 ~ 0.31-0.35 at low/mid h but ~0 or NEGATIVE at high h: the field
absorbs the drift entirely (and mode-sharpening overshoots it). This is
why a naive plug-in repair FAILS: keeping the (I) numerator and only
correcting the variance law (N^2 = phi + (1-phi)(1-beta)^2) yields a
gate h*' = 1 - (1-N)/(2*beta*kappa) ~ 0.95+ with cars's constants —
contradicting the observed gain at h = 0.81. The damage term is itself
h-gated (field absorption), beyond the explicit (1-h) factor. D3-prime
therefore needs BOTH corrected laws: the settled variance law AND a
field-aware signal law S(h) with an absorption factor — the latter is
the open half of G2.

**Consequence for the theory.** The (I)-model got both factors of the
high-h regime wrong in compensating directions (big fake dividend, big
fake damage), which is why its SIGN predictions held at high h while both
its factors were off; at mid h the errors stop compensating and the sign
flips (cars). The corrected mechanism story: **qe is not a variance
reducer; it is a mean-field mover** — it trims the ~5-20% iid shell and
shifts egos along the shared local field, helping when that field flows
toward the own-class mode (high h) and harming when it flows toward the
confusable class (low h). The G2 remainder lemma therefore has a concrete
target shape: a two-component noise model (shared field m(x) + iid
shell), with phi as its one new measurable constant, replacing 1/k_eff by
phi + (1-phi)/k_eff in the variance term and re-deriving the gate. V2
alignment is confirmed but secondary (corr(e_v, dbar_v) = 0.23-0.70,
largest where drift is smallest).
- Remaining formal debt (unchanged from the `dwt_theory.md` ledger): a
  rigorous (V1)/(V2) remainder lemma (G2 — route: [ACMP16] small-ball +
  bounded density ratio on the kNN ball; note [BJ21]-style kNN-regression
  bounds do not apply verbatim since pool vectors are both design and
  response); the moments -> right-tail-dominance bridge feeding Prop C
  (G6); the ratio-NCM (G7); kappa concentration across egos (new, mild —
  we use ratio-of-means over 4000 egos).
- Downstream hook: D3's conclusion — standardized margin improves — is
  EXACTLY the input Lemma B (§5) needs, whose output feeds Prop C (§6):
  dominance on R_alpha => E|C^T| <= E|C| via the order-statistic argument
  [D24]. D3 replaces the informal "Lemma A improvement condition" with a
  quantitative, measured gate; the rest of the chain is untouched.

---

## 9. Scorecard against the template

Matched (same epistemic grade as [Z23]):
- D1 = DAPS Theorem 2 verbatim in representation space (triangle
  inequality, deterministic, gap-free), with the graph-radius assumption
  replaced by a measured impurity budget.
- Validity companion free and exact (their Prop 2 = our Prop 1).

Added (beyond the template):
- The theorem lives at the object the method transforms (their footnote-5
  mismatch closed).
- The self-built-graph case they left to future work is handled under a
  named idealization (I) with SIGNED violation directions and a measurement
  protocol, instead of being assumed away.
- A second-moment theory (D2) exposing the averaging mechanism, plus the
  negative result that the norm level cannot gate.
- The operative gate (D3) is scale-free with zero fitted constants,
  derives the one-law/no-tunability/one-hop/knob-frontier phenomenology,
  and survives a five-dataset measured test with one falsifiable
  out-of-sample prediction (cars) registered before the experiment.

Owed (the honest debt, tracked in the G-ledger):
- G2 rigor (the (V1)/(V2) remainder lemma), G6 (margin dominance on
  R_alpha -> set size), G7 (ratio NCM), G8 (FCP-arm size identity), and
  the label-free surrogate for (h_w, kappa) — the deployability blocker
  the gate theorem now gives a precise shape to: predict sign(d'-ratio)
  without pool labels.

---

## 10. SNAPS Proposition 2 dissected: how a DAPS-style pointwise theorem became an E|C| statement, and what transplants (2026-08-16)

Weekly goal 4. Source: Song et al., NeurIPS 2024 ("SNAPS"), Prop 2 +
Appendix A.2 (docs/SNAPS-...-GNN.pdf, pp. 6, 14-15). DAPS Theorem 2 (our
Section 0 template) never mentions set size; SNAPS Prop 2 concludes
E[|C~(x)|] <= E[|C(x)|]. This section extracts how, grades the proof, and
states what our chain should import.

### 10.1 What Prop 2 actually is

Statement (translated to our notation): assume EVERY aggregated node has
the ego's label (h = 1, stated upfront). The aggregated score is idealized
in the proof (their eq. 6) as

```
S_hat_vk = (1-lam) * S_vk + lam * Ek[S_uk] ,
```

i.e. the realized neighbor average is replaced by the POPULATION
class-conditional mean score Ek[.] — a silent infinite-neighbor /
concentration idealization. Two conditions: (a) Ek[S_uk] < eta (the
class-mean true-label score sits below the base 1-alpha cal quantile eta);
(b) for false labels i, Ek[S_ui] >= (1-eps_ki)*Ek[pi_max] + Ek[xi*pi_i],
where eps_ki = fraction of class-k points whose top prediction is i
(their Lemma 1 — an unconditional APS identity, the only fully proved
ingredient). Conclusion: E|C~| <= E|C|.

Proof skeleton — a 2x2 case split at the level of SCORE ENTRIES, i.e. of
the individual membership events {S_vc <= eta}:

- True-label entry k: if S_vk >= Ek[S_uk], the convex combination LOWERS
  it (contraction toward a smaller target); else S_hat_vk < Ek[S_uk] < eta
  by (a) — already safe and stays safe. Direction favorable in both cases,
  deterministically.
- False-label entry i: if S_vi <= Ek[S_ui], contraction RAISES it; else
  S_hat_vi - eta > Ek[S_ui] - eta >= -Delta_S with
  Delta_S = eta - (Lemma-1 bound), and the proof says Delta_S "is very
  small", so the probability of the entry dropping below the new quantile
  "is very low". No bound is given. This is the leak.
- Quantile: "since false scores corresponding to ground-truth labels will
  decrease, eta_hat < eta" — asserted from "some cal scores decrease",
  with no order-statistic argument.
- Final line: "Finally E|C~| <= E|C|" — the expectation step is literally
  summing the entry-wise directions; it carries no probabilistic content.

Grading: the upgrade over DAPS Theorem 2 is NOT a new probabilistic tool.
It is (1) a change of OBJECT — from representation/probability-space error
to score entries, whose set membership IS a threshold event, so pointwise
score moves translate directly into |C| moves via
|C(x)| = sum_c 1[s(x,c) <= eta]; plus (2) assumptions strong enough
(h = 1, population-mean targets, conditions (a)+(b)) to make the move
direction of every entry deterministic; plus (3) two unproved steps (the
residual false-entry case and eta_hat < eta). Measured against our chain:
**Prop C (dwt_theory.md Section 6) is already strictly more rigorous than
SNAPS Prop 2** — its Binomial order-statistic monotone coupling IS the
missing proof of their eta_hat < eta step, its R_alpha restriction is
their implicit "only the crossing region matters", and its K*delta_n term
is an explicit version of their "very low probability". The literature
sweep's placement (litsweep: "SNAPS Prop 2 = h=1 corner") stands, now
with the proof-level autopsy.

### 10.2 The one device worth importing: contraction-toward-target

What Prop 2 has that our chain lacks: it never touches G6. Our (5c) route
needs "moments -> stochastic dominance on R_alpha", which moment
improvements alone cannot give (crossing CDFs). SNAPS sidesteps dominance
entirely: a convex combination with target m moves every score
MONOTONICALLY toward m, so conditional on WHICH SIDE of the threshold the
target sits, the direction of each membership event is known POINTWISE —
no distributional shape, no Gaussianity, no dominance. The distributional
question is compressed into a single event: "is the target on the safe
side?".

This transplants to our setting because the smoother is also a convex
combination, and D3 already identified the level at which it is LINEAR:
the pair-margin statistic g(x) = <x, v>,

```
g(x_hat) = (1-beta) * g(x) + beta * g(nu(x)) .
```

The ego's margin contracts toward the margin of its pool-local mean. The
SNAPS case split then runs verbatim per confusable pair — with our
measured objects in the roles their assumptions played:

| SNAPS ingredient                       | our object |
|----------------------------------------|------------|
| target Ek[S_uk] (population class mean) | g(nu(x)) — per-ego local-mean margin, E[g(nu)] = <mu_y,v> - D_y (D3 drift) |
| h = 1 assumption                        | (1-h) drift D_y, D_c — absorbed as reduced slack, measured |
| condition (a) Ek[S_uk] < eta            | anchor condition (A) below |
| Var of target = 0 (population mean)     | Var(g(nu)) ~ phi * sigma_v^2 — the Section-8b shared field; does NOT vanish with k |
| "very low probability" residual         | explicit Chebyshev remainder (below) |
| eta_hat < eta assertion                 | Prop C's order-statistic coupling (already proved) |

### 10.3 Corollary D4 (target shape): the contraction-anchor route to E|C|

Proposed statement shape, to be proved on top of D1 (the base lemma) and
slotted where G6's named condition currently sits:

**Anchor condition (A), per confusable pair (y,c):** with t* the
pair-threshold image of the cal quantile under Lemma B's score transfer,
and slack tau > 0:

```
for y-egos:  g(nu(x)) >= t* + tau     (local mean margin safely accepting)
for c-egos:  g(nu(x)) <= t* - tau     (local mean margin safely rejecting)
```

**Claim (shape).** Under (M1) + (A): every ego whose own margin is on the
safe side STAYS safe (contraction toward a safe target cannot cross);
egos on the wrong side move toward safety; the only leak is egos whose
TARGET is itself on the wrong side. Hence, per pair,

```
E|C^T| <= E|C| + sum_pairs P( g(nu(X)) on the wrong side | class )
              + K*delta_n + O(1/(n+1)) ,
```

with the middle term = the FIELD-LEVEL misclassification rate — exactly
Section 8b's mean-field-mover mechanism ("qe helps iff the local field
flows toward the own-class mode") appearing in the E|C| accounting. For
egos in the residual case, the overshoot probability is controlled by
Chebyshev on the (1-beta)-scaled ego deviation:
P(cross despite safe target) <= (1-beta)^2 * sigma_v^2 / tau^2 — an
explicit constant where SNAPS wrote "very low".

What this buys, precisely:

- **G6 reframed, not just discharged-by-assumption.** The named condition
  "score-margin dominance on R_alpha" is REPLACED by (A) + explicit
  remainder. (A) is a first-moment condition on measurable objects (D_y,
  D_c from the D3 instrument; Var(g(nu)) = phi-field variance from 8b) —
  a far easier target than tail dominance, and checkable with pool labels
  on every dataset. What remains of G6 is only locating t* in R_alpha
  (Prop C's condition (iii), unchanged).
- **Strictly stronger than Prop 2 where it applies:** h = 1 and zero
  target variance are the corner (A) with infinite slack and empty
  remainder; drift (h < 1) and field variance (phi > 0) enter as measured
  slack reductions instead of being assumed away.
- **The honest new difficulty (not in SNAPS):** our target g(nu) is a
  per-ego random variable correlated ~0.85-0.90 with the ego's own margin
  (8b), not a fixed population mean. The remainder is therefore governed
  by the FIELD distribution, not by 1/k concentration — there is no
  averaging rescue (the same lesson as 8b, resurfacing in the E|C| step).
  On cars/aircraft it is (A) that fails, and the field-misclassification
  term is the mechanism of harm — one more place the two-component noise
  model (G2's target) is the single missing ingredient.
- **Score-vs-margin conversion:** SNAPS aggregates scores (affine in
  scores); we aggregate representations (affine only in g, per pair). The
  cosine/softmax-LAC score is monotone-Lipschitz in g (Lemma B (5a)), so
  per-pair threshold events transfer; assembling K-1 pairs costs a union
  bound or the max-margin form — bookkeeping, not a gap.

### 10.4 Verdict (weekly goal 4)

How SNAPS upgraded a DAPS-style theorem to expected set size: move the
pointwise argument to score entries where membership is a threshold
event; assume perfect homophily and population-mean targets so every
entry's move is deterministic; sum entries and call it an expectation;
assert the quantile move; leave one case unbounded. The expectation
carries no probabilistic content, and our Prop C already exceeds the
proof's rigor on the two steps it does attempt. The importable content is
the CONTRACTION-ANCHOR DEVICE (10.2-10.3): it converts G6 from a
stochastic-dominance gap into a first-moment anchor condition (A) plus an
explicit remainder, both measurable with the instruments this document
already built (D3 drifts, 8b field variance). Proposed next theorem
target: Corollary D4 above, proved from D1 + (A), with the Gaussian case
of (5c) retained as a cross-check. The device also confirms, from an
independent source, that the R_alpha restriction (5c) is the right
formulation — SNAPS uses it implicitly and pays for skipping its proof.
