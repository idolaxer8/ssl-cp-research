# Theory — Validity of the SSL + Full-CP Pipeline

> Formal companion to `findings.md`. Pull from here when writing the
> Methods/Theory section. Goal: state the **coverage guarantee for the winning
> method** (Full CP + PCA + MS-CS) precisely, and show that each add-on
> preserves it. Citations live in `literature.md`.

---

## 0. Setup & notation

We observe a calibration set `Z_{1:n} = {(X_i, Y_i)}_{i=1}^n` and a test point
`(X_{n+1}, Y_{n+1})`, all taking values in `𝒳 × [K]` (here `𝒳 = ℝ^d` is the
frozen DINOv2 embedding space, `K` = #classes). We additionally hold a
**disjoint unlabeled pool** `U = {X_j^u}_{j=1}^m`, sampled independently of
`Z_{1:n+1}`. Target miscoverage `α = 0.1`.

A **nonconformity measure (NCM)** is a function `s(x, y; B)` that scores how
unusual the label `y` is for `x`, given a reference bag `B`. Larger `s` =
more nonconforming. Our headline NCM is the whitened geodesic top-k mean
(`GeodesicTopKMeanNCM`):

```
s(x, y) = arccos( mean_k_{i: Y_i = y}  ⟨z(x), z(X_i)⟩ )      (numerator: same-class)
          ────────────────────────────────────────────────
          arccos( mean_k_{i: Y_i ≠ y}  ⟨z(x), z(X_i)⟩ ) + ε  (denom: other-class)
```

where `z(·) = (w ⊙ ·) / ‖w ⊙ ·‖` is the whitened, L2-normalised embedding.
`w ∈ ℝ^d` is a **diagonal** per-dimension scale `w_j = 1/√(v_j + r)`, with `v`
the *pooled within-class* variance (mean over all `n` cal points of the squared
class-mean-centred residual) and `r = max(reg, 0.01·median(v))` an adaptive
regularization floor — **not** a full `d×d` covariance root (this matters for the
`O(1/n)` argument; see §1). `mean_k` averages the `k` largest cosine similarities. The
asymmetric variant (`geodesic_topk_asym`) uses a 1-NN `max` numerator. Both are
**symmetric functions of the same-class / other-class neighbour multisets** —
this is the only structural property the validity proof needs.

The exact key quantity:

```
exchangeability:  the joint law of (Z_1, …, Z_{n+1}) is invariant under any
                  permutation of the n+1 indices.
```

i.i.d. ⇒ exchangeable; exchangeability is strictly weaker and is all we use.

---

## 1. The coverage guarantee (Full / transductive CP) — **main result**

This is the guarantee the whole pipeline rests on. Full CP (Vovk, Gammerman &
Shafer; Shafer & Vovk 2008) is **transductive**: for each test point and each
candidate label it re-scores the *augmented* bag, with no train/calibration
split.

**Construction.** For a test input `x* = X_{n+1}` and a candidate label
`y ∈ [K]`, form the augmented bag

```
B^y = { (X_1, Y_1), …, (X_n, Y_n), (x*, y) }            (n+1 points)
```

Score every point in `B^y` *against `B^y` itself*:

```
σ_i^y = s(X_i, Y_i;  B^y),   i = 1, …, n          (calibration scores, recomputed)
σ_*^y = s(x*,  y;    B^y)                           (test score)
```

The conformal **p-value** for label `y` is the rank of the test score:

```
p(y) = ( #{ i ∈ [n] : σ_i^y ≥ σ_*^y } + 1 ) / (n + 1)        (Eq. 1)
```

and the **prediction set** is

```
C(x*) = { y ∈ [K] : p(y) > α }.                              (Eq. 2)
```

> Code: `FullConformalPredictor.predict` — `updated_calibration_scores_for`
> recomputes `{σ_i^y}` for the augmented bag, `score_x` gives `σ_*^y`, and the
> p-value is `(n_greater + 1)/(n_cal + 1)` exactly as in Eq. 1.

**Theorem 1 (marginal coverage).** *If `(Z_1, …, Z_n, (X_{n+1}, Y_{n+1}))` are
exchangeable and the score map `s(·;·)` is invariant to permutations of its
reference bag, then for the true label `Y_{n+1}`*

```
ℙ( Y_{n+1} ∈ C(X_{n+1}) )  ≥  1 − α,
```

*and if scores are a.s. distinct, the coverage is also `≤ 1 − α + 1/(n+1)`.*

**Proof sketch.** Evaluate the construction at the *true* label `y = Y_{n+1}`.
Then `B^{Y_{n+1}} = Z_{1:n+1}` is the full exchangeable sample, and because `s`
is a symmetric function of that bag, the scores `(σ_1, …, σ_n, σ_*)` are an
exchangeable sequence of `n+1` real numbers. Hence the rank of `σ_*` among them
is (sub-)uniform on `{1, …, n+1}`, so
`ℙ(p(Y_{n+1}) ≤ α) ≤ ⌊α(n+1)⌋/(n+1) ≤ α`. Since `Y_{n+1} ∈ C(X_{n+1})` iff
`p(Y_{n+1}) > α`, the lower bound follows; the upper bound is the no-ties case. ∎

**Why this is the *winning* method's guarantee.** The only requirement is
exchangeability of the bag + a symmetric score. Full CP never trains a model on
a held-out split, so there is no classifier-quality bottleneck — this is the
structural reason FCP dominates Split-CP / SemiCP / CV+ at small `n`, large `K`
(see §4). The guarantee holds at *every* `n`, including the few-shot regime
(`n = 2K … 8K`) that is our deployment setting.

**The one approximation: whitening.** The whitening scale `w` (and any RBF
bandwidth `σ`) is fit once on the cal set `Z_{1:n}` and *not* refit on each
augmented bag `B^y`. So `z(·)` is *applied* to all `n+1` bag points but
*estimated* from only the first `n`: swapping the test point for a cal point
would change `w` by `O(1/n)`, so the bag scores are exchangeable only up to an
`O(1/n)` term and coverage is `1−α + O(1/n)`, not exactly `≥ 1−α`.

This is a far *gentler* `O(1/n)` than "frozen whitening matrix" suggests, for
three implementation reasons:
- **Diagonal, not full.** `w` is `d` per-dimension scales, not a `d×d` matrix
  root. A full covariance is singular / ill-conditioned when `n < d` (our regime:
  `n = 200`, `d = 128`–`768`); the diagonal is always well-conditioned, so its
  per-point sensitivity is tiny and uniform across dimensions.
- **Pooled from residuals → uses all `n`.** `v` pools class-mean-centred
  residuals across *all* classes, so every cal point contributes regardless of
  `K`. Even at `k = 2` samples/class, `v` is estimated from `n` residuals, not
  `2` — this is what makes adding one point an `O(1/n)`, not `O(K/n)`, perturbation.
- **Adaptive regularization.** The floor `r = max(reg, 0.01·median(v))` inside
  the root shrinks the scales, further damping sensitivity to any single point.

`w` *does* use cal **labels** (to centre residuals), so it is label-dependent —
but only through a pooled *scale*, which one point barely moves. This is the safe
kind of label use; contrast LDA / pseudo-label heads (next paragraph), which use
labels to choose a discriminative *direction* that one point can swing materially
— **not** `O(1/n)`, hence structural under-coverage. Empirically coverage sits at
nominal for `n ≥ 200` (findings §1); Fan & Sesia (2025, arXiv:2512.15383) prove
`O(1/n)` validity for exactly this class of data-dependent standardizations. An
*exact* guarantee would refit `w` on `B^y` per candidate (`O(1/n)` cost, not
needed in practice) — same concession as `MahalNNRatio` /
`WhitenedGeodesicNNRatio`.

**Contrast — what breaks the guarantee.** Any preprocessing fit on the
calibration *labels* (LDA, a pseudo-label-trained projection/head) makes `s`
depend on the bag through a *label-dependent, non-`O(1/n)`* transform; the
symmetry argument fails and we observe structural under-coverage (70–85%,
findings §9, §10 negative results). Likewise, pool augmentation that feeds test
information back into the cal scores breaks exchangeability (`O(N/n)` leak).

---

## 2. Add-on A — PCA dimensionality reduction (exchangeability preserved)

We replace `X` by `Π X` where `Π ∈ ℝ^{d'×d}` projects onto the top `d'`
principal directions. The validity question is *what data `Π` is fit on*.

**Proposition 2 (unsupervised projection is free).** *Let `Π = Π(U)` be any
measurable function of the unlabeled pool `U`, with `U ⊥ Z_{1:n+1}`. Then
`{(Π X_i, Y_i)}_{i=1}^{n+1}` is exchangeable, and Theorem 1 applies verbatim to
the projected NCM `s(Πx, y)`. Coverage `≥ 1 − α` is preserved exactly — no
`O(1/n)` term.*

**Proof.** Condition on `U`. Given `U`, `Π` is a fixed deterministic map, and a
fixed map applied coordinatewise preserves exchangeability of
`Z_{1:n+1}`. Theorem 1 holds conditionally on `U`; marginalise over `U`. ∎

This is why we fit PCA on the **disjoint unlabeled pool**, never on cal/test.
Practically `m ≥ 500` is needed for a non-degenerate basis; the recipe plateaus
at `m ≈ 2500–5000` (findings §3). Optimal `d' = 128` (coarse) / `512`
(fine-grained); the U-shape is bias–variance in the NCM, **not** a coverage
effect — coverage is flat in `d'` by Prop. 2.

**Cal-fit PCA breaks it.** If `Π = Π(Z_{1:n})` (PCA on calibration features),
the projection is a non-`O(1/n)` function of the bag — the same mechanism as
LDA — and we observe under-coverage at high `d'` (findings §3, §10). The
unlabeled pool is what makes the add-on theoretically clean.

The whitening inside the NCM is complementary to PCA and not redundant: PCA
removes low-variance noise directions (total covariance), whitening rescales by
*within-class* covariance. Both are unsupervised wrt cal labels in the PCA-fit
sense above (whitening is the `O(1/n)` cal-fit term of §1).

---

## 3. Add-on B — MS-CS class-similarity penalty (formalized vs Fargion)

### 3.1 Reference: MA-CS (Fargion, Dabah & Tirer 2025, arXiv:2511.19359)

Fargion et al. add a group-error penalty to the score. With a semantic grouping
`g : [K] → [G]` they define a **binary** dissimilarity and penalized score:

```
d(y, y') = 𝟙{ g(y) ≠ g(y') }                                       (Fargion Eq. 1)
s_λ(x, y) = s(x, y) + λ · d( y, ŷ(x) )                             (Fargion Eq. 2)
```

where `ŷ(x)` is the predicted (1-NN) label. Their validity claim
(their Thm 3.1, after Vovk et al.): *`s_λ` is itself a valid nonconformity
score, so it inherits the coverage guarantee* — the penalty changes *which*
sets you get, never *whether* they cover. In their **split-CP** setting this is
immediate: any fixed deterministic transform of a valid score is a valid score.

### 3.2 Our MS-CS: label-free, continuous, data-driven `M`

We remove the need for a human taxonomy `g`. Build a **class-similarity matrix**
`M ∈ [0,1]^{K×K}` from k-means on the unlabeled pool `U`
(`build_cluster_similarity_matrix`):

```
1. k-means(U) → cluster centroids {μ_1, …, μ_C}, assignment a: [C clusters].
2. class centroid  c̄_y = mean{ X_i : Y_i = y }     (from cal; global mean if absent)
3. class→cluster   κ(y) = argmin_c ‖ c̄_y − μ_c ‖
4. M[y, y'] = 1                       if y = y' or κ(y) = κ(y')          (same cluster)
            = exp( −‖μ_{κ(y)} − μ_{κ(y')}‖² / τ )   otherwise           (Gaussian kernel)
   with τ = (scale) · median pairwise squared inter-cluster distance.
```

`M` is symmetric, `M[y,y] = 1`. The penalized score generalizes Fargion's
binary `d` to the continuous **dissimilarity** `1 − M`:

```
s_λ(x, y) = s(x, y) + λ · ( 1 − M[ y, ŷ(x) ] ),    ŷ(x) = 1-NN label.    (Eq. 3)
```

Fargion's MA-CS is the special case `M[y,y'] = 𝟙{g(y)=g(y')}` (cluster = group).
Best config: `λ = 0.05`, `C = 20` clusters, clustering in PCA-128 space
(findings §4). Gives 8–9% set reduction over FCP+PCA in the small-cal regime,
beating binary MA-CS by 2–7% with **no taxonomy labels**.

### 3.3 Validity of MS-CS — split CP vs Full CP (the subtle part)

**Split CP (easy direction, = Fargion).** If `M` and `ŷ` are fixed given the
unlabeled pool `U ⊥ Z_{1:n+1}`, then `s_λ` is a fixed deterministic transform of
a valid score, computed identically on cal and test. By the same conditioning
argument as Prop. 2, `s_λ` is a valid score and coverage `≥ 1−α` holds.

**Full CP (our setting — needs care).** In transductive CP the penalty itself
must be a **symmetric function of the augmented bag `B^y`**, because two of its
ingredients depend on the data:

- `ŷ(x)` is a 1-NN label *within the bag* — adding the candidate `(x*, y)` can
  change the LOO 1-NN of a calibration point;
- `M` is built from cluster *centroids*, and `c̄_y` shifts when `(x*, y)` joins
  class `y`: `c̄_y ← (n_y · c̄_y + x*)/(n_y + 1)`, which may flip `κ(y)`.

If we hold `M` and `ŷ` frozen at their cal-only values (the default,
non-exchangeable code path), the penalty is *not* bag-symmetric and the score
acquires the same `O(1/n)` asymmetry as whitening (§1) — empirically harmless
for `n ≥ 200`, coverage stays at nominal (findings §4).

The **exact** version (`run_fcp_with_mscs(..., exchangeable=True)`,
`update_M_for_candidate`) restores full symmetry: for each candidate `(x*, y)`
it (i) shifts `c̄_y`, re-runs `κ(y)`, and rebuilds the affected row/col of `M`;
(ii) recomputes each calibration point's penalty using the augmented-bag 1-NN
(if `x*` is now closer than its LOO neighbour, `ŷ` becomes `y`). Then the
*penalty* is a symmetric function of `B^y`; Theorem 1 then applies exactly up to
the base NCM's own whitening `O(1/n)` (§1), which `--exchangeable` does not touch.
Use `--exchangeable` when `n/K < 5` (where the `O(1/n)` slack is largest).

> Code mirrors the theory exactly: the test penalty `λ(1 − M[y, ŷ(x*)])` is
> added to `σ_*^y`, and `cal_penalty` (per-point `λ(1 − M[Y_i, ŷ_LOO(i)])`) is
> added to the recomputed cal scores *before* the rank in Eq. 1. Both sides get
> the penalty — that symmetry is what keeps the p-value valid.

---

## 4. Why Full CP beats the alternatives (validity-level explanation)

All three competitors satisfy *some* coverage guarantee; the issue is that at
small `n`, large `K` their guarantees are vacuous (sets of size `K`) while
FCP's stays tight.

- **Split CP / SemiCP.** Spend half the label budget `B` training an inner
  softmax classifier; with `K=100` and `≤300` cal the classifier accuracy is
  `<60%`, THR scores degenerate, sets collapse to size `K`. SemiCP's NNM
  unlabeled augmentation is null on THR in our regime (findings §2). FCP uses
  *all* of `B` for calibration of a frozen-feature NCM — no inner model.
- **CV+ / Jackknife+ (Barber et al. 2021).** Coverage `≥ 1 − 2α`; each fold
  trains on `(k−1)/k` of the data, so at low `n` classes vanish from folds →
  trivially full sets. FCP uses the whole cal bag transductively.

So FCP's advantage is not a tighter *theorem* — it is that its guarantee is the
one that remains **non-trivial** when labels are scarce and `K` is large.

### 4.1 Making a softmax head Full-CP-valid — the FCA family (prototype → ridge)

The split-CP failure above is about a softmax head trained *once on a disjoint
split*. A softmax head can instead be made **Full-CP-valid** by refitting it in
closed form per candidate — this is the idea of **FCA** (Full Conformal
Adaptation; Silva-Rodriguez et al., IPMI 2025, arXiv:2506.06076). FCA's SS-Text
probe is a class-mean prototype blended with a zero-shot **text** anchor,
`w_c = a·μ_c + b·t_c`, scored by LAC `s(x,y) = 1 − p(y|x)`. We have no text
encoder (pure SSL), so we keep the text-free skeleton and obtain two rungs:

- **Rung 3 — `PrototypeSoftmaxNCM` (vanilla).** Drop the text anchor: `w_c = μ_c`
  (class-mean prototype), `p(y|x) = softmax_c(⟨z, μ_c⟩/T)`, `s = 1 − p(y|x)`.
  No covariance term. `T` is a **fixed** temperature (FCA's `τ` analog).
- **Rung 4 — `RidgeSoftmaxNCM`.** Add back the feature covariance SS-Text discards,
  `w_c = (ZᵀZ + λI)⁻¹(Zᵀy_c + λ_a μ_c)` — a discriminative ridge probe (≈ cal-fit
  whitening, refit per candidate via Sherman–Morrison + PRESS LOO). Rung 3 is the
  `λ_a → ∞` limit; rung 4 is the clean ablation's "+covariance" arm.

**Exact exchangeability (rung 3).** This is an *instance* of Theorem 1, not a new
proof. Apply ONE symmetric score to every point `i` of a bag `B`:
`s_i = 1 − softmax_{y_i}(⟨z_i, μ_c^{(−i)}(B)⟩ / T)`, where `μ_c^{(−i)}(B)` is the
class-`c` mean over `B \ {i}` (leave `i` out of its own class only). Because the
prototype is **linear** in the data, this leave-one-out is the closed-form mean
update `μ_c^{(−i)} = (n_c μ_c − z_i)/(n_c − 1)` — no PRESS leverage needed (unlike
rung 4). Each `s_i` is a symmetric function of `B`, so the `n+1` augmented-bag
scores are exchangeable ⇒ cov ≥ 1−α for ANY fixed `T` (validity is `T`-independent;
`T` is an efficiency knob). The empty-class convention `μ_{y_i}^{(−i)}` undefined
⇒ `s_i = 1` fires identically for the held-out test point of an absent candidate
class and for a singleton cal class — the symmetric form of the missing-class fix.

**Consequence — bloat, not under-coverage.** With a fixed `T`, the same LOO rule,
and the symmetric empty-class rule, validity holds at every cal size. At small
**balanced** cal the prototypes are noisy → softmax ≈ uniform → high `(n+1)`
quantile → large sets. So the small-cal signature is *bloated sets at nominal
coverage*, never under-coverage. (A cal-fit `T` is the only O(1/n) break — it
routes through `warn_nonexchangeable`; under a *random* split a truly absent cal
class is structural under-coverage shared by all methods, a property of the split.)

---

## 5. Summary of approximations and their effects

Every step in the lead pipeline is either **exact** (preserves Theorem 1 with no
error term) or an **O(1/n)** concession (a bag asymmetry that vanishes as `n`
grows). The table inventories all of them; the two entries below the rule are the
failure modes an O(1/n) analysis does *not* cover — they break coverage outright.

| Component | Where fit / what it does | Exchangeability | Measured effect |
|-----------|--------------------------|-----------------|-----------------|
| Full-CP transductive p-value (§1) | per-candidate augmented-bag rank | **Exact** | the guarantee; cov ≥ 1−α at every `n` |
| PCA-128 on **unlabeled pool** (§2) | projection fit on `U ⊥ (cal,test)` | **Exact** (Prop. 2) | coverage flat in `d′`; sets −19% at cal=800 |
| Whitening inside the NCM (§1) | **diagonal** pooled-within-class scale `w`, fit on cal, not refit per bag | O(1/n) (gentle: diagonal + pooled + regularized) | cov at nominal for `n ≥ 200`; no measurable gap |
| **MS-CS frozen** penalty (§3.3) | `M`, `ŷ` fixed at cal-only values (default) | O(1/n) | **negligible** — table below |
| **MS-CS exact** (`exchangeable=True`) | `M`, `ŷ` recomputed per augmented bag | **Exact** | = frozen on 87–99.5% of sets; ~1.3–1.5× runtime |
| **FCP `update_calibration_scores=False`** (§5.1) | reuse static LOO cal scores at predict (no per-bag refit) | O(1/n) | = standard FCP set-for-set at all `B` |
| cal-fit PCA / LDA / pseudo-label head | projection fit on cal **labels** | **Breaks** (not O(1/n)) | structural under-coverage 70–85% |
| pool augmentation (test → cal feedback) | unlabeled points relabelled into cal | **Breaks** (O(N/n) leak) | invalid; archived |

**MS-CS frozen vs exact** (CIFAR-100, PCA-128, geodesic_topk_mean, λ=0.05,
5 trials, test=2000; `src/mscs_exchangeability_experiment.py`):

| cal (n/K) | frozen cov / sz / CovGap | exact cov / sz / CovGap | identical-set | Δcov (exact−frozen) |
|-----------|--------------------------|-------------------------|---------------|----------------------|
| 200 (2)   | 0.917 / 5.45 / 8.83      | 0.919 / 5.65 / 8.80     | 0.871         | +0.0021 |
| 400 (4)   | 0.888 / 1.92 / 8.93      | 0.889 / 1.93 / 8.93     | 0.980         | +0.0004 |
| 600 (6)   | 0.920 / 2.10 / 7.20      | 0.920 / 2.12 / 7.19     | 0.984         | +0.0001 |
| 800 (8)   | 0.900 / 1.59 / 7.68      | 0.900 / 1.59 / 7.69     | 0.995         | −0.0001 |

The frozen slack is sub-0.5 pp coverage, shrinks monotonically with `n`, and is
always validity-safe (exact ≥ frozen at small cal). So for the frozen lead
pipeline Theorem 1 holds to measurement precision at deployment cal (`n/K ≥ 4`);
only cal=200 shows a visible-but-tiny gap that `--exchangeable` closes. CovGap is
mode-independent to 2 decimals — the penalty path does not affect class-conditional
behaviour. (The cal=800 frozen row reproduces findings §2b exactly.)

### 5.1 `update_calibration_scores=False` (the option formerly called "SCP-geodesic")

Standard Full CP recomputes the calibration scores for each augmented bag
`{cal ∪ (x*, y)}` before ranking the test score (§1). Setting
`FullConformalPredictor.predict(update_calibration_scores=False)` skips that
refit: it reuses the **static leave-one-out** calibration scores from
`calibrate()` and ranks the test score against them. The NCM, the fit, and the
base `score_x` are identical — **not updating the cal scores at predict is the
*only* substantive difference** from FCP. (An earlier separate `SplitCPGeneric`
class called this "SCP-geodesic", which was misleading: there is *no* train/cal
split, so it was never inductive Split CP. The `n−1` (cal, self-excluded) vs `n`
(test) neighbour asymmetry sometimes quoted is merely the *consequence* of not
refitting, not a second independent difference.) Because the dropped augmentation
is `O(1/n)`, it reproduces standard FCP set-for-set at every `B` — verified on
CIFAR-100 (cal=400: cov 0.858 vs 0.862, sz 2.69 vs 2.70; memory
`[[scp-geodesic-isolates-ncm-vs-fcp]]`). It is kept as a flag for future tests
(e.g. directly measuring what the augmentation buys). The genuine inductive
baseline is instead **SCP-THR** (a softmax head on a disjoint split), which
*does* satisfy split exchangeability but collapses to sz ≈ K at small cal because
the head is undertrained.

---

## 6. Open questions

1. **cal=600 over-coverage** (all methods 92.5–92.9% on cluster). Suspected:
   stratified-split + ceiling-quantile `⌈(n+1)(1−α)⌉` interaction, which is
   method-independent and consistent with the upper bound in Theorem 1. A
   stratified-vs-random split ablation (findings §10 P1) would confirm.
2. ~~**Exact vs `O(1/n)` MS-CS**~~ — **Resolved (§5).** The 5-trial CIFAR-100
   sweep (`src/mscs_exchangeability_experiment.py`) shows frozen and exact agree
   on 87% (cal=200) → 99.5% (cal=800) of sets, Δcoverage ≤ +0.002 shrinking
   monotonically and always validity-safe; Theorem 1 holds to measurement
   precision for the frozen lead pipeline at deployment cal, and `--exchangeable`
   closes the residual cal=200 gap. Remaining: confirm on miniImageNet / CUB-200.
3. **Stratified (class-balanced) calibration** technically conditions on the
   label-count vector, so the relevant exchangeability is *within-class*. The
   marginal guarantee still holds; a class-conditional statement (à la Ding et
   al. 2023, findings §2b) is the cleaner object and is worth formalizing given
   our CovGap results.

---

*Sources: Vovk/Shafer full-CP guarantee (Shafer & Vovk 2008, arXiv:0706.3188);
Barber et al. 2021 (CV+); Fargion et al. 2025 (MA-CS, arXiv:2511.19359);
Fan & Sesia 2025 (O(1/n) standardization, arXiv:2512.15383). See `literature.md`.*
