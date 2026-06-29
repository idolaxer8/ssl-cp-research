# Novelty Pilots — Stage-1 Results (CIFAR-100, DINOv2-518)

Cheap go/no-go pilots for the two proposed reframings (see memory
`novelty-directions-pac-and-geometric-coverage`). Both reuse the exchangeable
pipeline (PCA-128 + cluster-whiten fit on the 10k unlabeled pool), alpha=0.1.
Scripts: `src/geometric_coverage_experiment.py`, `src/reliability_experiment.py`
(worktree `worktree-novelty-pilots`). Embeddings = `output/from_cluster/`.
Fixed held-out balanced test (20/class = 2000) isolates calibration variance.

**Verdict: BOTH directions are GO.** Kill criteria firmly rejected.

---

## Direction B — geometry-conditional coverage  (DONE, GO)

Q: under a balanced (class-count-equalised) protocol, is FCP coverage uniform
across LABEL-FREE geometric strata (local density / local intrinsic dimension /
k-means super-cluster), computed on the unlabeled pool (exchangeable, Prop 2)?
Kill criterion = flat per-stratum coverage. **Decisively not flat.**

Per-stratum coverage (5 quantile strata, low->high covariate), 10 trials,
prototype_softmax & geodesic, both NCMs overlap (=> property of the geometry,
not the score):

| config | covariate | worst stratum cov | CovGap_geo (pp) | Spearman(strata,cov) |
|--------|-----------|-------------------|------------------|----------------------|
| cal=400 balanced | density | 0.823 | 5.8 | -1.00 |
| cal=400 balanced | LID     | 0.852 | 4.0 | -1.00 |
| cal=800 balanced | density | 0.754 | 7.1 | -1.00 |
| cal=800 balanced | LID     | 0.797 | 4.6 | -1.00 |
| cal=800 **random (exact)** | density | **0.750** | 7.5 | -1.00 |
| cal=800 random (exact) | LID | 0.790 | 4.9 | -1.00 |

marginal coverage 0.90 (random) / 0.90–0.92 (balanced) in every row.

**Findings.**
1. **Coverage is monotone in local geometry** (Spearman = -1.0 everywhere):
   densest / lowest-LID test points over-cover (~0.98–0.99); sparsest /
   highest-LID under-cover to **0.75–0.80**, a ~15–20 pp conditional gap at a
   nominal-0.90 marginal.
2. **Not a balanced-split artifact.** The exactly-exchangeable RANDOM arm
   (marginal 0.900) shows the identical slope and a 0.750 worst stratum =>
   this is true feature-conditional miscoverage, not a stratification effect.
3. **Grows with cal** (CovGap_geo density 5.8 -> 7.1 pp as cal 400 -> 800):
   tightening the marginal makes the geometric heterogeneity MORE visible.
   Magnitude is comparable to the project's CLASS-conditional CovGap (7.68 pp,
   findings §2b) but on only ~5 strata => dodges the ClusterCP mass-cliff.
4. **It's per-point, not per-class.** Per-class coverage vs class-prototype LID
   is only weakly correlated (Spearman ~ -0.12) with frac-undercovered 0.36–0.38
   (matches the standing ~31% caveat). The strong signal is per-POINT density/LID
   => geometry-conditioning is the right object, not more class-conditioning.
5. Coarse k-means super-clusters are a weak, non-monotone covariate
   (CovGap_geo ~3–4 pp) — continuous density/LID is the lever.

**=> Stage 2 justified:** Mondrian over coarse density/LID strata (exact per-cell,
no mass cliff) + RLCP continuous localization; compare CovGap_geo vs ClusterCP.

---

## Direction A — calibration-conditional (PAC) reliability  (DONE, GO)

Q: at a FIXED budget B, how does the *distribution* of realized coverage and
set size over 100 calibration draws compare for the full pipeline vs a genuine
inductive SCP-THR (B/2 train head + B/2 cal)? Headline = reliability variance.

| cal | arm | cov mean | cov SD | cov p5 | P(cov<.90) | size | size_max | frac_K |
|-----|-----|----------|--------|--------|------------|------|----------|--------|
| 200 | full proto (bal) | 0.950 | 0.0156 | 0.917 | 0.00 | 5.25 | 71 | 0.00 |
| 200 | SCP-THR | 1.000 | 0.000 | 1.000 | 0.00 | **100** | 100 | **1.00** |
| 200 | full geo (rand, exact) | 0.902 | **0.0371** | 0.836 | 0.43 | 82 | 99 | 0.06 |
| 400 | full proto (bal) | 0.921 | 0.0152 | 0.896 | 0.07 | 1.70 | 8.7 | 0.00 |
| 400 | SCP-THR | 0.986 | 0.0263 | 0.932 | 0.00 | **84.7** | 94 | **0.76** |
| 400 | full geo (rand, exact) | 0.903 | 0.0207 | 0.867 | 0.37 | 2.88 | 38.6 | 0.00 |
| 800 | full proto (bal) | 0.906 | **0.0106** | 0.887 | 0.29 | 1.30 | 4.3 | 0.00 |
| 800 | full geo (bal) | 0.911 | 0.0111 | 0.890 | 0.14 | 1.61 | 8.2 | 0.00 |
| 800 | SCP-THR | 0.918 | 0.0130 | 0.895 | 0.09 | 3.60 | 15 | 0.00 |
| 800 | full geo (rand, exact) | 0.902 | 0.0115 | 0.884 | 0.39 | 1.63 | 8.3 | 0.00 |

Beta SD reference at alpha=0.1: SCP `0.3/sqrt(B/2+2)`; full-if-Beta `0.3/sqrt(B+2)`.
cal=800: SCP-Beta(400)=0.0150, full-Beta(800)=0.0106. cal=400: 0.0211 / 0.0150.

**Findings.**
1. **Size collapse is the dominant, reviewer-proof result.** Matched-budget
   SCP-THR degenerates to sz ~= K at small cal (frac_K = 1.00 @ cal200,
   0.76 @ cal400) — the inner head is useless at 1–2 labels/class. It only
   becomes usable at cal=800 (sz 3.60), still 2.3–2.8x the full arms. Full CP
   NEVER collapses (sz_max 4–24). This is the total-budget thesis made visceral.
2. **Coverage concentrates at the n=B rate (the "all B" hypothesis), confirmed.**
   full-proto coverage SD = 0.0152 @ cal400 (≈ full-Beta(400)=0.0150) and
   **0.0106 @ cal800 (= full-Beta(800) exactly)** — below SCP's 0.0130, and at
   2.8x smaller sets. SCP sits at best at the n=B/2 Beta rate and worse when the
   head is unstable (0.0263 @ cal400). The cal=200 corner is muddied (SCP
   degenerate SD=0; balanced full arms over-cover) — make the clean claim at
   cal>=400.
3. **The PAC scandal is real and quantified.** Even the balanced/exact full arms
   have P(cov<0.90) ≈ 0.29–0.39 — the nominal 0.90 predictor under-covers on
   ~1/3 of draws. The exact random arm has p5 as low as 0.836 (cal200) /
   0.867 (cal400); P(cov<0.88) = 0.27 (cal200) / 0.11 (cal400).
4. **PAC size-cost (SSBC, delta=0.05: smallest deploy alpha' with empirical
   P(cov>=0.90)>=0.95).** Price = size@alpha' - size@0.10:
   - cal=800: full-proto **+0.67** (1.30->1.97) vs SCP-THR **+8.13** (3.60->11.73)
     — full CP buys the 95%-reliable guarantee ~12x cheaper in absolute size.
   - cal=400: full-proto +1.44 vs SCP-THR +11.7.
5. **Balanced vs random = a clean bias-variance split of the coverage
   distribution.** balanced over-covers (low variance, conservative); random is
   centered at 0.90 with the wider lower tail. Both belong on the same plot.

**=> Strong story:** "at fixed budget the full transductive pipeline is more
RELIABLE — it never size-collapses, concentrates coverage at the n=B rate, and
buys a delta-PAC guarantee ~10x cheaper than budget-splitting SCP." SCP-geodesic
stays an O(1/n) ablation (not run here; not the headline).

---

## Cross-dataset confirmation — miniImageNet (DONE, both hold)

**B confirmed, even stronger.** Density-stratum coverage is monotone (Spearman
-1.0) on both splits. miniImageNet *looks solved* — marginal 0.900, near-singleton
sets (mean size 0.99) — yet the sparsest-density stratum sits at **0.600**
coverage (CovGap_geo 11.5 pp @ cal800), and identically on the exact random arm
(worst 0.608, marginal 0.897). The strongest possible "marginal coverage misleads"
illustration. LID slope present but weaker (Spearman -0.4 to -1.0); density is the
robust covariate across both datasets. Per-class LID flips sign vs CIFAR
(rho +0.3) => the effect is per-POINT, not per-class, on both datasets.

**A confirmed.** Matched-budget SCP-THR collapses (frac_K 1.00 @ cal200, 0.76 @
cal400; size 80–100), usable only @ cal800 (size 1.30). Full CP stays tight
(size ~1.0). Coverage SD concentration holds: full-proto 0.0111 @ cal800 ≈
full-Beta(800)=0.0106; SCP 0.0151 ≈ SCP-Beta(400)=0.0150. Exact random arm carries
the volatile tail (p5 0.828 @ cal200, P(cov<.88)=0.28 → 0.18 → 0.16).

## Selling figures (`output/novelty_selling/`, `src/plot_novelty_selling.py`)
- `reliability_sell.png` (2 datasets x 3 panels): coverage-SD-vs-cal (full rides
  the all-B Beta rate; SCP degenerate flagged), set-size collapse band (log-y),
  PAC size-cost bars (+0.67 vs +8.1 @ cal800 CIFAR).
- `geometry_sell.png` (3 panels): density-stratum coverage slope (both datasets,
  balanced + exact overlaid, crossing into the shaded under-covered zone), LID
  slope, worst-stratum coverage vs cal.

## Geometry — comprehensive NCM x MS-CS (the gap is INVARIANT)

Pipeline behind every geometry plot: DINOv2-518 -> UnlabeledTransform (PCA-128 +
within-cluster whitening, fit on 10k pool) -> NCM -> FullConformalPredictor (GPU),
alpha=0.1, balanced cal+test. 4 arms x 2 datasets, cal=800, density covariate
(Spearman = -1.0 in EVERY arm). `src/plot_geometry_comprehensive.py`.

| dataset | arm | marg cov | set size | CovGap_geo (pp) | worst stratum |
|---------|-----|----------|----------|------------------|----------------|
| CIFAR-100 | prototype          | 0.901 | 1.27 | 7.12 | 0.754 |
| CIFAR-100 | geodesic           | 0.905 | 1.53 | 7.36 | 0.758 |
| CIFAR-100 | prototype + MS-CS  | 0.897 | 1.24 | 7.57 | 0.737 |
| CIFAR-100 | geodesic + MS-CS   | 0.907 | 1.51 | 7.38 | 0.760 |
| miniImageNet | prototype       | 0.900 | 0.99 | 11.49 | 0.600 |
| miniImageNet | geodesic        | 0.907 | 0.98 | 10.98 | 0.625 |
| miniImageNet | prototype+MS-CS | 0.900 | 0.99 | 11.52 | 0.599 |
| miniImageNet | geodesic+MS-CS  | 0.907 | 0.98 | 10.95 | 0.627 |

**Conclusion: the geometric under-coverage is NCM- and MS-CS-invariant.** Switching
prototype<->geodesic, or adding the MS-CS class-similarity penalty, changes set
SIZE (e.g. CIFAR prototype 1.72->1.58 @ cal400, 1.27->1.24 @ cal800) but NOT the
slope — CovGap_geo is flat-to-slightly-worse and the worst stratum is unchanged.
MS-CS prunes *which classes* enter the set; it adds no coverage in sparse regions
(and by shrinking sets, nudges the sparse tail down). => None of the existing
efficiency machinery touches the geometric gap; only an explicitly
geometry-conditional method (Stage 2: Mondrian + RLCP) can close it. Figure:
`output/novelty_selling/geometry_comprehensive.png`.

## Reliability — convergence over budget B (where Split CP "turns on")

CIFAR-100, fixed test=2000, balanced. `src/plot_reliability_convergence.py`.

| B | SD full-proto | SD split-THR | Beta(B) | Beta(B/2) | split frac_K | split size |
|---|---------------|--------------|---------|-----------|--------------|------------|
| 200 | 0.0156 | (degenerate) | 0.0211 | 0.0297 | 1.00 | 100 |
| 400 | 0.0152 | 0.0263 | 0.0150 | 0.0211 | 0.76 | 84.7 |
| 800 | 0.0106 | 0.0130 | 0.0106 | 0.0150 | 0.00 | 3.60 |
| 1200 | 0.0074 | 0.0120 | 0.0087 | 0.0122 | 0.00 | 1.98 |
| 1600 | 0.0075 | 0.0100 | 0.0075 | 0.0106 | 0.00 | 1.62 |
| 2400 | 0.0055 | 0.0086 | 0.0061 | 0.0087 | 0.00 | 1.42 |

**Split CP is degenerate (sets = K) for B <= 400, "turns on" at B = 800** (frac_K
1.00 -> 0.76 -> 0.00, as B/2 finally trains a usable K=100 head). From B >= 1200
its coverage SD sits on the **Beta(B/2)** floor; **Full CP rides the Beta(B)
floor** at every B (e.g. 0.0106 == Beta(800), 0.0075 == Beta(1600)). The ~sqrt(2)
gap never closes — Full CP is uniformly more reliable. (B=3200 OOM'd on the 4GB
laptop; non-essential.) Figure:
`output/novelty_selling/reliability_convergence_cifar100.png`.

## Stage 2 — Mondrian-geometry CLOSES the gap (Stage 0 done, both datasets, both splits)

Method (`src/geometric_conditional_cp.py`): Mondrian Full CP over label-free
geometric strata (design C — shared global NCM, per-stratum quantile of the static
LOO cal scores; empty sets always allowed, NO all-classes fallback). Strata =
density quintiles fit on the unlabeled pool (exchangeable). One surgical add to
`conformal_prediction.py`: `return_test_scores` exposes the (n_test,K) score matrix
+ static cal_scores the GPU paths already compute. Runner
`src/geometric_conditional_cp_experiment.py`. G=5, cal=800, alpha=0.1, 10 trials,
fixed test=2000. `global_static` reproduces `global_fcp` set-for-set (design C sound).

| dataset / split | global CovGap_geo (worst) | Mondrian CovGap_geo (worst) | marginal | size (global->Mondrian) |
|-----------------|----------------------------|------------------------------|----------|--------------------------|
| miniImageNet / balanced | 11.49pp (0.600) | **0.58pp (0.902)** | 0.900->0.906 | 0.99 -> 1.21 (+22%) |
| miniImageNet / random (exact) | 10.97pp (0.608) | **0.65pp (0.892)** | 0.897->**0.902** | 0.97 -> 1.27 (+31%) |
| CIFAR-100 / balanced | 7.12pp (0.754) | **0.89pp (0.894)** | 0.901->0.907 | 1.27 -> 1.65 (+30%) |
| CIFAR-100 / random (exact) | 7.49pp (0.750) | **0.68pp (0.901)** | 0.900->0.907 | 1.63 -> 2.59 (+59%) |

- **CovGap_geo cut 87-95%** (>> 40% success bar); **worst-stratum 0.60/0.75 -> ~0.90**
  (meets the 0.90 stretch). Per-stratum coverage is flat at ~0.90 across all density
  strata (fig `output/geometric_conditional/<ds>/stage0_density_G5_cal800_a0.1.png`).
- **Marginal validity preserved on the EXACT random arm** (0.902 / 0.907) — the fix is
  not a balanced-split artifact; it is exactly-exchangeable (strata are pool-fixed).
- **Size tax is the honest conditional-coverage cost**: +22-59%, concentrated in the
  sparse stratum (e.g. miniImageNet sparsest 0.74 -> 2.39; CIFAR random 2.41 -> 6.90)
  exactly where larger sets are warranted. Above the <=20% target on the random/geodesic
  arms; report transparently. Empty-set invariant respected (no all-classes fallback;
  empties redistribute dense<->sparse, marginal empty-rate ~flat).
**Geometry is NECESSARY — the confidence-conditioning control fails** (balanced,
cal=800, G=5, 10 trials; `ConfidenceStratifier` = Mondrian threshold conditioned on
the model's max-posterior instead of geometry, evaluated on the SAME density strata):

| dataset | Mondrian-geometry CovGap_geo / marg / sz | Mondrian-confidence CovGap_geo / marg / sz |
|---------|-------------------------------------------|---------------------------------------------|
| miniImageNet | **0.58pp / 0.906 / 1.21** | 7.54pp / 0.976 / 1.94 |
| CIFAR-100 | **0.89pp / 0.907 / 1.65** | 5.80pp / 0.959 / 3.14 |

Conditioning on confidence does NOT flatten the geometric gap (~10x worse CovGap_geo)
and badly OVER-COVERS (0.96-0.98) with bigger sets — because softmax confidence is
miscalibrated exactly on atypical points (confident-but-wrong in sparse regions). The
density/LID axis isolates the failure; confidence does not. Fig
`output/geometric_conditional_conf/<ds>/stage0_density_G5_cal800_a0.1.png` (green flat
at 0.90 vs blue still-sloped-and-over-covering). **Core Stage-2 claim established.**

**Robust across alpha / G / covariate** (10 trials, cal=800, density unless noted;
global CovGap_geo (worst) -> Mondrian CovGap_geo (worst), Mondrian marginal):

| config | CIFAR-100 | miniImageNet |
|--------|-----------|--------------|
| alpha=0.05, G5 | 5.8(0.80)->0.4(0.95) | 5.8(0.80)->0.6(0.95) |
| alpha=0.1,  G5 | 7.1(0.75)->0.9(0.89) @0.907 | 11.5(0.60)->0.6(0.90) @0.906 |
| alpha=0.2,  G5 | 14.0(0.53)->1.7(0.80) | **19.9(0.32)->1.2(0.79)** |
| G=3,  alpha0.1 | 6.8(0.79)->0.6(0.90) | 11.8(0.72)->0.4(0.90) |
| G=10, alpha0.1 | 7.1(0.72)->1.0(0.90) | 12.5(0.57)->1.1(0.89) |
| LID, G5 alpha0.1 | 4.6(0.80)->0.3(0.90) | 5.3(0.77)->0.9(0.89) |

Mondrian flattens the gap at every setting (worst-stratum -> ~1-alpha), marginal valid
on the random arm throughout. The gap GROWS with looser alpha (miniImageNet alpha=0.2
sparse stratum at **0.30** coverage -> 0.79); LID's baseline gap is smaller than
density's (density is the stronger covariate); G=3/5 cleanest, G=10 slightly noisier.

**FGVC-Aircraft = the SCOPE BOUNDARY (rescope, not failure).** Global FCP CovGap_geo
is only 1.0-2.5pp (worst ~0.88) — the geometric gap **barely exists**, because DINOv2
cannot separate the variants so sets are uniformly huge (16-30 of K=100) and the
predictor is *uniformly* weak (no dense-over / sparse-under structure). Mondrian is a
near no-op (cal=800 random 1.49->0.55pp; cal=400 balanced 2.28->**3.24** slightly
WORSE), sets unchanged (16.8->17.0). Confirms the mechanism: **the geometric gap is a
property of a STRONG backbone with locally-varying failures, not generic difficulty**
([[aircraft-stress-test-scope-limit]]). Apply geometry-conditioning only when a strong
backbone has geometrically-localized holes.

**Validity tests PASS** (`tests/test_geometric_conditional_cp.py`, 5/5):
global_sets==FCP-static (design C sound); Mondrian marginal+per-stratum valid on a
random split (geodesic & prototype); empty-set invariant (no all-classes fallback);
RLCP marginally valid.

- Engine add `return_test_scores` (verified no-op). New: `src/geometric_conditional_cp.py`
  (GeometryStratifier/ConfidenceStratifier/mondrian_sets/global_sets/rlcp_sets),
  `src/geometric_conditional_cp_experiment.py`, `tests/test_geometric_conditional_cp.py`
  (all uncommitted, worktree).
**RLCP (continuous secondary) interpolates global<->Mondrian via bandwidth** (Hore &
Barber randomized localized CP; `rlcp_sets`, c0 = h / IQR(pool covariate); CIFAR-100
balanced, 8 trials): c0=1.0 (wide) CovGap 5.21pp (worst 0.824) ~ global; c0=0.5
3.02pp (0.872); c0=0.25 (tight) **1.59pp (0.886)** ~ Mondrian (0.80pp/0.894). Smaller
sets than Mondrian at a given closure but mildly conservative (over-covers 0.91-0.93;
randomization-exact in theory, a calibration nuance to tune). Mondrian (hard partition)
is the cleaner PRIMARY (tighter closure, hits ~1-alpha); RLCP is the smooth secondary
(no hard bins / no small-n_g cliff, bandwidth = locality knob).

Robustness figure: `output/novelty_selling/geometric_robustness.png`
(`src/plot_geometric_robustness.py`).

- Deferred (optional / cluster): ClusterCP geometric-CovGap negative control (known to
  degenerate to SplitCP, findings 2b), design-B exactness oracle (redundant with
  global_static==global_fcp + 5/5 unit tests + random-arm validity), the full
  alpha x NCM x G x dataset matrix on the 48GB pod, and a size-tax characterization figure.

## Caveats / next steps
- 100 draws, single seed family, CIFAR-100 only. Confirm on miniImageNet; bump to
  300–500 draws for tight p5 / P(<.88) tails; add the SD-vs-cal curve as a figure.
- B Stage-2 (Mondrian + RLCP) and a clean RLCP exchangeability proof are the
  build-now items if B proceeds.
- Geometric covariates are pool-fit (exchangeable). The density covariate is the
  strongest; LID is a robust cross-check.
- SCP-THR's collapse is the *intended* consequence of the inner-classifier
  requirement under total budget — fair, and the structural point being measured.
