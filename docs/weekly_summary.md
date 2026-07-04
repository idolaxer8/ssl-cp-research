# Weekly Progress Summary

Rolling weekly summary for instructor discussions. Newest week on top. Each
entry: per-topic key points + output (graph/table) locations. Math is plain
text (no LaTeX), per repo convention.

All experiments: CIFAR-100, DINOv2 embeddings, exchangeable pipeline
(PCA-128 + cluster-whiten fit off cal), alpha = 0.1, unless noted.

---

## Goals — upcoming week (planned, 06-30 to 07-04)

Forward-looking plan, not results. Each item converts into the dated entry above
as it lands. Four threads, building directly on last week's findings.

### 1. Class-similarity (MS-CS) on the softmax NCM — investigate, rethink M  ✔ RESOLVED (see "Week ending 2026-07-04" below)

**Why.** MS-CS is now a **no-op for prototype-softmax** (last week, topic 1): the
k-means cluster-M penalty neither shrinks nor helps once the y_hat bug is fixed,
while it stays a real lever for the geodesic NCM. Working hypothesis: the penalty's
class-similarity matrix M (k-means on the pool) is mismatched to the softmax score's
geometry — the softmax head already softly down-weights dissimilar classes via the
prototype cosines, so an external cluster-M adds nothing.
- Diagnose precisely why the penalty doesn't bite for the softmax head (interaction
  of the suspected-class y_hat with the LAC score).
- **Try a softmax-native M:** build class similarity from the **prototype cosines
  themselves** (M_ij = cos(mu_i, mu_j)) instead of pool k-means — a label-free,
  score-aligned similarity — and re-test on CIFAR-100. Keep exchangeability (fit on
  the unlabeled pool / per-candidate update).

### 2. Aircraft — class-similarity, Split-CP comparison, a second fine-grained set  ✔ RESOLVED (see "Week ending 2026-07-04" below)

**Why.** FGVC-Aircraft is our scope-limit dataset (DINOv2 can't separate the 100
variants; topic 2). We want to know if class-similarity helps where features can't,
and to confirm the limit is general, not Aircraft-specific.
- Run the **class-similarity penalty on Aircraft** (the new M from goal 1, post-fix
  y_hat) — does coarse structure help when the backbone can't?
- Add a **Split-CP baseline on Aircraft** (SCP-THR / SemiCP) for a full
  FCP-vs-Split head-to-head in the hard-backbone regime (so far we have only
  FCP-family numbers there).
- Find a **second fine-grained dataset** to replicate the scope limit (candidates:
  Stanford Cars, CUB-200, Oxford-Flowers-102 — we already have cub/flowers
  embeddings).

### 3. Geometric-conditional coverage — RESOLVED (worst metric + per-class effect)

**Verdict.** Both bullets closed; the fix is **one-sided (coverage-monotone) Mondrian**.
Full reference `docs/geometric_conditional_methods.md` + code on branch
`worktree-novelty-pilots` (commits e57d0f9..2321d83; unified harness
`src/geometric_methods_comparison.py`, 30-trial cifar100+mini).
- **"worst" metric = coverage FLOOR (not weird, just mislabeled).** "worst-stratum
  coverage" = MIN coverage over the G geometry strata = the field-standard safety floor
  (worst-slice, Cauchois 2021; worst-class, Ding 2023) — higher = safer, target
  >= 1-alpha. The odd-looking numbers were a LABEL trap: when a method OVER-covers, the
  min stratum is the one CLOSEST to target, so calling it "worst" inverts the intuition
  — renamed to **"coverage floor (min over strata)"**. `CovGap_geo` is TWO-SIDED (charges
  over- AND under-coverage), so an over-covering method scores a nonzero gap with a
  perfect floor. Metric is clean (min over G strata, pooled); floor lift 0.60/0.77 ->
  ~0.90 confirmed at 30 trials.
- **Per-class effect: hard Mondrian BREAKS the class axis; one-sided fixes BOTH.** Sparse
  strata ARE the badly-covered classes (Spearman(class cov, sparseness) = -0.63 cifar /
  -0.67 mini; worst classes at the 78th/87th sparseness pctile). BUT hard per-stratum
  Mondrian equalizes the stratum MARGINAL by reallocating coverage to intrinsically-easy
  classes -> **worst-class COLLAPSES** (mini cal=800: 0.625 -> 0.343) while CovGap_geo is
  fixed. **One-sided Mondrian** — per point take the more lenient of {global, its-stratum}
  threshold = `max(q_global, q_stratum)` = global set UNION stratum set — lifts sparse
  strata to the floor WITHOUT lowering dense strata, fixing the geo floor AND improving
  class coverage with no new victims (mini cal=800: floor 0.61 -> 0.91, worst-class 0.625
  -> 0.775, best CovGap_class). Cost = marginal over-coverage (~0.94-0.98) + bigger sparse
  sets — a floor guarantee MATHEMATICALLY requires some over-coverage (averaging arg).
  **one-sided AUTO** = cal-calibrated shrinkage k0 (guarantee-first default, held the test
  floor in 9/10 configs). **Shrinkage** buys back 15-49% set size at small cal. **RLCP
  does NOT beat one-sided** (exact-marginal can't guarantee a floor; plain RLCP worst-class
  0.367 on mini; monotone RLCP trades the floor guarantee for class-worst).
- **Naming:** "Mondrian" (Vovk 2003) = general per-TAXONOMY-cell quantile, NOT per-class;
  ours is the GEOMETRY/group-conditional taxonomy -> qualify as "Mondrian-over-geometry" /
  "stratum-conditional" to avoid the per-class connotation.

### 4. CAOS (Waldron, arXiv:2601.05219) — finish review + literature sweep

**Why.** CAOS is our nearest split-free one-shot ally (topic "Literature"); only the
Appendix-E "~52% of the win is not-splitting" decomposition is locked in so far.
- **Finish reading** CAOS and write a short **key-takeaways** summary for our
  positioning (validity argument, score/aggregation levers, what transfers to our
  few-shot many-class regime).
- **Review CAOS's related-work** for neighbours we've missed.
- **Fresh last-few-months scan** for more split-free / few-shot / transductive CP
  papers; fold the keepers into `literature.md` (§5 few-shot, §8 semi-sup).

---

## Week ending 2026-07-04 (work on 06-30 to 07-04)

Topics 1 and 2 of the week's plan resolved. Headline: (1) our MS-CS was not faithful
to the source paper; we built the faithful version, which settles the question — a
class-similarity M **cannot** help the (tight) softmax NCM, for a concrete structural
reason, but IS the **better** lever for the geodesic NCM; (2) on the hard-backbone
Aircraft set the same story holds (both the old and the faithful M inflate the softmax
sets), Full-CP dominates Split-CP in the few-shot regime, and the scope limit
replicates on a second fine-grained set (Stanford Cars, K=196). Defaults as in the
header unless noted.

**How each MS-CS M works.** The penalty is s_λ(x,y) = s(x,y) + λ·(1 − M[y, ŷ(x)]): it
raises the nonconformity of a candidate class y in proportion to its DISsimilarity to
the suspected class ŷ, pushing classes unlike ŷ out of the prediction set. The variants
differ only in how the class-similarity matrix M is built:
- **cluster** (our original "MS-CS"): k-means on the unlabeled pool; M[c,c'] = 1 if the
  two class means fall in the same cluster, else exp(−d²/τ) on the inter-cluster
  distance. Pool-derived, quantized.
- **centered_cosine** (= Fargion §5 "MS", the *faithful* port): M[c,c'] =
  cos(h_c − h_G, h_c' − h_G) — cosine of the class means *centered by the global mean*
  h_G, with h_G fit once on the pool (bag-independent → exactly exchangeable).
- **prototype** (softmax-native, the goal-1 attempt): M[c,c'] = cos(μ_c, μ_c') — cosine
  of the (uncentered) class-mean prototypes = the Gram of the prototype NCM's own cosine
  logits (redundant with the softmax score by construction).
- **MA** (Fargion §4, Model-Agnostic): binary M[c,c'] = 1{g(c)=g(c')} from coarse
  superclass labels g. (centroid: a pool-free Gaussian kernel on cal-centroid distances,
  a control.)

### 1. Class-similarity (MS-CS) on the softmax NCM — rethink M [RESOLVED]

**Fidelity correction (the starting point).** Reading Fargion, Dabah & Tirer (2025,
arXiv:2511.19359, ICML 2026) closely: their penalty has TWO M variants — **MA**
(Model-Agnostic, a binary superclass indicator 1{g(y)!=g(y')}, their §4) and **MS**
(Model-Specific, a continuous matrix from the model's embeddings, their §5 = cosine
of the CENTERED class means, cos(h_c-h_G, h_c'-h_G)). Our `macs_experiment.py`
reproduces their MA faithfully, but our "MS-CS" (k-means cluster-M and the
prototype-cosine M) is NOT their MS — we had taken the name but built a different,
uncentered/quantized matrix. This also retracts a novelty claim: we had positioned
"a continuous, label-free, embedding-derived M" as our generalization of their
binary d, but that is exactly their MS. Corrected in `theory.md` §3.1-3.2 and
`literature.md`.

**What we tried — the faithful port.** Implemented their MS exactly as
`--similarity centered_cosine`: M[c,c'] = cos(h_c-h_G, h_c'-h_G), with score-aligned
class means h_c from calibration and the global mean h_G fit ONCE on the unlabeled
pool. Because the pool is independent of cal+test, h_G is a bag-independent constant,
so the per-candidate Full-CP update shifts only the candidate class's mean ->
**exactly exchangeable** (verified: per-candidate update == full rebuild on the
augmented bag to 1.7e-16, incl. absent-class revival; regression test added).
Committed `d34c0e7`, branch `mscs-centered-cosine-m`.

**Result — no-op for the softmax NCM, and WHY it fails.** On prototype-softmax
(CIFAR-100, PCA-128) the faithful centered M is indistinguishable from the
uncentered prototype-M: at cal=800 (tight sets, sz 1.27) every M gives ~0% change
(+/-0.5%). The mechanism (diagnostic): a similarity penalty lambda*(1-M[y,yhat])
drops the LEAST-similar classes first, but the classes actually cluttering a tight
softmax set are the MOST-similar to yhat (mean M[c,yhat] = +0.42 for in-set extras
vs -0.01 for correctly-excluded classes) — so the penalty protects exactly the
clutter and can only raise the quantile -> no-op (or bloat). There is also little
headroom (77.5% of sets are already singletons), and the residual ambiguity is
per-instance, which no class-level (x-independent) M can resolve. We then tested the
one remaining M-shaped escape — an ASYMMETRIC confusion/correction matrix
M[y,yhat]=P(true=y|pred=yhat), built both from cal LOO errors and from pool
soft-confusion — and it also failed: at cal=800 all no-op; at cal=200 the plain
similarity M (+20%) even beats the confusion M (+6%). **Conclusion:** a class-level
M is HEADROOM-gated — it helps loose sets and is a no-op on tight ones, independent
of how M is built. The tight-softmax regime that carries our headline set sizes has
no M lever; the honest lever there is score-side (posterior smoothing / adaptivity),
not a class-distance penalty. (En route we corrected an earlier misattribution: the
"cal=200 bloat" was an artifact of the exact per-candidate update at m=2/class, not
the penalty — the frozen penalty reduces sets at small cal.)

**Positive spin-off — the faithful M is the BETTER lever for the geodesic NCM.**
Where M is not redundant with the score (the geodesic NCM ranks by NN-ratio, not
prototype cosine), the faithful centered M is a real, valid set-size lever and
competitive-to-better than the old k-means cluster-M (unwhitened_topk_mean, 3
trials): cal=200 cluster slightly ahead (22.1% vs 19.1% at lambda=0.1); **cal=800
(the tight-set regime) centered WINS and is lambda-monotone** (5.5% vs cluster 2.1%
at lambda=0.1; cluster peaks at lambda=0.05 = 3.3% then degrades). Coverage valid
(~0.90) throughout. **Decision: `centered_cosine` is now the default MS-CS M for the
geodesic NCM.** (To firm up: unwhitened_topk_asym + more trials; centered is still
CPU-only.)

**Outputs.** `theory.md` §3.1-3.2.1 (derivation + both results); commit `d34c0e7`
(faithful M + doc fidelity fixes); `tests/test_mscs_gpu_parity.py` (exchangeability
regression); memory `macs-paper-fidelity`.

### 2. Aircraft — class-similarity, Split-CP head-to-head, 2nd fine-grained set [RESOLVED]

All three parts run in the hard-backbone regime (DINOv2 cannot separate the fine-grained
classes, so every method's sets are large), single-label throughout — we did NOT add any
multi-label / multi-label-CP machinery. Aircraft = FGVC-Aircraft (K=100); the 2nd set is
Stanford Cars (K=196). Pipeline as in the header (PCA-128 + cluster-whiten off the pool),
20-50 trials on the cluster / 5 trials locally, α=0.1.

**(a) Class-similarity on Aircraft — confirms topic 1 on the hard backbone.** Ran
`prototype_softmax` with no-CS vs the old **cluster** M vs the faithful **centered_cosine**
M (post-`cf639b2`, 5 trials). Neither helps: under the exact (exchangeable) penalty BOTH
INFLATE the softmax sets — cluster red −4% / −25% / −40% at cal 400/800/1200; centered a
no-op at cal=400 (−0.2%) but +19% at cal=800 (milder than cluster, still no help).
Coverage holds ~0.89–0.90 (the earlier cal=400 → 0.79 "collapse" was the pre-`cf639b2`
y_hat bug, now confirmed gone). Same verdict as topic 1: a class-level M is not a lever
for the softmax head — its cosine score already encodes class similarity. Fig
`output/week_1_7_res/prototype_cs_aircraft.png`. **Sanity control (geodesic NCM):** the
SAME two penalties on `unwhitened_topk_mean` instead SHRINK the aircraft sets — cluster-M
red +9%, centered_cosine +16% at cal=800 (both valid ~0.91, λ=0.2), centered the better
lever — confirming the effect is NCM-specific (not dataset-specific) and mirroring the
CIFAR-100 geodesic result of topic 1. `output/week_1_7_res/geodesic/`.

**(b) Split-CP head-to-head — FCP dominates few-shot, converges by ~10 shots/class.**
Our FCP framework (`prototype_softmax` + `geodesic_topk_asym`, full label budget) vs
**SCP-THR** and **SemiCP-THR** (budget split 50/50 into train+cal; softmax classifier;
THR score = 1−p(y); SemiCP adds NNM scores from the unlabeled pool). Aircraft, size(cov):
at cal=400 (4 shots/class) Split-CP degenerates to the trivial full set (SCP 94 / SemiCP
100 vs FCP+proto 21 / geo_asym 28); the gap closes as cal grows, and by cal=1000 (10
shots) SemiCP-THR (18.4) is competitive with FCP+geo_asym (19.7) and trails only FCP+proto
(16.2). SemiCP's NNM is unreliable (hurts at cal=600, helps at cal=1000) — the unlabeled
pool does NOT rescue Split-CP. So FCP's win is a *few-shot* story. `geo_asym`+cluster-M
shrinks a further ~1–3.5% (the small real geodesic lever, per topic 1). Fig
`output/week_1_7_res/split_vs_fcp_size_cov.png`.

**(c) 2nd fine-grained set (Stanford Cars, K=196) — the scope limit generalizes.** Added a
loader for the HF mirror `tanganke/stanford_cars` (the official Stanford URL is dead);
train→labeled, test→unlabeled like aircraft. Scope sweep (`fca_family`, 50 trials,
balanced): sets stay huge (15–196 of K); `prototype_softmax` is tightest-on-average once
cal is adequate but DEGENERATES at 2-shot (sz = 196 = the full set at cal=400) and has the
worst class-conditional coverage; `geodesic_topk_asym` is the more honest NCM (best
CovGap). Old cluster-M on the prototype BALLOONS sets 3.6–5.4× at cal≥800 (coverage still
valid) — the "M does not help the softmax NCM" story, dramatized. The backbone-separability
scope limit is therefore NOT aircraft-specific. Fig
`output/week_1_7_res/compare_stanford_cars_balanced_both.png`.

**Outputs.** `output/week_1_7_res/` (3 figures + result JSONs: `results_aircraft.json`
[cluster-M], `results_aircraft_centered_cosine.json`); Stanford Cars loader in
`download_datasets.py` + `carve_unlabeled.py` + `cluster/run_aircraft_cs_splitcp.sh`
(committed `25709fb`); memory `prototype-cosine-mscs-M`. Topics 3-4 of the plan
(geometry-conditional audit, CAOS review) not covered this entry.

---

## Week ending 2026-06-29 (work on 06-23 to 06-29)

Two arcs this week, merged into one section. (A) The prototype-softmax NCM
matured (GPU paths), was stress-tested to its scope limit, and three side
threads closed — a validity bug fix (shipped) plus LATA and SAPS (both killed,
unmerged). (B) A new **geometry-conditional coverage** direction was piloted
end-to-end (the primary novelty result). Each topic gives the intuition, the
paper it builds on, and where to find the table/plot. Defaults as in the header
(CIFAR-100, DINOv2, balanced cal+test, alpha=0.1) unless noted.

### 1. prototype-softmax matured — GPU fast paths + a MS-CS validity bug fix (shipped)

**Intuition.** prototype-softmax is our FCA-style NCM (Silva-Rodriguez et al.,
IPMI 2025, arXiv:2506.06076): score a label by a softmax over cosine similarity to
class-mean prototypes, re-fit per candidate in closed form so Full CP stays exact.
This week it became fast and a subtle validity bug was found and fixed.
- **GPU fast paths + the softmax-update trick.** A candidate's refit moves only ONE
  class's prototype, so the softmax denominator is *swapped* (one exp term) instead
  of recomputed over all K classes — an O(1)-per-cal-point update. Result: **~27x
  faster (25.1s -> 0.93s/trial @ cal=800/K=100/test=1000), dropping prototype from
  46x slower than the geodesic NCM to ~1.7x** — the FCA-style head is now nearly as
  cheap as our ratio NCM. The MS-CS penalty got its own GPU kernel (~130x). All
  bit-exact, CUDA-parity tested. (`src/conformal_prediction.py`, `mscs_gpu.py`.)
- **MS-CS y_hat bug (fixed, `cf639b2`).** The penalty's suspected-class y_hat was
  argmax over the FULL prototypes (non-LOO): a calibration point "saw itself" in
  its own class while the test point did not, so the penalty was not a symmetric
  function of the bag -> coverage broke once the penalty had weight. Fix: y_hat =
  argmax of the NCM's own leave-one-out logits, with a coverage-under-domination
  regression test. **The geodesic NCM was never affected (its y_hat already
  excludes self) — all geodesic MS-CS results stand.**
- **Correction to last week.** The reported "MS-CS shrinks prototype sets" was a
  coverage-break artifact. Post-fix (20-trial ablation, see the table below),
  MS-CS is a **no-op for prototype** at lambda=0.05 — coverage now valid at every
  cal and set size essentially unchanged on PCA features (slight inflation on raw
  768-D). MS-CS stays a genuine efficiency lever for geodesic, not for prototype.

**Current status of the prototype NCM (CIFAR-100, post-fix) — one-glance summary,
plot `output/from_cluster/fca_ablation/ablation_cifar100.png`** (prototype vs
geodesic mean, PCA on/off x MS-CS on/off, 20 trials, regenerated 2026-06-30 with the
post-`cf639b2` code). Set size (coverage):

| arm | cal=200 | cal=400 | cal=800 |
|-----|---------|---------|---------|
| **prototype + PCA-128, no penalty (recommended)** | **4.25** (.95) | **1.67** (.92) | **1.30** (.90) |
| prototype + PCA-128, + MS-CS (lambda=0.05) | 4.24 (.94) | 1.67 (.92) | 1.30 (.90) |
| prototype, no PCA (full 768-D) | 10.80 (.96) | 2.17 (.93) | 1.40 (.91) |
| geodesic mean + PCA-128 | 5.21 (.92) | 2.18 (.91) | 1.58 (.91) |

- **Best valid recipe = prototype + PCA-128, penalty OFF** — beats geodesic mean at
  every cal; PCA-128 is the dominant lever (10.8 -> 4.25 @ cal=200). These
  no-penalty arms are unaffected by the bug, so the numbers stand (local GPU
  re-confirmed cal=800 sz 1.30).
- **MS-CS is a no-op for prototype (post-fix):** at lambda=0.05 on PCA features it
  neither shrinks nor helps (4.25 -> 4.24 @ cal=200; identical from cal=400) and on
  raw 768-D it slightly INFLATES (10.80 -> 11.62). Coverage is now valid at every
  cal incl. cal=800 (.902, vs the buggy .898) — the prior "MS-CS shrinks prototype"
  was purely the coverage-break artifact. So keep the penalty OFF for prototype;
  MS-CS stays a real lever only for the geodesic NCM.

### 2. Scope limit — prototype works on separable data, breaks on fine-grained FGVC-Aircraft

**Intuition.** The whole tight-set story assumes the backbone actually separates
the classes. We pinned down both sides: CIFAR-100 (DINOv2 separates) vs the new
FGVC-Aircraft (100 near-identical variants it does not).
- **CIFAR-100 (well-behaved), 20 trials.** PCA-128 is the dominant lever
  (prototype sz 10.8 -> 4.25 @ cal=200, valid) and **prototype beats geodesic mean
  at every cal** (4.25 vs 5.21 @200; 1.30 vs 1.58 @800). The MS-CS rows were
  regenerated post-fix (2026-06-30) and now show MS-CS is a no-op for prototype
  (see section 1). Table + plot:
  `output/from_cluster/fca_ablation/results_cifar100.json`, `ablation_cifar100.png`.
- **FGVC-Aircraft (stress test), 50 trials, set size (coverage):**

  | cal | prototype | geodesic asym | geodesic mean |
  |-----|-----------|---------------|---------------|
  | 200 | 74.6 (.974) | 52.3 (.950) | 47.0 (.917) |
  | 400 | 21.4 (.920) | 30.6 (.932) | 38.2 (.909) |
  | 800 | 17.2 (.909) | 21.2 (.918) | 29.3 (.923) |
  | 1800| 13.2 (.904) | 15.4 (.907) | 18.6 (.909) |

  Sets stay **HUGE — 13-75 of K=100** at every cal (vs ~1.3 on CIFAR-100): the
  **first dataset where the saturating tight-set story fails**. Prototype is
  tightest-on-average from cal>=400 but over-covers/bloats at cal=200 (sz 74.6) and
  has the **worst CovGap** (~10.5 pp vs geodesic asym ~7.0) — on hard data
  **geodesic asym is the more honest choice**. PCA's payoff ~vanishes (asym @
  cal=1000: 23.7 raw vs 24.0 PCA-128). Plot
  `output/from_cluster/fca_cluster/compare_aircraft_balanced_both.png`; ablation
  `output/from_cluster/aircraft_ablation/`.
- **Takeaway.** A backbone-separability scope limit to state plainly: the headline
  (tight valid sets) is contingent on class separability; on genuinely fine-grained
  data the method degrades gracefully on coverage but not on set size.

### 3. LATA posterior smoothing — KILLED for Full CP: a split-CP-only tool (unmerged)

**Intuition.** LATA (Bozorgtabar et al., arXiv:2602.17535) smooths the C-dim
posterior VECTOR over a kNN graph with a KL anchor — unlike our long-abandoned
"LATA score smoothing", which smoothed the scalar score and destroyed class
discrimination. We adapted it onto prototype-softmax.
- **Stage 0 (split CP):** modest, regime-specific gains — sets -21%/-9%/-5% @
  cal 200/400/800 (CIFAR-100) but mostly spending the balanced over-coverage
  cushion (under-covers on the exact random split); **no CovGap gain at K=100**
  (LATA's headline is a few-class effect). The scalar-score control reproduces the
  old failure -> confirms "smooth the vector, not the scalar". Plot
  `output/lata_posterior_smoothing/crossover.png`.
- **Stage 1 (full CP):** built and proven exactly exchangeable, but **~0 benefit**
  — smoothing the whole bag symmetrically per candidate cancels in the p-value
  rank (split CP moved only the test posterior vs fixed cal scores, hence its
  gain), at prohibitive cost O(B*K*iters*n^2). Plot
  `output/lata_fullcp_smoothing/fullcp_smoothing.png`.
- **Verdict:** posterior smoothing is a split-CP tool; do NOT adopt Stage 1. Branch
  `worktree-lata-posterior-smoothing`, NOT merged — awaiting go/no-go.

### 4. SAPS / APS rank scores on prototype — KILLED on the size/CovGap frontier; salvaged as an adaptivity tool (unmerged)

**Intuition.** LAC (1 - p(y)) bloats on hard data. Rank-based scores — APS (Romano,
Sesia & Candes, NeurIPS 2020, arXiv:2006.02544) and SAPS (Huang et al., ICML 2024,
arXiv:2310.06430) — score by label RANK to spread coverage across set sizes. We
added them to prototype-softmax (exactly exchangeable, GPU paths) plus a new
**SSCV** metric (size-stratified coverage violation).
- **Verdict (10 trials):** SAPS does NOT win the size/CovGap frontier — geodesic
  asym stays CovGap champion on Aircraft (~7.2 vs SAPS-cosine ~8.5); on CIFAR-100
  SAPS strictly loses to LAC (sz 2.7 vs 1.3); plain APS bloats (sz ~50).
- **Salvage:** SAPS-cosine wins SSCV decisively on hard large-set data (~0.5-1.7 vs
  ~10) — it is an adaptivity tool, not a size winner; and the result **strengthens
  the denominator-free geodesic thesis** (rank scoring can't fix the softmax CovGap
  bloat -> the pathology is intrinsic to the normalized-softmax score). Plot
  `output/saps_local_aircraft/compare_aircraft_balanced_both.png`. Branch
  `worktree-saps-prototype-ncm` (commit 13940be), archived, NOT merged.

### 5. NEW direction — geometry-conditional Full CP (primary novelty pilot, unmerged)

**Intuition.** Marginal 90% coverage is an AVERAGE over the test distribution; it
can hide regions that are badly under-covered. We stratify test points by a
LABEL-FREE geometric covariate — local density (kth-NN radius) or local intrinsic
dimension (Levina-Bickel MLE, 2004), computed on the unlabeled pool (a fixed
pool-function -> exactly exchangeable) — and measure per-stratum coverage
(CovGap_geo, the geometry-axis analogue of the class-conditional CovGap of Ding,
Tibshirani & Ramdas 2023, arXiv:2306.09335).
- **Diagnosis (CIFAR-100 + miniImageNet).** Per-stratum coverage is MONOTONE in
  geometry (Spearman -1.0): the sparse / high-LID 20% **under-cover to 0.60-0.75**
  at a 0.90 marginal (a 15-40 pp conditional gap); miniImageNet's sparsest stratum
  even gets mean set size 0.74 (<1) — the method hands EMPTY sets to its hardest
  points. The gap is split- and NCM-invariant, grows with cal, and is per-point.
  Plot `output/novelty_selling/geometry_sell.png`.
- **Fix — Mondrian CP** (Vovk): a SEPARATE conformal threshold per geometric
  stratum => group-conditional coverage. It cuts CovGap_geo **87-95%** (mini
  11.5 -> 0.58 pp; cifar 7.1 -> 0.89 pp), lifts worst-stratum coverage to ~0.90,
  keeps the marginal valid, at a +22-59% size tax confined to the sparse stratum
  (the distribution-free price; exact per-point conditional coverage is impossible,
  Foygel-Barber et al. 2021). A confidence-conditioning control fails (softmax is
  miscalibrated exactly on atypical points) -> geometry is the necessary axis.
  Robust across alpha x G x covariate; on Aircraft it is a no-op (no strong
  backbone -> no local holes). Secondary continuous variant RLCP (Hore & Barber
  2024, arXiv:2310.07850). Plot `output/novelty_selling/geometric_robustness.png`;
  dirs `output/geometric_conditional*/`; write-up `docs/novelty_pilots_findings.md`.
- **Regular (global FCP) vs Mondrian, BALANCED, G=5 density, 15 trials**
  (coverage / CovGap_geo pp / set size / worst-stratum coverage):

  | dataset | cal | global FCP | Mondrian |
  |---------|-----|------------|----------|
  | CIFAR-100    | 400 | 0.921 / 5.65 / 1.75 / 0.823 | 0.928 / 2.75 / 2.71 / 0.918 |
  | CIFAR-100    | 800 | 0.902 / 7.07 / 1.27 / 0.761 | 0.910 / 1.02 / 1.73 / 0.905 |
  | miniImageNet | 400 | 0.915 / 9.96 / 1.03 / 0.669 | 0.930 / 2.96 / 1.49 / 0.907 |
  | miniImageNet | 800 | 0.900 / 11.41 / 0.99 / 0.601 | 0.906 / 0.56 / 1.19 / 0.902 |

  (cal=400 residual CovGap is larger — n_g = 400/5 = 80 per stratum is a noisier
  quantile than 160 @ cal=800 — but worst-stratum is still lifted to ~0.91.)
- Validity gated: unit tests 5/5 + GPU-parity 7/7. Branch `worktree-novelty-pilots`,
  NOT merged.

### 6. (Secondary pilot) calibration-conditional reliability — full CP is ~12x cheaper

**Intuition.** At few-shot the calibration DRAW dominates coverage variance. Over
100 draws at fixed label budget B, full CP rides the Beta(n=B) coverage-SD floor
(split-CP coverage law, Vovk 2012) while matched-budget Split-CP-THR degenerates to
size-K sets at B<=400; the set-size cost of a 95%-reliable 0.90 guarantee (SSBC
small-sample Beta correction) is **~12x cheaper** for full CP. Plots
`output/novelty_selling/reliability_sell.png`,
`reliability_convergence_cifar100.png`. Branch `worktree-novelty-pilots`.

### Literature + maintenance

- **CAOS** (Waldron, arXiv:2601.05219): split-free one-shot conformal adaptation on
  frozen foundation-model features; its Appendix E decomposition shows **~52% of
  its set-size win is "not splitting"** (reusing all labels), with ~7% from
  aggregation — external quantitative support for our FCP-over-SplitCP thesis.
  Tracked in `literature.md` §5.
- Archived 4 watch-list candidates + the one-off `literature_update_semicp_2026-05.md`
  (2 paper recs still pending into `literature.md` §8); pruned a done findings item;
  synced the CLAUDE.md active-scripts list with `src/`.

---

## Week ending 2026-06-22 (work on 06-21 to 06-22)

### 1. Centroid (cal-only) vs Cluster (unlabeled pool) MS-CS — does the cluster M update correctly per test point?

**Question.** In transductive Full CP every test point x is added to the bag
once per candidate label yc. The MS-CS similarity matrix M must update for that
augmented point. We wanted to confirm the **cluster M** (from the unlabeled
pool's k-means) updates correctly when the test point enters — i.e. does not
silently break exchangeability or produce degenerate sets.

**Mechanism verified.** Adding one point labelled yc shifts only class yc's
centroid, so the LOO update touches **only row/col yc of M** (O(K); O(1) for
cluster-M if yc's assigned cluster is unchanged), and the yhat update touches
only column yc. Exact vs brute-force LOO semantics.

**Empirical confirmation** (balanced cal+test, 10 trials, lambda=0.05,
tau = 0.5 * median_d^2). Six arms = 2 feature sources (separate 10k **pool** vs
**transductive** cal+test fit) x {cluster M, centroid M, no-penalty FCP}. Set
size (coverage in parens):

| arm | cal=200 | cal=400 | cal=800 |
|-----|---------|---------|---------|
| cluster-pool         | 7.21 (.943) | **2.05** (.907) | **1.58** (.921) |
| centroid-pool        | 8.50 (.941) | 2.15 (.909) | 1.59 (.920) |
| FCP-pool (lam=0)     | 8.93 (.943) | 2.28 (.911) | 1.63 (.920) |
| cluster-transductive | 12.52 (.942)| 2.58 (.908) | 1.72 (.920) |
| centroid-transductive| 14.42 (.943)| 2.77 (.905) | 1.72 (.920) |
| FCP-transductive     | 14.85 (.944)| 2.99 (.907) | 1.77 (.920) |

**Key points:**
- **Cluster update is correct.** All six arms stay valid (coverage 0.90-0.94,
  over-covering as expected for the balanced split — never under). A broken
  per-test update would show up as under-coverage or set blow-up; neither
  appears. The cluster-pool penalty cleanly beats its FCP-pool baseline
  (1.58 vs 1.63, ~3% @ cal=800; 2.05 vs 2.28, ~10% @ cal=400).
- **Cluster ~= centroid from cal>=400.** The unlabeled pool's value *for the M
  source* is a small-cal insurance only: cluster beats centroid ~15% @ cal=200
  (7.21 vs 8.50) but only ~1% @ cal=800. PCA denoises the cal class-centroids,
  so the cal-only centroid M recovers almost all the gain at cal>=400 — and
  needs **no pool** (pool-free, still exchangeable via the per-test centroid
  update; small change in `src/exchangeable_fcp_experiment.py`).
- **Separate pool > transductive (cal+test) features** on this balanced
  protocol (1.58 vs 1.72 @ cal=800) — the 10k pool earns its keep through the
  feature transform, not the M source.

**Output:** `output/balanced_mscs_source/balanced_mscs_source.png` + `results.json`.

### 2. K-shot test — effect of balancing cal+test, and the new default split

**What we did.** Definitive 3-arm A/B over the k-shot (C x k) protocol, 30
trials, post-missing-class-fix pipeline:
- `random` — label-blind split (exactly-exchangeable reference)
- `balanced_cal` — balanced cal, random test
- `balanced_both` — balanced cal AND test, equal shots m_cal == m_test (new)

| shots (cal=test) | random cov / sz | balanced_both cov / sz |
|------------------|-----------------|------------------------|
| 2 (200) | .9055 / 20.96 | .9198 / **5.36** |
| 4 (400) | .9033 / 3.36  | .9053 / **2.15** |
| 6 (600) | .9011 / **1.96** | .9299 / 2.18 |
| 8 (800) | .9012 / 1.68  | .9159 / 1.62 |

**Key points:**
- **Random is exactly tight** at every cal (0.901-0.906) even at cal=200 with
  13-15 classes missing from cal — the exact guarantee holds.
- **Balanced over-covers, peak +3.1pp @ cal=600** (conservative, never under).
  Balancing the *test* set too does NOT remove it => the over-coverage is the
  **bag-dependence / anchor-count channel**, not marginal label-shift (clean
  mechanism result). The cal=600 spike reproduces the old "92.5% @ cal=600"
  anomaly; conjecture: peaks where m-1 = NCM's k (5 same-class LOO anchors).
- **Set size flips with regime:** balanced wins big at small cal (-40 to -73%
  @ cal<=400), ties/loses ~+18% @ cal=600, ties @ 800. CovGap: balanced always
  better (tail-class protection).
- **DECISION (2026-06-21): default = balanced cal + balanced test, equal
  shots/class, fresh each trial.** Rationale: comparability with few-shot CP
  lit (jusiro FCA / SCA-T, the C x k protocol), every class present in both,
  smaller sets, better CovGap. Cost is mild and safe-direction (over-covers
  ~1-3pp). **Always also report the random arm** for the exact-validity claim.
  Implemented as `balanced_both` split + pilot-fixed-T in
  `src/exchangeable_fcp_experiment.py`.

**Output:** `output/split_ablation/` — `split_ablation.png` (3-panel:
coverage-vs-band / log set size / CovGap), `split_ablation_3arm.png`,
`split_ablation_results.json`, `balanced_both_results.json`.

### 3. FCA-inspired prototype-softmax NCM — the new best method

**Idea.** FCA (Silva-Rodriguez / jusiro, arXiv:2506.06076 — conformal for medical
VLMs) makes a softmax classifier head valid for Full CP by **re-fitting it in
closed form for each candidate label**. Their probe is a class-mean prototype
blended with a zero-shot text anchor, scored by LAC = 1 - p(y|x). We have no text
encoder (pure SSL, DINOv2), so we keep just the **class-mean prototype**: score
each label by a softmax over cosine similarities to the per-class means. New NCM
`PrototypeSoftmaxNCM`, `create_ncm("prototype_softmax", ...)`.

**Algorithm — prototype-softmax NCM inside Full CP** (plain text; `<.,.>` = inner
product; `n_c` = #cal points in class c; `n` = #cal points; `K` = #classes):

```
Setup (once)
  z(x)  = embedding of x after the exchangeable transform (PCA-128 +
          cluster-whiten, fit on the unlabeled pool), L2-normalised.
  mu_c  = mean of the calibration embeddings in class c   (class "prototype").
  T     = softmax temperature, fixed once on a pilot draw (never re-fit on cal).

NCM score  s(x, y)  -- "how nonconforming is label y for input x"
  f_c(x)  = < z(x), mu_c >              for every class c     (cosine similarity)
  p(y|x)  = softmax_c( f_c(x) / T )                           (class posterior)
  s(x,y)  = 1 - p(y | x)            (LAC/THR: small = typical, large = atypical)

Prediction set for a test point x   (transductive Full CP, exact)
  for each candidate label y in {1..K}:
    1. add (x,y) to the calibration bag; update ONLY class y's prototype:
         mu_y'         = (n_y * mu_y + z(x)) / (n_y + 1)
    2. leave-one-out re-score every bag point i on its OWN class y_i, using the
       bag-minus-i prototype (closed form, no matrix inverse):
         mu_{y_i}^(-i) = (n_{y_i} * mu_{y_i} - z(x_i)) / (n_{y_i} - 1)
         s_i           = 1 - softmax( < z(x_i), mu^(-i) > / T )[y_i]
    3. p_value(y)    = ( #{ i : s_i >= s(x, y) } + 1 ) / (n + 1)
    4. keep y  iff  p_value(y) > alpha
  output  { y : p_value(y) > alpha }
```

**Why it is exactly valid.** Every point in the augmented bag — the test point
AND each calibration point — is scored by the *same* leave-one-out rule, so the
n+1 scores are exchangeable and marginal coverage >= 1 - alpha holds for any fixed
T. No model is trained on the calibration labels (the "probe" is just class
means), and the leave-one-out update is closed form, so the per-candidate re-fit
is cheap; a bit-exact GPU path runs the whole sweep on the cluster. Verified: fast
path vs brute-force leave-one-out = 3.55e-15; small balanced cal **over-covers
with bloated sets, never under-covers**.

**Cluster results** (50 trials, balanced cal+test, CIFAR-100 + miniImageNet,
cal 200-1800, PCA-128 + cluster-whiten). CIFAR-100 set size (coverage), vs our
geodesic NCMs:

| cal | prototype_softmax | geodesic asym | geodesic mean | proto vs best geo |
|-----|-------------------|---------------|---------------|-------------------|
| 200 | **4.58** (.949) | 8.63 (.954) | 5.65 (.921) | **-19%** |
| 400 | **1.64** (.913) | 2.29 (.923) | 2.12 (.905) | **-22%** |
| 800 | **1.31** (.904) | 1.63 (.909) | 1.63 (.915) | **-19%** |
| 1800| **1.20** (.903) | 1.38 (.903) | 1.27 (.906) | **-6%** |

**Key points:**
- **Prototype is the TIGHTEST method at every cal on CIFAR-100** (-6% to -22%
  vs the best geodesic; up to -47% vs asym at cal=200), valid throughout
  (cov 0.90-0.95), and **CovGap best-or-tied** with asym (~5.8-6.3pp) — tighter
  *without* worse class-conditional coverage.
- **miniImageNet saturates** (DINOv2 separates it ~perfectly, sets ~1): prototype
  wins at small cal (cal=200: 1.39 vs 1.44 mean / 1.81 asym) and ties within
  ~1-4% at cal>=400. Prototype best/tied on CovGap (geo_mean worst, ~8pp).
- **Validity as designed:** small balanced cal **over-covers with bloated sets,
  never under-covers** (CIFAR-100 cal=200: sz 4.58 at cov 0.949) — the 50-trial
  confirmation of the bloat-not-undercoverage property.
- **Shipped to main:** NCM, GPU path, unit tests (incl. CUDA parity), and the
  cluster script.

**Output:** `output/from_cluster/fca_family_cluster/{results_cifar100,
results_miniimagenet,results_all}.json` + `compare_*_balanced_both.png`.
Script `src/fca_family_cluster_experiment.py` (default = prototype vs geodesic,
balanced, 2 datasets, 50 trials, GPU). NCM in `src/conformal_prediction.py`,
tests `tests/test_prototype_softmax_ncm.py`, theory `docs/theory.md` §4.1.

### Cross-cutting note

Topics 1-3 now all run on the **same balanced cal+test default** (topic 2's
decision), which is exactly the few-shot protocol FCA uses (topic 3) — so the
centroid/cluster MS-CS and the prototype-softmax results are directly
lit-comparable. The week's headline: the **prototype-softmax NCM** is the tightest
valid NCM on CIFAR-100 across the whole cal range, beating our geodesic NCMs.
Standing rule: pair every balanced headline with the random arm for the
exact-validity statement.
