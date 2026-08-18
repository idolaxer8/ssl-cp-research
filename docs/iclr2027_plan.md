# ICLR 2027 submission plan

Status: v0.1 (2026-08-18, weekly goal 6). Deadline anchor: **abstracts due
Sep 18, full papers Sep 25 (AOE)** — 5 working weeks from this week (08-17).
This is the single distilled story for the submission plus the gap audit,
calendar, and code-organization ledger. Math plain ASCII per repo convention.

Companion docs: `docs/theory.md` (operating theory), `docs/findings.md`
(result log), `docs/dwt_denoise_theorem.md` (D1-D3), the W/T lemma draft
(`dwt_wt_lemmas.md`, on `worktree-theory-dwt-justification`),
`docs/weekly_summary.md` (goals 1-6 rationale).

---

## 0. The strategic call (read this first)

**The theory is not carrying the paper, and the plan is built around that.**
As of 08-18 only **D1** — the exact, assumption-free per-point denoise lemma
— stands as a genuine theory contribution. The W and T lemmas (W1a/W1b/T1a
exact cores + the chain-free "W/T serve D1's inputs" reframing) are
suggestive and measured, but their load-bearing quantitative layer (D3',
GW2b, GT1 constants) is unsupported, and the ideal endpoint
`E|C^T| <= E|C|` is an openly documented obstruction, not a result. Do NOT
stake the submission on a "DWT theorem".

**Therefore the paper is empirical-first.** Headline = a label-free,
exchangeable, pool-fit preprocessing pipeline that makes Full CP deliver
tight VALID sets at label-starved budgets where split CP / CV+ collapse.
Label-free dials (PR, homophily h, d'-ratio) appear as the **regime map** —
they explain and predict where each transform pays off — but we do NOT ship
an automatic transform-selection gate (user decision 08-18, §1 C4): the
transform choice is presented per-regime, the dials as its diagnosis. Theory
is a **supporting section**: D1 as an exact denoise lemma, W1a/W1b/T1a as
exact per-pair statements, validity-is-free as the clean structural result,
and an honest gap ledger. This framing is robust to the W/T rigor NOT
landing by the deadline.

**The one decision that gates the theory section** (resolve by wk 08-24, §4
D-1): do we claim a *partial DWT theorem* (D1 exact + W1a/W1b/T1a exact +
documented gaps) as a contribution, or demote theory to a "mechanism /
justification" subsection with no theorem-numbered W/T claims? Both are
defensible; the empirical contribution does not depend on the answer.

---

## 1. Distilled claim list (the working story)

Six claims, ordered by how much weight they carry. Each is a sentence a
reviewer can check against a figure/table.

- **C1 (predictor).** At label-starved budgets (K >= 100, cal < ~600) split
  CP and CV+ are invalid or trivial; **Full CP is the only method that is
  both valid and tight**. FCP is load-bearing precisely in our regime.
- **C2 (NCM).** Denominator-free prototype / geodesic NCMs on FROZEN SSL
  embeddings give tight valid sets on separable data (cifar100 sz 1.30 @
  cal-800) and **degrade gracefully** — coverage held, size large — on
  fine-grained data (aircraft/cars). The scope limit is stated, not hidden.
- **C3 (pipeline).** Pool-fit DWT transforms are **exchangeable by
  construction** (pool-measurable => coverage exact) and are the dominant
  efficiency levers: PCA-128 truncation on separable data, full-rank
  Ledoit-Wolf cluster-whitening on fine-grained, alpha-QE pool-neighbor
  denoising as a gated third lever (subsumes the SNAPS score correction).
- **C4 (regime map — OPTIONAL, descriptive only; user decision 08-18).**
  Label-free dials CHARACTERIZE the regimes: participation ratio PR tracks
  where truncation vs full-rank whitening pays off; weighted pool homophily
  h_w tracks where qe/SNAPS denoising helps vs harms; the d'-ratio
  diagnoses the denoise sign. Presented as ANALYSIS (the regime map behind
  the per-regime transform choice), **NOT as a deployed automatic
  selection gate** — we do not want to introduce a gate that automatically
  selects the optimal truncation. Automatic selection = future work.
- **C5 (transfer).** The pipeline **transfers across backbones**
  (dinov2 / dinov3 / clip-base / clip-large): all valid + tight. **PR is the
  transfer-robust dial** (7/8 cells); the raw homophily gate does NOT
  transfer (inverts on CLIP) — the paper's generalization caveat.
- **C6 (theory, supporting).** Validity is FREE (every transform
  pool-measurable, exchangeability preserved, FCP wrapping exact). **D1**:
  an exact per-point lemma certifying when pool-kNN smoothing contracts
  representation error (DAPS-Thm-2 analogue in representation space).
  **W1a/W1b/T1a**: exact per-pair whitening/truncation statements
  (Cauchy-Schwarz optimal metric; unlabeled fit exact by Sherman-Morrison;
  truncation alignment identity). The `E|C|` endpoint is open (documented
  obstruction, §4 G-T).

Working title direction: *"Label-free regime dials for exchangeable
pool-fit conformal prediction on frozen SSL embeddings."* (Finalize wk 08-31.)

---

## 2. Claim -> evidence -> gap table

Evidence = the committed experiment/figure backing the claim. Gap = what is
still missing before it is submission-grade. Status: [S] solid/committed,
[P] partial (needs a rerun/consolidation), [O] open (needs new work).

| Claim | Evidence (where) | Status | Gap to close |
|---|---|---|---|
| **C1** FCP load-bearing | FCP-vs-SCP-vs-CV+ comparisons; CAOS App-E external support (lit.md §5); `scp-geodesic-isolates-ncm-vs-fcp` | [S] | **Naming debt CLOSED 08-18** (THEORY.MD claim 2 + Key Discoveries fixed; theory.md §5.1 and findings.md §9 were already correct). Remaining nicety (not a gap): one consolidated head-to-head table at cal 200/400/800 x {cifar100, aircraft} for the paper draft. |
| **C2** NCM tight+graceful | prototype/geodesic ablations (`fca_ablation`, `prototype-softmax-ncm-rung3`); aircraft/cars scope limit (`aircraft-stress-test-scope-limit`) | [S] | Consolidate the 5-dataset NCM table (cifar100/mini/cifar10/aircraft/cars) at one seed budget; prototype_cosine (D1 goal-1) as the theory-faithful NCM row. |
| **C3** DWT exchangeable + levers | exchangeable pipeline (`exchangeable-unlabeled-pipeline`); PCA audit (`pca-rationalization-audit`); qe menu round 1 (`pool-repr-menu-round1`, qe subsumes SNAPS) | [P] | One reproduction script per table; qe results are on `worktree-pool-repr-menu` (unmerged). Confirm qe champion numbers at high trial on cluster. |
| **C4** regime map (OPTIONAL) | PR rule (PCA audit); h_w gate + d'-ratio (`dwt-denoise-theorem`, `d1_softmax_ablation`); transform-selection pilot | [P, optional] | **Descoped to descriptive analysis (user 08-18): no automatic transform/truncation-selection gate is shipped.** Needed only: the dial-vs-regime table as an analysis figure. Mid-PR (CUB) selector, dial->decision automation, and the (h_w, kappa) label-free surrogate all move to FUTURE WORK — no longer deploy blockers. |
| **C5** backbone transfer | goal-5 verdict (`backbone-dwt-gate-transfer`, `worktree-backbone-dwt`, 4 backbones x 2 paradigms) | [P] | MAE dropped (PR=2 degenerate) — decide whether to include as the negative pole or omit; beitv2 not run. Finalize the backbone x dataset table. |
| **C6a** validity free | Prop 1 (`dwt_theory.md` §2); exchangeability oracle in qe impl; FCP exactness oracle (MDCP 1.4e-14) | [S] | None material — this is the cleanest result. Just write it up. |
| **C6b** D1 denoise lemma | `dwt_denoise_theorem.md` D1; 5/5 pointwise validation; chain-free cert-rate table (`dwt_wt_lemmas.md` §7.1) | [S] | Move D1 doc onto main (currently on theory worktree). It is submission-grade as stated. |
| **C6c** W/T lemmas | W1a/W1b/T1a exact cores + measured `wt_phase_diagnostics` (aircraft wlw x3.13); chain-free reframe | [O] | **THE #1 THEORY RISK.** D3' quantitative gate unsupported; GW2b/GT1 constants owed; E|C| endpoint open. Decide (§4 D-1): claim partial theorem vs demote to mechanism. |
| **C6d** E\|C\| upgrade | SNAPS Prop 2 device (goal 4); Dhillon E\|C\| identity; Corollary D4 contraction-anchor reframing | [O] | Documented OBSTRUCTION, not a result. Do not claim. Optional stretch only. |

**Reading the table.** C6a/C6b/C2 are solid. C1/C3/C5 are partial — mostly
consolidation + one clean table each, not new science. C4 is optional
analysis (descoped 08-18). **C6c is the only true theory risk**, and the
paper is structured (§0) so that C6c landing or not does not sink the
submission.

### Expected-gap experiments still owed (the wk 08-24 run list)

1. **Backbone table finalize** (C5): fill beitv2 or drop; MAE-as-negative-pole
   decision; one table size@200/400/800 + coverage + dial values + gate-correct.
2. **qe champion consolidation** (C3/C4): high-trial cluster rerun of the qe
   menu numbers so the headline pipeline row is reproducible from main.
3. **Regime-map analysis figure** (C4, optional): dial values (PR/h/d') vs
   observed best transform across the 5 datasets + backbones — DESCRIPTIVE
   only (no selection gate; descoped 08-18). Cut first if time is short.
4. **FCP-vs-baselines clean table** (C1): fix the naming debt, one honest
   head-to-head.
5. **W/T rigor** (C6c, theory-thread, high-risk): at minimum land the
   chain-free W-N and T-A lemmas cleanly; the quantitative gate is a stretch.

(Removed from the run list 08-18: the dial->decision automation table and
the mid-PR CUB selector — both belonged to the descoped C4 auto-selection
story; now future work.)

---

## 3. Code organization — merge-or-archive ledger

13 branches were unmerged into main. **EXECUTED 2026-08-18 (pulled forward
from wk 08-24):** the 4 core branches + the 08-10 cleanup branch are merged
(conflicts in THEORY.MD / literature.md / weekly_summary.md resolved by
union); transform-selection turned out fully committed (stale memory) and is
contained in pool-repr-menu; archive verdicts recorded in `findings.md` §12.
Worktree DIRECTORIES left in place (gitignored `output/` inside — never
delete). Rule: one reproduction script per planned table/figure lives on
`main` after merge.

| Branch | Holds | Disposition | Note |
|---|---|---|---|
| `worktree-backbone-dwt` | goal-5 backbone table (C5) | **MERGE** | Core paper table; land script + `output/from_cluster/backbone_dwt_v2/`. |
| `worktree-pool-repr-menu` | qe menu, champion (C3/C4) | **MERGE** | Core pipeline lever; qe subsumes SNAPS. High-trial confirm first. |
| `worktree-theory-dwt-justification` | D1 doc + W1/T1 draft (C6) | **MERGE (D1 doc) + KEEP (W/T draft)** | D1/`dwt_denoise_theorem.md` is submission-grade -> main. W/T stays a draft until §4 D-1. |
| `worktree-g3-semisup` | reviewer-baseline arms | **MERGE (results doc)** | SemiCP = strongest baseline; needed for the baselines section. |
| `worktree-transform-selection` | label-free selector (C4) | **COMMIT + KEEP** | Uncommitted per memory — commit for provenance, but the selector is now FUTURE-WORK material (C4 auto-selection descoped 08-18); not on the paper's critical path. |
| `worktree-novelty-pilots` | geometric-conditional CP | **KEEP (doc committed)** | `docs/geometric_conditional_methods.md` exists; side result, likely not in main paper. |
| `worktree-snaps-pool` | SNAPS correction | **ARCHIVE (note)** | Subsumed by qe (`pool-repr-menu-round1`); keep as provenance for the "qe subsumes SNAPS" claim. |
| `worktree-mdcp-pool-pilot` | MDCP pool pilot | **ARCHIVE** | Direction parked; folded into literature. Keep branch as archive. |
| `worktree-mdcp-vanilla-split` | vanilla MDCP high-cal | **ARCHIVE** | Superseded; note verdict in findings. |
| `worktree-conformal-metric` | G1 conformal-metric | **ARCHIVE** | User called it a failure (08-03); keep branch, no merge. |
| `worktree-goal4-caos-litsweep` | CAOS lit sweep | **ARCHIVE** | Keepers already folded into `literature.md`; verify then archive. |
| `worktree-novelty-framing-review` | framing review notes | **ARCHIVE** | Notes only. |
| `worktree-saps-prototype-ncm` | SAPS NCM (killed) | **LEAVE** | Already closed/archived (`saps-prototype-ncm-diagnostic`). |
| `routine/repo-cleanup-2026-07-10`, `-08-10` | cleanup commits | **REVIEW + MERGE or DROP** | Bi-weekly cleanup branches; merge if clean, else drop. |

**Housekeeping actions (wk 08-24):** commit the transform-selection work;
merge the 3 core branches (backbone-dwt, pool-repr-menu, theory D1 doc);
write a one-paragraph archive note per archived branch into `findings.md`;
confirm every headline number has a reproduction script on main.

---

## 4. Theory distillation — the paper-shaped skeleton

Target shape: **assumptions -> per-phase lemmas -> validity -> gaps**, NOT a
grand DWT theorem. What each part can honestly claim today:

```
Setting (M1):  classes with anchors mu_c, shared within-class cov Sigma,
               pool U independent of cal/test; score = prototype-cosine /
               geodesic ratio; margin currency d'.

Validity      (Prop 1, EXACT, no gaps): every DWT transform is a fixed
  [SOLID]     measurable function of the pool; conditional on the pool,
               exchangeability of {(T(x_i), y_i)} holds => split CP coverage
               in [1-a, 1-a+1/(n+1)), FCP wrapping exact. "A wrong dial
               costs efficiency, never validity." <- the clean structural win.

Denoise D1    (EXACT, assumption-free): ||x_hat - mu_y|| < ||x - mu_y||
  [SOLID]     whenever  sum_u (w_u/W) r_u + (1-h_w) Delta_F(x) < eps.
               DAPS-Thm-2 in representation space; certified => improved held
               5/5 pointwise. This is THE theory anchor.

Whiten W1a/b  (EXACT per-pair cores): d'^2(M) <= delta' Sigma^-1 delta,
  [PARTIAL]   max at M = Sigma^-1 (Cauchy-Schwarz); total-cov whitening
               attains it label-free (Sherman-Morrison, two-class exact).
               GAP: K-class remainder (GW1), pool-estimation + impurity
               (GW2a/b) constants owed.

Truncate T1a  (EXACT identity): d'_m/d' = sqrt(a_m), a_m = ||P_m delta||^2 /
  [PARTIAL]   ||delta||^2; post-whitening truncation NEVER helps population
               d' (honest half the folklore skips). The finite-shot GAIN
               (T1b: anchor error m/s vs d/s) is the real mechanism but
               conditional; constants owed (GT1/GT2).

Chain-free    W/T justified WITHOUT the d'->LemmaB->PropC chain: W serves
  reframing   D1's neighborhoods (aircraft D1 cert rate .027 -> .150, x5.5),
  [PARTIAL]   T serves anchor estimation (s-shot err -25..40%). D1-cert
               argmax sorts 4/5 champion cells. Suggestive, measured, not
               yet a theorem.

Gaps (the honest ledger, stated in the paper):
  G-D3'  quantitative denoise gate: (I)-model overpredicts everywhere; open.
  GW2b   pool-whitening impurity/granularity constant.
  GT1    structured-bulk truncation (Chang-functional form).
  G-E|C| the E|C^T| <= E|C| endpoint: documented OBSTRUCTION (goal 4),
         NOT claimed. Optional stretch, not on the critical path.
```

**Decision D-1 (resolve wk 08-24):** claim a *partial DWT theorem*
(D1 exact + W1a/W1b/T1a exact + gap ledger) as a numbered contribution, OR
demote W/T to a "mechanism" subsection and claim only D1 + validity as
theory. Recommendation: **demote unless the chain-free W-N and T-A lemmas
land cleanly by 08-31** — the empirical contribution stands either way, and
an over-claimed theorem is a reviewer liability given the current gaps.

**Format note:** theory/lemma docs use LaTeX (VS Code / GitHub rendered) per
the 08-18 user preference; this PLAN doc uses plain ASCII (planning doc,
weekly-summary convention).

---

## 5. Calendar (5 weeks to Sep 25 AOE)

Today = 2026-08-18 (Tue, wk 08-17). Milestones anchored to the ICLR dates.

| Week | Dates | Focus | Exit criterion |
|---|---|---|---|
| **wk 08-17** | 08-17..08-23 | Gap audit (this doc) + finish theory goals 1-4 | This plan committed; goals 1-4 landed (1,2,4 done; 3 = W/T draft). Decide D-1 scope. |
| **wk 08-24** | 08-24..08-30 | **Freeze method.** Run the 6 owed experiments (§2); merge 3 core branches; commit transform-selection | Method frozen; every headline number has a repro script on main; backbone + dial tables done. |
| **wk 08-31** | 08-31..09-06 | **Draft method + theory sections first** (the risky ones); finalize title/abstract framing | Method + theory sections in full draft; D-1 resolved; theory demote-or-claim final. |
| **wk 09-07** | 09-07..09-13 | Full draft (intro/related/experiments/discussion) + internal review pass | Complete draft; self-review + one external read; figures final. |
| **wk 09-14** | 09-14..09-20 | Polish; **abstract submitted by Sep 18** | Abstract in by 09-18 (hard); paper near-final. |
| **wk 09-21** | 09-21..09-25 | Final polish, reproducibility check, **paper by Sep 25 AOE** | Paper submitted; supplementary + code archive ready. |

**Hard dates (VERIFIED 2026-08-18 vs official ICLR 2027 CFP):** abstract
**Sep 18, 2026 AOE**, paper **Sep 25, 2026 23:59:59 UTC-12 (AOE)**. (Reviews
Nov 5; author-reviewer discussion Nov 5-18.) Buffer is thin; the wk 08-24
freeze is the critical gate — if the owed experiments slip past 08-30, cut
scope (drop the C4 regime-map figure and the W/T quantitative gate first;
both are already flagged optional).

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| W/T theory does not reach theorem grade | **High** | Low (by design) | §0 framing: empirical-first; demote theory (D-1). Paper stands on C1-C5. |
| Reviewer asks "why no automatic transform selection?" (C4 descoped) | Medium | Low | State it plainly: dials are diagnosis, auto-selection is future work; the per-regime choice is fixed a priori and validity never depends on it (a wrong choice costs efficiency, not coverage). |
| Backbone table incomplete (beitv2/MAE) | Medium | Medium | Include 4 landed backbones; MAE as negative pole or omit; state scope. |
| qe champion numbers not reproducible from main | Low | High | wk 08-24 high-trial cluster rerun + merge before freeze. |
| ICLR dates slip / differ | Very low | High | Dates VERIFIED vs official CFP (08-18); calendar has a ~3-day buffer to Sep 25 AOE. |
| Naming debt (SCP-geodesic, "MS-CS") confuses reviewers | Medium | Medium | Fix THEORY.md/theory.md/findings.md naming in wk 08-31 draft pass. |

---

## 7. Immediate next actions (this week, 08-17)

- [x] Write this plan (`docs/iclr2027_plan.md`).
- [ ] User decision on **D-1** (claim partial DWT theorem vs demote W/T).
- [ ] User decision on paper framing/title direction (§1).
- [ ] Confirm goals 1-4 status closed for the week (1/2/4 done; 3 = draft
      kept open by user's dissatisfaction — feeds the D-1 decision).
- [ ] Queue the wk 08-24 experiment run list (§2) + branch merges (§3).
```
