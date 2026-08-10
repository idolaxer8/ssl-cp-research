# qe-upgrade pilots (2026-08-10): hub-debiased selection confirmed; reciprocal / 2-hop / z-norm killed

Follow-up to the alpha-QE citation sweep (`docs/alphaqe_citation_sweep.md`, branch
`worktree-theory-dwt-justification`): the sweep's ranked upgrade shortlist for the
qe (Denoise) stage, piloted on the round-1 champion pipelines. All arms are pool-fit
fixed maps -> exactly exchangeable; efficiency-only comparison.

## Design

Harness: `src/transform_control_experiment.py`, balanced splits, 10 trials, paired
seeds (seed 42), alpha=0.1, coverage verified ~0.90 everywhere. Champion arms:
cifar100 `qe_ldapool192` (prototype NCM), aircraft `qe_ldapool512` (geodesic
`unwhitened_topk_mean` + prototype), miniimagenet `qe_ldapool128` (both NCMs).
No-qe controls: same arms without the `qe_` prefix. Results:
`output/qe_upgrade_pilots/*.json` (worktree-local).

New `UnlabeledTransform` knobs (all additive; default path verified bit-identical
to HEAD on every legacy config; existing 8 repo tests pass):

- `qe_hub_gamma` (CSLS/NNN lineage): neighbor SELECTION score = cos(x, u) -
  gamma * hub(u), where hub(u) = u's mean top-qe_k cosine to the rest of the
  pool (pool-fit, cached). Weights/aggregation stay on raw cosine.
- `qe_iters`: multi-hop smoothing; hop i uses the i-times-smoothed pool as bank
  (first-order pieces of the Tikhonov filter (I + lam*L)^{-1}).
- `qe_znorm`: selection in per-vector z-scored (Pearson) space (Fei et al. ICCV
  2021 hubness reduction).
- (`qe_reciprocal` existed from round 1, first evaluated here.)

## Results (mean set size +- SE; cal 200 / 400 / 800)

aircraft, geodesic (harm regime, kNN homophily ~0.25):

```
no-qe            28.69+-.53   24.66+-.40   22.37+-.34
classic qe       28.74+-.77   26.96+-.36   24.89+-.31
safe qe (b=.3)   27.38+-.53   24.29+-.36   22.49+-.22   <- round-1 champion, reproduced
recip (classic)  35.13        30.78        28.93        KILL
safe+recip       30.12        25.83        23.96        KILL
znorm            29.25        27.65        25.13        KILL
iters2           32.66        30.34        27.41        KILL
hub gamma=0.5    27.23        26.03        23.89
hub gamma=1.0    26.55+-.44   25.31        23.98        <- NEW BEST @200
hub gamma=1.5    30.83        27.51        25.41        (non-monotone!)
hub gamma=2.0    28.27        24.48        22.12        (~ converges to no-qe)
safe + hub 1.0   26.60+-.67   24.23+-.37   22.39+-.24   <- >= safe at every cell
```

aircraft, prototype: no-qe ldapool512 = 21.50+-.28 / 18.22+-.33 @400/800 — beats
the round-1 champion cells (safe qe 21.91 / 19.81) by 2% / 8%. hub gamma=2.0
lands on top of it (21.72 / 18.37) via self-gating. At 200 prototype degenerates
(sz~100) in all arms except classic qe 65.5 (irrelevant; geodesic 26.6 rules).

cifar100, prototype (help regime, homophily ~0.8):

```
no-qe            2.98+-.24   1.49+-.04   1.24+-.03
classic qe       2.29+-.17   1.43+-.04   1.27+-.03   <- round-1 numbers, reproduced
safe qe          2.27+-.14   1.42+-.04   1.23+-.02
hub 0.5          2.42        1.40+-.04   1.24
hub 1.0          2.47        1.44        1.24
hub 1.5          2.47        1.40+-.04   1.22+-.03
znorm            2.45        1.52        1.31        KILL
iters2           2.94        1.55        1.42        KILL (~= no-qe @200)
recip            2.49        1.42        1.24        KILL (no gain)
```

miniimagenet (saturated, homophily ~0.92):

```
geodesic:  no-qe 1.19/1.02/0.98 | classic qe 1.18/1.09/0.99 | hub 1.0  1.12+-.07/0.99+-.01/0.97+-.01
prototype: no-qe 1.34/1.06/1.02 | classic qe 1.24/1.06/1.04 | hub 1.0  1.22+-.03/1.02+-.01/1.00+-.01
```

hub 1.0 wins EVERY mini cell on both NCMs, and beats the round-1 pre-qe menu best
(1.19/1.03/0.99) — closes the known "saturated-dataset residual deficit" of qe.

## Verdicts

1. **Hub-debiased selection (gamma ~= 1.0) = CONFIRMED upgrade direction, 3/3
   datasets.** Profile: (a) new best at the starved harm-regime cell (aircraft
   geodesic @200: 26.55 vs prior best 27.38, -3%; beats no-qe by -7%); (b)
   `safe + hub 1.0` dominates the deployed safe mode on aircraft (never worse,
   wins @200) — the levers compose (dose control via b, selection via gamma);
   (c) neutral-to-marginal-win on cifar (best cells @400/800 at gamma 0.5-1.5);
   (d) wins every miniimagenet cell. Mechanism: better CHOICE of neighbors, not
   smaller dose — the first lever that improves the harm regime through
   selection quality. Caveat: gamma response is NON-MONOTONE on aircraft
   (1.5 worse than both 1.0 and 2.0); at large gamma the selected low-hub
   neighbors have low raw cosine, clipped weights vanish, and the map
   self-gates toward identity (hub 2.0 ~ no-qe on both aircraft NCMs) — a
   soft OFF switch, interesting for the gate story but means gamma is a real
   knob (default 1.0; sweep before deploying elsewhere).
2. **Aircraft champion correction: under the prototype NCM, drop qe.** Plain
   ldapool512 = 21.50/18.22 @400/800, an 8% improvement over the round-1
   champion @800. Round 1 never ran prototype on the no-qe discriminant; its
   "safe mode has no known harm cell" claim dies here (19.81 vs 18.22). The
   label-free gate should be able to turn smoothing OFF, not just dilute it —
   at homophily ~0.25 OFF is the right setting under prototype (geodesic still
   prefers safe/hub smoothing at 200).
3. **k-reciprocal filter: KILL.** Ties cifar, worse than safe everywhere on
   aircraft (mutual edges are no purer at low homophily — the failure mode is
   plain neighborhood impurity, not hub asymmetry; reciprocity only cuts dose,
   beta does it better).
4. **2-hop (Tikhonov 2nd Neumann term): KILL, with a theory payoff.** Worse at
   every cell on both regimes; the second hop compounds wrong-class
   contamination and over-contracts toward pool centroids. Datum for the DWT
   denoise lemma: the optimal graph-filter truncation is FIRST order — one hop
   is not an approximation compromise, it is the optimum in both regimes tested.
5. **z-norm (Pearson) selection metric: KILL.** Neutral-to-worse everywhere;
   per-vector z-scoring is too crude for DINOv2 embedding geometry.

## Next steps

- hub gamma=1.0 confirmation at high trials + CUB (mid regime) + cars
  (homophily ~0.45, the remaining map point); then fold `safe+hub` into the
  deployment default and re-run the champion table.
- Non-monotone gamma: map the selection-overlap + effective-dose curves vs
  gamma (cheap diagnostics on the pool graph) before trusting any gamma != 1.
- Gate story upgrade: the gate now has three settings (classic / safe(+hub) /
  OFF) and two label-free inputs worth piloting from the citation sweep: HUGE
  attribute-space heterophily + hub-intersection (QB-Norm DIS) statistics.
- Theory: cite the one-hop-optimality datum in the DWT denoise lemma
  (first-order truncation of (I + lam*L)^{-1}); hub-debias enters as a
  selection-metric correction, orthogonal to the averaging analysis.
