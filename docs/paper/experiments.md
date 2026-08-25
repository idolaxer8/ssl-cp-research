# Paper-grade run ledger (R1-R6) ג€” launch protocol

Status as of 2026-08-20 (pipeline freeze). Maps each planned run
(`docs/iclr2027_plan.md` ֲ§2) to its driver on the `publish` branch, with the
exact launch command. Experiments are ISSUED FROM MAIN; a driver enters the
`publish` artifact branch only once its experiment is paper-ready. All runs target the cluster pod (RTX 6000 Ada, see
`cluster/README.md`); everything is resumable or cheap enough to re-run.

**Frozen pipeline** (all champion arms): Tג†’Wג†’D = pca128 ג†’ full-matrix LW
cluster whiten ג†’ alpha-QE `qe_stage=post` (separable regime; fine-grained =
full-rank `lw_cluster`, qe gated OFF by pool PR), scored by
`prototype_softmax` full CP. Re-validated 2026-08-20
(`output/freeze_validation/`, cifar100 cov 0.90-0.93, sz 1.88/1.36/1.21).

| Run | Priority | Driver | Output -> paper asset | Status |
|-----|----------|--------|----------------------|--------|
| R1 headline (v2, user spec 08-22): FROZEN champion vs SplitCP-THR/APS/RAPS + CV+ + SemiCP; 5 ds (separable cifar100/miniimagenet/cub200 = full frozen recipe; aircraft/cars = frozen w/o PCA); shots 2-14, 50 trials, alphas {0.1, 0.05}, metrics cov/size/CovGap | CRITICAL | `src/headline_experiment.py` via `cluster/run_headline.sh` | Table 2, Figs 1-2 | **COMPLETE (local run 08-23/24, `output/headline/`)**: all cells 50/50, 0 CPU fallbacks. Findings: (1) prototype_softmax auto-T COLLAPSES on fine-grained full-rank whitened geometry (T~0.002 -> full sets) -> aircraft/cars frozen rows repaired with `--frozen_ncm unwhitened_topk_asym` (both variants in the results files); (2) SplitCP-APS/RAPS at frac .75 under-covers 3-6pp on separable ds = balanced-split conditioning artifact (flatters the baseline; THR/CV+/SemiCP/frozen all valid) -> random-split companion pass pending user call; (3) cub200 qe control (`output/cub200_qe_control/`): as-run config right (qe wins @2sh, budget-dependent at mid-PR). Verdict: frozen dominates separable at all budgets (3.4x @2sh cifar100), crossover tracks PR (cub ~4sh, aircraft ~6sh, cars = scope limit) |
| R1b champion decomposition: {raw, wt, qe_wt} x 3 NCMs (the OLD R1; now the ablation companion) | high | `src/r1_headline_experiment.py` via `cluster/run_r1_headline.sh` | ablation table / Fig 2 inset | **COMPLETE for cifar100 + miniimagenet (local 08-25, 50 trials, balanced+random)**; paper outputs via `src/plot_r1b.py` -> `output/r1_headline/plots/` (cifar100 = MAIN paper, miniimagenet = APPENDIX, per-dataset files; "smoothing" naming). Fine-grained pending auto-T fix + NCM call |
| R2 baselines: selftrain probe rounds + pool-MLP reviewer arms (SCP/CV+/SemiCP moved INTO R1) | high | `src/g3_semisup_experiment.py` | Table 2 aux baseline rows | CV+/APS/RAPS now covered by R1 headline driver; g3 keeps selftrain/poolmlp only; raise --n_trials to 50 at launch |
| R3 transform menu + stanford_cars, 20-30 trials | high | `src/transform_control_experiment.py` | Table 4, Fig 3 | **COMPLETE for cifar100 + miniimagenet (local 08-25, 30 trials)** -> `output/r3_menu/`, Table 4 tex via `src/plot_r3_menu.py`; monotone raw < T < T+W < full at every cell, W-only << T-only at cal 200 (truncation load-bearing); cars pending |
| R4 backbone x cars; beitv2 in-or-drop | optional | `src/backbone_dwt_experiment.py` | Table 5 ext | as-is from v3 grid |
| R5 MS-CS/MA-CS penalty, 20 trials | optional | `src/mscs_unlabeled_experiment.py` | appendix D | as-is |
| R6 cifar10 appendix control top-up | cheap | `src/transform_control_experiment.py --dataset cifar10` | appendix | as-is |

## Launch commands

```bash
# R1 headline v2 (on the pod, from repo root; resumable per cell-trial,
# per dataset; cub200 needs output/pca_pilots/heldout_data synced to the pod)
nohup bash cluster/run_headline.sh > output/headline/run.log 2>&1 &

# R1b decomposition (after R1)
nohup bash cluster/run_r1_headline.sh > output/r1_headline/run.log 2>&1 &

# R3 (per dataset)
python src/transform_control_experiment.py --dataset <ds> \
    --data_dir output/from_cluster/embeddings \
    --arms raw768 pca128 pca128_lwcw qe_pca128_lwcw lw_cluster768 \
    --ncms prototype_softmax unwhitened_topk_asym --qe_stage post \
    --splits balanced_both --cal_sizes 200 400 800 --n_trials 30 --plot

# R6
python src/transform_control_experiment.py --dataset cifar10 \
    --arms qe_pca128_lwcw --ncms prototype_softmax --qe_stage post --n_trials 30
```

## R1 follow-ups (2026-08-24/25, all local, 50 trials unless noted)

- **W->D aircraft NCM ablation**: prototype cosine/softmax(T=0.07) on
  full-rank-W + qe-post do NOT beat topk-on-W overall, but both take the
  2-shot cell (26.0 / 27.5 vs 31.6); softmax ~= cosine at 2-4sh (not
  load-bearing there); qe hurts topk at every budget -> gate re-validated
  on aircraft. Rows in `results_aircraft.json` (+
  `output/headline_wd_softmax/` for the fixed-T softmax).
- **food101 (3rd separable ds, K=101, 336px local)**: frozen loses to CV+
  at ALL budgets despite pool PR 71.3 > qe-gate 64; qe-off control harms
  confirms qe hurts >=4sh yet qe-off still loses -> mid-separability or
  336px-extraction confound (cub200 + food101 = the two 336px locals;
  cifar100/mini at 518 win everywhere). 518 re-extract = decisive test,
  pending pod. Headline "wins at all budgets" claim currently rests on
  cifar100 + miniimagenet.
- **Per-arm timing** (`timing_{cifar100,stanford_cars}.json`): prototype
  full CP cheaper than CV+ from 8sh (1.25 vs 3.17 s/trial @14sh,
  cifar100), ~= SemiCP; one-off transform fit 8.4s. Geodesic topk @cars
  14sh: 28s ~ CV+ 24s. Runtime objection to full CP defused for the
  champion NCM.
- Paper figures: `src/plot_headline.py` -> `output/headline/plots/`
  (cifar100 done at both alphas).

## Pre-launch checklist

- [x] R1 specs user-fixed 08-22: 50 trials, shots 2-14, alphas {0.1, 0.05},
      metrics cov/size/CovGap; separable trio = cifar100/miniimagenet/cub200
      (full frozen recipe), fine-grained = aircraft/cars (frozen minus PCA);
      baselines SplitCP-THR/APS/RAPS + CV+ + SemiCP; FCA deferred to the
      ablation discussion.
- [x] SemiCP faithfulness: official repo (github.com/Shinning-Zhou/SemiCP,
      CVPR 2026, no LICENSE -> ported not vendored) matched exactly by
      `src/semicp_port.py`; documented adaptations = few-shot probe instead
      of their fully-trained frozen classifier, probs-not-logits input,
      non-randomized scores (their main-figure config), RAPS penalty=0.01
      kreg=2 (their CONFIG.py). Legacy `split_cp_baselines.compute_cp_sets`
      APS/RAPS off-by-one (boundary class included) does NOT affect the
      headline (port excludes it, per official rule).
- [ ] cars: 08-19 allocation variant (shot_lw, L>=800) as extra arm or
      main-table footnote ג€” user call.
- [ ] cub200: pod runs need `output/pca_pilots/heldout_data/` synced; the
      carve is local-336 (5977 labeled / 4000 pool). Re-extract at 518 for
      backbone parity with the other four ג€” user call (run can start at 336
      and be re-run cheaply).
- [ ] R3: confirm the menu subset for Table 4 (full 30-arm grid is not
      needed; the ablation column set is).
- [ ] R4/R5: in-or-out call.
- [ ] Embeddings: 4 main datasets present in
      `output/from_cluster/embeddings/` (cifar100, miniimagenet, aircraft
      matched-518 finals; stanford_cars via `_layers.pt` `final` key).
