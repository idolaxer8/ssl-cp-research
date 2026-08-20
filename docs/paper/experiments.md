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
| R1 headline: shots 2-14, 4 main ds, {raw, wt, qe_wt} x 3 NCMs, 50 trials | CRITICAL | `src/r1_headline_experiment.py` via `cluster/run_r1_headline.sh` | Table 2, Figs 1-2 | READY ג€” frozen pipeline encoded (6771b06 on main), checkpoint/resume verified; NOT yet launched |
| R2 baselines: SCP-THR/APS/RAPS, SemiCP, probe+CP, CV+ at matched budgets, 50 trials | CRITICAL | `src/g3_semisup_experiment.py` | Table 2 baseline rows | needs arm audit (CV+/APS/RAPS coverage) before launch |
| R3 transform menu + stanford_cars, 20-30 trials | high | `src/transform_control_experiment.py` | Table 4, Fig 3 | READY (champion arm `qe_pca128_lwcw --qe_stage post`) |
| R4 backbone x cars; beitv2 in-or-drop | optional | `src/backbone_dwt_experiment.py` | Table 5 ext | as-is from v3 grid |
| R5 MS-CS/MA-CS penalty, 20 trials | optional | `src/mscs_unlabeled_experiment.py` | appendix D | as-is |
| R6 cifar10 appendix control top-up | cheap | `src/transform_control_experiment.py --dataset cifar10` | appendix | as-is |

## Launch commands

```bash
# R1 (on the pod, from repo root; resumable per trial, per dataset)
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

## Pre-launch checklist (blocked on user specifications)

- [ ] R1: confirm trial count (50), shot grid (2-14), and whether cars uses
      the 08-19 allocation variant (shot_lw, L>=800) as an extra arm or
      main-table footnote.
- [ ] R2: audit `g3_semisup_experiment.py` for CV+/APS/RAPS arms; add if
      missing (plan flags this explicitly).
- [ ] R3: confirm the menu subset for Table 4 (full 30-arm grid is not
      needed; the ablation column set is).
- [ ] R4/R5: in-or-out call.
- [ ] Embeddings: all 4 main datasets present in
      `output/from_cluster/embeddings/` (cifar100, miniimagenet, aircraft
      matched-518 finals; stanford_cars via `_layers.pt` `final` key). CUB
      local-336 only ג€” re-extract at 518 if CUB moves out of the appendix.
