# Archived scripts

Completed, superseded, or abandoned investigations. Kept runnable from the
repo root (`python src/archive/<script>.py`) but no longer maintained — they
may lag behind API changes in `src/conformal_prediction.py`. Results they
produced are recorded in `docs/findings.md` and `archive/findings_archive.md`.

## Superseded one-off comparisons (archived 2026-06-10)

| Script | What it was | Superseded by / outcome |
|---|---|---|
| `run_multi_dataset_experiments.py` | FCP vs CV+ vs SCP across cifar10/100, stl10, eurosat (April 2026 headline) | `semicp_experiment.py` multi-dataset runs + cluster matched-518 reproduction (findings §1-2) |
| `compare_ncms.py` | NCM comparison on full 768-d embeddings | `ncm_comparison_reduced.py` (same NCMs × full/PCA/AE reductions, stratified) |
| `fcp_vs_aps_raps.py` | FCP vs SCP-APS/RAPS, labeled budget only | `semicp_experiment.py` (adds SemiCP + NNM augmentation, multi-dataset) |
| `fcp_vs_scp_mlp.py` | FCP+PCA vs SCP with LR/MLP heads at matched budget | SCP-geodesic isolation: `FullConformalPredictor.predict(update_calibration_scores=False)` (findings §9) |
| `pca_vs_semicp.py` | Fair pool-usage comparison (FCP / +PCA / +MS-CS / SemiCP) + AE-vs-PCA driver | `cs_ablation.py` (reduction × MS-CS) + exchangeable pool ablation (findings §3-4) |
| `visualize_embeddings.py` | t-SNE / PCA embedding scatter plots | unused since 2026-02; revive for paper figures if needed |

## RBF NCM development iterations (archived 2026-06-10, were in `tests/`)

`rbf_bandwidth_sweep.py`, `rbf_iter2_capacity.py`, `rbf_iter3_whitened.py`,
`rbf_iter4_confirm.py`, `rbf_iter5_asym.py`, `rbf_iter6_paired.py` —
bandwidth/capacity/whitening iterations that produced the tuned
`rbf_density` config (sigma_scale=0.20, ratio mode). Consolidated driver:
`src/rbf_ncm_experiment.py`; result: findings §5 (AE-128 + wRBF wins at
cal=600).

## Abandoned / negative results (pre-2026-06)

| Script | Outcome |
|---|---|
| `semisup_fcp_experiment.py` | Pool augmentation — feedback loop breaks exchangeability (O(N/n)) |
| `lata_fcp_experiment.py` | Naive label-free score smoothing destroys class discrimination (n.b. distinct from Fargion-group "LATA") |
| `ppi_rcps.py` | PPI-RCPS — deprioritized 2026-05-18, split-CP-oriented |
| `split_audit_experiment.py` | Split audit — concluded |
| `run_few_shot_experiment.py` | Few-shot regime — concluded |
| `compare_ssl_models.py` | Old SSL backbone comparison — superseded by `src/compare_backbones.py` |
| `rank_diagnostic.py` | Embedding rank diagnostic — concluded |
| `plot_macs_sweep.py` | Old MA-CS sweep plots |
