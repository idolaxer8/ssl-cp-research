# Archived scripts

Completed, superseded, or abandoned investigations. Kept runnable from the
repo root (`python src/archive/<script>.py`) but no longer maintained — they
may lag behind API changes in `src/conformal_prediction.py`. Results they
produced are recorded in `docs/findings.md` and `archive/findings_archive.md`.

## Watch list — archive candidates (review dates per row)

Files still in `src/` that get archived at the next bi-weekly cleanup sweep
if their review date passes without use. Each carries a matching
`# ARCHIVE-CANDIDATE (review YYYY-MM-DD)` header note (greppable). Entries
marked HOLD are tied to a pending item in `docs/findings.md` §10 — archive
them only when that item completes or is dropped, even past the review date.

| File | Why candidate | Condition |
|---|---|---|
| `ncm_comparison_reduced.py` | results recorded 2026-05-14 | HOLD: P0.3 (topk_asym cluster confirmation) |
| `cs_ablation.py` | reduction×MS-CS ablation recorded | HOLD: P0.4 (per-dataset AE retrain) |
| `compare_backbones.py` | backbone comparison recorded (findings §7) | HOLD: P2.11 (cluster backbone sensitivity) |
| `rbf_ncm_experiment.py` | RBF result recorded (findings §5) | HOLD: P1.5 (RBF multi-dataset) |
| `conditional_coverage_experiment.py` | class-conditional results recorded (findings §2b); §10 item 8 done | plain: archive after 2026-08-24 if unused |
| `pool_source_comparison.py` | pool-source question settled (findings §4d) | plain: archive after 2026-08-24 if unused |
| `pool_source_limits.py` | pool-source limits arm recorded (findings §4d) | plain: archive after 2026-08-24 if unused |
| `plot_pool_source.py` | plots only the two pool_source_* result JSONs | plain: archive with the two above (2026-08-24) |
| `pool_ablation_hightrial.py` | high-trial pool ablation recorded (findings §4e); question settled | plain: archive after 2026-08-24 if unused |
| `plot_unlabeled_pool_ablation.py` | plots only the §4e pool-ablation results | plain: archive with `pool_ablation_hightrial.py` (2026-08-24) |
| `ridge_softmax_cluster_experiment.py` | superseded by `fca_family_cluster_experiment.py` (ridge = rung 4 of the same ladder) | plain: archive after 2026-08-24 if unused |
| `plot_ridge_softmax_compare.py` | plots only the ridge_softmax_compare comparison | plain: archive with `ridge_softmax_cluster_experiment.py` (2026-08-24) |

The four `pool_source_*`/`conditional_coverage` rows were first flagged by the
2026-07-10 sweep (review 2026-07-24), but that sweep's branch
(`routine/repo-cleanup-2026-07-10`) never merged, so the notes never reached
main; the 2026-08-10 sweep re-flagged them with a fresh review window.

Never candidates: library modules imported by active code
(`conformal_prediction`, `split_cp_baselines`, `exchangeable_features`,
`autoencoder_utils`, `mscs_gpu`, `macs_experiment`, `mscs_unlabeled_experiment`,
`semicp_experiment`), pipeline infra (`extract_features`, `download_datasets`,
`run_conformal_experiment`), and anything with uncommitted changes.

## Watch-list candidates archived 2026-06-24

Review date (2026-06-24) reached with no intervening use or new finding;
moved here from the watch list above.

| Script | What it was | Superseded by / outcome |
|---|---|---|
| `cs_comparison.py` | MA-CS vs MS-CS on CIFAR-100 (legacy non-exchangeable engine) | `mscs_vs_macs_experiment.py` on the exchangeable engine (findings sec 4) |
| `unlabeled_size_sweep.py` | FCP / +PCA / +MS-CS vs unlabeled-pool size N | pool-size question settled (findings sec 3; plateau N=2500-5000) |
| `linearity_diagnostics.py` | SVD spectrum + AE-vs-PCA reconstruction diagnostics | AE-vs-PCA linearity question settled (findings sec 3) |
| `mscs_exchangeability_experiment.py` | MS-CS exact vs O(1/n) frozen penalty path | exact ~= frozen settled (findings sec 4b) |

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
