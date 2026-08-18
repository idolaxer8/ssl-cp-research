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

*(empty — the entire 2026-08-10 list was archived early on 2026-08-18 as
ICLR-2027 code-reduction round 1; see that section below.)*

Never candidates: library modules imported by active code
(`conformal_prediction`, `split_cp_baselines`, `exchangeable_features`,
`autoencoder_utils`, `mscs_gpu`, `macs_experiment`, `mscs_unlabeled_experiment`),
pipeline infra (`extract_features`, `download_datasets`,
`run_conformal_experiment`), and anything with uncommitted changes.

## ICLR-2027 code-reduction round 1 (archived 2026-08-18)

Whole watch list archived early (user-directed reduction toward the compact
publishable version, `docs/iclr2027_plan.md` §3b). The four HOLD conditions
were retired: their findings-§10 items (2026-05 plan) are superseded by the
ICLR-2027 plan — P0.3/P2.11 by the champion/backbone tables
(`backbone_dwt_experiment.py`), P1.5/P0.4 by the closed RBF/AE lines.

| Script | What it was | Superseded by / outcome |
|---|---|---|
| `ncm_comparison_reduced.py` | NCM comparison across reductions | champion tables (`fca_family_cluster_experiment.py`, `transform_control_experiment.py`) |
| `cs_ablation.py` | reduction x MS-CS ablation (legacy engine) | `mscs_vs_macs_experiment.py` + exchangeable engine; `cluster/run_small_cal_sweep.sh` path updated |
| `compare_backbones.py` | old SSL backbone comparison (findings §7) | `backbone_dwt_experiment.py` (goal-5 backbone x DWT table) |
| `rbf_ncm_experiment.py` | RBF density NCM iteration | RBF line closed; not in the paper |
| `conditional_coverage_experiment.py` | class-conditional CovGap + ClusterCP baseline (findings §2b) | results recorded; CovGap now computed inside the active experiment drivers |
| `pool_source_comparison.py` / `pool_source_limits.py` / `plot_pool_source.py` | pool-source ablations (findings §4d) | question settled |
| `pool_ablation_hightrial.py` / `plot_unlabeled_pool_ablation.py` | high-trial pool ablation (findings §4e) | question settled |
| `ridge_softmax_cluster_experiment.py` / `plot_ridge_softmax_compare.py` | ridge-softmax NCM benchmark | `fca_family_cluster_experiment.py` (ridge excluded by default) |
| `ica_negent_pilot.py` | TAFSSL negentropy dim selection | KILLED 2026-07-08 (few-way tool; see THEORY.MD Open Questions) |
| `ae_lowpr_pilot.py` | AE on low-PR (aircraft) | KILLED 2026-07-06 (nonlinearity not the missing ingredient) |
| `transform_selection_pilot.py` / `plot_framework_summary.py` | label-free transform selector pilot | C4 auto-selection DESCOPED 2026-08-18 (future work); provenance on `worktree-transform-selection` |
| `semicp_experiment.py` | SemiCP vs FCP comparison driver | `g3_semisup_experiment.py` (scaled reviewer-baseline arms; `SemiCP` class stays in `split_cp_baselines.py`) |

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
