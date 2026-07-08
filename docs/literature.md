# Literature Tracker — SSL + Conformal Prediction Research

Legend: `[ ]` unread · `[~]` in progress · `[x]` read · `*` closest setting

---

## 1. Core CP Theory

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Shafer & Vovk** — "A Tutorial on Conformal Prediction" | JMLR | 2008 | [arXiv:0706.3188](https://arxiv.org/abs/0706.3188) | Full CP, ICP, k-NN NCM. Theory base. |
| [ ] | **Barber, Candes, Tibshirani & Wager** — "Predictive Inference with the Jackknife+" | Ann. Statistics | 2021 | [arXiv:1905.02928](https://arxiv.org/abs/1905.02928) | CV+ and 1-2a guarantee. |
| [ ] | **Angelopoulos & Bates** — "A Gentle Introduction to Conformal Prediction" | FnTML | 2022 | [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) | Best tutorial. Read before writing methods. |

---

## 2. Directly Adjacent Work — Closest Settings

The Silva-Rodriguez et al. trilogy is the nearest neighbour to our work: transductive, split-free CP on frozen foundation-model embeddings, few-shot calibration, optimizing set size. They win via **feature/probe adaptation** (info-max, SS-Text, OT) to close the pretrain->task gap; we keep the backbone frozen and win via **SSL-geometry NCM + unlabeled-pool PCA + a class-similarity penalty (MS-CS, adapted from Fargion et al. to transductive FCP)**. They are VLM/zero-shot (often medical); we are pure SSL (DINOv2) on many-class natural images.

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [~] | * **Silva-Rodriguez, Fillioux et al.** — "Full Conformal Adaptation of Medical VLMs" (FCA) | IPMI | 2025 | [arXiv:2506.06076](https://arxiv.org/abs/2506.06076) | **Closest full-CP precedent.** Transductive **full** CP on frozen VLM features, few-shot, training-free SS-Text linear-probe solver. **+27% set efficiency** at matched coverage. **READ FIRST.** |
| [~] | * **Silva-Rodriguez, Ben Ayed, Dolz** — "Conformal Prediction for Zero-Shot Models" (Conf-OT) | CVPR | 2025 | [arXiv:2505.24693](https://arxiv.org/abs/2505.24693) | **Closest natural-image sibling.** Transductive split-free CP on CLIP; OT bridges domain gap. 15 datasets, 3 NCM scores, **+20% efficiency, 15x faster** than transductive baselines. **READ FIRST.** |
| [ ] | **Silva-Rodriguez, Ben Ayed, Dolz** — "Trustworthy Few-Shot Transfer of Medical VLMs via Split CP" (SCA-T) | MICCAI | 2025 | [arXiv:2506.17503](https://arxiv.org/abs/2506.17503) | Transductive **split** conformal adaptation; info-max joint cal+test + label-marginal reg. Code: [jusiro/SCA-T](https://github.com/jusiro/SCA-T). Same scarce-cal/transductive/set-size spirit; split-CP + feature adaptation, medical. |
| [~] | **Fillioux, Cherian et al.** — "Are Foundation Models Good Conformal Predictors?" | arXiv | 2024 | [arXiv:2412.06082](https://arxiv.org/abs/2412.06082) | Survey-style closest work. 17 FMs + Split CP only. We differ: Full CP, geodesic NCMs, few-shot regime. |
| [ ] | **Fisch et al.** — "Few-Shot Conformal Prediction with Auxiliary Tasks" | ICML | 2021 | [arXiv:2102.08898](https://arxiv.org/abs/2102.08898) | Seminal few-shot CP (meta-learning). We extend with SSL embeddings instead. |
| [ ] | **Angelopoulos et al.** — "Uncertainty Sets for Image Classifiers" (RAPS) | ICLR | 2021 | [arXiv:2009.14193](https://arxiv.org/abs/2009.14193) | RAPS baseline. **Must implement for paper.** |
| [x] | **Fargion, Dabah & Tirer** — "Enhancing CP via Class Similarity" (MA-CS / MS-CS) | ICML | 2026 | [arXiv:2511.19359](https://arxiv.org/abs/2511.19359) | **Read 2026-06-30.** TWO variants: **MA** (binary superclass penalty, §4) + **MS** (continuous `1−M`, `M`=cosine of **centered** class means `cos(h_c−h_G, h_c'−h_G)`, §5; no taxonomy). Split CP, `M` on disjoint training data → no per-candidate update. Our `macs_experiment.py` = faithful MA; our k-means `mscs_unlabeled` is a DIFFERENT M (not their MS). Faithful MS port (`centered_cosine`) planned; theory §3.1–3.2.1. |
| [ ] | **Fan & Sesia** — "Transductive Standardization in CP" | arXiv | 2025 | [arXiv:2512.15383](https://arxiv.org/abs/2512.15383) | Supports our O(1/n) exchangeability argument for whitened NCMs. |

---

## 3. Set-Size / Efficiency Optimization (our objective)

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Wang, Sun & Dobriban** — "Singleton-Optimized Conformal Prediction" (SOCOP) | arXiv | 2026 | [arXiv:2509.24095](https://arxiv.org/abs/2509.24095) | NCM that minimizes P(non-singleton set), O(K). Image classification. Alternative route to our set-size goal — NCM-design baseline. |
| [ ] | **(anon.)** — "Direct Prediction Set Minimization via Bilevel Conformal Training" (DPSM) | arXiv | 2025 | [arXiv:2506.06599](https://arxiv.org/abs/2506.06599) | Bilevel conformal training, ~20% set-size cut. Requires *training* the classifier — contrast for our frozen-backbone constraint. |
| [ ] | **Liu et al.** — "C-Adapter: Adapting Deep Classifiers for Efficient CP Sets" | arXiv | 2024 | [arXiv:2410.09408](https://arxiv.org/abs/2410.09408) | Adapter tuning for efficient CP sets (trainable head). Landscape reference. |

---

## 4. NCM Design & Geometry — Neighborhood / Distance NCMs

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Ghosh et al.** — "Neighborhood Conformal Prediction" (NCP) | AAAI | 2023 | [arXiv:2303.10694](https://arxiv.org/abs/2303.10694) | k-NN in learned representation + distance-weighted adaptive sets. **Prior art for our geodesic NN-ratio NCM — cite.** |
| [ ] | **(DANCE)** — "Doubly Adaptive Neighborhood Conformal Estimation" | arXiv | 2026 | [arXiv:2602.20652](https://arxiv.org/abs/2602.20652) | Latest neighborhood-adaptive CP. Watch for NCM positioning. |
| [ ] | **Katsios et al.** — "Multi-label CP with a Mahalanobis NCM" | COPA (PMLR v230) | 2024 | [proceedings](https://proceedings.mlr.press/v230/katsios24a.html) | Mahalanobis-distance NCM — theory/contrast for `MahalNNRatio` + whitening. |
| [ ] | **Bateni et al.** — "Improved Few-Shot Visual Classification" | CVPR | 2020 | — | Pooled Mahalanobis distance. Theory for `MahalNNRatio`. |
| [x] | **Wang & Isola** — "Alignment and Uniformity on the Hypersphere" | ICML | 2020 | [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) | SSL = uniform on S^{d-1}. Justifies geodesic metric. |

---

## 5. Few-Shot / Scarce-Cal CP

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **CAOS** — "Conformal Aggregation of One-Shot Predictors" | arXiv | 2026 | [arXiv:2601.05219](https://arxiv.org/abs/2601.05219) | One-shot FM adaptation, **leave-one-out calibration**, smaller sets than split-CP. Extreme of our "don't waste B on a train/cal split" thesis. |
| [ ] | **Park et al.** — "Few-Shot Calibration via Meta-Learned CV-Based CP" (meta-XB) | IEEE TSP | 2023 | [arXiv:2210.03067](https://arxiv.org/abs/2210.03067) | Few-shot CP via meta-learned CV+. Same motivation (split-CP bloats sets at low data), different machinery. |
| [ ] | **Seedat et al.** — "Improving Adaptive CP Using Self-Supervised Learning" (SSCP) | AISTATS | 2023 | [arXiv:2302.12238](https://arxiv.org/abs/2302.12238) | SSL pretext tasks improve NCM. Designed for regression — needs classification adaptation. |

---

## 6. Adaptive & Class-Conditional CP

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Romano, Sesia & Candes** — "Valid and Adaptive Coverage" (APS) | NeurIPS | 2020 | [arXiv:2006.02544](https://arxiv.org/abs/2006.02544) | Main competitor score. |
| [ ] | **Ding, Tibshirani & Ramdas** — "Class-Conditional CP with Many Classes" (ClusterCP) | NeurIPS | 2023 | [arXiv:2306.09335](https://arxiv.org/abs/2306.09335) | CovGap metric + ClusterCP baseline. Implemented; ClusterCP degenerates to SplitCP at our cal/K. |
| [ ] | **(anon.)** — "Fundamental Bounds on Efficiency-Confidence Trade-off for Transductive CP" | arXiv | 2025 | [arXiv:2509.04631](https://arxiv.org/abs/2509.04631) | Finite-sample set-size vs confidence bounds for transductive CP. Theory for our full-CP efficiency story (and its limits). |

---

## 7. SSL Backbones

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Oquab et al.** — "DINOv2" | TMLR | 2024 | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | Primary backbone. |
| [ ] | **Caron et al.** — "DINO" | ICCV | 2021 | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | Predecessor. k-NN on frozen ViT motivates our NCMs. |

---

## 8. Semi-Supervised CP (Unlabeled Data)

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [x] | **Zhou et al.** — "SemiCP" (NNM score) | CVPR | 2026 | [arXiv:2505.21147](https://arxiv.org/abs/2505.21147) | Pool augmentation for Split CP. We tested for FCP — negative result. |
| [x] | **Einbinder et al.** — "PPI for Risk Control" | arXiv | 2024 | [arXiv:2412.11174](https://arxiv.org/abs/2412.11174) | PPI-based lambda tuning. **Deprioritized 2026-05-18** — archived to `src/archive/ppi_rcps.py`. |
| [x] | **Bozorgtabar et al.** — "LATA" (kNN graph smoothing) | arXiv | 2026 | [arXiv:2602.17535](https://arxiv.org/abs/2602.17535) | Graph smoothing on SSL embeddings. Tested — naive approach fails for FCP NCM scores. |

---

## 9. Pool-Fit Unsupervised Projections — the "fit stage on unlabeled data" lineage (sweep 2026-07-06)

Citation architecture for rationalizing the transform stage (PCA-d'/whitening fit on
the unlabeled pool). Full annotated sweep in memory `pca-rationalization-audit`;
verified via web search 2026-07-06.

**Framing ("unlabeled data defines the geometry, labels only calibrate"):**
- Belkin & Niyogi, NIPS 2002 (partially-labeled manifold classification) — basis fit on
  unlabeled points, labels fit only expansion coefficients; structurally our pipeline.
- Castelli & Cover 1995/96 — once the mixture geometry is known from unlabeled data,
  labels only "name components" (exponential value). Our cal set fits ONE threshold.
- Belkin/Niyogi/Sindhwani JMLR 2006 (manifold regularization); Cai/He/Han ICCV 2007
  (SDA — unlabeled data stabilizes the projection LDA overfits, mirrors our LDA kill);
  Zhang/Zhou/Chen SDM 2007 (SSDR).

**Method ancestry (few-shot frozen-feature transforms fit on auxiliary data):**
- SimpleShot (Wang et al. 1911.04623) — center-by-base-mean + L2-norm before NN;
  the canonical "the transform, not the classifier, is the win".
- **TAFSSL (Lichtenstein et al., ECCV 2020, 2003.06670) — PCA/ICA fit on task-adjacent
  UNLABELED data before prototype classification; the single closest few-shot match.**
- PT-MAP (Hu et al. 2006.03806), Distribution Calibration (Yang et al., ICLR 2021) —
  label-free reshaping / borrowed second-order statistics (= our LW-whitening logic).
- Mensink et al. TPAMI 2013 (NCM + transferable Mahalanobis metric); Tian et al. ECCV
  2020 ("good embedding is all you need"); Chen et al. 2010.02430 (no base labels).

**Theory (when unlabeled fitting provably helps — and when not):**
- Positive: Saunshi/Arora ICML 2019 (contrastive theory guarantees the MEAN classifier
  — exactly our prototype rules); HaoChen et al. NeurIPS 2021 (SSL features ~ spectral
  embedding of the augmentation graph -> our PCA = second spectral truncation); Lee et
  al. NeurIPS 2021 (reconstruction pretexts).
- Negative/conditional: Ben-David/Lu/Pal COLT 2008; Singh/Nowak/Zhu NeurIPS 2008
  ("now it helps, now it doesn't") — maps 1:1 onto our cifar-vs-aircraft inversion.
  Paper framing: **validity assumption-free, efficiency assumption-dependent.**

**Pool-only d'-selection (closes the "why 128?" hole):**
- Gavish & Donoho IEEE-TIT 2014 (optimal hard threshold; sigma-unknown form = 2.858 x
  median singular value — pure pool statistic); Marchenko-Pastur eigenvalue clipping;
  Dobriban & Owen JRSS-B 2019 (deflated parallel analysis) + Hong et al. 2012.02985
  (signflip PA) — assumption-light. KEY: G-D/MP assume a spiked spectrum + noise bulk
  — holds on cifar/mini, PROVABLY inapplicable on aircraft's collapsed fat-tail
  spectrum -> the three-regime map becomes "truncate where the spiked model holds;
  where it fails, truncation is undefined and full-rank decorrelation (Ledoit-Wolf
  2004) is the remaining lever".
- Participation ratio formalized: Litwin-Kumar et al. Neuron 2017; Gao et al. 2017
  (bioRxiv). Finite-sample bias caveat for small pools: arXiv:2509.26560.
- Intrinsic dim (Levina-Bickel 2004; TwoNN, Facco 2017) with the Ansuini et al.
  NeurIPS 2019 caveat (deep reps low-ID but CURVED -> local ID != linear rank).

**Decorrelation as first-class operation (supports the aircraft/LW finding):**
- W-MSE (Ermolov ICML 2021, hard whitening), Barlow Twins (2103.03230), VICReg
  (2105.04906); Hua et al. ICCV 2021 (decorrelation prevents dimensional collapse);
  Kalapos & Gyires-Toth 2408.07519 (ZCA on frozen SSL features improves kNN 1-5%).

**CP-side recent (2024-26), beyond the trilogy/SemiCP/SSCP:**
- Correia & Louizos, NeurIPS 2025 (2507.10425) — unlabeled data for shift-robust
  thresholds (they spend unlabeled on threshold robustness; we spend it on score
  efficiency). [Resolves the pending §8 fold from the 2026-06 archive note.]
- Flechsig & Pilz 2509.10321 — CP with zero labeled calibration (1-alpha-beta
  guarantee); brackets the label-budget axis.
- CONFIDE 2604.08885 — kNN scores on LM representations with PCA/Mahalanobis/metric
  GRID SEARCH; closest concurrent analogue; our pool-spectrum selection is the delta.
- Fadnavis 2602.12693 (leverage-weighted CP) — notes full-rank whitening ~ identity
  for Mahalanobis scores while TRUNCATION changes geometry (independent corroboration
  of our probe attribution). Also: COMPASS 2509.22240, Contrastive Conformal Sets
  2603.26261, Katsios COPA 2024 (Mahalanobis NCM).

**NEGATIVE FINDING (novelty slot):** no published work fits a projection/metric on an
INDEPENDENT unlabeled pool and runs full/transductive CP with distance/prototype NCMs
on the projected features. Nearest relatives are cal+test transductive adapters
(SCA-T, Conf-OT) and score-level unlabeled uses (SemiCP, Flechsig, Correia). The
pool-fit-transform-for-FCP slot appears open.

---

## Reading Priority

**Read first — closest setting (FM-embedding + transductive + set size):**
1. **FCA** (Silva-Rodriguez 2506.06076) — closest full-CP precedent; positioning anchor
2. **Conf-OT** (Silva-Rodriguez 2505.24693) — closest natural-image sibling; CVPR'25
3. **SCA-T** (Silva-Rodriguez 2506.17503) — split-CP variant of the same idea
4. **Fillioux/Cherian** (2412.06082) — FM + Split CP survey; positioning

**Then — method building blocks:**
5. **RAPS** (Angelopoulos 2009.14193) — must implement as SCP baseline
6. **Neighborhood CP** (Ghosh 2303.10694) — prior art for our NN-ratio NCM
7. **SOCOP** (2509.24095) — set-size NCM baseline
8. **Fargion** (2511.19359) — MA-CS theory we validate
9. **Fisch** (2102.08898) / **CAOS** (2601.05219) — few-shot framing
10. **Angelopoulos & Bates** (2107.07511) — tutorial for methods section

---

## Key Differentiators vs Closest Work

**vs Silva-Rodriguez trilogy (FCA / Conf-OT / SCA-T):**
1. **Frozen backbone, no feature adaptation** — they adapt features/probe (info-max, SS-Text, OT); we change *only* the NCM + an unsupervised PCA, keeping DINOv2 fully frozen.
2. **SSL geometry, not VLM zero-shot** — geodesic NN-ratio NCM for hyperspherical DINOv2 embeddings; they use a text-derived linear classifier.
3. **Class-similarity penalty (MS-CS)** — none of the three use a similarity penalty. (We adapt Fargion et al.'s MS-CS to transductive Full CP; our k-means variant is additionally label-free. NB: Fargion's MS is *also* continuous + taxonomy-free — see §2 — so the differentiator vs *them* is the transductive/Full-CP exact-exchangeable port + geodesic NCMs, not "continuous, label-free".)
4. **Many-class natural images at small cal/K** — CIFAR-100/miniImageNet/CUB-200 (K>=100); they target medical (FCA/SCA-T) or generic VLM transfer.

**vs Fillioux/Cherian (2024):** Full CP not Split CP; geodesic NCMs for SSL geometry; few-shot regime (k=3-5/class) they don't test; backbone-geometry analysis (DINOv2 > CLIP for NN-ratio CP).

---

*Last updated: 2026-07-06 (added §9 pool-fit unsupervised projection lineage).*
