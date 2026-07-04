# Literature Tracker — SSL + Conformal Prediction Research

Legend: `[ ]` unread · `[~]` in progress · `[x]` read · `*` closest setting

---

## 1. Core CP Theory

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Shafer & Vovk** — "A Tutorial on Conformal Prediction" | JMLR | 2008 | [arXiv:0706.3188](https://arxiv.org/abs/0706.3188) | Full CP, ICP, k-NN NCM. Theory base. |
| [ ] | **Barber, Candes, Tibshirani & Wager** — "Predictive Inference with the Jackknife+" | Ann. Statistics | 2021 | [arXiv:1905.02928](https://arxiv.org/abs/1905.02928) | CV+ and 1-2a guarantee. |
| [ ] | **Angelopoulos & Bates** — "A Gentle Introduction to Conformal Prediction" | FnTML | 2022 | [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) | Best tutorial. Read before writing methods. |
| [ ] | **Vovk** — "Cross-Conformal Predictors" | Ann. Math. & AI | 2015 | [arXiv:1208.0806](https://arxiv.org/abs/1208.0806) | Foundational **data-reuse CP** (split-CP + CV): reuse all labels for fit + calibration, no train/cal split. Basis of our "don't split" thesis and CAOS's LOO scheme. |

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
| [ ] | **Liu et al.** — "C-Adapter: Adapting Deep Classifiers for Efficient CP Sets" | arXiv | 2024 | [arXiv:2410.09408](https://arxiv.org/abs/2410.09408) | Adapter tuning for efficient CP sets (trainable head). Landscape reference. **NOTE: withdrawn by authors ("experimental results not sufficient") — do not cite as a result.** |
| [ ] | **Conformal aggregation / selection family** (Rivera; Alami; Hegazy; Vovk) | NeurIPS'25 / AAAI'26 / arXiv | 2024–26 | — | Score-level aggregation + valid selection from a POOL of CP predictors for tighter sets: Rivera [2405.16246](https://arxiv.org/abs/2405.16246) (NeurIPS'25), Alami SACP [2512.06945](https://arxiv.org/abs/2512.06945) (AAAI'26), Hegazy [2506.20173](https://arxiv.org/abs/2506.20173), Vovk e-class [2605.07963](https://arxiv.org/abs/2605.07963). **CAOS's family** — tangential to our single-NCM full CP; landscape, not a baseline we run. |

---

## 4. NCM Design & Geometry — Neighborhood / Distance NCMs

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Ghosh et al.** — "Neighborhood Conformal Prediction" (NCP) | AAAI | 2023 | [arXiv:2303.10694](https://arxiv.org/abs/2303.10694) | k-NN in learned representation + distance-weighted adaptive sets. **Prior art for our geodesic NN-ratio NCM — cite.** |
| [ ] | * **Marazov, Balieva, Tanchev, Lazarova & Rankova** — "Conformal Predictions for Visual Animal Identification" | Technologies (MDPI) 14(4):232 | 2026 | [DOI:10.3390/technologies14040232](https://doi.org/10.3390/technologies14040232) | **Nearest non-VLM published sibling to our NN-ratio NCM.** Split CP on **frozen ResNet-50 / Swin-T** embeddings; **cosine-distance** nonconformity, 3 scores (NN-**ratio** / min / mean); cow + goat re-ID. **Ratio score cuts mean set size up to 79% vs min/mean** — external validation of ratio NCMs; Swin-T beats ResNet-50 by up to 14pp singleton rate. **Identity-level split** (disjoint IDs, open-set re-ID; gallery/query): cow cal/test = 55/55 IDs (229/183 gallery refs + 55/55 queries), goat = 36/36 IDs (113/118 refs + 36/36 queries) — conformal n = #queries ≈ 36–55 (tiny). We differ: **Full** CP, **arccos geodesic** ratio, PCA + MS-CS, many-class natural images at few-shot cal. |
| [ ] | **(DANCE)** — "Doubly Adaptive Neighborhood Conformal Estimation" | arXiv | 2026 | [arXiv:2602.20652](https://arxiv.org/abs/2602.20652) | Latest neighborhood-adaptive CP. Watch for NCM positioning. |
| [ ] | **Katsios et al.** — "Multi-label CP with a Mahalanobis NCM" | COPA (PMLR v230) | 2024 | [proceedings](https://proceedings.mlr.press/v230/katsios24a.html) | Mahalanobis-distance NCM — theory/contrast for `MahalNNRatio` + whitening. |
| [ ] | **Bateni et al.** — "Improved Few-Shot Visual Classification" | CVPR | 2020 | — | Pooled Mahalanobis distance. Theory for `MahalNNRatio`. |
| [x] | **Wang & Isola** — "Alignment and Uniformity on the Hypersphere" | ICML | 2020 | [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) | SSL = uniform on S^{d-1}. Justifies geodesic metric. |

---

## 5. Few-Shot / Scarce-Cal CP

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [x] | **Waldron** — "CAOS: Conformal Aggregation of One-Shot Predictors" | arXiv | 2026 | [arXiv:2601.05219](https://arxiv.org/abs/2601.05219) | **Reviewed 06-30 (goal 4).** One-shot adaptation on **frozen DINOv3**: 1-shot score `1 - cos(z_test, z_ref)`, aggregate = **sum of k=3 smallest**, calibrate **leave-one-out over all n (no split)**. Valid by **monotonicity** (C_full contained in C_caos; full CP only a proof device -> conservative / over-covers). **App. E: of the ~55% set-size win over split-CP, ~52pp is NOT-splitting, ~7pp the aggregation** — quantified support for our no-split thesis. Equivalent to our `update_calibration_scores=False` static-LOO path with a sum-of-k-min cosine score; one-shot **patch**-classification, not many-class natural images. |
| [ ] | **Gasparin & Ramdas** — "Improving the Statistical Efficiency of Cross-Conformal Prediction" | ICML | 2025 | [arXiv:2503.01495](https://arxiv.org/abs/2503.01495) | Efficient p-value combination for CV+ / cross-conformal -> smaller sets, same guarantees. The split-free data-reuse efficiency machinery behind CAOS's "reuse all labels". |
| [ ] | **Bashari et al.** — "Synthetic-Powered Predictive Inference" | arXiv | 2025 | [arXiv:2505.13432](https://arxiv.org/abs/2505.13432) | Score-transporter aligns real/synthetic scores -> tighter sets when calibration is scarce; demoed on image classification. Scarce-cal efficiency; contrast to our unlabeled-pool route. |
| [ ] | **Bhattacharyya, Zhang & Barber** — "Approximating Full CP via the Tournament Correction" | arXiv | 2026 | [arXiv:2605.29200](https://arxiv.org/abs/2605.29200) | Cheap full-CP approximation generalizing LOO cross-conformal; distribution-free 1-2α, tightens under stability. Full-CP efficiency reference. |
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

*Last updated: 2026-06-30.*
