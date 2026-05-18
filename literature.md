# Literature Tracker — SSL + Conformal Prediction Research

Legend: `[ ]` unread · `[~]` in progress · `[x]` read

---

## 1. Core CP Theory

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Shafer & Vovk** — "A Tutorial on Conformal Prediction" | JMLR | 2008 | [arXiv:0706.3188](https://arxiv.org/abs/0706.3188) | Full CP, ICP, k-NN NCM. Theory base. |
| [ ] | **Barber, Candes, Tibshirani & Wager** — "Predictive Inference with the Jackknife+" | Ann. Statistics | 2021 | [arXiv:1905.02928](https://arxiv.org/abs/1905.02928) | CV+ and 1-2a guarantee. |
| [ ] | **Angelopoulos & Bates** — "A Gentle Introduction to Conformal Prediction" | FnTML | 2022 | [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) | Best tutorial. Read before writing methods. |

---

## 2. Directly Adjacent Work

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [~] | **Fillioux, Cherian et al.** — "Are Foundation Models Good Conformal Predictors?" | arXiv | 2024 | [arXiv:2412.06082](https://arxiv.org/abs/2412.06082) | **Closest work.** 17 FMs + Split CP only. We differ: Full CP, geodesic NCMs, few-shot regime. |
| [ ] | **Fisch et al.** — "Few-Shot Conformal Prediction with Auxiliary Tasks" | ICML | 2021 | [arXiv:2102.08898](https://arxiv.org/abs/2102.08898) | Seminal few-shot CP. We extend with SSL embeddings instead of meta-learning. |
| [ ] | **Angelopoulos et al.** — "Uncertainty Sets for Image Classifiers" (RAPS) | ICLR | 2021 | [arXiv:2009.14193](https://arxiv.org/abs/2009.14193) | RAPS baseline. **Must implement for paper.** |
| [~] | **Fargion, Dabah & Tirer** — "Enhancing CP via Class Similarity" (MA-CS) | arXiv | 2025 | [arXiv:2511.19359](https://arxiv.org/abs/2511.19359) | MA-CS penalty theory. We validate in FCP + geodesic NCMs. |
| [ ] | **Fan & Sesia** — "Transductive Standardization in CP" | arXiv | 2025 | [arXiv:2512.15383](https://arxiv.org/abs/2512.15383) | Supports our O(1/n) exchangeability argument for whitened NCMs. |

---

## 3. NCM Design & Geometry

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Bateni et al.** — "Improved Few-Shot Visual Classification" | CVPR | 2020 | — | Pooled Mahalanobis distance. Theory for `MahalNNRatio`. |
| [x] | **Wang & Isola** — "Alignment and Uniformity on the Hypersphere" | ICML | 2020 | [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) | SSL = uniform on S^{d-1}. Justifies geodesic metric. |

---

## 4. Adaptive & Class-Conditional CP

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Romano, Sesia & Candes** — "Valid and Adaptive Coverage" (APS) | NeurIPS | 2020 | [arXiv:2006.02544](https://arxiv.org/abs/2006.02544) | Main competitor score. |
| [ ] | **Ding, Tibshirani & Ramdas** — "Class-Conditional CP with Many Classes" | NeurIPS | 2023 | [arXiv:2306.09335](https://arxiv.org/abs/2306.09335) | Related to MA-CS. Groups classes for finite-sample behavior. |

---

## 5. SSL Backbones

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [ ] | **Oquab et al.** — "DINOv2" | TMLR | 2024 | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | Primary backbone. |
| [ ] | **Caron et al.** — "DINO" | ICCV | 2021 | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | Predecessor. k-NN on frozen ViT motivates our NCMs. |

---

## 6. Semi-Supervised CP (Unlabeled Data)

| Status | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|
| [x] | **Zhou et al.** — "SemiCP" (NNM score) | CVPR | 2026 | [arXiv:2505.21147](https://arxiv.org/abs/2505.21147) | Pool augmentation for Split CP. We tested for FCP — negative result. |
| [x] | **Einbinder et al.** — "PPI for Risk Control" | arXiv | 2024 | [arXiv:2412.11174](https://arxiv.org/abs/2412.11174) | PPI-based lambda tuning. **Deprioritized 2026-05-18** — implementation archived to `src/archive/ppi_rcps.py`. Split-CP-oriented; orthogonal to our FCP pipeline. |
| [x] | **Bozorgtabar et al.** — "LATA" (kNN graph smoothing) | arXiv | 2026 | [arXiv:2602.17535](https://arxiv.org/abs/2602.17535) | Graph smoothing on SSL embeddings. Tested — naive approach fails for FCP NCM scores. |

---

## Reading Priority

1. **RAPS** (Angelopoulos 2009.14193) — must implement as SCP baseline
2. **Fillioux/Cherian** (2412.06082) — positioning vs closest work
3. **Fargion** (2511.19359) — MA-CS theory we validate
4. ~~**Einbinder** (2412.11174)~~ — Deprioritized 2026-05-18 (see §6 note)
5. **Fisch** (2102.08898) — few-shot framing
6. **Wang & Isola** (2005.10242) — geodesic justification (review for paper)
7. **Angelopoulos & Bates** (2107.07511) — tutorial for methods section

---

## Key Differentiators vs Fillioux/Cherian (2024)

1. **Full CP, not Split CP** — they only test inductive methods; we show FCP dominates at small cal
2. **NCM design for SSL geometry** — geodesic NCMs for hyperspherical DINOv2 embeddings
3. **Few-shot regime** — they don't test below ~1000 cal points; we show FCP works at k=3-5/class
4. **Backbone geometry analysis** — why DINOv2 > CLIP for NN-ratio CP

---

*Last updated: 2026-05-12*
