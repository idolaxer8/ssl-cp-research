# Literature Tracker — SSL + Conformal Prediction Research

Legend: `[ ]` unread · `[~]` in progress · `[x]` read · ⭐ priority

---

## 1. Core Conformal Prediction Theory

| Status | Priority | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|---|
| [ ] | ⭐⭐⭐ | **Shafer & Vovk** — "A Tutorial on Conformal Prediction" | JMLR | 2008 | [arXiv:0706.3188](https://arxiv.org/abs/0706.3188) | Foundational: Full CP, ICP, k-NN NCM. Direct theory base for entire project. |
| [ ] | ⭐⭐ | **Vovk, Gammerman & Shafer** — "Algorithmic Learning in a Random World" | Springer (book) | 2005/2022 | [alrw.net](https://www.alrw.net/) | The book. Full CP theory, exchangeability, validity proofs. |
| [ ] | ⭐⭐⭐ | **Barber, Candès, Tibshirani & Wager** — "Predictive Inference with the Jackknife+" | Ann. Statistics | 2021 | [arXiv:1905.02928](https://arxiv.org/abs/1905.02928) | Defines CV+ and 1−2α guarantee. Direct theory for `CrossValidationPlusPredictor`. |
| [ ] | ⭐⭐ | **Papadopoulos et al.** — "Inductive Confidence Machines for Regression" | ECML | 2002 | — | Original Split/Inductive CP (ICP) paper. Theory base for SplitCP baseline. |

---

## 2. Most Directly Adjacent to This Project

| Status | Priority | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|---|
| [ ] | ⭐⭐⭐ | **Cherian et al.** — "Are Foundation Models for Computer Vision Good Conformal Predictors?" | arXiv | 2024 | [arXiv:2412.06082](https://arxiv.org/abs/2412.06082) | **Closest existing work.** 17 vision foundation models (incl. DINOv2, CLIP) + TPS/APS/RAPS on CIFAR-10/100. Understand how our approach differs: Full CP, NCM geometry, few-shot regime. |
| [ ] | ⭐⭐⭐ | **Fisch, Schuster, Jaakkola & Barzilay** — "Few-Shot Conformal Prediction with Auxiliary Tasks" | ICML | 2021 | [arXiv:2102.08898](https://arxiv.org/abs/2102.08898) | Seminal few-shot CP. Our CIFAR-100 k=3–5 results extend this using SSL embeddings instead of meta-learning. |
| [ ] | ⭐⭐⭐ | **Angelopoulos, Bates, Jordan & Malik** — "Uncertainty Sets for Image Classifiers using Conformal Prediction" (RAPS) | ICLR | 2021 | [arXiv:2009.14193](https://arxiv.org/abs/2009.14193) | Introduces RAPS — penalizes unlikely classes, analogous to our class-similarity penalty. Must read to position CS contribution. |
| [ ] | ⭐⭐ | **Fisch et al.** — "Efficient Conformal Prediction via Cascaded Inference with Expanded Admission" | ICLR | 2021 | [arXiv:2007.03114](https://arxiv.org/abs/2007.03114) | CP efficiency via cascaded label pruning. Relevant for computational comparison of Full CP vs CV+. |

---

## 3. NCM Design & Feature Space Geometry

| Status | Priority | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|---|
| [ ] | ⭐⭐⭐ | **Bateni et al.** — "Improved Few-Shot Visual Classification" | CVPR | 2020 | [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bateni_Improved_Few-Shot_Visual_Classification_CVPR_2020_paper.pdf) | Pooled Mahalanobis distance on frozen features improves few-shot k-NN. Theory for `MahalNNRatio`. |
| [x] | ⭐⭐⭐ | **Wang & Isola** — "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere" | ICML | 2020 | [arXiv:2005.10242](https://arxiv.org/abs/2005.10242) | SSL representations are uniform on S^{d-1} → arccos/geodesic is the natural metric. Theory for `WhitenedGeodesicNNRatio`. |
| [ ] | ⭐⭐ | **Cherubin** — "Majority Vote Ensembles of Conformal Predictors" | ML (journal) | 2019 | [Springer](https://link.springer.com/article/10.1007/s10994-018-5752-y) | Simplified k-NN NCM (sum of k distances) → `SimplifiedKNNNonconformity`. Relevant for k>1 investigation. |
| [ ] | ⭐ | **Weinberger & Saul** — "Distance Metric Learning for Large Margin Nearest Neighbor Classification" (LMNN) | JMLR | 2009 | [JMLR](https://jmlr.org/papers/v10/weinberger09a.html) | Mahalanobis metric learning for k-NN. Background theory for whitened NCMs. |
| [ ] | ⭐ | **Goldberger et al.** — "Neighbourhood Components Analysis" (NCA) | NeurIPS | 2004 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf) | Differentiable metric learning for k-NN. Background theory for whitened NCMs. |

---

## 4. Adaptive & Class-Conditional CP

| Status | Priority | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|---|
| [ ] | ⭐⭐ | **Romano, Sesia & Candès** — "Classification with Valid and Adaptive Coverage" (APS) | NeurIPS | 2020 | [arXiv:2006.02544](https://arxiv.org/abs/2006.02544) | Introduces APS (adaptive prediction sets). Main competitor score to our NCM-based approach. |
| [ ] | ⭐⭐ | **Angelopoulos et al.** — "Conformal Risk Control" | ICLR | 2024 | [arXiv:2208.02814](https://arxiv.org/abs/2208.02814) | Generalizes CP to any monotone risk. Motivates class-similarity penalty as a risk objective. |
| [ ] | ⭐ | **Tibshirani et al.** — "Conformal Prediction Under Covariate Shift" | NeurIPS | 2019 | [arXiv:1904.06019](https://arxiv.org/abs/1904.06019) | Importance-weighted CP under distribution shift. Relevant for EuroSAT/CUB-200 domain shift analysis. |

---

## 5. SSL Backbone Models

| Status | Priority | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|---|
| [ ] | ⭐⭐⭐ | **Oquab et al.** — "DINOv2: Learning Robust Visual Features without Supervision" | TMLR | 2024 | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | The primary backbone used. Cite for embedding extraction. Shows frozen features near-SOTA on ImageNet with linear head. |
| [ ] | ⭐⭐ | **Caron et al.** — "Emerging Properties in Self-Supervised Vision Transformers" (DINO) | ICCV | 2021 | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | DINOv2 predecessor. Frozen ViT features achieve 78% top-1 ImageNet with plain k-NN → motivates k-NN NCMs. |
| [ ] | ⭐ | **Radford et al.** — "Learning Transferable Visual Models From Natural Language Supervision" (CLIP) | ICML | 2021 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | CLIP backbone used in SSL model comparisons. Zero-shot transfer via language-vision pretraining. |

---

## 6. Tutorials & Surveys (reference material)

| Status | Priority | Paper | Venue | Year | Link | Notes |
|---|---|---|---|---|---|---|
| [ ] | ⭐⭐⭐ | **Angelopoulos & Bates** — "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" | FnTML | 2022 | [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) | Best comprehensive tutorial. Read before writing methods section. Covers split CP, full CP, CV+, adaptive scores. |

---

## Reading Priority Order

1. **Cherian et al. 2412.06082** — understand the closest existing work first
2. **Fisch et al. 2102.08898** — frame the few-shot contribution
3. **Angelopoulos et al. 2009.14193 (RAPS)** — position the class-similarity penalty
4. **Barber et al. 1905.02928 (Jackknife+)** — solidify CV+ theory
5. **Shafer & Vovk 0706.3188** — Full CP theory foundations
6. **Wang & Isola 2005.10242** — geodesic NCM justification
7. **Bateni et al. CVPR 2020** — Mahalanobis NCM justification
8. **Angelopoulos & Bates 2107.07511** — write methods section

---

## My Notes

<!-- Add your personal notes, key insights, and connections to the project here -->

### Connections to the project

- **Cherian et al.** use Split CP only → our Full CP approach is underexplored in this context
- **Fisch et al.** require auxiliary tasks for few-shot CP → we show frozen SSL features alone suffice
- **RAPS** penalizes by softmax rank → our CS penalty uses embedding-space geometry instead
- **Wang & Isola** hyperspherical uniformity → justifies `whitened_geodesic` as the theoretically correct NCM for DINOv2

---

*Last updated: 2026-03-17*
