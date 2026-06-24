# Literature Update: SemiCP Citation Network & Semi-Supervised CP Landscape

**Date:** 2026-05-17  
**Scope:** Zhou et al. (2025/2026, arXiv:2505.21147) — backward + forward citation analysis, broader semi-supervised CP landscape, and relevance to FCP-on-SSL-embeddings research.

---

## 1. The Paper Under Review

**Zhou, X., Shi, Z., Zeng, H., Xia, X., Jing, B., & Wei, H. (2026). Semi-supervised conformal prediction with unlabeled nonconformity score. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)*. arXiv:2505.21147**

### Method Summary

SemiCP extends split conformal prediction to the semi-supervised setting. The core contribution is the **Nearest Neighbor Matching (NNM) score**: unlabeled samples receive estimated nonconformity scores by finding their closest pseudo-labeled counterpart in the calibration set, then substituting that labeled sample's score. The combined labeled + pseudo-labeled calibration pool defines the quantile threshold.

**Theoretical guarantee**: The average coverage gap (|empirical coverage − (1−α)|) decreases at rate *O*(1/√N), where *N* is the unlabeled pool size, converging to a residual error term tied to NNM approximation quality.

**Key limitation (noted by authors)**: The theoretical framework assumes i.i.d. data — a strictly stronger assumption than the exchangeability required by standard CP. Semi-supervised validity under plain exchangeability remains an open problem.

**Experimental setting**: CIFAR-10, CIFAR-100, ImageNet. As few as 20 labeled calibration samples; 4,000 unlabeled. Primary metric: CovGap. Score functions: THR, APS, RAPS on ResNet50 (+ ViT, MobileNet, etc.). Reported 77% reduction in CovGap with 4,000 unlabeled examples at 20 labeled.

---

## 2. Backward Citations — Foundations SemiCP Builds On

### 2a. Conformal Prediction Core

**Angelopoulos, A. N., & Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *Foundations and Trends in Machine Learning, 16*(4), 494–591. arXiv:2107.07511**

The primary CP tutorial SemiCP cites. Covers split CP, score functions, and the exchangeability condition. Central to framing why the i.i.d. requirement in SemiCP is an additional burden. *[Already in literature.md §1.]*

---

**Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer.**

Originating work for transductive (full) CP and the ICP/split variant. SemiCP positions itself as an extension of the split variant. *[Foundational; not in literature.md — low priority to add.]*

---

**Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association, 113*(523), 1094–1111.**

Split CP theory for regression; coverage proof template used in SemiCP's propositions.

---

**Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous set-valued classifiers with bounded error levels. *Journal of the American Statistical Association, 114*(525), 223–234.**

Source of the **THR score** (1 − p(ŷ)). SemiCP uses THR as one of three score functions. *[Implicit in our softmax NCM; no direct dependency.]*

---

### 2b. Score Functions

**Romano, Y., Sesia, M., & Candès, E. (2020). Classification with valid and adaptive coverage. *Advances in Neural Information Processing Systems, 33*, 3581–3591. arXiv:2006.02544**

Introduces **APS** (Adaptive Prediction Sets): cumulative sum of sorted class probabilities. SemiCP tests APS and is the main failure mode — APS produces bloated sets (sz = 64–100) on K ≥ 100 classes. *[Already in literature.md §4.]*

---

**Angelopoulos, A. N., Bates, S., Jordan, M. I., & Malik, J. (2021). Uncertainty sets for image classifiers using conformal prediction. *International Conference on Learning Representations (ICLR)*. arXiv:2009.14193**

Introduces **RAPS** (Regularized APS): adds rank penalty *λ·(rank − k_reg)*₊ to discourage large sets. SemiCP tests RAPS. Our empirical result (findings.md §11): RAPS also bloated (sz = 17–43) for K ≥ 100, even with NNM augmentation. *[Already in literature.md §2 — must implement as SCP baseline.]*

---

### 2c. Semi-Supervised / PPI Context

**Kim, J., et al. (2025). Semi-supervised calibration methods. [specific venue not resolved]**  
**Wen, Q., et al. (2025). Semi-supervised CP extension. [specific venue not resolved]**

SemiCP cites two 2025 contemporaneous papers as semi-supervised learning context but does not compare against them experimentally. Neither appears in a resolvable arXiv search as of 2026-05-17 — likely concurrent workshop papers or not yet indexed.

---

## 3. Forward Citations — Work Citing SemiCP (as of 2026-05-17)

Semantic Scholar indexes one confirmed forward citation:

**Hur, Y. H., Nath, A., & Allen, G. I. (2026). Inference for clustering: Conformal sets for cluster labels. arXiv:2604.03488**

Uses split conformal prediction for uncertainty quantification of cluster assignments in single-cell RNA-seq data. Cites SemiCP in passing as a semi-supervised CP reference. **Not relevant** to image classification or SSL embeddings.

**Assessment**: SemiCP's forward citation count is effectively zero in image or SSL domains as of May 2026. The paper is too recent (submitted May 2025, CVPR 2026 accepted March 2026) for follow-up work to have appeared. The lack of domain-specific uptake is consistent with our empirical finding that NNM augmentation is neutral.

---

## 4. Broader Semi-Supervised CP Landscape (2025–2026)

The following papers were not cited by SemiCP but form the current research frontier:

---

### 4a. Split Conformal with Unsupervised Calibration

**Mazuelas, S. (2025). Split conformal classification with unsupervised calibration. arXiv:2510.07185**

Proposes using fully unlabeled calibration samples combined with supervised training data to define split-CP prediction sets. Unlike SemiCP (which still requires labeled cal samples), this method targets the extreme case of *zero* labeled cal data. Coverage guarantees are weaker ("moderate degradation") vs. supervised calibration. Computationally less efficient than standard split CP.

**Relevance ★★☆**: Directly addresses our regime concern — what happens when cal → 0? But weaker guarantees and no result for K ≥ 100 classes make it unsuitable as our direct comparator. Worth monitoring.

---

### 4b. Non-Exchangeable CP with Optimal Transport

**Correia, A. H. C., & Louizos, C. (2025). Non-exchangeable conformal prediction with optimal transport: Tackling distribution shifts with unlabeled data. *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. arXiv:2507.10425**

Uses unlabeled *test* data + optimal transport distances to reweight calibration scores, providing coverage guarantees under distribution shift. Two upper bounds on coverage gap derived from OT distances between calibration and test nonconformity score distributions.

**Relevance ★☆☆**: Our pipeline assumes in-distribution test data (same domain as cal). Distribution shift framing is not our current focus. The OT reweighting idea is orthogonal to our NCM design. Not worth implementing now.

---

### 4c. PPI for Semi-Supervised Risk Control

**Einbinder, B.-S., Ringel, L., & Romano, Y. (2024). Semi-supervised risk control via prediction-powered inference. arXiv:2412.11174**

Extends the Risk-Controlling Prediction Sets (RCPS) framework using prediction-powered inference to incorporate unlabeled data for hyperparameter (λ) tuning. Reduces the over-conservatism caused by small labeled cal sets. Demonstrated on few-shot image classification and early time-series classification.

**Relevance ★☆☆ — DEPRIORITIZED 2026-05-18**: Implementation archived to `src/archive/ppi_rcps.py`. The framework is split-CP-oriented and targets RCPS λ-tuning rather than FCP nonconformity score design. Our empirical line (findings.md §11) already shows FCP+PCA dominating semi-supervised split-CP baselines without any λ-tuning machinery. Re-evaluate only if MS-CS λ tuning becomes a bottleneck on a multi-dataset run.

---

### 4d. Self-Supervised CP for Imaging Inverse Problems

**Everink, J. M., Tamo Amougou, B., & Pereyra, M. (2025). Self-supervised conformal prediction for uncertainty quantification in imaging problems. arXiv:2502.05127**

Uses Stein's Unbiased Risk Estimator (SURE) to self-calibrate CP without ground truth labels, targeting image denoising/deblurring (inverse problems). Not classification; not SSL embeddings.

**Relevance ✗**: Wrong task domain. Name overlap with SSL can mislead — this is "self-supervised" in the SURE sense, not contrastive/masked-prediction SSL.

---

### 4e. Extending PPI Through Conformal Prediction

**[Authors TBD]. (2025). Extending prediction-powered inference through conformal prediction. arXiv:2510.16166**

Generalizes PPI using CP as the inference engine, strengthening Einbinder et al.'s framework. If this subsumes arXiv:2412.11174, it may be the preferred citation for PPI-RCPS.

**Relevance ★★☆**: Monitor — if it improves on Einbinder et al. for the λ-tuning task, upgrade to primary reference.

---

## 5. Project Relevance Assessment

| Paper | Direction | Status | Relevance | Action |
|---|---|---|---|---|
| Zhou et al. 2025 — SemiCP | NNM score aug for Split CP | Tested, negative | ★★☆ | Cite as baseline we beat; note i.i.d. limitation |
| Einbinder et al. 2024 — PPI-RCPS | λ tuning for RCPS | **Deprioritized (2026-05-18); archived** | ★☆☆ | See `src/archive/ppi_rcps.py` |
| Mazuelas 2025 — Unsupervised Cal | Zero-label cal CP | Track | ★★☆ | Monitor; not implement now |
| Correia & Louizos 2025 — OT-CP | Distribution shift reweighting | Track | ★☆☆ | Out of scope (in-distribution) |
| Angelopoulos (RAPS) | Image classifier CP baseline | Must implement | ★★★ | Add to SCP comparison |
| Fan & Sesia 2025 — Transductive Std | Supports whitening theory | Track | ★★☆ | Cite for O(1/n) argument |

---

## 6. Key Synthesis: Why FCP Dominates SemiCP in Our Setting

SemiCP's NNM augmentation is designed for the **extreme label-scarcity regime** (cal ≤ 20 per *entire dataset*), where Split CP's coverage instability (CovGap) is the dominant failure mode. In that regime, 4,000 unlabeled samples can dramatically stabilize the threshold estimate.

Our setting differs structurally:

1. **Calibration size**: We operate at cal = 300–800, meaning 3–8 labeled samples per class (K = 100). At this scale, Split CP's CovGap is already small; the dominant problem is **prediction set efficiency** (sz >> 1), not coverage instability.

2. **Full CP vs. Split CP**: Full CP eliminates the cal/train split entirely and uses all labeled data transductively. This achieves both valid coverage *and* small sets simultaneously. NNM augmentation applied to Split CP cannot close this gap — we showed FCP+PCA sz = 1.59 vs. SemiCP-THR sz = 3.22 at cal = 800, CIFAR-100 (findings.md §11).

3. **NNM approximation quality**: NNM assumes the unlabeled sample's true score ≈ its nearest pseudo-labeled neighbor's score. For DINOv2 embeddings (768-d, hyperspherical, 100-class), this approximation introduces variance that is *not* reduced by PCA — whereas PCA directly sharpens the NCM geometry. PCA-128 reduces sz by 24% (CIFAR-100); NNM adds 0%.

4. **i.i.d. vs. exchangeability**: SemiCP requires i.i.d. data. FCP only requires exchangeability. For our multi-dataset experiments with stratified splits (equal samples per class), exchangeability holds; i.i.d. is an unnecessary additional constraint.

**Conclusion**: SemiCP occupies a different niche (extreme label scarcity, single-model fixed embeddings) than our work (moderate calibration, geometry-aware NCMs, dimensionality reduction). The two are not competing solutions — they address different bottlenecks. This distinction should be made explicit in the paper's related work section.

---

## 7. Recommended Updates to `literature.md`

Add to **§6. Semi-Supervised CP**:
- Mazuelas (arXiv:2510.07185) — unsupervised calibration split CP
- Correia & Louizos (arXiv:2507.10425) — OT-based non-exchangeable CP (NeurIPS 2025)

~~Add to §7 (new section) — PPI and Risk Control:~~ **Removed 2026-05-18.** PPI direction deprioritized; Einbinder et al. implementation archived. PPI+CP extension (arXiv:2510.16166) similarly deferred.

**No new NCM design papers** discovered — the NNM approach in SemiCP uses simple k-NN lookup, unrelated to geodesic / whitened geometry.

---

## References (APA 7.0)

Angelopoulos, A. N., & Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *Foundations and Trends in Machine Learning, 16*(4), 494–591. https://arxiv.org/abs/2107.07511

Angelopoulos, A. N., Bates, S., Jordan, M. I., & Malik, J. (2021). Uncertainty sets for image classifiers using conformal prediction. *International Conference on Learning Representations*. https://arxiv.org/abs/2009.14193

Correia, A. H. C., & Louizos, C. (2025). Non-exchangeable conformal prediction with optimal transport: Tackling distribution shifts with unlabeled data. *Advances in Neural Information Processing Systems, 38*. https://arxiv.org/abs/2507.10425

Einbinder, B.-S., Ringel, L., & Romano, Y. (2024). Semi-supervised risk control via prediction-powered inference. https://arxiv.org/abs/2412.11174

Everink, J. M., Tamo Amougou, B., & Pereyra, M. (2025). Self-supervised conformal prediction for uncertainty quantification in imaging problems. https://arxiv.org/abs/2502.05127

Hur, Y. H., Nath, A., & Allen, G. I. (2026). Inference for clustering: Conformal sets for cluster labels. https://arxiv.org/abs/2604.03488

Mazuelas, S. (2025). Split conformal classification with unsupervised calibration. https://arxiv.org/abs/2510.07185

Romano, Y., Sesia, M., & Candès, E. (2020). Classification with valid and adaptive coverage. *Advances in Neural Information Processing Systems, 33*, 3581–3591. https://arxiv.org/abs/2006.02544

Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous set-valued classifiers with bounded error levels. *Journal of the American Statistical Association, 114*(525), 223–234.

Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer.

Zhou, X., Shi, Z., Zeng, H., Xia, X., Jing, B., & Wei, H. (2026). Semi-supervised conformal prediction with unlabeled nonconformity score. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://arxiv.org/abs/2505.21147

---

*Generated: 2026-05-17. Search coverage: Semantic Scholar, arXiv cs.LG/cs.CV 2025–2026.*
