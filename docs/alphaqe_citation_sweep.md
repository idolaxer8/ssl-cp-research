# Citation sweep: Radenovic et al. "Fine-tuning CNN Image Retrieval with No Human Annotation" (2026-08-10)

Target: Radenovic, Tolias, Chum, TPAMI 2019 (arXiv:1711.02512) — source of GeM pooling and
**alpha-weighted query expansion (alpha-QE)**, which our DWT pipeline's Denoise step (`qe`)
ports to SSL embeddings as a test-time, label-free, pool-neighbor smoother before
whitening + truncation + full CP.

Method: 3 parallel web sweeps — (A) the paper's citation graph (~1,600 citing works via
OpenAlex; Semantic Scholar rate-limited), (B) the CP/UQ intersection (incl. the full
citation list of Feature CP, Teng et al. ICLR 2023), (C) successor techniques regardless
of citation. Companion doc: `dwt_theory_litsweep.md` (same-day theory-anchor sweep).

---

## 1. Headline verdict

1. **The conformal slot is empty.** No work in the alpha-QE citation graph applies
   conformal prediction to QE-processed embeddings, and no work citing Feature CP does
   neighbor-based *representation* smoothing before CP. Everything nearby operates on the
   score/calibration side (SNAPS, DAPS, NCP, localized/cluster CP, GraphLCP) or averages
   views of the same input (TTA-CP, randomized-smoothing CP). "Smooth embeddings toward
   unlabeled-pool neighbors, then conformalize, with an explicit help/hurt condition" is
   an unclaimed intersection — consistent with the theory sweep's finding that alpha-QE
   has zero theory.
2. **The community learned the QE step rather than analyzing it.** Line: LAttQE (ECCV
   2020, attention over neighbors) -> CSA (NeurIPS 2021, second-order affinity features)
   -> SuperGlobal (ICCV 2023, training-free GeM-mean neighbor aggregation, two-sided) ->
   QuARI / LOCORE (2025, query-adaptive linear maps / listwise long-context rerankers).
   Live through 2026, largely from the original Tolias/Radenovic circle.
3. **A parallel line reframed QE as one hop of graph smoothing** (GNN-perspective paper,
   GCN re-ranking, diffusion precomputation, 2024 affinity-learning survey) — exactly the
   framing the DWT denoise lemma needs (qe = first-order truncation of a Tikhonov graph
   filter; imports CSBM/homophily theory).
4. **Concrete upgrade levers exist** for both the smoother (SuperGlobal GeM aggregation,
   Tikhonov/spectral filtering, CSLS/NNN hubness correction, k-reciprocal edge filtering)
   and — critically — the **blocking label-free eta gate** (HUGE heterophily estimate,
   QB-Norm dynamic activation, hypergraph community-selection, SUE neighbor-spread).
5. **Watch item:** Fargion et al. (MA-CS/MS-CS source paper) reportedly published at
   ICML 2026 (arXiv:2511.19359) with a new *embedding-discovered* class-similarity
   variant — re-read for overlap with our centered-cosine M (verify venue/version).

---

## 2. Where the citation graph took alpha-QE (agent A)

### 2a. Learning / improving the aggregation step
- **LAttQE — Attention-Based Query Expansion Learning** (Gordo, Radenovic, Berg; ECCV
  2020; arXiv:2007.08019). Self-attention transformer learns per-neighbor weights over
  query + top-k; shows alpha-QE/AQE are special cases of a learnable weight family;
  analyzes alpha-QE's sensitivity to k and false-positive neighbors (= our low-homophily
  failure mode). Canonical "learn our smoothing weights" path.
- **UGQE — Uncertainty Guided Query Expansion** (Oncel et al.; ICIAR 2022). [CP-adjacent]
  Adds an uncertainty network that modulates per-neighbor aggregation weights; claims
  first uncertainty-aware QE. Heuristic UQ, no guarantee — natural baseline/citation when
  arguing qe weights should be reliability-weighted.
- **CSA — Contextual Similarity Aggregation** (Ouyang et al.; NeurIPS 2021;
  arXiv:2110.13430). Re-ranks from each neighbor's *affinity vector* (second-order
  neighborhood statistics) via transformer. Precedent for feeding neighborhood-structure
  statistics (our purity/homophily diagnostics) into the correction step.
- **SuperGlobal** (Shao et al.; ICCV 2023; arXiv:2308.06954). Training-free reranking
  with global descriptors only: GeM-style pooled aggregation of each top-k image with its
  own neighbors + QE refinement of the query. Two-sided (database side too) = retrieval
  analogue of our smoothing of cal+test against the pool. Strongest modern endorsement of
  pure embedding-space neighbor smoothing.
- **QuARI** (Xing et al.; 2025; arXiv:2505.21647). Query-specific *linear transformation*
  of embedding space for re-scoring (replicates alpha-QE as baseline). Suggests a learned/
  input-adaptive version of our W and T stages.
- **ADORE** (ICASSP 2026). Distills reranking/QE effects back into base descriptors —
  the train-time alternative to test-time qe.

### 2b. Transformer/listwise reranking (context)
- **RRT** (ICCV 2021, arXiv:2103.12236), **CVNet** (CVPR 2022, arXiv:2204.01458; TPAMI
  2024 ext.), **AMES** (ECCV 2024, arXiv:2408.03282), **LOCORE** (CVPR 2025,
  arXiv:2503.21772 — first *listwise* long-context reranker over the whole shortlist).
  Supervised/local-feature line; mostly INSPIRATION-ONLY for us (AMES's asymmetric
  compute budget — spend more on the test point than the pool — is worth stealing).

### 2c. Graph / diffusion / manifold smoothing (qe's mechanism family)
- **GCN re-ranking / GCR** (Zhang et al.; IEEE TMM 2023; arXiv:2306.08792, predecessor
  arXiv:2012.07620). Re-ranking as GCN-style *feature propagation*: descriptor updated by
  aggregating k-reciprocal neighbor features, training-free, GPU-parallel. Literally
  alpha-QE-as-embedding-update = one graph-convolution hop.
- **Understanding Image Retrieval Re-Ranking: A GNN Perspective** (Zhang, Zheng et al.;
  arXiv:2012.07620, ACM TOMM). Proves QE and diffusion re-ranking are instances of GNN
  message passing on the affinity graph. **Load-bearing citation for the DWT denoise
  lemma** — the formal bridge importing GNN homophily theory (Baranwal/CSBM anchors).
- **Efficient Diffusion on Region Manifolds / Fast Spectral Ranking** (Iscen et al.;
  CVPR 2017/2018). Multi-step QE via random-walk diffusion; spectral form h(L) applied to
  the pool graph converts directly into a feature smoother (precompute eigenvectors on
  pool, filter every embedding).
- **CAS — Cluster-Aware Similarity Diffusion** (Luo et al.; ICML 2024; arXiv:2406.02343).
  Confines diffusion to local clusters to stop cross-manifold leakage — structural fix
  for exactly our low-homophily harm regime.
- **LPMT** (Luo et al.; ICML 2025; arXiv:2506.05196). SOTA of the diffusion-reranking
  line (multi-hop signal decay fix); relevant only if we go multi-hop.
- **R-DiP** (DEXA 2024). Precomputes diffusion/reranked neighborhoods for the *database*
  offline — engineering precedent for amortizing qe smoothing of a fixed pool.
- **Continuous-CRF NN-graph denoising** (arXiv:2412.13875, 2024) — denoises the kNN
  affinity graph before propagation; retrieval-side confirmation of our "low-purity pool
  graph propagates wrong-class signal" mechanism.
- **Survey: Unsupervised Affinity Learning for Image Retrieval** (Pereira-Ferrero et al.;
  Computer Science Review 2024). Best single positioning citation for the post-processing
  family (QE / diffusion / rank-affinity / GCN reranking), still active.

### 2d. QE machinery outside retrieval
- **Label Propagation for Deep Semi-Supervised Learning** (Iscen, Tolias, Avrithis, Chum;
  CVPR 2019; arXiv:1904.04717). Same authors bridge retrieval-diffusion to
  classification-with-unlabeled-pool — closest philosophical ancestor (they propagate
  labels; we propagate features and keep exchangeability).
- **MeTTA — Mean Embeddings with Test-Time Augmentation** (Ashukha et al.; 2021;
  arXiv:2106.08038). Averages embeddings of augmented views — sibling denoiser and clean
  control arm for "is qe's gain just averaging?".
- **DeDrift** (ICCV 2023; arXiv:2308.02752). Content drift in embedding indices —
  citable threat model for pool-vs-test exchangeability aging.
- **ICMR 2023 encoder benchmarking** (Schall et al.) — grounds "foundation encoder +
  kNN classification" as a recognized regime.

### 2e. Uncertainty on retrieval embeddings (heuristic/Bayesian cluster)
- **Bayesian Triplet Loss** (Warburg et al.; ICCV 2021; arXiv:2011.12663) and **Bayesian
  Metric Learning** (Warburg et al.; 2023; arXiv:2302.01332) — embedding-space UQ exists
  but is approximate; sharpens our "conformal gives finite-sample validity" contrast.
- **SUE** (Zaffar et al.; CVPR 2024). Spatial spread of the query's top-k reference
  neighbors beats learned uncertainty estimators for place-recognition confidence —
  **direct methodological support for label-free neighborhood-statistic gates**.
- **ILIAS** (CVPR 2025; arXiv:2502.11748) — 2025 instance-retrieval benchmark for
  foundation models incl. DINOv2; fixes the empirical landscape.

---

## 3. The CP intersection (agent B)

### What exists (score/calibration side — none touch the embedding)
- **TTA-CP — Test-time Augmentation Improves Efficiency in Conformal Prediction**
  (Shanmugam et al.; arXiv:2505.22764, 2025). Aggregates TTA probabilities before
  calibration, symmetric => exchangeability preserved, −10-14% set size. **Closest
  published cousin of qe** — but averages augmentations of the same input in probability
  space, not pool neighbors in embedding space. Cite + differentiate.
- **RSCP / RSCP+** (Gendler et al. ICLR 2022; Yan et al. ICLR 2024, arXiv:2404.19651).
  Scores as expectations over Gaussian-randomized inputs with finite-sample validity —
  the main "smoothed score + validity proof" formalism. Their smoothing distribution is
  isotropic noise; ours is the data manifold (pool neighbors). Theory anchor.
- **NCP — Neighborhood Conformal Prediction** (Ghosh et al.; AAAI 2023;
  arXiv:2303.10694). kNN-weights the *calibration examples* in representation space;
  proves smaller sets under representation conditions. Complementary, non-overlapping
  with qe; useful theorem style ("conditions on the representation").
- **Graph CP 2025-2026**: GraphLCP (arXiv:2605.08074; PPR-kernel localized CP, argues
  pure embedding-proximity localization unreliable on sparse graphs), RR-CP for GNNs
  (UAI 2025, arXiv:2506.07854), RoCP-GNN (arXiv:2408.13825), graph-CP benchmark/audit
  (arXiv:2409.18332). Successors to DAPS/SNAPS, all score-side.
- **Unlabeled-pool CP beyond SemiCP**: StCP (arXiv:2605.01452, 2026 — stabilizes
  localized CP with unlabeled target data, transductive; reviewer-baseline candidate),
  PPI semi-supervised risk control (arXiv:2412.11174), label-free shift adaptation
  (arXiv:2406.01416). **Converging on the same unlabeled-pool resource — position qe as
  INPUT-space denoising with an explicit help/hurt condition, which none has.**
- **CP on foundation-model embeddings**: Localized CP for VLMs (EUVIP 2025,
  arXiv:2606.31577; cosine-weighted calibration scores), Cluster-Frequency CP
  (arXiv:2605.24872). Same substrate, score-side localization only.
- **Conformal retrieval / RAG**: Any2Any (arXiv:2411.10513), two-stage risk control for
  ranked retrieval (IJCAI 2025, arXiv:2404.17769 — "recall-guaranteed candidate pool"
  tool if we ever want guarantees on the qe neighbor set), TRAQ (NAACL 2024,
  arXiv:2307.04642), **C-RAG** (ICML 2024, arXiv:2402.03181 — conditions under which
  retrieval augmentation provably helps a distribution-free bound; the closest template
  for a "when does neighbor info help" theorem), SAFEVPR (arXiv:2605.28048), conformal
  CIR (arXiv:2605.24634). All conformalize retrieval *outputs*.
- **Feature-CP line** (full citation list checked, ~39 works): Fast Feature CP
  (arXiv:2412.00653), COMPASS (ICLR 2026, arXiv:2509.22240 — perturbs intermediate
  features along metric-sensitive subspaces), SCD-split (arXiv:2509.22529 — "smoothing"
  = merging regression subintervals). **None does neighbor-based representation
  denoising** — direct evidence the slot is unoccupied.
- **kNN-CP theory in metric spaces** (Lugosi & Matabuena; arXiv:2507.15741). Finite-
  sample conformal + locally adaptive kNN regions in general metric spaces with oracle
  rates — cleanest theory anchor for kNN nonconformity; candidate scaffold for a
  qe-Lipschitz lemma (DWT Step B).
- **SSL signal in CP scores**: Seedat et al. AISTATS 2023 (arXiv:2302.12238) — precedent
  for label-free SSL signal in nonconformity, no neighbor structure.
- **DAC** (Tomani et al.; ICML 2023; arXiv:2302.05118). kNN density in feature space as
  post-hoc calibration signal — non-CP baseline for the label-free gate panel.
- **Fargion et al. class-similarity CP** — reportedly ICML 2026, arXiv:2511.19359,
  camera-ready adds an embedding-discovered class-similarity variant (no human
  partitions). Re-read for overlap with our centered-cosine M. (Verify.)

### Confirmed-empty axes (novelty signals)
- Classical IR query expansion / pseudo-relevance feedback + CP: **empty** (only RAG-side
  CP exists). Closest: Collins-Thompson CIKM 2009 robust-QE risk analysis (pre-deep,
  no CP) — canonical "QE helps on average, hurts on tails", our homophily gate
  modernizes it.
- Neighbor-based representation smoothing before CP: **empty** (verified against Feature
  CP citations; flanked by TTA-CP and RSCP+).
- Formal help/hurt conditions for QE in deep embedding spaces: **empty** (only the GNN-
  perspective reframing arXiv:2012.07620 + 2009 text-IR analysis).

---

## 4. Upgrade candidates for the qe step (agent C, ranked)

1. **Tikhonov / spectral graph filtering, pool-fit** — X_smooth = (I + lambda*L)^{-1} X,
   where L = Laplacian of the pool kNN graph (Embedding Propagation, Rodriguez et al.
   ECCV 2020, arXiv:2003.04151; GSP-denoising view of GNNs; AGE arXiv:2007.01594).
   One-step qe is the first-order Neumann truncation of exactly this. Closed-form,
   label-free, single dial lambda, and the denoiser becomes an explicit low-pass graph
   filter — a spectral condition can replace the homophily heuristic in the DWT theory.
   ZLaP (CVPR 2024, arXiv:2404.04072) shows the inductive dual trick: precompute
   diffusion on the pool once, apply per test embedding in O(k) — the missing piece for
   exchangeability-friendly (pool-only-fit) multi-hop smoothing.
2. **SuperGlobal-style GeM neighbor aggregation** — replace the similarity^alpha weighted
   mean with a tuned generalized-mean (p) aggregation of neighbors; training-free,
   per-embedding, and its two-sided idea (refine pool side too) is free to test.
3. **Hub-corrected neighbor selection inside qe** — CSLS (arXiv:1710.04087) or NNN
   (EMNLP 2024, arXiv:2410.24114; one cached subtraction per pool point) to select and
   weight neighbors; Mutual Proximity / local scaling as alternatives (Feldbauer &
   Flexer KAIS 2019, scikit-hubness). Orthogonal to and stackable with any aggregation
   rule; directly attacks wrong-neighbor selection, our identified failure mechanism.
   Also: per-vector z-score normalization provably reduces hubness (Fei et al. ICCV
   2021) — one-line pre-step before building the kNN graph. SBERT-hubness NLDL 2024
   shows corrections transfer to modern transformer embedding spaces.
4. **k-reciprocal neighbor filtering (GCR-style)** — keep only mutual neighbors as the
   qe neighbor set; one-line structural gate on edges that removes hub/cross-class
   edges; GPU-cheap, label-free, proven at retrieval scale.
5. **Label-free homophily gate (the blocking eta/deploy problem)** — combine:
   (a) HUGE (arXiv:2502.13308) — label-free heterophily measure from attribute space
   with proven alignment to true edge heterophily, modulates aggregation; port from
   fraud graphs to embedding kNN graphs = dataset/regime-level dial;
   (b) QB-Norm's Dynamic Inverted Softmax (CVPR 2022, arXiv:2112.12777) — activates
   normalization only when the query's top-k intersects the hub set = per-query
   activation template;
   (c) hypergraph community-selection (An et al., NeurIPS 2021) — graph-cohesion score
   deciding whether to propagate;
   (d) SUE (CVPR 2024) — neighbor spatial spread as the best label-free confidence
   signal. All pool-computable, O(k) per test point.

Other noted arms: PT power transform (arXiv:2006.03806, Gaussianizing pre-conditioner
before W/T), EASE (CVPR 2022, unsupervised discriminant subspace = learned W+T
replacement, same homophily failure axis), TransCLIP (NeurIPS 2024, arXiv:2406.01837 —
score-space Laplacian smoothing with a built-in KL "don't drift" anchor = harm-limiting
idea), CAS cluster-confinement (port to feature averaging: mask neighbors outside the
point's pool-cluster), noHub (CVPR 2023, arXiv:2303.09352 — if hubness, not noise, turns
out dominant), ADC (NeurIPS 2021 — per-channel diffusion-time learning; drive with a
label-free surrogate like our margin-tail stat), MeTTA (control arm), protoLP/iLPC
(pool-side cleaning, pseudo-label machinery — exchangeability caution).

---

## 5. Positioning + action items

**Positioning sentence for the qe paper:** input-space, exchangeability-preserving
pool-neighbor denoising before CP, with an explicit label-free help/hurt (homophily)
condition — flanked by TTA-CP (augmentation-averaging, probability space) and RSCP+
(noise-averaging with validity proofs), neither of which uses data-manifold neighbors;
no occupant found.

**Citation stack (minimal):** Radenovic 2019 (mechanism) + LAttQE + SuperGlobal
(learned/modern QE) + arXiv:2012.07620 GNN-perspective + GCR (graph-smoothing view) +
2024 affinity survey (family positioning) + UGQE + SUE (uncertainty-aware neighbors, no
guarantees) + TTA-CP + RSCP+ + NCP (CP flank) + C-RAG (help-condition template) +
Iscen 2019 label propagation (unlabeled-pool ancestor).

**Action items:**
1. THEORY week tie-in: use arXiv:2012.07620 as the formal QE->message-passing bridge in
   the DWT denoise lemma; consider stating the lemma for the Tikhonov filter and
   deriving one-step qe as its first-order truncation (upgrades both theory and method).
2. Pilot queue (cheap, in expected-value order): (i) k-reciprocal edge filter inside qe;
   (ii) CSLS/NNN neighbor weighting; (iii) GeM-p aggregation vs similarity^alpha;
   (iv) two-sided smoothing (pool side too, amortized offline a la R-DiP);
   (v) gate panel: HUGE-style heterophily proxy + hub-intersection + neighbor-spread vs
   our hom(k) oracle on the 6-dataset regime map.
3. Re-read Fargion arXiv:2511.19359 (ICML 2026 version) for the embedding-discovered
   class-similarity variant vs our centered-cosine M.
4. Baseline watch: StCP + GraphLCP + localized-CP-on-VLM-embeddings are converging on
   the unlabeled-pool resource — likely reviewer baselines for any 2026+ submission.
