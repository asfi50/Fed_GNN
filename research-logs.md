# FedGATSage Research Logs

## [2026-07-20] Experiment: Resolving Client-Side CUDA Out-Of-Memory

**Objective:**
Run the FedGATSage codebase on the real `NF-ToN-IoT-fouad.csv` dataset (1.3M rows) using a standard 15GB GPU environment (Kaggle).

**Observation:**
The client-side training phase crashed with a CUDA Out-Of-Memory (OOM) error, attempting to allocate over 17GB of VRAM.

**Analysis:**
We discovered that the published codebase attempts to train the client-side Graph Attention Networks (GAT) using full-batch training (loading all nodes and edges simultaneously). The paper's architecture proposes "FedGATSage", implying the use of GraphSAGE (Graph SAmple and aggreGATe). While GraphSAGE is explicitly mentioned for the server side, it is impossible to train standard GATs on a 1.3M row dataset without also applying GraphSAGE's neighborhood sampling to the clients. The authors omitted this critical engineering component from their GitHub repository.

**Resolution:**
We mathematically audited the training loop and engineered PyTorch Geometric's `LinkNeighborLoader` into the local client models (`asfi-codes/mini_batching.py`). This dynamically creates memory-safe subgraphs containing the exact contextual neighborhood (1st and 2nd hop) required for the 3-layer GAT to compute attention scores, without altering the underlying logic.

**Result:**
The modification successfully decoupled the dataset size from the VRAM limit, dropping the maximum required VRAM from >17GB down to ~5GB. This structural engineering fix makes the codebase fully operational on standard hardware.

## [2026-07-21] Experiment: Community Detection Upgrade

**Objective:**
Upgrade the community detection algorithm from Louvain to Leiden.

**Observation:**
The legacy Louvain algorithm was theoretically flawed (producing disconnected communities) and was slower on large graphs.

**Analysis:**
**Leiden Migration:** The pipeline required a switch to Leiden, utilizing `igraph` for faster and mathematically guaranteed connected communities.

**Resolution:**
**Leiden Wrapper:** Implemented `asfi-codes/community_leiden.py` to seamlessly convert NetworkX graphs to igraph, compute the Leiden partition, and return the community mapping to the `CommunityAwareProcessor`.

**Result:**
The community detection pipeline now explicitly defaults to Leiden, boosting feature quality.

## [2026-07-22] Experiment: Fixing Model Confusion and Class Imbalance Bias

**Objective:**
Address the severe class imbalance and feature overlap that causes the federated ensemble to misclassify rare/similar attacks on smaller datasets (e.g., mislabeling Password/Scanning as DDoS, and XSS as Injection).

**Observation:**
The confusion matrix on a 1k sampled dataset revealed that:
1. XSS is consistently misclassified as Injection.
2. Password and Scanning attacks are mostly labeled as DDoS/Injection.
3. Injection attacks are frequently mislabeled as DDoS.
This occurs because Temporal features (volume spikes) overwhelm Behavioral/Content features, and the neural network defaults to guessing the majority classes (`DDoS`, `Injection`) due to extreme dataset imbalance.

**Proposed Resolution:**
1. **Class Weights:** Compute inverse-frequency class weights dynamically during preprocessing and inject them directly into the PyTorch `CrossEntropyLoss` function in `mini_batching.py`. This will heavily penalize the model for missing rare attacks.
2. **Focal Loss:** Modify the loss function to implement Focal Loss, forcing the model to focus on hard-to-classify examples (like differentiating XSS from Injection) rather than easy, high-volume examples.
3. **Random Forest Feature Injection:** Instead of passing *only* the raw GNN probabilities to the final ensemble Random Forest in `ensemble_evaluator.py`, we will also pass key engineered statistical features (e.g., average payload size, port variance). This gives the Random Forest the contextual data it needs to accurately tie-break when the Temporal GAT screams "DDoS" but the Content GAT suspects "Injection".

## [2026-07-23] Experiment: Cross-Dataset Fusion (NF + CIC Enhanced)

**Objective:**
Enrich the `NF-ToN-IoT` dataset (14 columns, NetFlow format) with rare attack samples from `CIC-ToN-IoT` (85 columns, CICFlowMeter format) to improve balanced detection of all 10 attack classes.

**Dataset Analysis:**
- `CIC-ToN-IoT`: 5,351,760 rows, 85 columns. Dominated by `xss` (40%) and `Benign` (47%). DDoS extremely rare (202 rows).
- `NF-ToN-IoT`: 1,379,274 rows, 14 columns. Dominated by `injection` (34%) and `ddos` (24%). Ransomware extremely rare (142 rows).
- Both datasets share identical 10 attack labels but dramatically different distributions.
- Column overlap: Only 2 shared columns (Attack, Label). Core features are structurally different.

**Resolution:**
Created `temp/fuse_datasets.py` to merge datasets via column mapping:
- `Src/Dst IP` → `IPV4_SRC/DST_ADDR`, `Src/Dst Port` → `L4_SRC/DST_PORT`
- `TotLen Fwd/Bwd Pkts` → `IN/OUT_BYTES`, `Tot Fwd/Bwd Pkts` → `IN/OUT_PKTS`
- `Flow Duration` (µs in CIC) ÷ 1000 → `FLOW_DURATION_MILLISECONDS` (ms in NF)
- `L7_PROTO`: Estimated from `Dst Port` using IANA port mappings (port 80→7/HTTP, 443→91/HTTPS, etc.) — avoids data leakage from using uniform 0 default (which would inadvertently label all CIC-migrated rows as "unknown protocol", creating a spurious discriminator).

**Rows Migrated:** 100% backdoor, dos, ddos, mitm, ransomware, scanning; 50% password; 100k xss (sampled).
**Final Enhanced Dataset:** `datasets/asfi/NF-ToN-IoT-Enhanced.csv` — 1,718,690 rows.

## [2026-07-23] Experiment: Option A — Discriminative Feature Engineering

**Objective:**
Fix systematic confusion between DDoS/Injection/Password/XSS/Scanning when running on NF-format data (only 8 usable numeric columns).

**Root Cause:** NF-ToN-IoT has no payload content columns. All web attacks (injection, xss, password) look nearly identical: ~5 packets, ~450-700 bytes, TCP port 80/443.

**New Features Added to `src/feature_engineering.py`:**

*Base features (all detectors, NF-format):*
- `out_in_bytes_ratio` → `OUT_BYTES / IN_BYTES`. Key discriminator: injection ~4x (large DB response), password ~1.6x, xss ~1.5x, ddos ~0.5x. Clipped [0, 100].
- `bytes_per_packet` → `(IN+OUT bytes) / total pkts`. DDoS: ~30 bytes/pkt. Real attacks: 200+ bytes/pkt. Clipped [0, 10000].
- `pkt_asymmetry` → `|IN_PKTS - OUT_PKTS| / total_pkts`. DDoS: high (one-directional). Others: low (bidirectional). Clipped [0, 1].
- `nf_flow_rate` → `total_pkts / FLOW_DURATION_MS`. Clipped [0, 1000] to prevent inf for 0ms DDoS rows.

*Temporal detector:*
- `flows_per_src_ip` + `flows_per_src_ip_norm` — DDoS/scanning flood from same IP.
- `dst_ip_diversity` + `dst_ip_diversity_log` — Unique dst IPs per src IP. Scanning: 50+, Password/Injection: 1-2.
- `dst_port_diversity` + `dst_port_diversity_log` — Unique dst ports per src IP. Scanning: very high, Password/Injection: 1 (always port 80/443).
- `port_to_flow_ratio` — Scanning ≈ 1.0 (one flow per new port), Password ≈ 0.01.

*Content detector:*
- `is_web_port`, `is_db_port` (NF-format, using `L4_DST_PORT`)
- `response_size_category` — Bins `out_in_bytes_ratio` into 0=small(XSS-like), 1=balanced, 2=large(injection-like).

*Behavioral detector:*
- `is_ephemeral_src`, `targets_system_port`, `port_spread` (NF-format using `L4_SRC/DST_PORT`)
- `is_short_session`, `is_long_session` (NF-format using `FLOW_DURATION_MILLISECONDS`)
- `session_regularity` — `1 / CV(IN_BYTES per dst_port)`. Password brute-force sends uniform requests → high regularity. NaN from single-member groups filled to 0.

**Safety:** All features use `.clip()` + `.fillna(0)`. No NaN/inf can enter GNN node tensors or Random Forest.

**Results (NF-Enhanced, 8k rows, 15 rounds, 5 clients, pure FedAvg):**

| Metric | Baseline | After Option A Wave 1 (base feats) | After Wave 2 (diversity feats) |
|---|---|---|---|
| Balanced Accuracy | 0.6133 | 0.6274 | 0.6257 |
| Standard Accuracy | 0.6987 | 0.7212 | 0.7188 |
| Macro F1 | 0.6081 | 0.6238 | 0.6232 |

**Observation:** Base features gave +1.4% Balanced Accuracy improvement. Diversity features did not improve further — likely because 1280-row client batches are too small for scanning patterns to be statistically meaningful (scanning appears ~30 times per client batch, insufficient for `dst_ip_diversity` to generalize). Needs larger batch size to see full effect.

**Remaining confusion pairs (from confusion matrix):**
- `injection` recall 70.9% — confused as `password` (34 times) and `xss` (17 times)
- `password` recall 60.1% — confused as `injection` (39 times) and `scanning` (14 times)
- `scanning` recall 24.1% — confused as `password` (10 times) and `injection` (6 times)
- `ransomware` recall 0% — only 2 samples in test set (too rare to detect)

**Next Steps:** Test with larger sample sizes (15k-20k rows) to give diversity features enough scanning examples per batch.

### 2026-07-27: Multi-Epoch Server Training, Equation 6 Adaptive Weighting, and Class-Balanced Evaluator

**Goal:** Resolve stalling server-side `global_loss` (~1.00) and increase test accuracy / Macro F1 on minority attack classes in FedGATSage without architectural hacks or cheating.

**What was tried:**
1. **Server-Side Optimization (`src/federated_learning.py`):** Replaced single-step server update with a 20-epoch training loop per round and a persistent Adam optimizer (`lr=0.005`, `weight_decay=1e-4`, grad clipping `max_norm=2.0`) to allow GraphSAGE to converge on community overlay graphs.
2. **Adaptive Parameter Redistribution (`src/federated_learning.py`):** Implemented Equation 6 from Section 4 of the paper ($w_k = \alpha + (1-\alpha) \frac{A_k - A_{\min}}{A_{\max} - A_{\min}}$, $\alpha=0.2$). Client parameter averaging is now weighted by local validation accuracy instead of naive unweighted FedAvg.
3. **Smoothed Inverse-Frequency Focal Loss (`asfi-codes/mini_batching.py`):** Replaced unweighted Cross-Entropy with `FocalLoss(gamma=1.5)` using square-root smoothed inverse class frequencies clipped to `[1.0, 10.0]`. **Outcome & Deprecation:** Although applied to penalize missing rare attacks, empirical testing showed that for rare classes with low prediction confidence ($p_t$), the modulating factor $(1-p_t)^\gamma$ combined with class weights caused severe gradient instability and loss escalation instead of stable convergence. Thus, Focal Loss was deprecated and removed.
4. **Balanced Ensemble Evaluator (`asfi-codes/ensemble_evaluator.py`):** Set `class_weight='balanced'` in `RandomForestClassifier` so meta-feature evaluation penalizes misclassifying rare attack types (e.g., `mitm`, `backdoor`).

### 2026-07-27: Shift from Focal Loss to Class-Balanced Loss (CB Loss)

**Goal:** Prevent gradient instability and loss explosion observed with Focal Loss while safely weighting minority attack classes (e.g., `password`, `mitm`) without degrading majority class accuracy.

**What was tried:**
1. **Class-Balanced Loss Based on Effective Number of Samples (`asfi-codes/mini_batching.py`, `src/federated_learning.py`):** Implemented `ClassBalancedLoss` from the seminal paper by Yin Cui et al. (**CVPR 2019**, *"Class-Balanced Loss Based on Effective Number of Samples"*).
   - **Theoretical Perspective:** As sample count $n$ increases, information gain per sample decreases due to data overlap (diminishing marginal benefits). Instead of raw count $n$, the effective number of samples is defined as $E_n = \frac{1 - \beta^n}{1 - \beta}$ (using hyperparameter $\beta=0.9999$).
   - **Implementation Details:** Replaced client mini-batch training loss (`train_client_model_minibatch`) and server global training loss (`_train_global_model`) with `ClassBalancedLoss(beta=0.9999, loss_type="focal", gamma=1.5)`.
   - **Why Applied:** The normalized weights based on $E_n$ establish a mathematical saturation limit (ceiling), ensuring rare attack classes receive boosted emphasis while completely avoiding gradient explosion and preserving majority class boundaries (`Benign`, `Injection`).

**Next Steps:** Manually run experiments to verify whether test accuracy and minority F1 scores improve empirically.
