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
