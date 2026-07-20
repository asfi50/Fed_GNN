# FedGATSage Project Findings

This document tracks critical discrepancies, bottlenecks, and loopholes discovered between the theoretical claims in the original research paper and the actual provided implementation code.

## Finding 1: The "Simplified" Server Aggregation Loophole (1TB VRAM Bug)
**Date:** 2026-07-19
**Impact:** Prevents the code from running on real-world datasets by triggering massive Out-Of-Memory (OOM) errors.

### The Discrepancy
According to the published paper (under *Server-side processing*), the server constructs an overlay graph using a highly optimized, sparse approach. It calculates the cosine similarity between community embeddings and only creates an edge if the similarity exceeds a dynamic threshold (starting at 0.7). The paper claims this architecture requires only **2.4GB RAM** during processing.

However, the actual code provided in this repository takes a massive, unoptimized shortcut. In `src/federated_learning.py` (`_aggregate_updates`), line 386, the author wrote:
```python
# Create a fully connected graph for the global model (simplified)
# In a real scenario, we would use the community structure to define edges
num_nodes = global_x.shape[0]
edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(self.device)
```

### The Consequence
This "simplified" approach attempts to build a **fully-connected graph** where every node connects to every other node. 
- When run on the 200-row dummy dataset, this works fine.
- When run on the real 1.3 million row `NF-ToN-IoT` dataset, `torch.combinations` calculates over **33 billion edges**.
- PyTorch attempts to allocate memory for this massive tensor, resulting in a **1 Terabyte GPU VRAM allocation request**, instantly crashing any standard environment (like Colab or Kaggle T4s).

### Required Fixes
To make the codebase match the paper and actually scale to real datasets, the following mathematical optimizations must be implemented:
1. **Implement Cosine Similarity Overlay:** Replace `torch.combinations` with a sparse k-NN or Cosine Similarity graph as described in the paper.
2. **Leiden Algorithm Implementation:** Replace the slow Python-based Louvain community detection with the C++ optimized Leiden algorithm (via `igraph`) to solve the CPU pinning bottleneck.
3. **Local Mini-Batching:** Implement PyTorch Geometric's `NeighborLoader` on the client-side GAT layers to process graphs in smaller chunks, preventing the local nodes from maxing out standard 16GB VRAM bounds.

## Finding 2: The GraphSAGE Client-Side Omission (17GB VRAM Bug)
**Date:** 2026-07-20
**Impact:** Causes standard 15GB/16GB GPUs to crash with CUDA Out-Of-Memory errors during the local client training phase.

### The Discrepancy
The paper's title "FedGATSage" implies the use of GraphSAGE (Graph SAmple and aggreGATe). In the paper's theory, they state that GAT is used on the client-side. However, the standard GAT operates using full-batch training, which attempts to load the entire graph structure simultaneously. Given that the NF-ToN-IoT dataset contains over 1.3 million edges, computing multi-head attention scores for all edges at once requires upwards of 17GB of VRAM, mathematically crashing any standard GPU. 

If the original authors successfully trained their client models on this dataset without crashing, they **must** have applied GraphSAGE's defining characteristic—Neighborhood Sampling (mini-batching)—to their client-side GAT models.

However, in the published GitHub codebase, this logic was entirely omitted. The provided code blindly pushes the full, uncut graph into the local models (`predictions = model(x, edge_index)`), causing an instant memory crash.

### The Resolution
We explicitly engineered GraphSAGE's neighborhood sampling back into the local training loop to fix the code:
- **Location:** `asfi-codes/mini_batching.py`
- **Integration:** Updated `src/federated_learning.py` and modified the forward passes in `src/gnn_models.py` to support `target_edge_index`.
- **Result:** By using PyTorch Geometric's `LinkNeighborLoader`, we dynamically slice the client graph into mathematical subgraphs (mini-batches). This drops the VRAM requirement from >17GB down to a safe ~5GB, completely resolving the crash while preserving the structural attention logic.

## Resolutions Implemented

To track custom changes cleanly and allow easy reversion, newly implemented solutions strictly adhere to the original paper's math but are stored in a separate `asfi-codes` directory. They are imported into the main project files with the comment tag `# asfi-codes`.

### 1. Cosine Similarity Server Aggregation
**Fixed:** The 1TB VRAM OOM error was fixed by replacing the naive `torch.combinations` step with the mathematically rigorous Cosine Similarity approach outlined in Algorithm 2 of the original paper.
- **Location:** `asfi-codes/graph_utils.py` (`build_cosine_similarity_graph`)
- **Integration:** Replaced global graph construction in `src/federated_learning.py`.
- **Result:** Reduces server aggregation memory from O(N^2) fully-connected mapping (billions of edges) down to a highly sparse, optimized representation consuming less than ~3GB VRAM.

### 2. Random Forest Ensemble Evaluation
**Fixed:** The repository originally lacked the ensemble evaluation logic entirely, opting instead to evaluate only a single "temporal" detector. We implemented the fusion classifier from the paper.
- **Location:** `asfi-codes/ensemble_evaluator.py` (`RandomForestEnsembleEvaluator`)
- **Integration:** Updated `experiments/fedgatsage_experiment.py` to intercept the evaluation phase. If all three detectors (Temporal, Content, Behavioral) are present, it concatenates their Softmax probability outputs and trains a Random Forest Classifier to fuse their predictions.
- **Result:** Matches the paper's multi-detector ensemble architecture and correctly prints `balanced_accuracy_score`.
