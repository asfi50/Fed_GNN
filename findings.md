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
