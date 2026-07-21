import torch
import torch.nn.functional as F
import gc

def build_cosine_similarity_graph(embeddings: torch.Tensor, device: torch.device, initial_threshold: float = 0.7) -> torch.Tensor:
    """
    Builds a sparse graph based on Cosine Similarity.
    Matches Algorithm 2 in the paper.
    Uses chunked computation and strict k-NN capping (max 20 neighbors per node) 
    to prevent dense graph VRAM explosion during SAGEConv message passing.
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    num_nodes = embeddings.shape[0]
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Cap maximum neighbors to prevent SAGEConv OOM (e.g., 17GB crash)
    max_neighbors = min(20, num_nodes - 1)
    
    edge_src = []
    edge_dst = []

    # If the graph is small, we can just do the dense approach
    if num_nodes < 5000:
        sim_matrix = torch.matmul(norm_embeddings, norm_embeddings.t())
        sim_matrix.fill_diagonal_(-1.0)
        
        top_k_vals, top_k_indices = torch.topk(sim_matrix, max_neighbors, dim=1)
        
        for i in range(num_nodes):
            for rank, j in enumerate(top_k_indices[i]):
                sim_val = top_k_vals[i, rank].item()
                # Connect if sim > threshold OR it's in the top 3 (ensures minimum connectivity)
                if sim_val > initial_threshold or rank < 3:
                    edge_src.append(i)
                    edge_dst.append(j.item())
                    
        if not edge_src:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
        return edge_index

    # For large graphs, chunk the matrix multiplication
    chunk_size = 5000
    
    for i in range(0, num_nodes, chunk_size):
        end_i = min(i + chunk_size, num_nodes)
        
        # Calculate sim for this chunk against all nodes
        chunk = norm_embeddings[i:end_i]
        sim_chunk = torch.matmul(chunk, norm_embeddings.t())
        
        # Remove self loops
        for j in range(end_i - i):
            sim_chunk[j, i + j] = -1.0
            
        top_k_vals, top_k_indices = torch.topk(sim_chunk, max_neighbors, dim=1)
        
        for local_i in range(end_i - i):
            actual_i = i + local_i
            for rank, j in enumerate(top_k_indices[local_i]):
                sim_val = top_k_vals[local_i, rank].item()
                if sim_val > initial_threshold or rank < 3:
                    edge_src.append(actual_i)
                    edge_dst.append(j.item())
                    
        # Free memory aggressively
        del sim_chunk
        del top_k_vals
        del top_k_indices
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    # Convert lists to tensor
    if not edge_src:  
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    
    # Clean up
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return edge_index
