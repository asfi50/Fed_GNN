import torch
import torch.nn.functional as F

import gc

def build_cosine_similarity_graph(embeddings: torch.Tensor, device: torch.device, initial_threshold: float = 0.7) -> torch.Tensor:
    """
    Builds a sparse graph based on Cosine Similarity.
    Matches Algorithm 2 in the paper.
    Uses chunked computation to prevent O(N^2) VRAM explosion (e.g., 150GB+ crashes).
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    num_nodes = embeddings.shape[0]
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # If the graph is small, we can just do the dense approach (faster)
    if num_nodes < 5000:
        sim_matrix = torch.matmul(norm_embeddings, norm_embeddings.t())
        sim_matrix.fill_diagonal_(-1.0)
        
        threshold = initial_threshold
        while threshold >= 0.3:
            degrees = (sim_matrix > threshold).sum(dim=1)
            if (degrees < 3).any():
                threshold -= 0.05
            else:
                break
                
        edge_index = (sim_matrix > threshold).nonzero(as_tuple=False).t().contiguous()
        return edge_index.to(device)

    # For large graphs, chunk the matrix multiplication
    chunk_size = 5000
    edge_src = []
    edge_dst = []
    
    threshold = initial_threshold
    # Since we can't efficiently run the while loop on a chunked matrix without re-calculating everything,
    # we'll approximate the paper's requirement: nodes with < 3 connections get their top-k neighbors.
    
    for i in range(0, num_nodes, chunk_size):
        end_i = min(i + chunk_size, num_nodes)
        
        # Calculate sim for this chunk against all nodes
        chunk = norm_embeddings[i:end_i]
        sim_chunk = torch.matmul(chunk, norm_embeddings.t())
        
        # Remove self loops
        for j in range(end_i - i):
            sim_chunk[j, i + j] = -1.0
            
        mask = sim_chunk > threshold
        degrees = mask.sum(dim=1)
        
        low_degree_mask = degrees < 3
        
        if low_degree_mask.any():
            # Grab top 3 neighbors for these specific nodes
            top_k = min(3, num_nodes - 1)
            sim_chunk_low = sim_chunk[low_degree_mask]
            _, top_indices = torch.topk(sim_chunk_low, top_k, dim=1)
            
            low_idx_rel = low_degree_mask.nonzero(as_tuple=True)[0]
            for idx_rel, neighbors in zip(low_idx_rel, top_indices):
                actual_i = i + idx_rel.item()
                for neighbor in neighbors:
                    edge_src.append(actual_i)
                    edge_dst.append(neighbor.item())
                    
            # Avoid double adding
            mask[low_degree_mask] = False
            
        # Extract edges that passed the threshold
        src_rel, dst = mask.nonzero(as_tuple=True)
        src = src_rel + i
        
        edge_src.extend(src.tolist())
        edge_dst.extend(dst.tolist())
        
        # Free memory aggressively
        del sim_chunk
        del mask
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    # Convert lists to tensor
    if not edge_src:  # Fallback if somehow no edges exist
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    
    # Clean up
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return edge_index
