import torch
import torch.nn.functional as F

def build_cosine_similarity_graph(embeddings: torch.Tensor, device: torch.device, initial_threshold: float = 0.7) -> torch.Tensor:
    """
    Builds a sparse graph based on Cosine Similarity.
    Matches Algorithm 2 in the paper.
    Reduces threshold if a node does not have at least 3 connections.
    """
    num_nodes = embeddings.shape[0]
    
    # Calculate pairwise cosine similarity matrix
    # Using broadcasting to compute all pairs efficiently
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(norm_embeddings, norm_embeddings.t())
    
    # Remove self loops
    sim_matrix.fill_diagonal_(-1.0)
    
    threshold = initial_threshold
    
    # Loop to ensure degree condition without allocating dense float matrices
    while threshold >= 0.3:
        # Check degrees directly on the boolean mask to save 4.4 GB VRAM
        degrees = (sim_matrix > threshold).sum(dim=1)
        if (degrees < 3).any():
            threshold -= 0.05
        else:
            break
            
    # Convert directly from boolean mask to edge_index (sparse COO format)
    # This avoids creating the intermediate dense `adj` matrix
    edge_index = (sim_matrix > threshold).nonzero(as_tuple=False).t().contiguous()
    
    return edge_index.to(device)
