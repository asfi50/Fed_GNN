import torch
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_client_model_minibatch(model, data: dict, device: torch.device, num_epochs: int = 3) -> dict:
    """
    Trains a GAT client model using GraphSAGE-style neighborhood sampling.
    This resolves the >17GB CUDA Out Of Memory crash by dynamically creating 
    memory-safe subgraphs instead of feeding the entire 1.3M edge graph at once.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    import os
    weight_tensor = None
    if os.path.exists('data/class_weights.pt'):
        weight_tensor = torch.load('data/class_weights.pt').to(device)
        
    criterion = FocalLoss(weight=weight_tensor, gamma=2.0)
    
    # Extract tensors and keep them on CPU to save GPU memory
    # Call .contiguous() to satisfy pyg-lib C++ backend memory layout requirements
    x = data['features'].contiguous()
    edge_index = data['edge_index'].contiguous()
    edge_labels = data['edge_labels'].contiguous()
    
    # Create the PyTorch Geometric Data object required by LinkNeighborLoader
    graph_data = Data(x=x, edge_index=edge_index)
    
    # Initialize the SAGE neighborhood sampler
    # We sample 15 neighbors for the first hop, and 10 for the second hop
    loader = LinkNeighborLoader(
        graph_data,
        num_neighbors=[15, 10],
        edge_label_index=edge_index,
        edge_label=edge_labels,
        batch_size=2048,
        shuffle=True,
        num_workers=0
    )
    
    total_loss = 0.0
    batches = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for batch in loader:
            # Only move the small sampled subgraph to the GPU
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Predict using the target edges
            _, predictions = model(batch.x, batch.edge_index, target_edge_index=batch.edge_label_index)
            
            loss = criterion(predictions, batch.edge_label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            
        total_loss += (epoch_loss / max(1, epoch_batches))
        batches += 1
        
    avg_loss = total_loss / max(1, batches)
    
    return {'loss': avg_loss}
