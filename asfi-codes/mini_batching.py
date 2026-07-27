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
    # Persist the optimizer across federated rounds.
    # Note: federated_learning.py resets this after each FedAvg round to avoid
    # stale Adam momentum conflicting with the newly averaged weights.
    if not hasattr(model, 'client_optimizer'):
        model.client_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = model.client_optimizer
    
    # Compute smoothed inverse-frequency class weights clipped to [1.0, 10.0] to avoid gradient explosions
    unique_labels, counts = torch.unique(data['edge_labels'], return_counts=True)
    num_classes = list(model.edge_classifier[-1].parameters())[0].shape[0]
    
    weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    max_count = counts.max().item()
    for label, count in zip(unique_labels, counts):
        if label.item() < num_classes:
            val = (max_count / max(1, count.item())) ** 0.5
            weights[label.item()] = min(10.0, max(1.0, val))
            
    criterion = FocalLoss(weight=weights, gamma=1.5)
    
    # Extract tensors and keep them on CPU to save GPU memory
    # Call .contiguous() to satisfy pyg-lib C++ backend memory layout requirements
    x = data['features'].contiguous()
    edge_index = data['edge_index'].contiguous()
    edge_labels = data['edge_labels'].contiguous()
    
    # Create the PyTorch Geometric Data object required by LinkNeighborLoader
    graph_data = Data(x=x, edge_index=edge_index)
    
    # Initialize the SAGE neighborhood sampler.
    # batch_size=512 (not 2048): gives ~8 batches per epoch instead of 2,
    # resulting in ~24 gradient updates per round instead of 6.
    # More gradient updates = better local learning before FedAvg wipes it out.
    loader = LinkNeighborLoader(
        graph_data,
        num_neighbors=[15, 10],
        edge_label_index=edge_index,
        edge_label=edge_labels,
        batch_size=512,
        shuffle=True,
        num_workers=0
    )
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
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
            
            # Clip gradients to prevent explosion from focal loss / dense graphs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            
            with torch.no_grad():
                preds = predictions.argmax(dim=1)
                total_correct += (preds == batch.edge_label).sum().item()
                total_samples += len(batch.edge_label)
            
        total_loss += (epoch_loss / max(1, epoch_batches))
        batches += 1
        
    avg_loss = total_loss / max(1, batches)
    avg_acc = total_correct / max(1, total_samples)
    
    return {'loss': avg_loss, 'accuracy': avg_acc}
