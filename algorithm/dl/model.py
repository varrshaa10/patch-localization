"""
Siamese CNN for NCC candidate re-ranking.

Architecture:
- Two parallel CNN towers (shared weights)
- Each processes reference_patch and candidate_patch independently
- Extract feature embeddings from each tower
- Combine embeddings (concatenate) and pass through classifier head
- Output: sigmoid probability "is this candidate the true match for this reference?"
"""

import torch
import torch.nn as nn


class SiameseCNN(nn.Module):
    """
    Siamese network for patch matching.
    
    Input: (batch_size, 1, 128, 128) for reference and candidate patches
    Output: (batch_size, 1) sigmoid probability
    """
    
    def __init__(self, embedding_dim=128):
        super(SiameseCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Shared CNN tower: processes both reference and candidate patches
        self.tower = nn.Sequential(
            # Block 1: Conv + BatchNorm + ReLU + MaxPool
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64
            
            # Block 2: Conv + BatchNorm + ReLU + MaxPool
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
            
            # Block 3: Conv + BatchNorm + ReLU + MaxPool
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            
            # Block 4: Conv + BatchNorm + ReLU + MaxPool
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature embedding dimension after tower: 256 (output channels)
        tower_output_dim = 256
        
        # Classifier head: takes concatenated embeddings from both towers
        # Concatenation: [ref_embedding (256) || cand_embedding (256)] = 512
        self.classifier = nn.Sequential(
            nn.Linear(tower_output_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: probability (0-1)
        )
    
    def forward(self, ref_patch, cand_patch):
        """
        Forward pass.
        
        Args:
            ref_patch: (batch_size, 1, 128, 128) reference patches
            cand_patch: (batch_size, 1, 128, 128) candidate patches
        
        Returns:
            (batch_size, 1) sigmoid probabilities
        """
        # Extract feature embeddings from both patches using shared tower
        ref_embedding = self.tower(ref_patch)  # (batch_size, 256, 1, 1)
        cand_embedding = self.tower(cand_patch)  # (batch_size, 256, 1, 1)
        
        # Flatten embeddings
        ref_embedding = ref_embedding.view(ref_embedding.size(0), -1)  # (batch_size, 256)
        cand_embedding = cand_embedding.view(cand_embedding.size(0), -1)  # (batch_size, 256)
        
        # Concatenate embeddings
        combined = torch.cat([ref_embedding, cand_embedding], dim=1)  # (batch_size, 512)
        
        # Classify
        output = self.classifier(combined)  # (batch_size, 1)
        
        return output


class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross-entropy loss.
    
    Upweights positive class to handle imbalanced data (11.8% positive, 88.2% negative).
    Weight positive class by ~7.5x (inverse of 11.8% / 88.2%).
    """
    
    def __init__(self, pos_weight=7.5):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        # Note: PyTorch's BCEWithLogitsLoss has pos_weight parameter,
        # but we're using Sigmoid output, so use regular BCE with weights
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch_size, 1) predictions from model with sigmoid
            target: (batch_size, 1) binary labels {0, 1}
        
        Returns:
            scalar loss
        """
        loss = self.bce(pred, target)
        
        # Weight positive samples more heavily
        weights = torch.where(
            target == 1,
            torch.tensor(self.pos_weight, device=target.device),
            torch.tensor(1.0, device=target.device)
        )
        
        weighted_loss = loss * weights
        return weighted_loss.mean()


if __name__ == "__main__":
    # Test: instantiate model and do a forward pass
    model = SiameseCNN()
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy inputs
    ref = torch.randn(4, 1, 128, 128)
    cand = torch.randn(4, 1, 128, 128)
    
    # Forward pass
    output = model(ref, cand)
    print(f"Input shapes: ref {ref.shape}, cand {cand.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test loss
    loss_fn = WeightedBCELoss(pos_weight=7.5)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss = loss_fn(output, target)
    print(f"Loss: {loss:.4f}")
