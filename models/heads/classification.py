import torch
import torch.nn as nn
from typing import Dict


class ClassificationHead(nn.Module):
    """
    Classification head with embedding layer and dual outputs.
    """
    def __init__(
        self,
        in_features: int,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        use_auxiliary_depth: bool = True,
        depth_map_size: int = 14,
    ):
        """
        Args:
            in_features: input feature dimension (from fusion)
            embedding_dim: embedding layer dimension
            dropout: dropout rate
            use_auxiliary_depth: whether to output auxiliary depth map
            depth_map_size: spatial size of auxiliary depth map (14x14)
        """
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(embedding_dim, 1)
        
        self.use_auxiliary_depth = use_auxiliary_depth
        if use_auxiliary_depth:
            self.depth_head = nn.Linear(embedding_dim, depth_map_size * depth_map_size)
            self.depth_map_size = depth_map_size
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: fused features [B, in_features]
        Returns:
            {
                "logits": [B, 1] - raw logits (apply sigmoid for score)
                "embedding": [B, embedding_dim] - for visualization/analysis
                "depth_map": [B, 1, H, W] - auxiliary depth prediction (if enabled)
            }
        """
        embedding = self.embedding(x)
        logits = self.classifier(embedding)
        
        output = {
            "logits": logits,
            "embedding": embedding,
        }
        
        if self.use_auxiliary_depth:
            depth = self.depth_head(embedding)
            depth = depth.view(-1, 1, self.depth_map_size, self.depth_map_size)
            output["depth_map"] = depth
            
        return output
