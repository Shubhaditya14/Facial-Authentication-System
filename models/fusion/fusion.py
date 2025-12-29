import torch
import torch.nn as nn
from typing import Dict, Optional


class LateFusion(nn.Module):
    """
    Late fusion: concatenate features from all modalities.
    """
    def __init__(self, feature_dims: Dict[str, int]):
        """
        Args:
            feature_dims: {"rgb": 960, "depth": 576, "ir": 576}
        """
        super().__init__()
        self.feature_dims = feature_dims
        self.total_dim = sum(feature_dims.values())
        
    def forward(self, features: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            features: {"rgb": [B, 960], "depth": [B, 576] or None, "ir": [B, 576] or None}
        Returns:
            fused: [B, total_dim] - only concatenates non-None features
        """
        non_none_features = [v for v in features.values() if v is not None]
        
        if not non_none_features:
            raise ValueError("At least one feature tensor must be non-None")
        
        fused = torch.cat(non_none_features, dim=1)
        return fused
        
    def get_output_dim(self, available_modalities: list[str]) -> int:
        """Get output dim based on which modalities are available."""
        return sum(self.feature_dims[m] for m in available_modalities if m in self.feature_dims)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion: learnable weights for each modality.
    Placeholder for Phase 2.
    """
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()
        raise NotImplementedError("AttentionFusion will be implemented in Phase 2")
