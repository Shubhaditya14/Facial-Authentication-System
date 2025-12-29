import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from .backbones.backbone import get_backbone
from .fusion.fusion import LateFusion
from .heads.classification import ClassificationHead


class MultiModalFASModel(nn.Module):
    """
    Multi-Modal Face Anti-Spoofing Model.
    
    Supports RGB, Depth, IR inputs with configurable backbones and fusion.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: model configuration dict with keys:
                - backbone_rgb: str (default: "mobilenetv3_large")
                - backbone_depth: str (default: "mobilenetv3_small")
                - backbone_ir: str (default: "mobilenetv3_small")
                - pretrained: bool
                - embedding_dim: int
                - dropout: float
                - use_auxiliary_depth: bool
                - depth_map_size: int
                - fusion_type: str ("late" or "attention")
        """
        super().__init__()
        self.config = config
        
        backbone_rgb = config.get("backbone_rgb", "mobilenetv3_large")
        backbone_depth = config.get("backbone_depth", "mobilenetv3_small")
        backbone_ir = config.get("backbone_ir", "mobilenetv3_small")
        pretrained = config.get("pretrained", True)
        
        self.rgb_backbone, rgb_dim = get_backbone(backbone_rgb, pretrained, in_channels=3)
        self.depth_backbone, depth_dim = get_backbone(backbone_depth, pretrained, in_channels=1)
        self.ir_backbone, ir_dim = get_backbone(backbone_ir, pretrained, in_channels=1)
        
        self.feature_dims = {
            "rgb": rgb_dim,
            "depth": depth_dim,
            "ir": ir_dim,
        }
        
        fusion_type = config.get("fusion_type", "late")
        if fusion_type == "late":
            self.fusion = LateFusion(self.feature_dims)
        elif fusion_type == "attention":
            raise NotImplementedError("Attention fusion in Phase 2")
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        max_fusion_dim = sum(self.feature_dims.values())
        
        self.head = ClassificationHead(
            in_features=max_fusion_dim,
            embedding_dim=config.get("embedding_dim", 256),
            dropout=config.get("dropout", 0.2),
            use_auxiliary_depth=config.get("use_auxiliary_depth", True),
            depth_map_size=config.get("depth_map_size", 14),
        )
        
        self.active_illum_module = None
        
    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        ir: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with flexible modality inputs.
        
        Args:
            rgb: [B, 3, H, W] or None
            depth: [B, 1, H, W] or None
            ir: [B, 1, H, W] or None
            
        Returns:
            {
                "logits": [B, 1]
                "score": [B, 1] (sigmoid applied)
                "embedding": [B, embedding_dim]
                "depth_map": [B, 1, 14, 14] (if auxiliary depth enabled)
            }
        """
        features = {}
        
        if rgb is not None:
            features["rgb"] = self.rgb_backbone(rgb)
            
        if depth is not None:
            features["depth"] = self.depth_backbone(depth)
            
        if ir is not None:
            features["ir"] = self.ir_backbone(ir)
        
        if not features:
            raise ValueError("At least one modality must be provided")
        
        fused = self.fusion(features)
        
        expected_dim = sum(self.feature_dims.values())
        if fused.shape[1] < expected_dim:
            padding = torch.zeros(
                fused.shape[0], 
                expected_dim - fused.shape[1],
                device=fused.device,
                dtype=fused.dtype
            )
            fused = torch.cat([fused, padding], dim=1)
        
        output = self.head(fused)
        
        output["score"] = torch.sigmoid(output["logits"])
        
        return output
    
    def forward_dict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass accepting batch dict from DataLoader.
        
        Args:
            batch: {"rgb": tensor, "depth": tensor or None, "ir": tensor or None, "label": tensor}
        """
        return self.forward(
            rgb=batch.get("rgb"),
            depth=batch.get("depth"),
            ir=batch.get("ir"),
        )


def create_model(config: Dict[str, Any]) -> MultiModalFASModel:
    """Factory function to create model from config."""
    return MultiModalFASModel(config)
