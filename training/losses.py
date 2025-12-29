import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BCEWithLogitsLoss(nn.Module):
    """Binary cross entropy with optional label smoothing."""
    def __init__(self, label_smoothing: float = 0.0, pos_weight: Optional[float] = None):
        """
        Args:
            label_smoothing: smoothing factor (0.0 = no smoothing)
            pos_weight: weight for positive class (for imbalanced data)
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight else None
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 1] raw model output
            targets: [B, 1] or [B] ground truth (0=spoof, 1=real)
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class AuxiliaryDepthLoss(nn.Module):
    """
    Auxiliary depth map supervision loss.
    Real faces have depth variation, spoofs are flat.
    """
    def __init__(self, loss_type: str = "mse"):
        """
        Args:
            loss_type: "mse" or "l1"
        """
        super().__init__()
        self.loss_fn = nn.MSELoss() if loss_type == "mse" else nn.L1Loss()
        
    def forward(
        self, 
        pred_depth: torch.Tensor, 
        target_depth: Optional[torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_depth: [B, 1, H, W] predicted depth map
            target_depth: [B, 1, H, W] ground truth depth (if available)
            labels: [B] - used to create pseudo labels if no GT depth
            
        If no target_depth provided:
            - Real faces (label=1): target = ones (has depth)
            - Spoof faces (label=0): target = zeros (flat)
        """
        if target_depth is None:
            b, _, h, w = pred_depth.shape
            target_depth = labels.view(b, 1, 1, 1).expand(-1, -1, h, w).float()
        
        return self.loss_fn(pred_depth, target_depth)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to push real/spoof embeddings apart.
    """
    def __init__(self, margin: float = 1.0, mining: str = "hard"):
        """
        Args:
            margin: margin for contrastive loss
            mining: "hard" or "all" pairs
        """
        super().__init__()
        self.margin = margin
        self.mining = mining
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embedding_dim] normalized embeddings
            labels: [B] ground truth (0=spoof, 1=real)
            
        Pulls same-class embeddings together, pushes different-class apart.
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        batch_size = embeddings.shape[0]
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        labels = labels.view(-1, 1)
        label_mask = (labels == labels.T).float()
        
        positive_mask = label_mask - torch.eye(batch_size, device=embeddings.device)
        negative_mask = 1 - label_mask
        
        positive_pairs = dist_matrix * positive_mask
        negative_pairs = self.margin - dist_matrix
        negative_pairs = F.relu(negative_pairs) * negative_mask
        
        if self.mining == "hard":
            positive_loss = positive_pairs.max(dim=1)[0].sum() / (positive_mask.sum() + 1e-6)
            negative_loss = negative_pairs.max(dim=1)[0].sum() / (negative_mask.sum() + 1e-6)
        else:
            positive_loss = positive_pairs.sum() / (positive_mask.sum() + 1e-6)
            negative_loss = negative_pairs.sum() / (negative_mask.sum() + 1e-6)
        
        return (positive_loss + negative_loss) / 2


class FASLoss(nn.Module):
    """
    Combined loss for Face Anti-Spoofing.
    
    Total = λ1 * BCE + λ2 * DepthLoss + λ3 * ContrastiveLoss
    """
    def __init__(
        self,
        bce_weight: float = 1.0,
        depth_weight: float = 0.5,
        contrastive_weight: float = 0.1,
        label_smoothing: float = 0.0,
        depth_loss_type: str = "mse",
        contrastive_margin: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.depth_weight = depth_weight
        self.contrastive_weight = contrastive_weight
        
        self.bce_loss = BCEWithLogitsLoss(label_smoothing=label_smoothing)
        self.depth_loss = AuxiliaryDepthLoss(loss_type=depth_loss_type)
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: model output dict {"logits", "depth_map", "embedding"}
            targets: {"label": [B], "depth": [B,1,H,W] or None}
            
        Returns:
            {"total": total_loss, "bce": bce, "depth": depth, "contrastive": contr}
        """
        losses = {}
        
        losses["bce"] = self.bce_loss(outputs["logits"], targets["label"].unsqueeze(1))
        
        if "depth_map" in outputs and self.depth_weight > 0:
            losses["depth"] = self.depth_loss(
                outputs["depth_map"],
                targets.get("depth"),
                targets["label"]
            )
        else:
            losses["depth"] = torch.tensor(0.0, device=outputs["logits"].device)
        
        if "embedding" in outputs and self.contrastive_weight > 0:
            losses["contrastive"] = self.contrastive_loss(
                outputs["embedding"],
                targets["label"]
            )
        else:
            losses["contrastive"] = torch.tensor(0.0, device=outputs["logits"].device)
        
        losses["total"] = (
            self.bce_weight * losses["bce"] +
            self.depth_weight * losses["depth"] +
            self.contrastive_weight * losses["contrastive"]
        )
        
        return losses
