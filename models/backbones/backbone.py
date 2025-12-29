import torch
import torch.nn as nn
import timm


class BackboneWrapper(nn.Module):
    """
    Wrapper for timm backbones with feature extraction.
    """
    def __init__(
        self,
        name: str = "mobilenetv3_large_100",
        pretrained: bool = True,
        in_channels: int = 3,
    ):
        """
        Args:
            name: timm model name
            pretrained: use pretrained weights
            in_channels: input channels (3 for RGB, 1 for depth/IR)
        """
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.model(dummy_input)
            self.feature_dim = dummy_output.shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor [B, C, H, W]
        Returns:
            features: [B, feature_dim]
        """
        if self.in_channels != 3:
            x = self.input_adapter(x)
        
        features = self.model(x)
        return features


def get_backbone(name: str, pretrained: bool = True, in_channels: int = 3) -> tuple[nn.Module, int]:
    """
    Factory function to get backbone and its output dimension.
    
    Returns:
        (backbone, feature_dim)
    """
    name_map = {
        "mobilenetv3_large": "mobilenetv3_large_100",
        "mobilenetv3_small": "mobilenetv3_small_100",
        "efficientnet_b0": "efficientnet_b0",
        "resnet18": "resnet18",
    }
    
    timm_name = name_map.get(name, name)
    backbone = BackboneWrapper(timm_name, pretrained=pretrained, in_channels=in_channels)
    return backbone, backbone.feature_dim
