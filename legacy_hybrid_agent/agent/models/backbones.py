"""
Pretrained CNN backbones for feature extraction.
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple


SUPPORTED_BACKBONES = {
    "resnet18": {"feature_dim": 512},
    "efficientnet_b0": {"feature_dim": 1280},
    "mobilenetv3_large_100": {"feature_dim": 1280},
}


def get_backbone(
    name: str = "resnet18",
    pretrained: bool = True,
    freeze: bool = True
) -> Tuple[nn.Module, int]:
    """
    Load a pretrained backbone model.
    
    Args:
        name: Backbone name (resnet18, efficientnet_b0, mobilenetv3_large_100)
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone parameters
    
    Returns:
        Tuple of (backbone module, feature dimension)
    """
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {name}. "
            f"Supported: {list(SUPPORTED_BACKBONES.keys())}"
        )
    
    # Load model from timm
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    feature_dim = SUPPORTED_BACKBONES[name]["feature_dim"]
    
    # Freeze backbone if requested
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    return model, feature_dim


class BackboneWrapper(nn.Module):
    """
    Wrapper for backbone that handles different input formats.
    """
    
    def __init__(self, name: str = "resnet18", pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.backbone, self.feature_dim = get_backbone(name, pretrained, freeze)
        self.name = name
        self.frozen = freeze
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        if self.frozen:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        
        return features
    
    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if specified."""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self
