"""
Classifier head and full model for binary classification.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from .backbones import BackboneWrapper, SUPPORTED_BACKBONES


class ClassifierHead(nn.Module):
    """
    Simple classifier head for binary classification.
    
    Architecture:
    - Linear(feature_dim, 256)
    - ReLU
    - Dropout(0.3)
    - Linear(256, 1)
    """
    
    def __init__(self, feature_dim: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, feature_dim)
        
        Returns:
            Logit tensor of shape (B, 1)
        """
        return self.classifier(x)


class BinaryClassifier(nn.Module):
    """
    Full binary classification model combining backbone and classifier head.
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.backbone = BackboneWrapper(backbone_name, pretrained, freeze_backbone)
        self.head = ClassifierHead(self.backbone.feature_dim, dropout)
        self.backbone_name = backbone_name
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logit tensor of shape (B, 1)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Probability tensor of shape (B,) with values in [0, 1]
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs
    
    def save(self, path: str) -> None:
        """Save model state dict."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "backbone_name": self.backbone_name,
            "state_dict": self.state_dict(),
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "BinaryClassifier":
        """Load model from checkpoint."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            backbone_name=checkpoint["backbone_name"],
            pretrained=False,
            freeze_backbone=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        
        return model


def create_model(
    backbone_name: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: Optional[str] = None
) -> BinaryClassifier:
    """
    Factory function to create a binary classifier.
    
    Args:
        backbone_name: Name of backbone architecture
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        device: Device to place model on
    
    Returns:
        BinaryClassifier instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BinaryClassifier(
        backbone_name=backbone_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    model.to(device)
    
    return model
