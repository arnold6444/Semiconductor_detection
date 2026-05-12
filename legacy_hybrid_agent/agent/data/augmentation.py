"""
Data augmentation for training and test-time augmentation (TTA).
Conservative augmentations to preserve defect features.
NO CutMix/CutOut - these can destroy defect features!
"""

import torch
from PIL import Image
from torchvision import transforms
from typing import List
import random
import numpy as np

from ..config import get_config


def get_training_augmentation(config=None) -> transforms.Compose:
    """
    Get training-time augmentation transform.
    
    CONSERVATIVE augmentations to preserve defect features:
    - Gentle crop (scale 0.9-1.0 to keep defects visible)
    - Limited rotation (±10°)
    - Light brightness/contrast
    - Horizontal/Vertical flip
    
    DISABLED (can destroy defects):
    - CutMix
    - CutOut
    - RandomErasing
    - Aggressive cropping
    """
    if config is None:
        config = get_config()
    
    size = config.preprocessing.size
    
    return transforms.Compose([
        # VERY gentle random crop (preserve defect regions)
        transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        
        # Limited rotation to preserve defect orientation
        transforms.RandomRotation(degrees=10),
        
        # Flips
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # LIGHT color adjustments only
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.01
        ),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize
        transforms.Normalize(
            mean=config.preprocessing.mean,
            std=config.preprocessing.std
        ),
        
        # NO RandomErasing - can destroy small defects!
    ])


class TTAAugmentation:
    """
    Test-Time Augmentation (TTA) with deterministic transformations.
    
    Light TTA (4 transforms):
    - Original
    - Horizontal flip
    - Vertical flip
    - 180-degree rotation
    """
    
    def __init__(self, config=None):
        if config is None:
            config = get_config()
        
        self.size = config.preprocessing.size
        self.mean = config.preprocessing.mean
        self.std = config.preprocessing.std
        
        # Base transform (resize + normalize)
        self.base_transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def get_augmented_versions(self, image: Image.Image) -> List[torch.Tensor]:
        """Generate all TTA versions of an image."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        versions = []
        
        # Original
        versions.append(self.base_transform(image))
        
        # Horizontal flip
        img_hflip = image.transpose(Image.FLIP_LEFT_RIGHT)
        versions.append(self.base_transform(img_hflip))
        
        # Vertical flip
        img_vflip = image.transpose(Image.FLIP_TOP_BOTTOM)
        versions.append(self.base_transform(img_vflip))
        
        # 180-degree rotation
        img_rot180 = image.transpose(Image.ROTATE_180)
        versions.append(self.base_transform(img_rot180))
        
        return versions
    
    def get_augmented_batch(self, image: Image.Image) -> torch.Tensor:
        """Get TTA versions as a single batch tensor."""
        versions = self.get_augmented_versions(image)
        return torch.stack(versions)
    
    def get_predictions_with_variance(
        self,
        image: Image.Image,
        model: torch.nn.Module,
        device: str = "cpu"
    ) -> tuple:
        """Get TTA predictions with mean and variance."""
        batch = self.get_augmented_batch(image).to(device)
        
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(batch).cpu().numpy()
        
        mean_prob = float(np.mean(probs))
        variance = float(np.var(probs))
        min_prob = float(np.min(probs))
        max_prob = float(np.max(probs))
        
        return mean_prob, variance, min_prob, max_prob


class ExtendedTTAAugmentation(TTAAugmentation):
    """Extended TTA for ambiguous cases (8 transforms)."""
    
    def get_augmented_versions(self, image: Image.Image) -> List[torch.Tensor]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        versions = super().get_augmented_versions(image)
        
        # 90-degree rotation
        img_rot90 = image.transpose(Image.ROTATE_90)
        versions.append(self.base_transform(img_rot90))
        
        # 270-degree rotation
        img_rot270 = image.transpose(Image.ROTATE_270)
        versions.append(self.base_transform(img_rot270))
        
        # Horizontal flip + 90
        img_hflip_rot90 = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
        versions.append(self.base_transform(img_hflip_rot90))
        
        # Vertical flip + 90
        img_vflip_rot90 = image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_90)
        versions.append(self.base_transform(img_vflip_rot90))
        
        return versions
