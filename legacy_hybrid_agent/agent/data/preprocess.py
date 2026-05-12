"""
Image preprocessing and validation module.
"""

import torch
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple
import numpy as np

from ..config import get_config


def get_preprocess_transform(config=None) -> transforms.Compose:
    """Get the standard preprocessing transform for inference."""
    if config is None:
        config = get_config()
    
    return transforms.Compose([
        transforms.Resize((config.preprocessing.size, config.preprocessing.size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.preprocessing.mean,
            std=config.preprocessing.std
        )
    ])


def validate_image(image: Image.Image) -> Tuple[bool, Optional[str]]:
    """
    Validate an image for processing.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    # Check mode
    if image.mode not in ["RGB", "L", "RGBA"]:
        return False, f"Unsupported image mode: {image.mode}"
    
    # Check size
    if image.size[0] < 32 or image.size[1] < 32:
        return False, f"Image too small: {image.size}"
    
    # Check for corrupted images
    try:
        np.array(image)
    except Exception as e:
        return False, f"Corrupted image: {e}"
    
    return True, None


def preprocess_image(
    image: Image.Image,
    config=None
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """
    Preprocess a single image for model inference.
    
    Returns:
        Tuple of (tensor or None, error message or None)
    """
    # Validate
    is_valid, error = validate_image(image)
    if not is_valid:
        return None, error
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply transforms
    try:
        transform = get_preprocess_transform(config)
        tensor = transform(image)
        return tensor, None
    except Exception as e:
        return None, f"Preprocessing failed: {e}"


def preprocess_batch(
    images: list,
    config=None
) -> Tuple[Optional[torch.Tensor], list]:
    """
    Preprocess a batch of images.
    
    Returns:
        Tuple of (batch tensor or None, list of errors for failed images)
    """
    tensors = []
    errors = []
    
    for i, img in enumerate(images):
        tensor, error = preprocess_image(img, config)
        if tensor is not None:
            tensors.append(tensor)
            errors.append(None)
        else:
            errors.append(error)
    
    if tensors:
        batch = torch.stack(tensors)
        return batch, errors
    
    return None, errors
