"""
Stage 1: DINOv2 Feature Extraction
===================================
Extracts feature vectors from images using frozen DINOv2 backbone.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from typing import Union, List
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DINO_MODEL, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


class DinoFeatureExtractor:
    """
    DINOv2 기반 이미지 특징 추출기
    
    - Model: dinov2_vits14 (frozen, no fine-tuning)
    - Output: 384차원 특징 벡터 (ViT-S/14)
    - Self-supervised pre-training으로 미세 패턴 인식에 강함
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or DINO_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load DINOv2 model from torch hub
        logger.info(f"Loading DINOv2 model: {self.model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing (DINOv2 optimal: 518x518 with ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        logger.info(f"DINOv2 loaded on {self.device}")
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """URL에서 이미지 로드"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image from {url}: {e}")
            raise
    
    def load_image(self, source: Union[str, Image.Image]) -> Image.Image:
        """이미지 로드 (URL, 파일 경로, 또는 PIL Image)"""
        if isinstance(source, Image.Image):
            return source.convert('RGB')
        elif source.startswith('http'):
            return self.load_image_from_url(source)
        else:
            return Image.open(source).convert('RGB')
    
    @torch.no_grad()
    def extract_features(self, source: Union[str, Image.Image], normalize: bool = True) -> torch.Tensor:
        """
        단일 이미지에서 특징 벡터 추출
        
        Args:
            source: 이미지 URL, 파일 경로, 또는 PIL Image
            normalize: L2 정규화 여부 (cosine distance 사용 시 필수)
            
        Returns:
            torch.Tensor: [1, feature_dim] 형태의 특징 벡터 (L2 정규화됨)
        """
        image = self.load_image(source)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass through DINOv2
        features = self.model(tensor)
        
        # ★ 핵심: L2 normalize - cosine distance 사용을 위해 필수
        if normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        
        return features.cpu()
    
    @torch.no_grad()
    def extract_batch_features(self, sources: List[Union[str, Image.Image]], normalize: bool = True) -> torch.Tensor:
        """
        배치 이미지에서 특징 벡터 추출
        
        Args:
            sources: 이미지 소스 리스트
            normalize: L2 정규화 여부 (cosine distance 사용 시 필수)
            
        Returns:
            torch.Tensor: [batch_size, feature_dim] 형태의 특징 벡터 (L2 정규화됨)
        """
        tensors = []
        for source in sources:
            image = self.load_image(source)
            tensor = self.transform(image)
            tensors.append(tensor)
        
        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch)
        
        # ★ 핵심: L2 normalize - cosine distance 사용을 위해 필수
        if normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        
        return features.cpu()
    
    def get_feature_dim(self) -> int:
        """특징 벡터 차원 반환"""
        dim_map = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
        }
        return dim_map.get(self.model_name, 384)


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = DinoFeatureExtractor()
    print(f"Feature dimension: {extractor.get_feature_dim()}")
