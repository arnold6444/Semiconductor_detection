"""
Stage 2: EfficientNet Filter
==============================
High-confidence filter using EfficientNet for binary classification.
NOT a final classifier - only confirms clear normal/abnormal samples.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import requests
from io import BytesIO
from typing import Union, Tuple, List, Optional
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EFFICIENTNET_MODEL, EFFICIENTNET_HIGH_CONF, EFFICIENTNET_LOW_CONF,
    EFFICIENTNET_EPOCHS, EFFICIENTNET_LR, EFFICIENTNET_DROPOUT,
    EFFICIENTNET_CACHE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
)

logger = logging.getLogger(__name__)


class TransistorDataset(Dataset):
    """학습용 데이터셋"""
    
    def __init__(self, image_sources: List[str], labels: List[int], transform=None):
        self.image_sources = image_sources
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_sources)
    
    def __getitem__(self, idx):
        source = self.image_sources[idx]
        label = self.labels[idx]
        
        # Load image
        if source.startswith('http'):
            response = requests.get(source, timeout=30)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(source).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class EfficientNetFilter:
    """
    EfficientNet 기반 2차 필터
    
    목적: 고확신 판정만 수행
    - p(anomaly) >= 0.9 → 비정상 확정
    - p(anomaly) <= 0.1 → 정상 확정
    - 그 외 → 3차 LLM 필터로 전달
    
    주의: 이진 분류기가 아니라 "확신 필터"로 사용
    """
    
    def __init__(
        self,
        model_name: str = None,
        high_conf: float = None,
        low_conf: float = None,
        device: str = None
    ):
        self.model_name = model_name or EFFICIENTNET_MODEL
        self.high_conf = high_conf or EFFICIENTNET_HIGH_CONF
        self.low_conf = low_conf or EFFICIENTNET_LOW_CONF
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self._fitted = False
        
        # Transforms - 강력한 증강으로 다양한 정상 패턴 학습
        self.train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # 약간 크게 리사이즈
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 랜덤 크롭
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # 추가
            transforms.RandomRotation(30),  # 더 강한 회전
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 이동/스케일
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # 더 강한 색상 변환
            transforms.RandomGrayscale(p=0.1),  # 가끔 그레이스케일
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # 블러
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 랜덤 지우기
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        logger.info(f"EfficientNetFilter initialized: {self.model_name}")
    
    def _build_model(self):
        """모델 구조 생성"""
        # Load pretrained EfficientNet
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=2,
            drop_rate=EFFICIENTNET_DROPOUT
        )
        self.model = self.model.to(self.device)
        logger.info(f"EfficientNet model loaded on {self.device}")
    
    def fit(
        self,
        image_sources: List[str],
        labels: List[int],
        epochs: int = None,
        lr: float = None,
        augment_multiplier: int = 5  # 정상 샘플을 몇 배로 증강할지
    ):
        """
        dev.csv 데이터로 파인튜닝
        
        Args:
            image_sources: 이미지 URL/경로 리스트
            labels: 0 (정상) / 1 (비정상) 리스트
            epochs: 학습 에폭 (default: 3, 과적합 방지)
            lr: 학습률
            augment_multiplier: 정상 샘플 증강 배수 (다양한 정상 패턴 학습)
        """
        epochs = epochs or EFFICIENTNET_EPOCHS
        lr = lr or EFFICIENTNET_LR
        
        if self.model is None:
            self._build_model()
        
        # ===== 정상 샘플 오버샘플링 =====
        # Stage 2로 오는 샘플 중 "새로운 정상"을 잘 판단하려면
        # 다양한 정상 패턴을 학습해야 함
        augmented_sources = []
        augmented_labels = []
        
        for source, label in zip(image_sources, labels):
            if label == 0:  # 정상
                # 정상은 여러 번 추가 (증강 시마다 다른 변환 적용)
                for _ in range(augment_multiplier):
                    augmented_sources.append(source)
                    augmented_labels.append(label)
            else:  # 비정상
                # 비정상도 어느 정도 증강 (2배)
                for _ in range(2):
                    augmented_sources.append(source)
                    augmented_labels.append(label)
        
        n_normal = sum(1 for l in augmented_labels if l == 0)
        n_abnormal = sum(1 for l in augmented_labels if l == 1)
        
        logger.info(f"Fine-tuning EfficientNet with augmentation:")
        logger.info(f"  Original: {len(image_sources)} samples")
        logger.info(f"  Augmented: {len(augmented_sources)} samples (Normal: {n_normal}, Abnormal: {n_abnormal})")
        logger.info(f"  Epochs: {epochs}")
        
        # Create dataset with strong augmentation
        dataset = TransistorDataset(augmented_sources, augmented_labels, self.train_transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Loss & Optimizer (balanced weights)
        if n_abnormal > 0:
            weight = torch.tensor([1.0, n_normal / n_abnormal]).to(self.device)
        else:
            weight = torch.tensor([1.0, 1.0]).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Accuracy tracking
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            scheduler.step()
            
            avg_loss = total_loss / len(loader)
            acc = 100 * correct / total
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.1f}%")
        
        self._fitted = True
        logger.info("EfficientNet fine-tuning complete (with augmentation)")
    
    def load_image(self, source: Union[str, Image.Image]) -> Image.Image:
        """이미지 로드"""
        if isinstance(source, Image.Image):
            return source.convert('RGB')
        elif source.startswith('http'):
            response = requests.get(source, timeout=30)
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            return Image.open(source).convert('RGB')
    
    @torch.no_grad()
    def predict_proba(self, source: Union[str, Image.Image]) -> float:
        """
        비정상 확률 반환
        
        Returns:
            p(anomaly): 0.0 ~ 1.0
        """
        if self.model is None:
            self._build_model()
        
        self.model.eval()
        
        image = self.load_image(source)
        tensor = self.eval_transform(image).unsqueeze(0).to(self.device)
        
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        
        # Return P(anomaly) = P(class=1)
        return probs[0, 1].item()
    
    def classify_stage2(self, prob: float) -> Tuple[str, str]:
        """
        2차 필터 판정
        
        Returns:
            (decision, reason)
            decision: 'normal', 'abnormal', 'uncertain'
        """
        if prob >= self.high_conf:
            return 'abnormal', f'P(anomaly)={prob:.4f} >= {self.high_conf}'
        elif prob <= self.low_conf:
            return 'normal', f'P(anomaly)={prob:.4f} <= {self.low_conf}'
        else:
            return 'uncertain', f'P(anomaly)={prob:.4f} in ambiguous range [{self.low_conf}, {self.high_conf}]'
    
    def predict(self, source: Union[str, Image.Image]) -> Tuple[str, float, str]:
        """
        전체 예측 파이프라인
        
        Returns:
            (decision, probability, reason)
        """
        prob = self.predict_proba(source)
        decision, reason = self.classify_stage2(prob)
        return decision, prob, reason
    
    def save_cache(self, path: Path = None):
        """모델 가중치 저장"""
        path = path or EFFICIENTNET_CACHE
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            logger.info(f"EfficientNet weights saved to {path}")
    
    def load_cache(self, path: Path = None) -> bool:
        """모델 가중치 로드"""
        path = path or EFFICIENTNET_CACHE
        
        if not path.exists():
            return False
        
        try:
            if self.model is None:
                self._build_model()
            
            self.model.load_state_dict(torch.load(path, weights_only=True))
            self.model.eval()
            self._fitted = True
            logger.info(f"EfficientNet weights loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load EfficientNet cache: {e}")
            return False


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    filter_model = EfficientNetFilter()
    
    # Test inference (without fine-tuning)
    print("EfficientNetFilter ready for testing")
