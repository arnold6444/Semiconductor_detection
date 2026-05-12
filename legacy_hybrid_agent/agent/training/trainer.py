"""
Training loop for binary classifier with advanced strategies.
Includes:
- 2-Stage Fine-tuning (freeze then partial unfreeze)
- Focal Loss for class imbalance
- WeightedRandomSampler
- Label Smoothing
- Cosine Annealing LR
- Gradient Clipping
- EMA (Exponential Moving Average)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List, Callable
from tqdm import tqdm
import logging
import copy

from ..models.classifier import BinaryClassifier, create_model
from ..data.augmentation import get_training_augmentation
from ..config import get_config

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal term
        focal_term = (1 - pt) ** self.gamma
        
        loss = alpha_t * focal_term * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingBCELoss(nn.Module):
    """
    BCE Loss with Label Smoothing.
    Reduces overconfidence by softening labels.
    """
    
    def __init__(self, smoothing: float = 0.1, pos_weight: Optional[float] = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        if self.pos_weight is not None:
            weight = torch.where(
                targets > 0.5,
                torch.tensor(self.pos_weight, device=inputs.device),
                torch.tensor(1.0, device=inputs.device)
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets_smooth, weight=weight
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets_smooth)
        
        return loss


class EMA:
    """
    Exponential Moving Average for model weights.
    Helps stabilize predictions.
    Handles dynamically unfrozen parameters (e.g., in 2-stage fine-tuning).
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        self._update_shadow_dict()
    
    def _update_shadow_dict(self):
        """Update shadow dict with any new trainable parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights. Handles newly unfrozen parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    # New parameter just became trainable (e.g., Stage 2)
                    self.shadow[name] = param.data.clone()
                else:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights to model for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class ImageDataset(Dataset):
    """Simple dataset for training with augmentation."""
    
    def __init__(
        self,
        images: List[Image.Image],
        labels: List[int],
        transform: Optional[Callable] = None
    ):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler for balanced sampling.
    Each sample gets weight inversely proportional to class frequency.
    """
    class_counts = [labels.count(0), labels.count(1)]
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    
    logger.info(
        f"WeightedRandomSampler: class_counts={class_counts}, "
        f"class_weights=[{class_weights[0]:.3f}, {class_weights[1]:.3f}]"
    )
    
    return sampler


class Trainer:
    """
    Advanced trainer for binary classification with frozen backbone.
    
    Features:
    - 2-Stage Fine-tuning
    - Focal Loss / Label Smoothing BCE
    - WeightedRandomSampler
    - Cosine Annealing LR
    - Gradient Clipping
    - EMA
    """
    
    def __init__(
        self,
        model: BinaryClassifier,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: Optional[float] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        label_smoothing: float = 0.1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        gradient_clip: float = 1.0
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.gradient_clip = gradient_clip
        
        # Only train classifier head initially
        trainable_params = [p for p in model.head.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function selection
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})")
        elif label_smoothing > 0:
            self.criterion = LabelSmoothingBCELoss(smoothing=label_smoothing, pos_weight=pos_weight)
            logger.info(f"Using Label Smoothing BCE (smoothing={label_smoothing})")
        else:
            if pos_weight is not None:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight]).to(device)
                )
                logger.info(f"Using BCEWithLogitsLoss (pos_weight={pos_weight:.2f})")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        
        # EMA
        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        if use_ema:
            logger.info(f"Using EMA (decay={ema_decay})")
    
    def unfreeze_backbone_last_layers(self, num_layers: int = 1):
        """
        Stage 2: Unfreeze last N layers of backbone for fine-tuning.
        """
        # Get all backbone parameters
        backbone_params = list(self.model.backbone.backbone.parameters())
        
        # Unfreeze last N layers
        for param in backbone_params[-num_layers:]:
            param.requires_grad = True
        
        # Update optimizer to include unfrozen backbone params
        unfrozen_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Use smaller learning rate for backbone
        self.optimizer = optim.AdamW([
            {'params': self.model.head.parameters(), 'lr': 3e-4},
            {'params': [p for p in self.model.backbone.parameters() if p.requires_grad], 'lr': 1e-5}
        ], weight_decay=1e-4)
        
        logger.info(f"Stage 2: Unfroze last {num_layers} backbone layers")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        # Keep most of backbone frozen
        self.model.backbone.backbone.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        use_ema: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on validation data.
        
        Returns:
            Tuple of (loss, predictions, labels)
        """
        # Use EMA weights for evaluation if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits).squeeze(-1)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.squeeze(-1).cpu().numpy())
        
        # Restore original weights
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        return (
            total_loss / max(num_batches, 1),
            np.array(all_probs),
            np.array(all_labels)
        )
    
    def train(
        self,
        train_images: List[Image.Image],
        train_labels: List[int],
        val_images: Optional[List[Image.Image]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 50,
        batch_size: int = 4,
        early_stopping_patience: int = 10,
        use_weighted_sampler: bool = True,
        two_stage_finetune: bool = True,
        stage2_start_epoch: int = 20,
        config=None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Full training loop with 2-stage fine-tuning.
        
        Stage 1 (epochs 1 to stage2_start_epoch): Train only classifier head
        Stage 2 (remaining epochs): Fine-tune with last backbone layers
        """
        if config is None:
            config = get_config()
        
        # Create training dataset
        train_transform = get_training_augmentation(config)
        train_dataset = ImageDataset(train_images, train_labels, train_transform)
        
        # Create sampler for balanced training
        if use_weighted_sampler:
            sampler = create_weighted_sampler(train_labels)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
        
        # Validation loader
        val_loader = None
        if val_images is not None and val_labels is not None:
            from ..data.preprocess import get_preprocess_transform
            val_transform = get_preprocess_transform(config)
            val_dataset = ImageDataset(val_images, val_labels, val_transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Cosine Annealing LR Scheduler
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        
        best_val_loss = float("inf")
        best_val_probs = None
        best_val_labels = None
        patience_counter = 0
        best_state = None
        stage2_started = False
        
        for epoch in range(epochs):
            # Stage 2: Unfreeze backbone
            if two_stage_finetune and epoch == stage2_start_epoch and not stage2_started:
                self.unfreeze_backbone_last_layers(num_layers=2)
                stage2_started = True
                patience_counter = 0  # Reset patience for stage 2
            
            train_loss = self.train_epoch(train_loader)
            scheduler.step()
            
            if val_loader is not None:
                val_loss, val_probs, val_labels_arr = self.evaluate(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_probs = val_probs
                    best_val_labels = val_labels_arr
                    # Save EMA weights if using EMA
                    if self.ema is not None:
                        self.ema.apply_shadow()
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        self.ema.restore()
                    else:
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                stage_str = "S2" if stage2_started else "S1"
                logger.debug(
                    f"[{stage_str}] Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.debug(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return best_val_loss, best_val_probs, best_val_labels
