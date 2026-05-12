"""
Stratified K-Fold cross-validation with improved stability for small datasets.
Uses Repeated Stratified K-Fold and median threshold selection.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import logging

from ..models.classifier import create_model, BinaryClassifier
from .trainer import Trainer
from .threshold import optimize_threshold_per_fold, analyze_predictions
from ..config import get_config

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Repeated Stratified K-Fold cross-validation with improved stability.
    
    Key features for small/imbalanced data:
    - Multiple repeats for stable threshold selection
    - Per-fold confusion matrix logging
    - Median threshold (not mean) for robustness
    - Skip folds with no positive samples
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 5,  # Increased for stability
        random_state: int = 42,
        backbone_name: str = "efficientnet_b0"
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.backbone_name = backbone_name
        self.fold_results = []
    
    def run(
        self,
        images: List[Image.Image],
        labels: List[int],
        epochs: int = 50,
        batch_size: int = 4,
        device: str = "cpu",
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        two_stage_finetune: bool = True
    ) -> Tuple[float, float, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Run Repeated Stratified K-Fold cross-validation.
        
        Returns:
            Tuple of (mean_f1, optimal_threshold, fold_results)
        """
        config = get_config()
        
        # Log initial data distribution
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        logger.info("=" * 60)
        logger.info("TRAINING DATA DISTRIBUTION")
        logger.info(f"  Total: {len(labels)}")
        logger.info(f"  Normal (0): {n_neg}")
        logger.info(f"  Abnormal (1): {n_pos}")
        logger.info(f"  Imbalance ratio: {n_neg}:{n_pos} = {n_neg/max(n_pos,1):.1f}:1")
        logger.info("=" * 60)
        
        images_arr = np.array(images, dtype=object)
        labels_arr = np.array(labels)
        
        # Use Repeated Stratified K-Fold for stability
        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
        
        fold_results = []
        total_folds = self.n_splits * self.n_repeats
        
        for fold, (train_idx, val_idx) in enumerate(rskf.split(images_arr, labels_arr)):
            repeat = fold // self.n_splits + 1
            fold_in_repeat = fold % self.n_splits + 1
            
            train_images = [images[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_images = [images[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Log class distribution for this fold
            train_pos = sum(train_labels)
            train_neg = len(train_labels) - train_pos
            val_pos = sum(val_labels)
            val_neg = len(val_labels) - val_pos
            
            logger.info(f"\n[Repeat {repeat}/{self.n_repeats}, Fold {fold_in_repeat}/{self.n_splits}]")
            logger.info(f"  Train: {train_neg} normal + {train_pos} abnormal")
            logger.info(f"  Val: {val_neg} normal + {val_pos} abnormal")
            
            # Skip if validation has no positive samples
            if val_pos == 0:
                logger.warning(f"  ⚠️ Skipping fold - no positive samples in validation")
                fold_results.append((None, None))
                continue
            
            # Create fresh model for each fold
            model = create_model(
                backbone_name=self.backbone_name,
                pretrained=True,
                freeze_backbone=True,
                device=device
            )
            
            # Calculate pos_weight for imbalanced data
            pos_weight = train_neg / max(train_pos, 1)
            logger.info(f"  pos_weight: {pos_weight:.2f}")
            
            trainer = Trainer(
                model=model,
                device=device,
                pos_weight=pos_weight,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                use_ema=True
            )
            
            _, val_probs, val_labels_arr = trainer.train(
                train_images=train_images,
                train_labels=train_labels,
                val_images=val_images,
                val_labels=val_labels,
                epochs=epochs,
                batch_size=batch_size,
                use_weighted_sampler=True,
                two_stage_finetune=two_stage_finetune,
                stage2_start_epoch=epochs // 2,
                config=config
            )
            
            fold_results.append((val_probs, val_labels_arr))
        
        self.fold_results = fold_results
        
        # Use per-fold threshold optimization with MEDIAN
        logger.info("\n" + "=" * 60)
        logger.info("THRESHOLD OPTIMIZATION (Per-Fold with Median)")
        logger.info("=" * 60)
        
        optimal_threshold, mean_f1, fold_details = optimize_threshold_per_fold(
            fold_results,
            min_threshold=0.3,
            max_threshold=0.9
        )
        
        # Aggregate all predictions for final analysis
        all_probs = []
        all_labels = []
        for probs, labels in fold_results:
            if probs is not None and labels is not None:
                all_probs.extend(probs)
                all_labels.extend(labels)
        
        if len(all_probs) > 0:
            logger.info("\n" + "=" * 60)
            logger.info("AGGREGATED CROSS-VALIDATION RESULTS")
            logger.info("=" * 60)
            analyze_predictions(
                np.array(all_labels),
                np.array(all_probs),
                optimal_threshold
            )
        
        return mean_f1, optimal_threshold, fold_results
    
    def train_final_model(
        self,
        images: List[Image.Image],
        labels: List[int],
        epochs: int = 50,
        batch_size: int = 4,
        device: str = "cpu",
        use_focal_loss: bool = True,
        two_stage_finetune: bool = True
    ) -> BinaryClassifier:
        """Train final model on all data."""
        config = get_config()
        
        model = create_model(
            backbone_name=self.backbone_name,
            pretrained=True,
            freeze_backbone=True,
            device=device
        )
        
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        pos_weight = neg_count / max(pos_count, 1)
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING FINAL MODEL ON ALL DATA")
        logger.info(f"  Samples: {len(labels)} ({neg_count} normal + {pos_count} abnormal)")
        logger.info(f"  pos_weight: {pos_weight:.2f}")
        logger.info("=" * 60)
        
        trainer = Trainer(
            model=model,
            device=device,
            pos_weight=pos_weight,
            use_focal_loss=use_focal_loss,
            use_ema=True
        )
        
        # More epochs for final model
        trainer.train(
            train_images=images,
            train_labels=labels,
            epochs=int(epochs * 1.5),
            batch_size=batch_size,
            use_weighted_sampler=True,
            two_stage_finetune=two_stage_finetune,
            stage2_start_epoch=epochs // 2,
            config=config
        )
        
        return model
