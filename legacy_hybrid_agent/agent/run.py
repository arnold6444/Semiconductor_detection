"""
CLI entry point for the Hybrid AI Agent.

Usage:
    Training mode (requires labeled data with 'label' column):
        python -m agent.run --train --train_csv train.csv
    
    Inference mode:
        python -m agent.run --input_csv eval.csv --output submission.csv
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import Config, get_config, set_config
from .data.download import download_image
from .data.preprocess import preprocess_image
from .models.classifier import create_model, BinaryClassifier
from .training.crossval import CrossValidator
from .inference.predictor import Predictor
from .inference.fallback import FallbackPolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_url_column(df: pd.DataFrame) -> str:
    """Find the URL column in dataframe (supports multiple column names)."""
    url_columns = ["url", "img_url", "image_url", "URL", "IMG_URL", "IMAGE_URL"]
    for col in url_columns:
        if col in df.columns:
            return col
    raise ValueError(f"CSV must have one of these columns: {url_columns}")


def get_id_column(df: pd.DataFrame) -> str:
    """Find the ID column in dataframe (supports multiple column names)."""
    id_columns = ["id", "ID", "image_id", "img_id", "sample_id"]
    for col in id_columns:
        if col in df.columns:
            return col
    return None  # ID column is optional


def train(args):
    """Run training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 60)
    
    # Load or create config
    config = Config()
    config.seed = args.seed
    config.model.backbone = args.backbone
    set_config(config)
    set_seed(config.seed)
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load training data
    logger.info(f"Loading training data from: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    
    # Find URL column
    url_col = get_url_column(train_df)
    logger.info(f"Using URL column: {url_col}")
    
    # Check for label column (required for training)
    if "label" not in train_df.columns:
        raise ValueError(
            "Training CSV must have 'label' column with 0 (Normal) or 1 (Abnormal) values.\n"
            "For inference without labels, use: python -m agent.run --input_csv your_file.csv"
        )
    
    urls = train_df[url_col].tolist()
    labels = train_df["label"].tolist()
    
    logger.info(f"Found {len(urls)} training samples")
    logger.info(f"Class distribution: Normal={labels.count(0)}, Abnormal={labels.count(1)}")
    
    # Download images
    logger.info("Downloading training images...")
    images = []
    valid_labels = []
    
    for url, label in tqdm(zip(urls, labels), total=len(urls), desc="Downloading"):
        img, error = download_image(url)
        if img is not None:
            images.append(img)
            valid_labels.append(label)
        else:
            logger.warning(f"Failed to download: {url}")
    
    logger.info(f"Successfully loaded {len(images)}/{len(urls)} images")
    
    if len(images) < 10:
        raise ValueError("Not enough valid images for training")
    
    # Run Repeated Stratified K-Fold cross-validation
    logger.info(f"Running {args.repeats}x{args.folds}-Fold cross-validation...")
    cv = CrossValidator(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=config.seed,
        backbone_name=config.model.backbone
    )
    
    mean_f1, optimal_threshold, fold_results = cv.run(
        images=images,
        labels=valid_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        two_stage_finetune=args.two_stage
    )
    
    logger.info(f"Cross-validation F1: {mean_f1:.4f}")
    logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
    
    # Train final model on all data
    logger.info("Training final model on all data...")
    final_model = cv.train_final_model(
        images=images,
        labels=valid_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        use_focal_loss=args.focal_loss,
        two_stage_finetune=args.two_stage
    )
    
    # Save artifacts
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = checkpoint_dir / "model.pth"
    final_model.save(str(model_path))
    logger.info(f"Saved model to: {model_path}")
    
    # Update and save config
    config.threshold = optimal_threshold
    config.model.checkpoint = "model.pth"
    config.save(str(checkpoint_dir / "config.yaml"))
    logger.info(f"Saved config to: {checkpoint_dir / 'config.yaml'}")
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  F1 Score: {mean_f1:.4f}")
    logger.info(f"  Threshold: {optimal_threshold:.3f}")
    logger.info(f"  Checkpoint: {checkpoint_dir}")
    logger.info("=" * 60)


def inference(args):
    """Run inference pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Inference Pipeline")
    logger.info("=" * 60)
    
    # Load config and set seed
    checkpoint_dir = Path(args.checkpoint_dir)
    config = Config.load(str(checkpoint_dir / "config.yaml"))
    set_config(config)
    set_seed(config.seed)
    
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Threshold: {config.threshold:.3f}")
    logger.info(f"Margin: {config.margin:.3f}")
    logger.info(f"Alpha: {config.alpha:.3f}")
    
    # Load predictor
    logger.info("Loading model...")
    predictor = Predictor.load(
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        use_llm=not args.no_llm
    )
    
    if args.no_llm:
        logger.info("LLM decision support: DISABLED")
    else:
        if predictor.llm_client and predictor.llm_client.is_available():
            logger.info("LLM decision support: ENABLED")
        else:
            logger.warning("LLM decision support: UNAVAILABLE (API key missing?)")
    
    # Load evaluation data
    logger.info(f"Loading evaluation data from: {args.input_csv}")
    eval_df = pd.read_csv(args.input_csv)
    
    # Find URL and ID columns
    url_col = get_url_column(eval_df)
    id_col = get_id_column(eval_df)
    
    logger.info(f"Using URL column: {url_col}")
    if id_col:
        logger.info(f"Using ID column: {id_col}")
    
    urls = eval_df[url_col].tolist()
    ids = eval_df[id_col].tolist() if id_col else [f"sample_{i}" for i in range(len(urls))]
    
    logger.info(f"Found {len(urls)} evaluation samples")
    
    # Run inference
    results = []
    fallback_count = 0
    
    for sample_id, url in tqdm(zip(ids, urls), total=len(urls), desc="Processing"):
        # Download image
        img, error = download_image(url)
        
        if img is None:
            # Handle download failure
            fallback = FallbackPolicy.handle_download_failure(url)
            results.append({
                "id": sample_id,
                "img_url": url,
                "label": fallback.label
            })
            fallback_count += 1
            continue
        
        # Predict using Agent Flow
        label, prob, outcome, fallback = predictor.predict_single(img)
        
        results.append({
            "id": sample_id,
            "img_url": url,
            "label": label
        })
        
        if fallback is not None:
            fallback_count += 1
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(args.output)
    output_df.to_csv(output_path, index=False)
    
    # Print Agent Statistics
    predictor.print_stats()
    
    # Summary
    label_counts = output_df["label"].value_counts()
    
    logger.info("=" * 60)
    logger.info("AI Agent Inference Complete!")
    logger.info(f"  Total samples: {len(output_df)}")
    logger.info(f"  Normal (0): {label_counts.get(0, 0)}")
    logger.info(f"  Abnormal (1): {label_counts.get(1, 0)}")
    logger.info(f"  Output saved to: {output_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid AI Agent for Semiconductor Image Classification"
    )
    
    # Mode selection
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run in training mode"
    )
    
    # Training arguments
    parser.add_argument(
        "--train_csv",
        type=str,
        help="Path to training CSV file (requires 'label' column)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of CV repeats for stability (default: 5)"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["resnet18", "efficientnet_b0", "mobilenetv3_large_100"],
        help="Backbone architecture (default: efficientnet_b0)"
    )
    parser.add_argument(
        "--focal_loss",
        action="store_true",
        default=True,
        help="Use Focal Loss for class imbalance (default: True)"
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter (default: 2.0)"
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.75,
        help="Focal Loss alpha parameter (default: 0.75)"
    )
    parser.add_argument(
        "--two_stage",
        action="store_true",
        default=True,
        help="Use 2-stage fine-tuning (default: True)"
    )
    
    # Inference arguments
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to evaluation CSV file (columns: id, img_url)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Disable LLM decision support"
    )
    
    # Common arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train:
        if not args.train_csv:
            parser.error("--train_csv is required for training mode")
        train(args)
    else:
        if not args.input_csv:
            parser.error("--input_csv is required for inference mode")
        inference(args)


if __name__ == "__main__":
    main()
