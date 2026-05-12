"""
Threshold optimization and evaluation metrics for binary classification.
Includes confusion matrix, per-fold metrics, and stable threshold selection.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix components.
    
    Returns:
        Tuple of (TN, FP, FN, TP)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def print_confusion_matrix(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    prefix: str = ""
):
    """Print detailed confusion matrix and metrics."""
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    logger.info(f"{prefix}Confusion Matrix (threshold={threshold:.3f}):")
    logger.info(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    logger.info(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Also show probability distribution
    pos_probs = y_probs[y_true == 1]
    neg_probs = y_probs[y_true == 0]
    
    if len(pos_probs) > 0:
        logger.info(f"  P(abnormal) dist: mean={np.mean(pos_probs):.3f}, min={np.min(pos_probs):.3f}, max={np.max(pos_probs):.3f}")
    if len(neg_probs) > 0:
        logger.info(f"  P(normal) dist: mean={np.mean(neg_probs):.3f}, min={np.min(neg_probs):.3f}, max={np.max(neg_probs):.3f}")
    
    return tp, fp, fn, tn, f1


def compute_f1(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float
) -> float:
    """Compute F1 score at given threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    
    # Handle edge case where no positive predictions
    if np.sum(y_pred) == 0 or np.sum(y_true) == 0:
        return 0.0
    
    return f1_score(y_true, y_pred, zero_division=0)


def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_threshold: float = 0.3,
    max_threshold: float = 0.9,
    step: float = 0.01
) -> float:
    """
    Find threshold that maximizes F1 score.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        min_threshold: Minimum threshold to search
        max_threshold: Maximum threshold to search
        step: Step size for threshold search
    
    Returns:
        Optimal threshold
    """
    thresholds = np.arange(min_threshold, max_threshold, step)
    
    best_f1 = 0
    best_threshold = 0.5
    
    all_results = []
    
    for t in thresholds:
        f1 = compute_f1(y_true, y_probs, t)
        all_results.append((t, f1))
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    # Log top 5 thresholds
    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)[:5]
    logger.debug(f"Top 5 thresholds: {[(f'{t:.2f}', f'{f:.3f}') for t, f in sorted_results]}")
    
    return best_threshold


def optimize_threshold_per_fold(
    fold_results: List[Tuple[np.ndarray, np.ndarray]],
    min_threshold: float = 0.3,
    max_threshold: float = 0.9
) -> Tuple[float, float, List[dict]]:
    """
    Find optimal threshold using median across folds for stability.
    
    Returns:
        Tuple of (median_threshold, mean_f1, fold_details)
    """
    fold_thresholds = []
    fold_f1s = []
    fold_details = []
    
    for fold_idx, (y_probs, y_true) in enumerate(fold_results):
        if y_probs is None or y_true is None:
            continue
        
        # Skip folds with no positive samples
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0:
            logger.warning(f"Fold {fold_idx}: No positive samples in validation, skipping")
            continue
        
        # Find best threshold for this fold
        best_t = optimize_threshold(y_true, y_probs, min_threshold, max_threshold)
        best_f1 = compute_f1(y_true, y_probs, best_t)
        
        fold_thresholds.append(best_t)
        fold_f1s.append(best_f1)
        
        # Confusion matrix for this fold
        y_pred = (y_probs >= best_t).astype(int)
        tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
        
        fold_details.append({
            "fold": fold_idx,
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "threshold": best_t,
            "f1": best_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })
        
        logger.info(
            f"Fold {fold_idx}: n_pos={n_pos}, n_neg={n_neg}, "
            f"threshold={best_t:.3f}, F1={best_f1:.4f}, "
            f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
        )
    
    if len(fold_thresholds) == 0:
        logger.error("No valid folds found!")
        return 0.5, 0.0, []
    
    # Use MEDIAN for stability (not mean!)
    median_threshold = float(np.median(fold_thresholds))
    mean_f1 = float(np.mean(fold_f1s))
    
    logger.info(f"Threshold selection: median={median_threshold:.3f} from {fold_thresholds}")
    logger.info(f"Mean F1 across folds: {mean_f1:.4f}")
    
    return median_threshold, mean_f1, fold_details


def analyze_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float
):
    """
    Detailed analysis of predictions for debugging.
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    # Overall metrics
    tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
    
    logger.info("=" * 50)
    logger.info("Prediction Analysis")
    logger.info("=" * 50)
    logger.info(f"Threshold: {threshold:.3f}")
    logger.info(f"Total samples: {len(y_true)}")
    logger.info(f"Actual positives: {np.sum(y_true)}")
    logger.info(f"Actual negatives: {len(y_true) - np.sum(y_true)}")
    logger.info("-" * 50)
    logger.info(f"Confusion Matrix:")
    logger.info(f"  TP (correct abnormal): {tp}")
    logger.info(f"  TN (correct normal): {tn}")
    logger.info(f"  FP (false alarm): {fp}")
    logger.info(f"  FN (missed defect): {fn}")
    logger.info("-" * 50)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("-" * 50)
    
    # Probability distribution analysis
    pos_probs = y_probs[y_true == 1]
    neg_probs = y_probs[y_true == 0]
    
    if len(pos_probs) > 0:
        logger.info(f"Abnormal probs: mean={np.mean(pos_probs):.3f}, std={np.std(pos_probs):.3f}")
        logger.info(f"  range: [{np.min(pos_probs):.3f}, {np.max(pos_probs):.3f}]")
    
    if len(neg_probs) > 0:
        logger.info(f"Normal probs: mean={np.mean(neg_probs):.3f}, std={np.std(neg_probs):.3f}")
        logger.info(f"  range: [{np.min(neg_probs):.3f}, {np.max(neg_probs):.3f}]")
    
    # Check if model is just predicting all zeros or all ones
    if tp + fp == 0:
        logger.warning("⚠️ Model predicts ALL NORMAL (no positive predictions)")
    if tn + fn == 0:
        logger.warning("⚠️ Model predicts ALL ABNORMAL (no negative predictions)")
    
    logger.info("=" * 50)
    
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1
    }
