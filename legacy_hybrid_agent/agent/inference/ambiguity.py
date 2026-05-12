"""
Ambiguity detection for hybrid decision making.
Includes margin-based detection and TTA variance check.
"""

from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def is_ambiguous(
    p_local: float,
    threshold: float,
    margin: float,
    tta_variance: float = 0.0,
    tta_variance_threshold: float = 0.04  # (max-min > 0.2 means var > 0.04)
) -> bool:
    """
    Determine if a prediction is ambiguous.
    
    A prediction is ambiguous if either:
    1. The local model probability is within 'margin' of the decision threshold
    2. TTA variance exceeds the threshold (high prediction instability)
    
    Args:
        p_local: Local model abnormal probability
        threshold: Fixed decision threshold
        margin: Ambiguity margin
        tta_variance: Variance of TTA predictions
        tta_variance_threshold: Maximum acceptable TTA variance
    
    Returns:
        True if prediction is ambiguous
    """
    # Margin-based ambiguity
    margin_ambiguous = abs(p_local - threshold) <= margin
    
    # TTA variance-based ambiguity
    variance_ambiguous = tta_variance > tta_variance_threshold
    
    if margin_ambiguous:
        logger.debug(f"Ambiguous (margin): |{p_local:.3f} - {threshold:.3f}| = {abs(p_local - threshold):.3f} <= {margin}")
    if variance_ambiguous:
        logger.debug(f"Ambiguous (variance): {tta_variance:.4f} > {tta_variance_threshold}")
    
    return margin_ambiguous or variance_ambiguous


def get_ambiguity_score(
    p_local: float,
    threshold: float
) -> float:
    """
    Calculate ambiguity score (lower = more ambiguous).
    
    Args:
        p_local: Local model abnormal probability
        threshold: Fixed decision threshold
    
    Returns:
        Distance from threshold (0 = most ambiguous)
    """
    return abs(p_local - threshold)


def categorize_confidence(
    p_local: float,
    threshold: float,
    margin: float,
    tta_variance: float = 0.0,
    tta_variance_threshold: float = 0.04
) -> Tuple[str, float]:
    """
    Categorize prediction confidence level.
    
    Returns:
        Tuple of (category, score) where category is one of:
        - "high_confidence_normal"
        - "high_confidence_abnormal"
        - "ambiguous_margin"
        - "ambiguous_variance"
    """
    distance = abs(p_local - threshold)
    
    if tta_variance > tta_variance_threshold:
        return "ambiguous_variance", tta_variance
    elif distance <= margin:
        return "ambiguous_margin", distance
    elif p_local < threshold:
        return "high_confidence_normal", distance
    else:
        return "high_confidence_abnormal", distance


def should_use_llm(
    p_local: float,
    threshold: float,
    margin: float,
    tta_variance: float = 0.0,
    tta_min: float = 0.0,
    tta_max: float = 1.0,
    tta_range_threshold: float = 0.20
) -> Tuple[bool, str]:
    """
    Determine if LLM should be invoked for this sample.
    
    Conditions for LLM invocation:
    1. |p_local - threshold| <= margin
    2. max(p_tta) - min(p_tta) > tta_range_threshold
    
    Returns:
        Tuple of (should_invoke, reason)
    """
    tta_range = tta_max - tta_min
    
    margin_trigger = abs(p_local - threshold) <= margin
    variance_trigger = tta_range > tta_range_threshold
    
    if margin_trigger:
        return True, f"margin_trigger (|{p_local:.3f} - {threshold:.3f}| <= {margin})"
    elif variance_trigger:
        return True, f"variance_trigger (range={tta_range:.3f} > {tta_range_threshold})"
    else:
        return False, "high_confidence"
