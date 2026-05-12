"""
Decision fusion for combining local model and LLM predictions.
Includes Confidence Gate for LLM results.
"""

from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# LLM Confidence Gate threshold
LLM_CONFIDENCE_GATE = 0.65


def fuse_predictions(
    p_local: float,
    p_llm: float,
    llm_confidence: float,
    alpha: float,
    confidence_gate: float = LLM_CONFIDENCE_GATE
) -> Tuple[float, bool]:
    """
    Fuse local model and LLM predictions using weighted average.
    Applies Confidence Gate: if LLM confidence < gate, ignore LLM.
    
    Formula: p_final = α * p_local + (1 - α) * p_llm
    
    Args:
        p_local: Local model abnormal probability
        p_llm: LLM-derived abnormal probability
        llm_confidence: LLM's reported confidence
        alpha: Weight for local model (0 to 1)
        confidence_gate: Minimum LLM confidence to use LLM
    
    Returns:
        Tuple of (fused_probability, llm_used)
    """
    # Confidence Gate: ignore low-confidence LLM predictions
    if llm_confidence < confidence_gate:
        logger.info(
            f"LLM confidence {llm_confidence:.2f} < {confidence_gate}, using local only"
        )
        return p_local, False
    
    # Weighted fusion
    p_final = alpha * p_local + (1 - alpha) * p_llm
    
    logger.info(
        f"Fusion: {alpha:.2f}*{p_local:.3f} + {1-alpha:.2f}*{p_llm:.3f} = {p_final:.3f}"
    )
    
    return p_final, True


def make_decision(
    p_final: float,
    threshold: float
) -> int:
    """
    Make final binary decision based on fused probability.
    
    Args:
        p_final: Final abnormal probability
        threshold: Decision threshold
    
    Returns:
        0 for Normal, 1 for Abnormal
    """
    return 1 if p_final >= threshold else 0


def fuse_and_decide(
    p_local: float,
    p_llm: float,
    llm_confidence: float,
    alpha: float,
    threshold: float,
    confidence_gate: float = LLM_CONFIDENCE_GATE
) -> Tuple[float, int, bool]:
    """
    Fuse predictions and make decision in one step.
    
    Args:
        p_local: Local model abnormal probability
        p_llm: LLM-derived abnormal probability
        llm_confidence: LLM's reported confidence
        alpha: Weight for local model
        threshold: Decision threshold
        confidence_gate: Minimum LLM confidence to use LLM
    
    Returns:
        Tuple of (fused_probability, label, llm_was_used)
    """
    p_final, llm_used = fuse_predictions(
        p_local, p_llm, llm_confidence, alpha, confidence_gate
    )
    label = make_decision(p_final, threshold)
    return p_final, label, llm_used
