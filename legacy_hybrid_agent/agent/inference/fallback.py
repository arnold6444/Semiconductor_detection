"""
Failure handling and fallback policies for robust inference.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur during inference."""
    DOWNLOAD_FAILED = "download_failed"
    PREPROCESSING_FAILED = "preprocessing_failed"
    MODEL_INFERENCE_FAILED = "model_inference_failed"
    LLM_FAILED = "llm_failed"
    INVALID_PREDICTION = "invalid_prediction"  # NaN, Inf


@dataclass
class FallbackResult:
    """Result of a fallback decision."""
    label: int
    probability: float
    reason: str
    failure_type: FailureType


class FallbackPolicy:
    """
    Deterministic fallback policy for handling failures.
    
    All policies are predefined and evaluation-data independent.
    """
    
    # Default fallback label (0 = Normal, conservative choice)
    DEFAULT_LABEL = 0
    DEFAULT_PROBABILITY = 0.0
    
    @classmethod
    def handle_download_failure(cls, url: str) -> FallbackResult:
        """Handle image download failure after all retries exhausted."""
        logger.warning(f"Download failed for {url}, using fallback label")
        return FallbackResult(
            label=cls.DEFAULT_LABEL,
            probability=cls.DEFAULT_PROBABILITY,
            reason=f"Image download failed: {url}",
            failure_type=FailureType.DOWNLOAD_FAILED
        )
    
    @classmethod
    def handle_preprocessing_failure(cls, error: str) -> FallbackResult:
        """Handle preprocessing failure (corrupted image, etc.)."""
        logger.warning(f"Preprocessing failed: {error}, using fallback label")
        return FallbackResult(
            label=cls.DEFAULT_LABEL,
            probability=cls.DEFAULT_PROBABILITY,
            reason=f"Preprocessing failed: {error}",
            failure_type=FailureType.PREPROCESSING_FAILED
        )
    
    @classmethod
    def handle_model_inference_failure(cls, error: str) -> FallbackResult:
        """Handle model inference failure."""
        logger.error(f"Model inference failed: {error}, using fallback label")
        return FallbackResult(
            label=cls.DEFAULT_LABEL,
            probability=cls.DEFAULT_PROBABILITY,
            reason=f"Model inference failed: {error}",
            failure_type=FailureType.MODEL_INFERENCE_FAILED
        )
    
    @classmethod
    def handle_llm_failure(
        cls,
        p_local: float,
        threshold: float,
        error: str
    ) -> FallbackResult:
        """
        Handle LLM failure - fall back to local model prediction.
        
        Unlike other failures, we use the local model's prediction
        rather than the default conservative label.
        """
        logger.warning(f"LLM failed: {error}, using local model prediction only")
        label = 1 if p_local >= threshold else 0
        return FallbackResult(
            label=label,
            probability=p_local,
            reason=f"LLM failed, using local model: {error}",
            failure_type=FailureType.LLM_FAILED
        )
    
    @classmethod
    def handle_invalid_prediction(
        cls,
        prediction: float,
        error: str
    ) -> FallbackResult:
        """Handle invalid predictions (NaN, Inf)."""
        logger.error(f"Invalid prediction {prediction}: {error}")
        return FallbackResult(
            label=cls.DEFAULT_LABEL,
            probability=cls.DEFAULT_PROBABILITY,
            reason=f"Invalid prediction: {error}",
            failure_type=FailureType.INVALID_PREDICTION
        )
    
    @staticmethod
    def is_valid_probability(p: float) -> bool:
        """Check if a probability value is valid."""
        import math
        return not (math.isnan(p) or math.isinf(p)) and 0.0 <= p <= 1.0
