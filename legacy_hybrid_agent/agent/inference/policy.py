"""
AI Agent Policy for handling edge cases and defining clear rules.
Implements the decision policies required for a proper AI Agent flow.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DecisionOutcome(Enum):
    """Possible outcomes of the agent decision process."""
    HIGH_CONFIDENCE_LOCAL = "high_confidence_local"
    AMBIGUOUS_LLM_USED = "ambiguous_llm_used"
    AMBIGUOUS_LLM_IGNORED = "ambiguous_llm_ignored"
    AMBIGUOUS_RETRY_TTA = "ambiguous_retry_tta"
    STILL_AMBIGUOUS_CONSERVATIVE = "still_ambiguous_conservative"
    FALLBACK_ERROR = "fallback_error"


@dataclass
class AgentPolicy:
    """
    AI Agent Policy Configuration.
    Defines all the rules and thresholds for decision making.
    """
    # Ambiguity detection thresholds
    margin: float = 0.10
    tta_range_threshold: float = 0.20
    
    # LLM integration
    llm_confidence_gate: float = 0.65
    alpha: float = 0.7  # Local model weight in fusion
    
    # Retry and termination
    max_llm_retries: int = 2
    use_extended_tta_on_ambiguous: bool = True
    
    # Conservative fallback policy
    conservative_label: int = 0  # Fall back to "normal" when still ambiguous
    
    def is_ambiguous(
        self,
        p_local: float,
        threshold: float,
        tta_min: float = 0.0,
        tta_max: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Check if prediction is ambiguous based on policy rules.
        
        Returns:
            Tuple of (is_ambiguous, reason)
        """
        tta_range = tta_max - tta_min
        distance = abs(p_local - threshold)
        
        if distance <= self.margin:
            return True, f"margin_trigger (|{p_local:.3f} - {threshold:.3f}| = {distance:.3f} <= {self.margin})"
        
        if tta_range > self.tta_range_threshold:
            return True, f"variance_trigger (range={tta_range:.3f} > {self.tta_range_threshold})"
        
        return False, "high_confidence"
    
    def should_use_llm_result(
        self,
        llm_confidence: float
    ) -> Tuple[bool, str]:
        """
        Apply confidence gate to decide if LLM result should be used.
        
        Returns:
            Tuple of (should_use, reason)
        """
        if llm_confidence >= self.llm_confidence_gate:
            return True, f"confidence_ok ({llm_confidence:.2f} >= {self.llm_confidence_gate})"
        else:
            return False, f"confidence_low ({llm_confidence:.2f} < {self.llm_confidence_gate})"
    
    def fuse_predictions(
        self,
        p_local: float,
        p_llm: float
    ) -> float:
        """
        Fuse local and LLM predictions using alpha weighting.
        
        Formula: p_final = alpha * p_local + (1 - alpha) * p_llm
        """
        return self.alpha * p_local + (1 - self.alpha) * p_llm
    
    def is_still_ambiguous(
        self,
        p_final: float,
        threshold: float
    ) -> bool:
        """
        Check if final prediction is still ambiguous after LLM fusion.
        Uses a tighter margin (half of normal margin).
        """
        tight_margin = self.margin / 2
        return abs(p_final - threshold) <= tight_margin
    
    def get_conservative_decision(
        self,
        p_final: float,
        threshold: float
    ) -> Tuple[int, str]:
        """
        Make conservative decision for still-ambiguous cases.
        
        Policy options:
        1. Always return 0 (normal) - conservative
        2. Use extended TTA for final decision
        3. Trust the probability even if close
        
        Returns:
            Tuple of (label, reason)
        """
        # Current policy: Trust the probability but log that it was borderline
        if p_final >= threshold:
            return 1, f"borderline_abnormal (p={p_final:.3f} >= {threshold:.3f})"
        else:
            return 0, f"borderline_normal (p={p_final:.3f} < {threshold:.3f})"


class AgentTermination:
    """
    Defines clear termination conditions for the AI Agent.
    """
    
    @staticmethod
    def should_terminate(
        llm_attempts: int,
        max_attempts: int,
        has_valid_result: bool,
        error_occurred: bool
    ) -> Tuple[bool, str]:
        """
        Check if agent should terminate current sample processing.
        
        Termination conditions:
        1. Valid result obtained
        2. Max LLM attempts reached
        3. Unrecoverable error occurred
        
        Returns:
            Tuple of (should_terminate, reason)
        """
        if has_valid_result:
            return True, "valid_result_obtained"
        
        if llm_attempts >= max_attempts:
            return True, f"max_attempts_reached ({llm_attempts}/{max_attempts})"
        
        if error_occurred:
            return True, "unrecoverable_error"
        
        return False, "continue_processing"
    
    @staticmethod
    def get_loop_status(
        current_phase: str,
        samples_processed: int,
        total_samples: int
    ) -> dict:
        """
        Get current status of the agent processing loop.
        """
        return {
            "phase": current_phase,
            "processed": samples_processed,
            "total": total_samples,
            "remaining": total_samples - samples_processed,
            "progress_pct": (samples_processed / max(total_samples, 1)) * 100
        }


# Default policy instance
DEFAULT_POLICY = AgentPolicy()
