"""
Main inference predictor implementing complete AI Agent Flow.

Agent Flow Phases:
1. 관찰 (Observation): Download and preprocess image
2. 판단 (Decision): Local model + TTA inference
3. 검증/보정 (Verification): Ambiguity check → LLM if needed → Fusion
4. 출력 (Output): Final label with confidence

Clear distinction from single LLM call:
- Multi-phase decision pipeline
- Conditional LLM invocation (only for ambiguous cases)
- Policy-based action selection
- Termination conditions
- Retry logic for failures
"""

import torch
import logging
from PIL import Image
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import numpy as np

from ..config import Config, get_config
from ..models.classifier import BinaryClassifier
from ..data.preprocess import preprocess_image
from ..data.augmentation import TTAAugmentation, ExtendedTTAAugmentation
from ..llm.client import LLMClient
from .policy import AgentPolicy, AgentTermination, DecisionOutcome, DEFAULT_POLICY
from .fallback import FallbackPolicy, FallbackResult

logger = logging.getLogger(__name__)


class AgentPredictor:
    """
    AI Agent Predictor - Complete Agent Flow Implementation.
    
    This is NOT a simple LLM wrapper. It implements a multi-phase
    decision pipeline with clear termination conditions and policies.
    
    Flow:
    ┌──────────────────────────────────────────────────────────────┐
    │  Phase 1: OBSERVATION                                        │
    │  - Load image                                                 │
    │  - Validate & preprocess                                      │
    └──────────────────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  Phase 2: LOCAL DECISION                                     │
    │  - Run local model with TTA (4 augmentations)                │
    │  - Calculate mean probability and variance                    │
    └──────────────────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  Phase 3: AMBIGUITY CHECK (Policy-based)                     │
    │  - Check margin: |p - threshold| <= margin?                  │
    │  - Check TTA variance: max-min > threshold?                  │
    │  - If NOT ambiguous → Go to Phase 5                          │
    │  - If ambiguous → Go to Phase 4                              │
    └──────────────────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  Phase 4: LLM CONSULTATION (Conditional)                     │
    │  - Call external LLM with image + prompt                     │
    │  - Apply Confidence Gate (>= 0.65?)                          │
    │  - If confidence low → Use Extended TTA instead              │
    │  - Fuse predictions: p_final = α*p_local + (1-α)*p_llm       │
    │  - Check if still ambiguous → Conservative decision          │
    └──────────────────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  Phase 5: OUTPUT                                             │
    │  - Apply threshold: label = 1 if p >= threshold else 0       │
    │  - Record outcome and statistics                              │
    └──────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        model: BinaryClassifier,
        config: Optional[Config] = None,
        device: str = "cpu",
        use_llm: bool = True,
        policy: Optional[AgentPolicy] = None
    ):
        self.model = model
        self.config = config or get_config()
        self.device = device
        self.use_llm = use_llm
        self.policy = policy or DEFAULT_POLICY
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            model=self.config.llm.model,
            max_retries=self.config.llm.max_retries,
            timeout=self.config.llm.timeout
        ) if use_llm else None
        
        # TTA augmentations
        self.tta = TTAAugmentation(self.config)
        self.extended_tta = ExtendedTTAAugmentation(self.config)
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Statistics tracking
        self.stats = {
            "total": 0,
            "high_confidence": 0,
            "ambiguous_margin": 0,
            "ambiguous_variance": 0,
            "llm_invoked": 0,
            "llm_used": 0,
            "llm_ignored_low_conf": 0,
            "llm_failed": 0,
            "still_ambiguous_after_llm": 0,
            "extended_tta_used": 0,
            "fallbacks": 0
        }
        
        # Decision log for debugging
        self.decision_log = []
    
    def _phase1_observation(
        self,
        image: Image.Image
    ) -> Tuple[Image.Image, Optional[str]]:
        """
        Phase 1: Observation - Validate and preprocess image.
        """
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image, None
        except Exception as e:
            return None, f"Image preprocessing failed: {e}"
    
    def _phase2_local_decision(
        self,
        image: Image.Image,
        extended: bool = False
    ) -> Tuple[float, float, float, float]:
        """
        Phase 2: Local Decision - Run model with TTA.
        
        Returns:
            Tuple of (mean_prob, variance, min_prob, max_prob)
        """
        self.model.eval()
        tta_module = self.extended_tta if extended else self.tta
        
        with torch.no_grad():
            batch = tta_module.get_augmented_batch(image)
            batch = batch.to(self.device)
            probs = self.model.predict_proba(batch).cpu().numpy()
            
            mean_prob = float(np.mean(probs))
            variance = float(np.var(probs))
            min_prob = float(np.min(probs))
            max_prob = float(np.max(probs))
        
        return mean_prob, variance, min_prob, max_prob
    
    def _phase3_ambiguity_check(
        self,
        p_local: float,
        threshold: float,
        tta_min: float,
        tta_max: float
    ) -> Tuple[bool, str]:
        """
        Phase 3: Ambiguity Check - Decide if LLM is needed.
        """
        return self.policy.is_ambiguous(p_local, threshold, tta_min, tta_max)
    
    def _phase4_llm_consultation(
        self,
        image: Image.Image,
        p_local: float,
        threshold: float
    ) -> Tuple[float, DecisionOutcome, str]:
        """
        Phase 4: LLM Consultation - Get LLM prediction and fuse.
        
        Returns:
            Tuple of (final_probability, outcome, reason)
        """
        self.stats["llm_invoked"] += 1
        
        # Step 4a: Call LLM
        if not self.llm_client or not self.llm_client.is_available():
            # LLM not available → Use extended TTA
            self.stats["extended_tta_used"] += 1
            p_ext, _, _, _ = self._phase2_local_decision(image, extended=True)
            return p_ext, DecisionOutcome.AMBIGUOUS_RETRY_TTA, "llm_unavailable_used_extended_tta"
        
        label_llm, confidence_llm, error = self.llm_client.analyze_image(image)
        
        if error is not None:
            # LLM failed → Use extended TTA
            self.stats["llm_failed"] += 1
            self.stats["extended_tta_used"] += 1
            p_ext, _, _, _ = self._phase2_local_decision(image, extended=True)
            return p_ext, DecisionOutcome.AMBIGUOUS_RETRY_TTA, f"llm_error:{error}"
        
        # Step 4b: Apply Confidence Gate
        should_use, gate_reason = self.policy.should_use_llm_result(confidence_llm)
        
        if not should_use:
            # LLM confidence too low → Use extended TTA
            self.stats["llm_ignored_low_conf"] += 1
            self.stats["extended_tta_used"] += 1
            p_ext, _, _, _ = self._phase2_local_decision(image, extended=True)
            return p_ext, DecisionOutcome.AMBIGUOUS_LLM_IGNORED, gate_reason
        
        # Step 4c: Convert LLM label to probability
        p_llm = confidence_llm if label_llm == 1 else (1.0 - confidence_llm)
        
        # Step 4d: Fuse predictions
        self.stats["llm_used"] += 1
        p_final = self.policy.fuse_predictions(p_local, p_llm)
        
        logger.info(f"LLM Fusion: local={p_local:.3f}, llm={p_llm:.3f}, final={p_final:.3f}")
        
        # Step 4e: Check if still ambiguous
        if self.policy.is_still_ambiguous(p_final, threshold):
            self.stats["still_ambiguous_after_llm"] += 1
            # Apply conservative decision
            label, reason = self.policy.get_conservative_decision(p_final, threshold)
            return p_final, DecisionOutcome.STILL_AMBIGUOUS_CONSERVATIVE, reason
        
        return p_final, DecisionOutcome.AMBIGUOUS_LLM_USED, f"llm_fused (p_llm={p_llm:.3f})"
    
    def _phase5_output(
        self,
        p_final: float,
        threshold: float
    ) -> int:
        """
        Phase 5: Output - Apply threshold to get final label.
        """
        return 1 if p_final >= threshold else 0
    
    def predict_single(
        self,
        image: Image.Image
    ) -> Tuple[int, float, DecisionOutcome, Optional[FallbackResult]]:
        """
        Run complete agent flow for a single image.
        
        Returns:
            Tuple of (label, probability, outcome, fallback_result)
        """
        threshold = self.config.threshold
        self.stats["total"] += 1
        
        # Phase 1: Observation
        image, error = self._phase1_observation(image)
        if error:
            self.stats["fallbacks"] += 1
            fallback = FallbackPolicy.handle_preprocessing_failure(error)
            return fallback.label, fallback.probability, DecisionOutcome.FALLBACK_ERROR, fallback
        
        # Phase 2: Local Decision
        try:
            p_local, tta_var, tta_min, tta_max = self._phase2_local_decision(image)
            
            if not FallbackPolicy.is_valid_probability(p_local):
                self.stats["fallbacks"] += 1
                fallback = FallbackPolicy.handle_invalid_prediction(p_local, "NaN/Inf")
                return fallback.label, fallback.probability, DecisionOutcome.FALLBACK_ERROR, fallback
                
        except Exception as e:
            self.stats["fallbacks"] += 1
            fallback = FallbackPolicy.handle_model_inference_failure(str(e))
            return fallback.label, fallback.probability, DecisionOutcome.FALLBACK_ERROR, fallback
        
        # Phase 3: Ambiguity Check
        is_ambiguous, ambiguity_reason = self._phase3_ambiguity_check(
            p_local, threshold, tta_min, tta_max
        )
        
        if not is_ambiguous:
            # High confidence → Direct output
            self.stats["high_confidence"] += 1
            label = self._phase5_output(p_local, threshold)
            
            self.decision_log.append({
                "p_local": p_local,
                "outcome": DecisionOutcome.HIGH_CONFIDENCE_LOCAL,
                "label": label
            })
            
            return label, p_local, DecisionOutcome.HIGH_CONFIDENCE_LOCAL, None
        
        # Track ambiguity type
        if "margin" in ambiguity_reason:
            self.stats["ambiguous_margin"] += 1
        else:
            self.stats["ambiguous_variance"] += 1
        
        logger.info(f"Ambiguous sample: {ambiguity_reason}")
        
        # Phase 4: LLM Consultation (only if ambiguous)
        if self.use_llm:
            p_final, outcome, reason = self._phase4_llm_consultation(
                image, p_local, threshold
            )
        else:
            # No LLM → Use extended TTA
            self.stats["extended_tta_used"] += 1
            p_final, _, _, _ = self._phase2_local_decision(image, extended=True)
            outcome = DecisionOutcome.AMBIGUOUS_RETRY_TTA
            reason = "llm_disabled_used_extended_tta"
        
        # Phase 5: Output
        label = self._phase5_output(p_final, threshold)
        
        self.decision_log.append({
            "p_local": p_local,
            "p_final": p_final,
            "outcome": outcome,
            "reason": reason,
            "label": label
        })
        
        return label, p_final, outcome, None
    
    def predict_batch(
        self,
        images: List[Image.Image],
        show_progress: bool = True
    ) -> List[Tuple[int, float, DecisionOutcome, Optional[FallbackResult]]]:
        """
        Run agent flow for a batch of images.
        Implements the automation requirement: process 100 images without human intervention.
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(images, desc="Agent Processing") if show_progress else images
        
        for i, image in enumerate(iterator):
            result = self.predict_single(image)
            results.append(result)
            
            # Log progress
            status = AgentTermination.get_loop_status(
                "inference", i + 1, len(images)
            )
            
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {status['processed']}/{status['total']} "
                    f"({status['progress_pct']:.1f}%)"
                )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get prediction statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset prediction statistics."""
        self.stats = {k: 0 for k in self.stats}
        self.decision_log = []
    
    def print_stats(self):
        """Print detailed prediction statistics."""
        logger.info("=" * 60)
        logger.info("AI Agent Prediction Statistics")
        logger.info("=" * 60)
        logger.info(f"  Total samples: {self.stats['total']}")
        logger.info("-" * 40)
        logger.info("Decision Distribution:")
        logger.info(f"  High confidence (local only): {self.stats['high_confidence']}")
        logger.info(f"  Ambiguous (margin): {self.stats['ambiguous_margin']}")
        logger.info(f"  Ambiguous (variance): {self.stats['ambiguous_variance']}")
        logger.info("-" * 40)
        logger.info("LLM Usage:")
        logger.info(f"  LLM invoked: {self.stats['llm_invoked']}")
        logger.info(f"  LLM used (fused): {self.stats['llm_used']}")
        logger.info(f"  LLM ignored (low conf): {self.stats['llm_ignored_low_conf']}")
        logger.info(f"  LLM failed: {self.stats['llm_failed']}")
        logger.info(f"  Still ambiguous after LLM: {self.stats['still_ambiguous_after_llm']}")
        logger.info("-" * 40)
        logger.info("Fallbacks:")
        logger.info(f"  Extended TTA used: {self.stats['extended_tta_used']}")
        logger.info(f"  Error fallbacks: {self.stats['fallbacks']}")
        logger.info("=" * 60)
    
    @classmethod
    def load(
        cls,
        checkpoint_dir: str,
        device: str = "cpu",
        use_llm: bool = True
    ) -> "AgentPredictor":
        """Load agent predictor from checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        
        config = Config.load(checkpoint_path / "config.yaml")
        model_path = checkpoint_path / config.model.checkpoint
        model = BinaryClassifier.load(str(model_path), device)
        
        return cls(model, config, device, use_llm)


# Alias for backward compatibility
Predictor = AgentPredictor
