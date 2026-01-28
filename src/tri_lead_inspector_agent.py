"""
TriLeadInspectorAgent
=====================
The Final AI Agent for Transistor Inspection (DINOv2 + LLM)
"""
import torch
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
from PIL import Image

# Import Components
from dino_extractor import DinoFeatureExtractor
from reference_manager import ReferenceManager
from anomaly_scorer import AnomalyScorer
from policy import Policy
from luxia_mcp_client import LuxiaMCPClient

logger = logging.getLogger(__name__)

class TriLeadInspectorAgent:
    """
    DINOv2 + LLM 2-Stage Hybrid Agent
    
    Workflow:
    1. Preprocess: Resize & Normalize
    2. Stage 1: DINOv2 Feature Extractor -> (L2 norm) -> Features
    3. Scorer: Multi-Reference Cosine Distance (Min Dist to Centroids)
    4. Gate: PASS / REVIEW / FAIL_CAND
    5. Stage 2 (Conditional): LLM Logic Verification
    6. Final Decision
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Init Components
        self.feature_extractor = DinoFeatureExtractor(device=self.device)
        self.ref_manager = ReferenceManager(n_clusters=3)
        self.scorer = AnomalyScorer()
        self.policy = Policy()
        self.llm_client = LuxiaMCPClient()
        
        self.is_ready = False
        logger.info("TriLeadInspectorAgent initialized.")

    def auto_tune(self, normal_paths: List[str], abnormal_paths: List[str] = None):
        """
        자동 등록 및 튜닝 절차 (16장 학습)
        """
        logger.info("Beginning Auto-Tune...")
        
        # 1. Extract Features
        t0 = time.time()
        normal_feats = self.feature_extractor.extract_batch_features(normal_paths, normalize=True)
        if abnormal_paths:
            abnormal_feats = self.feature_extractor.extract_batch_features(abnormal_paths, normalize=True)
        else:
            abnormal_feats = None
        
        normal_ids = [Path(p).name for p in normal_paths]
        abnormal_ids = [Path(p).name for p in abnormal_paths] if abnormal_paths else []
        
        # 2. Fit Reference Manager (K-Means)
        self.ref_manager.fit(
            normal_urls=normal_paths,
            normal_ids=normal_ids,
            abnormal_urls=abnormal_paths or [],
            abnormal_ids=abnormal_ids,
            normal_features=normal_feats,
            abnormal_features=abnormal_feats
        )
        
        # 3. Fit Scorer (Centroids)
        centroids = self.ref_manager.get_centroids() # [K, D]
        self.scorer.fit(centroids)
        
        # 4. Compute Thresholds (T_low, T_high)
        # 중요: Centroid 기반 거리로 다시 Threshold 계산 필요?
        # -> AnomalyScorer.compute_threshold_bounds는 normal_vectors(centroids) 끼리의 거리를 잼.
        #    하지만 우리가 필요한 건 '개별 Normal Sample'과 'Centroids' 사이의 거리 분포임.
        
        dist_list = []
        for i in range(len(normal_paths)):
            sample_feat = normal_feats[i] # [D]
            dist = self.scorer.compute_score(sample_feat)
            dist_list.append(dist)
            
        import numpy as np
        dists = np.array(dist_list)
        t_low = float(np.percentile(dists, 95)) # 95% percentile
        t_high = float(np.percentile(dists, 99)) + 0.05 # Margin
        
        self.policy.set_gate_thresholds(t_low, t_high)
        
        logger.info(f"Auto-Tune Complete ({time.time()-t0:.2f}s)")
        logger.info(f"Thresholds: T_low={t_low:.4f}, T_high={t_high:.4f}")
        self.is_ready = True

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Agent Prediction Pipeline (LLM Orchestrator)
        Step 0: Triage (Plan)
        Step 1: Conditional Act (DINO / LLM)
        Step 2: Review (Multi-Expert)
        """
        if not self.is_ready:
            raise RuntimeError("Agent not tuned. Call auto_tune() first.")
            
        result = {
            'image_path': image_path,
            'timestamp': time.time(),
            'llm_called': False
        }
        
        # === Step 0: LLM Plan / Triage ===
        # Always call Triage to decide the best route
        try:
            triage_res = self.llm_client.run_triage(image_path)
            route = triage_res.get('route', 'DINO_THEN_LLM') # Default standard route
            result['triage'] = triage_res
            
            logger.info(f"[TRIAGE RESULT] {triage_res}") # Debug log

            # --- ROUTE LOGIC (Deterministic) ---
            # 0-1 issues -> DINO_THEN_LLM (Standard/Hybrid)
            # 2+ issues  -> LLM_STRICT (Safe)
            
            negatives = 0
            if triage_res.get('lead_visibility') == 'unclear': negatives += 1
            if triage_res.get('chip_size') == 'small': negatives += 1
            if triage_res.get('blur_or_noise') is True: negatives += 1
            if triage_res.get('exposure_issue') is True: negatives += 1
            if triage_res.get('background_dominant') is True: negatives += 1
            
            if negatives <= 1:
                route = 'DINO_THEN_LLM'
                logger.info(f"[Agent] Triage: Standard Quality ({negatives} issue) -> {route}")
            else:
                route = 'LLM_STRICT'
                logger.warning(f"[Agent] Triage: Bad Quality ({negatives} issues) -> {route}")
            # -----------------------------------
            
        except Exception as e:
            logger.error(f"Triage Failed: {e}")
            route = 'LLM_STRICT' # Fallback to strict mode
            result['triage_error'] = str(e)

        logger.info(f"[Agent] Triage Route: {route}")
        result['route'] = route

        # Pre-Load DINO Features (Used in most paths, so extract once)
        feat = self.feature_extractor.extract_features(image_path, normalize=True)
        score = self.scorer.compute_score(feat)
        result['dino_score'] = score
        
        # === Step 1: Conditional Execution based on Route ===
        
        should_run_llm_ensemble = False
        
        if route == 'DINO_THEN_LLM':
            # Standard Path: Pass -> Normal, Else -> LLM Review
            gate, gate_conf, gate_reason = self.policy.decide_stage1(score)
            
            if gate == 'PASS':
                result.update({
                    'final_decision': 'NORMAL',
                    'final_conf': gate_conf,
                    'final_reason': f"[DINO_THEN_LLM] {gate_reason}"
                })
            else:
                should_run_llm_ensemble = True
                
        elif route == 'LLM_STRICT':
            # Direct Path: Ignore DINO Pass, force LLM Review
            # (We still log DINO score but don't use it to auto-pass)
            should_run_llm_ensemble = True
            
        # === Step 2: Multi-Expert Verification (If routed) ===
        if should_run_llm_ensemble:
            normal_refs, abnormal_refs = self.ref_manager.get_best_refs(feat)
            
            try:
                # 1. Run Expert Ensemble (A, B, C)
                ensemble_results = self.llm_client.run_expert_ensemble(
                    query_image=image_path,
                    normal_refs=normal_refs,
                    abnormal_refs=abnormal_refs
                )
                
                # 2. Decide based on Ensemble
                # Pass 'FAIL_CAND' effectively as gate to force strict checks, 
                # or just reuse the logic from policy.
                final_decision, final_conf, final_reason = self.policy.decide_ensemble(ensemble_results, gate='FAIL_CAND')
                
                # 3. Arbitration (if needed)
                arbiter_res = None
                if final_decision == 'NEEDS_ARBITRATION':
                    logger.info("Ensemble Conflict -> Invoking Arbiter")
                    arbiter_res = self.llm_client.run_arbiter(
                        query_image=image_path,
                        normal_refs=normal_refs,
                        expert_results=ensemble_results
                    )
                    final_decision, final_conf, final_reason = self.policy.decide_arbiter(arbiter_res)
                
                result.update({
                    'final_decision': final_decision.upper(),
                    'final_conf': final_conf,
                    'final_reason': final_reason,
                    'llm_called': True,
                    'llm_ensemble': ensemble_results,
                    'llm_arbiter': arbiter_res
                })
                
            except Exception as e:
                logger.error(f"LLM Ensemble Failed: {e}")
                # Fallback: Safety First -> ABNORMAL
                result.update({
                    'final_decision': 'ABNORMAL',
                    'final_conf': 0.0,
                    'final_reason': f"LLM Error Fallback: {e}",
                    'llm_called': True,
                    'error': str(e)
                })

        return result
