"""
TransistorInspectorAgent - 2-Stage Anomaly Detection
=====================================================
Multi-Reference DINOv2 + LLM 기반 2단계 검증 에이전트

핵심 변경:
- EfficientNet (2차 필터) 제거
- 3구간 게이팅: PASS / REVIEW / FAIL_CAND
- Cosine Distance 사용 (L2 정규화 필수)
- 점수 범위: 0~2 (기존 3000~4000 → 0.x로 개선)
"""
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CACHE_DIR, OUTPUT_DIR, PARAMS_CACHE, NORMAL_VECTORS_CACHE,
    REFERENCE_CACHE,
    DINO_NORMAL_PERCENTILE, DINO_ABNORMAL_PERCENTILE,
    LUXIA_API_KEY, NORMAL_REF_COUNT, ABNORMAL_REF_COUNT
)

from .dino_extractor import DinoFeatureExtractor
from .anomaly_scorer import AnomalyScorer
from .luxia_mcp_client import LuxiaMCPClient
from .reference_manager import ReferenceManager
from .policy import Policy

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent:
    """
    TransistorInspectorAgent - 2-Stage 3-Gate 이상탐지 시스템
    
    "정상 16장으로 정상 범주를 Multi-Reference로 정의하고,
     DINOv2로 대량 처리(PASS 확대),
     LLM은 필요한 경우(REVIEW/FAIL_CAND)만 개입"
    
    2단계 Funnel:
    Stage 1: DINOv2 Multi-Reference (Cosine Distance)
      - PASS: 정상 확정, LLM 차단
      - REVIEW: LLM 검증 필요 (애매한 영역)
      - FAIL_CAND: 비정상 후보, LLM으로 근거/유형 기록
    
    Stage 2: LLM Reviewer
      - 체크리스트 기반 검증 (lead_ok, contamination 등)
      - 최종 라벨 + severity + reason 반환
    """
    
    def __init__(self):
        # Tools
        self.dino: Optional[DinoFeatureExtractor] = None
        self.scorer: Optional[AnomalyScorer] = None
        self.llm_client: Optional[LuxiaMCPClient] = None
        self.ref_manager: Optional[ReferenceManager] = None
        
        # Policy
        self.policy = Policy()
        
        # State
        self._tools_loaded = False
        self._trained = False
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'pass_count': 0,
            'review_count': 0,
            'fail_cand_count': 0,
            'llm_calls': 0,
        }
        
        logger.info(f"TransistorInspectorAgent initialized on {self.device}")
        logger.info("  - Architecture: 2-Stage (DINOv2 Cosine + LLM)")
        logger.info("  - Gating: PASS / REVIEW / FAIL_CAND")
    
    def _load_tools(self, load_llm: bool = False):
        """도구 로딩"""
        if not self._tools_loaded:
            logger.info("🔧 Loading tools...")
            
            # Stage 1: DINOv2 + Cosine Scorer
            self.dino = DinoFeatureExtractor(device=self.device)
            self.scorer = AnomalyScorer()
            
            # Reference Manager (for LLM comparison)
            self.ref_manager = ReferenceManager()
            
            self._tools_loaded = True
            logger.info("✓ Tools loaded (DINOv2 + Cosine Scorer)")
        
        # Stage 2: LLM (lazy loading)
        if load_llm and self.llm_client is None:
            if LUXIA_API_KEY:
                self.llm_client = LuxiaMCPClient()
                logger.info("✓ Luxia LLM client loaded")
            else:
                logger.warning("⚠️ LUXIA_API_KEY not set. LLM stage will be skipped.")
    
    def train(
        self,
        train_df: pd.DataFrame,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        정상 이미지로 정상 공간 학습 + 3-Gate threshold 계산
        
        Args:
            train_df: id, img_url, label 컬럼 포함
            force_retrain: True면 캐시 무시하고 재학습
        
        Returns:
            학습 결과 (t_low, t_high, normal_distance_stats 등)
        """
        logger.info("=" * 60)
        logger.info("📚 Training Started (2-Stage Cosine Distance)")
        logger.info("=" * 60)
        
        self._load_tools(load_llm=False)
        
        # Check cache
        if not force_retrain and self._try_load_cache():
            logger.info("📦 Loaded from cache")
            return self._get_cached_params()
        
        # Split by label
        normal_df = train_df[train_df['label'] == 0]
        defect_df = train_df[train_df['label'] == 1]
        
        logger.info(f"Normal images: {len(normal_df)}, Defect images: {len(defect_df)}")
        
        # Step 1: Extract all features (with L2 normalization)
        logger.info("Extracting features (L2 normalized)...")
        all_features = {}
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Feature extraction"):
            features = self.dino.extract_features(row['img_url'])  # Already L2 normalized
            all_features[row['id']] = {
                'features': features,
                'label': row['label'],
                'url': row['img_url']
            }
        
        # Step 2: Fit normal space with normalized vectors
        normal_features = torch.cat([
            all_features[idx]['features']
            for idx in normal_df['id'].tolist()
        ], dim=0)
        
        self.scorer.fit(normal_features)
        
        # Step 3: Compute 3-Gate thresholds
        t_low, t_high = self.scorer.compute_threshold_bounds(
            DINO_NORMAL_PERCENTILE, DINO_ABNORMAL_PERCENTILE
        )
        self.policy.set_gate_thresholds(t_low, t_high)
        
        # Step 4: Set up reference manager
        logger.info("Setting up reference manager...")
        central_indices = self.scorer.find_central_samples(n=5)
        normal_urls = normal_df['img_url'].tolist()
        normal_ids = normal_df['id'].tolist()
        abnormal_urls = defect_df['img_url'].tolist()
        abnormal_ids = defect_df['id'].tolist()
        
        # 비정상 특징 벡터
        abnormal_features = None
        if not defect_df.empty:
            abnormal_features = torch.cat([
                all_features[idx]['features']
                for idx in abnormal_ids
            ], dim=0)

        self.ref_manager.fit(
            normal_urls, normal_ids,
            abnormal_urls, abnormal_ids,
            normal_features=normal_features,
            abnormal_features=abnormal_features,
            central_indices=central_indices
        )
        
        # Step 5: Evaluate on training set
        logger.info("Evaluating with 3-Gate...")
        results = self._evaluate_training(train_df, all_features)
        
        # Save cache
        self._save_cache(results)
        
        self._trained = True
        
        logger.info("=" * 60)
        logger.info("✓ Training Complete! (Cosine Distance)")
        logger.info(f"  - T_low (PASS if <=): {t_low:.4f}")
        logger.info(f"  - T_high (FAIL_CAND if >=): {t_high:.4f}")
        logger.info(f"  - Normal distance stats: {self.scorer.normal_distance_stats}")
        logger.info("=" * 60)
        
        return results
    
    def _evaluate_training(
        self,
        train_df: pd.DataFrame,
        all_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """학습 데이터 평가 (3-Gate 기준)"""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        y_true = []
        y_pred = []
        gate_counts = {'PASS': 0, 'REVIEW': 0, 'FAIL_CAND': 0}
        
        for _, row in train_df.iterrows():
            features = all_features[row['id']]['features']
            score = self.scorer.compute_score(features)
            gate, conf, reason = self.policy.decide_stage1(score)
            
            gate_counts[gate] += 1
            
            y_true.append(row['label'])
            # PASS → 0 (normal), REVIEW/FAIL_CAND → 1 (abnormal candidate)
            y_pred.append(0 if gate == 'PASS' else 1)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            't_low': self.policy.t_low,
            't_high': self.policy.t_high,
            'gate_counts': gate_counts,
            'normal_distance_stats': self.scorer.normal_distance_stats,
            'trained_at': datetime.now().isoformat()
        }
    
    def predict(
        self,
        image_url: str,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        단일 이미지 예측 (2-Stage 3-Gate)
        
        Args:
            image_url: 이미지 URL 또는 경로
            use_llm: True면 REVIEW/FAIL_CAND에서 LLM 사용
        
        Returns:
            {
                'label': 0 or 1,
                'decision': 'normal' or 'abnormal',
                'gate': 'PASS', 'REVIEW', or 'FAIL_CAND',
                'route': 'PASS', 'REVIEW', 'FAIL_CAND', or 'NEEDS_REVIEW',
                'confidence': float,
                'dino_score': float,
                'llm_used': bool,
                'reason': str
            }
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        result = {
            'image_url': image_url,
            'dino_score': None,
            'llm_decision': None,
            'llm_used': False,
        }
        
        # ===== Stage 1: DINOv2 Cosine Distance =====
        features = self.dino.extract_features(image_url)
        dino_score = self.scorer.compute_score(features)
        result['dino_score'] = round(dino_score, 4)
        
        gate, conf, reason = self.policy.decide_stage1(dino_score)
        result['gate'] = gate
        
        # PASS → 즉시 정상 확정 (LLM 차단)
        if gate == 'PASS':
            self.stats['pass_count'] += 1
            result.update({
                'label': 0,
                'decision': 'normal',
                'funnel_stage': 1,
                'route': 'PASS',
                'confidence': round(conf, 4),
                'reason': reason
            })
            return result
        
        # ===== Stage 2: LLM Reviewer (REVIEW or FAIL_CAND) =====
        self.stats['review_count' if gate == 'REVIEW' else 'fail_cand_count'] += 1
        
        if use_llm and self.llm_client is not None:
            # RAG for Vision: 쿼리와 가장 유사한 레퍼런스 추출
            best_normal, best_abnormal = self.ref_manager.get_best_refs(
                features, 
                top_n_normal=NORMAL_REF_COUNT, 
                top_n_abnormal=ABNORMAL_REF_COUNT
            )
            
            # 가장 가까운 정상 레퍼런스 ID 기록
            nearest_idx, nearest_dist = self.scorer.find_nearest_reference(features)
            
            llm_result = self.llm_client.compare_and_decide(
                query_image=image_url,
                normal_refs=best_normal,
                abnormal_refs=best_abnormal
            )
            
            self.stats['llm_calls'] += 1
            result['llm_used'] = True
            result['llm_decision'] = llm_result.get('decision')
            result['llm_confidence'] = llm_result.get('confidence')
            result['llm_evidence'] = llm_result.get('visual_evidence', [])
            result['nearest_reference_idx'] = nearest_idx
            
            decision, conf, reason = self.policy.decide_llm(llm_result, gate)
            
            result.update({
                'label': self.policy.get_final_label(decision),
                'decision': decision,
                'funnel_stage': 2,
                'route': gate,
                'confidence': round(conf, 4),
                'reason': reason
            })
            return result
        
        # LLM 사용 불가 → 보수적 판정
        result.update({
            'label': 1,  # Conservative: assume abnormal
            'decision': 'abnormal',
            'funnel_stage': 1,
            'route': 'NEEDS_REVIEW',
            'confidence': 0.5,
            'reason': f'{gate} but LLM unavailable → conservative ABNORMAL'
        })
        return result
    
    def predict_batch(
        self,
        test_df: pd.DataFrame,
        use_llm: bool = True
    ) -> pd.DataFrame:
        """
        배치 예측
        
        Args:
            test_df: id, img_url 컬럼 포함
            use_llm: LLM 사용 여부
        
        Returns:
            예측 결과 DataFrame
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if use_llm:
            self._load_tools(load_llm=True)
        
        # Reset stats
        self.stats = {
            'total_samples': len(test_df),
            'pass_count': 0,
            'review_count': 0,
            'fail_cand_count': 0,
            'llm_calls': 0,
        }
        
        results = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            try:
                pred = self.predict(row['img_url'], use_llm=use_llm)
                pred['id'] = row['id']
                results.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {row['id']}: {e}")
                results.append({
                    'id': row['id'],
                    'label': 1,  # Conservative
                    'decision': 'abnormal',
                    'gate': 'ERROR',
                    'route': 'ERROR',
                    'funnel_stage': 0,
                    'confidence': 0,
                    'reason': f'Error: {str(e)}'
                })
        
        # Log statistics
        logger.info(f"\n📊 Prediction Statistics:")
        logger.info(f"  - Total: {self.stats['total_samples']}")
        logger.info(f"  - PASS (no LLM): {self.stats['pass_count']} ({self.stats['pass_count']/self.stats['total_samples']*100:.1f}%)")
        logger.info(f"  - REVIEW: {self.stats['review_count']}")
        logger.info(f"  - FAIL_CAND: {self.stats['fail_cand_count']}")
        logger.info(f"  - LLM calls: {self.stats['llm_calls']} ({self.stats['llm_calls']/self.stats['total_samples']*100:.1f}%)")
        
        return pd.DataFrame(results)
    
    def evaluate(
        self,
        eval_df: pd.DataFrame,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        평가 모드: 레이블이 있는 데이터로 성능 평가
        """
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
        
        if use_llm:
            self._load_tools(load_llm=True)
        
        predictions = self.predict_batch(eval_df, use_llm=use_llm)
        
        # Merge with actual labels
        predictions = predictions.merge(
            eval_df[['id', 'label']].rename(columns={'label': 'actual'}),
            on='id'
        )
        
        y_true = predictions['actual'].tolist()
        y_pred = predictions['label'].tolist()
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Gate statistics
        gate_counts = predictions['gate'].value_counts().to_dict() if 'gate' in predictions.columns else {}
        
        return {
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': cm.tolist(),
            'gate_counts': gate_counts,
            'stage_counts': predictions['funnel_stage'].value_counts().to_dict() if 'funnel_stage' in predictions.columns else {},
            'predictions': predictions,
            'stats': self.stats
        }
    
    def validate_all_stages(
        self,
        eval_df: pd.DataFrame,
        use_llm: bool = False
    ) -> pd.DataFrame:
        """
        모든 샘플에 대해 Stage 1, (Stage 2) 결과 수집
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if use_llm:
            self._load_tools(load_llm=True)
        
        results = []
        
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Validating all stages"):
            try:
                record = {
                    'id': row['id'],
                    'actual': row.get('label', None),
                    'img_url': row['img_url']
                }
                
                # === Stage 1: DINOv2 Cosine ===
                features = self.dino.extract_features(row['img_url'])
                dino_score = self.scorer.compute_score(features)
                gate, s1_conf, s1_reason = self.policy.decide_stage1(dino_score)
                
                record['dino_score'] = round(dino_score, 4)
                record['gate'] = gate
                record['stage1_conf'] = round(s1_conf, 4)
                
                # Nearest reference
                nearest_idx, nearest_dist = self.scorer.find_nearest_reference(features)
                record['nearest_ref_idx'] = nearest_idx
                record['nearest_ref_dist'] = round(nearest_dist, 4)
                
                # === Stage 2: LLM (선택) ===
                if use_llm and self.llm_client is not None and gate != 'PASS':
                    best_normal, best_abnormal = self.ref_manager.get_best_refs(
                        features,
                        top_n_normal=NORMAL_REF_COUNT,
                        top_n_abnormal=ABNORMAL_REF_COUNT
                    )
                    
                    llm_result = self.llm_client.compare_and_decide(
                        query_image=row['img_url'],
                        normal_refs=best_normal,
                        abnormal_refs=best_abnormal
                    )
                    
                    s2_decision, s2_conf, s2_reason = self.policy.decide_llm(llm_result, gate)
                    
                    record['llm_decision'] = llm_result.get('decision')
                    record['llm_confidence'] = llm_result.get('confidence')
                    record['stage2_decision'] = s2_decision
                    record['stage2_conf'] = round(s2_conf, 4)
                    record['llm_used'] = True
                else:
                    record['llm_used'] = False
                
                # === Final decision ===
                if gate == 'PASS':
                    final_decision = 'normal'
                    final_stage = 1
                elif record.get('llm_used') and 'stage2_decision' in record:
                    final_decision = record['stage2_decision']
                    final_stage = 2
                else:
                    final_decision = 'abnormal'  # Conservative
                    final_stage = 1
                
                record['final_decision'] = final_decision
                record['final_label'] = self.policy.get_final_label(final_decision)
                record['final_stage'] = final_stage
                
                results.append(record)
                
            except Exception as e:
                logger.error(f"Validation failed for {row['id']}: {e}")
                results.append({
                    'id': row['id'],
                    'actual': row.get('label', None),
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def _try_load_cache(self) -> bool:
        """캐시 로드 시도"""
        if not PARAMS_CACHE.exists():
            return False
        
        try:
            self.scorer.load_cache(NORMAL_VECTORS_CACHE)
            self.ref_manager.load_cache(REFERENCE_CACHE)
            
            params = json.loads(PARAMS_CACHE.read_text())
            self.policy.set_gate_thresholds(
                params.get('t_low', params.get('dino_normal_upper')),
                params.get('t_high', params.get('dino_abnormal_lower'))
            )
            
            self._trained = True
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return False
    
    def _get_cached_params(self) -> Dict[str, Any]:
        """캐시된 파라미터 반환"""
        return json.loads(PARAMS_CACHE.read_text())
    
    def _save_cache(self, results: Dict[str, Any]):
        """캐시 저장"""
        self.scorer.save_cache(NORMAL_VECTORS_CACHE)
        self.ref_manager.save_cache(REFERENCE_CACHE)
        
        PARAMS_CACHE.write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"Cache saved to {CACHE_DIR}")


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = AnomalyDetectionAgent()
    print("TransistorInspectorAgent ready (2-Stage 3-Gate)")
