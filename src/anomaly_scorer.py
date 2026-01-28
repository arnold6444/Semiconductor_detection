"""
Stage 1: Anomaly Scoring (Cosine Distance)
============================================
Calculates anomaly scores based on cosine distance from normal feature space.
Uses L2-normalized embeddings for stable scoring (0-2 range instead of 3000+).
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANOMALY_THRESHOLD, K_NEIGHBORS, USE_MAHALANOBIS,
    NORMAL_VECTORS_CACHE, CACHE_DIR
)

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """
    정상 공간 기반 이상 점수 계산기 (Cosine Distance 버전)
    
    핵심 변경:
    - L2 거리 → Cosine Distance (1 - cosine_similarity)
    - 점수 범위: 0 ~ 2 (0에 가까울수록 정상, 1 이상이면 이상)
    - L2 정규화된 임베딩 필수
    
    3구간 게이팅:
    - PASS: score <= T_low (정상 확정, LLM 차단)
    - REVIEW: T_low < score < T_high (LLM 검증)
    - FAIL_CAND: score >= T_high (비정상 후보)
    """
    
    def __init__(
        self,
        threshold: float = None,
        k_neighbors: int = None,
        use_mahalanobis: bool = None
    ):
        self.threshold = threshold or ANOMALY_THRESHOLD
        self.k_neighbors = k_neighbors or K_NEIGHBORS
        # Mahalanobis는 cosine distance에서는 사용 안 함 (호환성 유지용)
        self.use_mahalanobis = False  # Cosine distance 고정
        
        # Normal space statistics
        self.normal_vectors: Optional[torch.Tensor] = None  # L2 정규화된 상태
        self.mean: Optional[torch.Tensor] = None
        
        # Threshold bounds for 3-gate decision
        self.t_low: Optional[float] = None   # 이하 = PASS
        self.t_high: Optional[float] = None  # 이상 = FAIL_CAND
        
        # 진단용: 정상끼리 거리 분포
        self.normal_distance_stats: Dict[str, float] = {}
        
        logger.info(f"AnomalyScorer initialized (Cosine Distance mode): k={self.k_neighbors}")
    
    def fit(self, normal_vectors: torch.Tensor):
        """
        정상 이미지 벡터 로드 (Centroids)
        
        Args:
            normal_vectors: ReferenceManager에서 계산된 Centroids [K, 384]
        """
        # L2 정규화 강제 (Double Check)
        self.normal_vectors = F.normalize(normal_vectors, p=2, dim=-1)
        self.n_refs = self.normal_vectors.shape[0]
        
        logger.info(f"AnomalyScorer fitted with {self.n_refs} references (Centroids)")
    
    def compute_cosine_distance(self, test_vector: torch.Tensor) -> float:
        """
        Multi-Reference Cosine Distance
        
        score = min( dist(test, Ref1), dist(test, Ref2), ... )
              = min( 1 - sim(test, Ref1), ... )
        
        Args:
            test_vector: [1, 384] or [384]
            
        Returns:
            score (0.0 ~ 2.0)
        """
        if self.normal_vectors is None:
            raise ValueError("Scorer not fitted")
            
        # 1. Input Normalize
        if test_vector.dim() == 1:
            test_vector = test_vector.unsqueeze(0)
        z_new = F.normalize(test_vector, p=2, dim=-1) # [1, D]
        
        # 2. Similarity with all Normal References (Centroids)
        # z_refs: [K, D] -> sim: [1, K]
        sims = torch.mm(z_new, self.normal_vectors.t())
        
        # 3. Distance = 1 - Similarity
        dists = 1.0 - sims
        
        # 4. Multi-Reference Score = Min Distance
        score = dists.min().item()
        
        # 5. ★ Structural Safety Check (Fail-safe)
        if score < -1e-6 or score > 2.0 + 1e-6:
             # 이론상 불가능 (Cosine Distance는 0~2)
             # 수치 오차 감안해도 -1e-6 ~ 2.000001
             logger.critical(f"INVALID SCORE DETECTED: {score}")
             raise RuntimeError(f"Score explosion detected! value={score}")

        return max(0.0, min(2.0, score)) # Clamp purely for floating point safety
    
    def compute_score(self, test_vector: torch.Tensor) -> float:
        return self.compute_cosine_distance(test_vector)
    
    def compute_threshold_bounds(self, t_low_percentile: float = 70, t_high_percentile: float = 95):
        """
        정상 샘플 점수 분포 기반 3-Gate threshold 계산 (LOO)
        
        Args:
            t_low_percentile: 이 percentile 이하 = PASS (정상 확정)
            t_high_percentile: 이 percentile 이상 = FAIL_CAND (비정상 후보)
        
        Returns:
            (t_low, t_high)
        """
        if self.normal_vectors is None:
            raise ValueError("Normal space not fitted.")
        
        # Leave-one-out 점수 계산
        loo_scores = []
        for i in range(self.normal_vectors.shape[0]):
            mask = torch.arange(self.normal_vectors.shape[0]) != i
            temp_vectors = self.normal_vectors[mask]
            test_vec = self.normal_vectors[i:i+1]  # Keep 2D
            
            # Cosine similarity with remaining normals
            similarities = test_vec @ temp_vectors.T
            k = min(self.k_neighbors, temp_vectors.shape[0])
            topk_sims, _ = torch.topk(similarities[0], k, largest=True)
            
            # Cosine distance
            loo_scores.append((1.0 - topk_sims.mean()).item())
        
        scores = np.array(loo_scores)
        self.t_low = float(np.percentile(scores, t_low_percentile))
        self.t_high = float(np.percentile(scores, t_high_percentile))
        
        logger.info(f"3-Gate Thresholds (Cosine Distance):")
        logger.info(f"  - T_low (PASS if <=): {self.t_low:.4f} (P{t_low_percentile})")
        logger.info(f"  - T_high (FAIL_CAND if >=): {self.t_high:.4f} (P{t_high_percentile})")
        logger.info(f"  - REVIEW range: ({self.t_low:.4f}, {self.t_high:.4f})")
        logger.info(f"  - LOO scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        return self.t_low, self.t_high
    
    def classify_3gate(self, score: float) -> Tuple[str, str]:
        """
        3구간 게이팅 판정
        
        Returns:
            (gate, reason)
            gate: 'PASS', 'REVIEW', 'FAIL_CAND'
        """
        if self.t_low is None or self.t_high is None:
            # Fallback: 단순 threshold
            if score <= 0.3:
                return 'PASS', f'Score {score:.4f} very low (normal-like)'
            elif score >= 0.7:
                return 'FAIL_CAND', f'Score {score:.4f} very high (abnormal candidate)'
            else:
                return 'REVIEW', f'Score {score:.4f} in ambiguous zone'
        
        if score <= self.t_low:
            return 'PASS', f'Score {score:.4f} <= T_low {self.t_low:.4f} → PASS (normal confirmed)'
        elif score >= self.t_high:
            return 'FAIL_CAND', f'Score {score:.4f} >= T_high {self.t_high:.4f} → FAIL_CAND'
        else:
            return 'REVIEW', f'Score {score:.4f} in REVIEW zone [{self.t_low:.4f}, {self.t_high:.4f}]'
    
    # Legacy compatibility
    def classify_stage1(self, score: float) -> Tuple[str, str]:
        """Legacy wrapper for classify_3gate"""
        gate, reason = self.classify_3gate(score)
        # Map to old format
        if gate == 'PASS':
            return 'normal', reason
        elif gate == 'FAIL_CAND':
            return 'abnormal', reason
        else:
            return 'uncertain', reason
    
    def find_central_samples(self, n: int = 3) -> List[int]:
        """정상 이미지 중 가장 중심에 가까운 n개 이미지 인덱스 반환"""
        if self.normal_vectors is None or self.mean is None:
            raise ValueError("Normal space not fitted.")
        
        # Cosine similarity with mean
        similarities = self.normal_vectors @ self.mean.unsqueeze(1)
        _, indices = torch.topk(similarities.squeeze(), n, largest=True)
        
        return indices.tolist()
    
    def find_nearest_reference(self, test_vector: torch.Tensor) -> Tuple[int, float]:
        """테스트 벡터와 가장 가까운 정상 레퍼런스 인덱스와 거리 반환"""
        if self.normal_vectors is None:
            raise ValueError("Normal space not fitted.")
        
        test_vector = F.normalize(test_vector.view(1, -1), p=2, dim=-1)
        similarities = test_vector @ self.normal_vectors.T
        best_idx = similarities.argmax().item()
        best_sim = similarities[0, best_idx].item()
        
        return best_idx, 1.0 - best_sim  # Return index and distance
    
    def save_cache(self, path: Path = None):
        """정상 공간 캐시 저장"""
        path = path or NORMAL_VECTORS_CACHE
        path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'normal_vectors': self.normal_vectors,
            'mean': self.mean,
            'threshold': self.threshold,
            'k_neighbors': self.k_neighbors,
            't_low': self.t_low,
            't_high': self.t_high,
            'normal_distance_stats': self.normal_distance_stats,
            'scoring_method': 'cosine_distance',  # 버전 표시
        }
        torch.save(cache_data, path)
        logger.info(f"Normal space cache saved to {path}")
    
    def load_cache(self, path: Path = None) -> bool:
        """정상 공간 캐시 로드"""
        path = path or NORMAL_VECTORS_CACHE
        
        if not path.exists():
            return False
        
        try:
            cache_data = torch.load(path, weights_only=False)
            self.normal_vectors = cache_data['normal_vectors']
            self.mean = cache_data['mean']
            self.t_low = cache_data.get('t_low')
            self.t_high = cache_data.get('t_high')
            self.normal_distance_stats = cache_data.get('normal_distance_stats', {})
            
            # 버전 체크
            if cache_data.get('scoring_method') != 'cosine_distance':
                logger.warning("Cache was created with old L2 distance method. May need retraining.")
            
            logger.info(f"Normal space cache loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scorer = AnomalyScorer(k_neighbors=3)
    
    # Synthetic test with normalized vectors
    normal = torch.randn(16, 384)
    normal = F.normalize(normal, p=2, dim=-1)  # L2 normalize
    
    scorer.fit(normal)
    scorer.compute_threshold_bounds()
    
    # Test with similar vector (should be low score)
    test_normal = normal[0:1] + torch.randn(1, 384) * 0.05
    test_normal = F.normalize(test_normal, p=2, dim=-1)
    score_normal = scorer.compute_score(test_normal)
    gate_normal, reason_normal = scorer.classify_3gate(score_normal)
    print(f"Normal-ish: score={score_normal:.4f}, gate={gate_normal}")
    
    # Test with random vector (should be higher score)
    test_random = torch.randn(1, 384)
    test_random = F.normalize(test_random, p=2, dim=-1)
    score_random = scorer.compute_score(test_random)
    gate_random, reason_random = scorer.classify_3gate(score_random)
    print(f"Random: score={score_random:.4f}, gate={gate_random}")
