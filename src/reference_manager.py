"""
Reference Image Manager
========================
Manages reference images for LLM comparison-based inference.
"""
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NORMAL_REF_COUNT, ABNORMAL_REF_COUNT,
    REFERENCE_CACHE
)

logger = logging.getLogger(__name__)


class ReferenceManager:
    """
    레퍼런스 이미지 관리자 (Multi-Reference K-Means Version)
    
    정책:
    - 정상 기준 생성: 정상 이미지 16장을 K-Means Clustering (k=2~4) -> Centroids
    - 신규 샘플 매칭: 신규 샘플과 모든 Centroids 간 거리 중 최소값 사용
    - RAG용 레퍼런스: 시각적 유사도 기반으로 원본 이미지 중 선별
    """
    
    def __init__(
        self,
        normal_count: int = None,
        abnormal_count: int = None,
        n_clusters: int = 3
    ):
        self.normal_count = normal_count or NORMAL_REF_COUNT
        self.abnormal_count = abnormal_count or ABNORMAL_REF_COUNT
        self.n_clusters = n_clusters
        
        # Raw Data (URLs & IDs)
        self.all_normal_urls: List[str] = []
        self.all_normal_ids: List[str] = []
        self.all_abnormal_urls: List[str] = []
        self.all_abnormal_ids: List[str] = []
        
        # Features & Centroids
        self.normal_features: Optional[torch.Tensor] = None    # Raw embeddings (normalized)
        self.normal_centroids: Optional[torch.Tensor] = None   # Cluster centroids
        self.abnormal_features: Optional[torch.Tensor] = None
        
        self._fitted = False
        
        logger.info(f"ReferenceManager: k_clusters={self.n_clusters}")
    
    def fit(
        self,
        normal_urls: List[str],
        normal_ids: List[str],
        abnormal_urls: List[str],
        abnormal_ids: List[str],
        normal_features: torch.Tensor = None,
        abnormal_features: torch.Tensor = None,
        central_indices: List[int] = None # Legacy ignored
    ):
        """
        K-Means 기반 레퍼런스 생성
        """
        import torch
        import torch.nn.functional as F

        self.all_normal_urls = normal_urls
        self.all_normal_ids = normal_ids
        self.all_abnormal_urls = abnormal_urls
        self.all_abnormal_ids = abnormal_ids
        
        # 1. Normal Features & Clustering
        if normal_features is not None:
            # L2 Normalize input
            self.normal_features = F.normalize(normal_features, p=2, dim=-1)
            
            # K-Means Clustering (PyTorch implementation)
            if self.normal_features.shape[0] >= self.n_clusters:
                self.normal_centroids = self._kmeans_torch(self.normal_features, k=self.n_clusters)
            else:
                logger.warning(f"Not enough samples for K-Means (n={self.normal_features.shape[0]}, k={self.n_clusters}). Using all samples.")
                self.normal_centroids = self.normal_features
            
            # Centroids L2 Normalize (Important!)
            self.normal_centroids = F.normalize(self.normal_centroids, p=2, dim=-1)
            
        # 2. Abnormal Features
        if abnormal_features is not None:
            self.abnormal_features = F.normalize(abnormal_features, p=2, dim=-1)
        
        self._fitted = True
        logger.info(f"RefMgr fitted: {len(normal_urls)} normal -> {len(self.normal_centroids)} centroids")

    def _kmeans_torch(self, X: torch.Tensor, k: int, max_iters: int = 100) -> torch.Tensor:
        """Simple K-Means in PyTorch"""
        import torch
        
        # Initialize centroids randomly
        indices = torch.randperm(X.shape[0])[:k]
        centroids = X[indices].clone()
        
        for _ in range(max_iters):
            # Assign clusters
            distances = torch.cdist(X, centroids) # (N, K)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = []
            for i in range(k):
                mask = (labels == i)
                if mask.sum() > 0:
                    new_centroids.append(X[mask].mean(dim=0))
                else:
                    # Random re-init for empty cluster
                    new_centroids.append(X[torch.randint(0, X.shape[0], (1,))].squeeze(0))
            
            new_centroids = torch.stack(new_centroids)
            
            # Check convergence
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
            
        return centroids

    def get_centroids(self) -> torch.Tensor:
        """DINOv2 Scoring용 Centroid 반환"""
        if not self._fitted or self.normal_centroids is None:
            raise ValueError("ReferenceManager not fitted")
        return self.normal_centroids

    def get_best_refs(self, query_features, top_n_normal: int = 3, top_n_abnormal: int = 2) -> Tuple[List[str], List[str]]:
        """
        RAG용: 쿼리와 시각적으로 가장 유사한 '원본' 이미지 검색
        """
        import torch
        import torch.nn.functional as F

        if not self._fitted or self.normal_features is None:
            return self.all_normal_urls[:top_n_normal], self.all_abnormal_urls[:top_n_abnormal]

        query_norm = F.normalize(query_features, p=2, dim=-1)
        
        # 1. Normal Raw Images
        sim_normal = torch.mm(query_norm, self.normal_features.t())[0]
        top_indices_normal = torch.topk(sim_normal, min(top_n_normal, len(sim_normal))).indices
        best_normal = [self.all_normal_urls[i] for i in top_indices_normal.tolist()]

        # 2. Abnormal Raw Images
        best_abnormal = self.all_abnormal_urls[:top_n_abnormal]
        if self.abnormal_features is not None and len(self.abnormal_features) > 0:
            sim_abnormal = torch.mm(query_norm, self.abnormal_features.t())[0]
            top_indices_abnormal = torch.topk(sim_abnormal, min(top_n_abnormal, len(sim_abnormal))).indices
            best_abnormal = [self.all_abnormal_urls[i] for i in top_indices_abnormal.tolist()]

        return best_normal, best_abnormal
    
    def get_context_for_query(self, query_url: str) -> Dict:
        """Legacy Wrapper"""
        # Note: Actual RAG logic happens in Agent, this is just a helper
        return {
            'query_image': query_url,
            'normal_references': self.all_normal_urls[:3], # Default fallback
            'abnormal_references': self.all_abnormal_urls[:2]
        }
    
    def save_cache(self, path: Path = None):
        """레퍼런스 캐시 저장"""
        path = path or REFERENCE_CACHE
        path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'normal_urls': self.all_normal_urls,
            'normal_ids': self.all_normal_ids,
            'abnormal_urls': self.all_abnormal_urls,
            'abnormal_ids': self.all_abnormal_ids,
            'n_clusters': self.n_clusters,
            # Centroids는 재계산 가능하므로 URL/ID만 저장해도 됨 (Feature는 별도 DB가 아니므로)
            # 여기서는 편의상 ID/URL만 저장하고 실행 시 다시 Feature Extract 하거나
            # Feature를 별도 저장해야 함.
            # * User has separate cache for vectors usually.
        }
        
        path.write_text(json.dumps(cache_data, indent=2))
        logger.info(f"Reference cache saved to {path}")
    
    def load_cache(self, path: Path = None) -> bool:
        """레퍼런스 캐시 로드"""
        path = path or REFERENCE_CACHE
        if not path.exists():
            return False
        try:
            cache_data = json.loads(path.read_text())
            self.all_normal_urls = cache_data['normal_urls']
            self.all_normal_ids = cache_data['normal_ids']
            self.all_abnormal_urls = cache_data['abnormal_urls']
            self.all_abnormal_ids = cache_data['abnormal_ids']
            self.n_clusters = cache_data.get('n_clusters', 3)
            # Cannot set _fitted=True until we have features
            return True
        except Exception:
            return False
    
    def update_abnormal_refs(self, new_urls: List[str], new_ids: List[str]):
        """비정상 레퍼런스 업데이트"""
        for url, id_ in zip(new_urls, new_ids):
            if id_ not in self.all_abnormal_ids:
                self.all_abnormal_urls.append(url)
                self.all_abnormal_ids.append(id_)


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = ReferenceManager()
    
    # Test with dummy data
    normal_urls = [f"http://example.com/normal_{i}.png" for i in range(10)]
    normal_ids = [f"NORMAL_{i}" for i in range(10)]
    abnormal_urls = [f"http://example.com/abnormal_{i}.png" for i in range(3)]
    abnormal_ids = [f"ABNORMAL_{i}" for i in range(3)]
    
    manager.fit(normal_urls, normal_ids, abnormal_urls, abnormal_ids)
    
    print("Normal refs:", manager.get_normal_refs())
    print("Abnormal refs:", manager.get_abnormal_refs())
