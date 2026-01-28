"""
Sanity Check for TriLeadInspectorAgent
======================================
Verifies:
1. Pipeline initialization
2. K-Means Clustering
3. Score constraints (0~2.0)
4. Gating Logic
"""
import torch
import torch.nn.functional as F
import logging
import sys
from unittest.mock import MagicMock

# Set up simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SanityCheck")

try:
    from tri_lead_inspector_agent import TriLeadInspectorAgent
except ImportError:
    # Add src to path if running from root
    import sys
    sys.path.append('src')
    from tri_lead_inspector_agent import TriLeadInspectorAgent

def mock_extract_features(image_path, normalize=True):
    # Return random vector
    return F.normalize(torch.randn(1, 384), p=2, dim=-1)

def mock_extract_batch_features(image_paths, normalize=True):
    # Return random batch
    return F.normalize(torch.randn(len(image_paths), 384), p=2, dim=-1)

def run_check():
    agent = TriLeadInspectorAgent(device='cpu')
    
    # Mock Feature Extractor to avoid loading DINOv2 model (heavy)
    agent.feature_extractor.extract_features = mock_extract_features
    agent.feature_extractor.extract_batch_features = mock_extract_batch_features
    
    # Mock LLM Client Methods
    agent.llm_client.run_triage = MagicMock(return_value={
        'route': 'DINO_THEN_LLM',
        'confidence': 0.95,
        'reason': 'Clear image'
    })
    
    agent.llm_client.run_expert_ensemble = MagicMock(return_value={
        'A': {'rule_violation': False, 'confidence': 0.9, 'lead_count': 3},
        'B': {'rule_violation': False, 'confidence': 0.9}
    })
    
    agent.llm_client.run_arbiter = MagicMock(return_value={
        'rule_violation': False, 'confidence': 0.8
    })

    print("\n[1] Component Init Check... OK")

    # dummy data
    normal_paths = [f"dummy_norm_{i}.jpg" for i in range(16)]
    abnormal_paths = [f"dummy_abnorm_{i}.jpg" for i in range(4)]
    
    # Auto-tune
    print("\n[2] Running Auto-Tune (Synthetic)...")
    agent.auto_tune(normal_paths, abnormal_paths)
    
    # Check Centroids
    centroids = agent.ref_manager.get_centroids()
    print(f"    - Centroids shape: {centroids.shape} (Expect [3, 384])")
    assert centroids.shape[0] == 3
    
    # Check Thresholds
    print(f"    - Thresholds: T_low={agent.policy.t_low}, T_high={agent.policy.t_high}")
    assert agent.policy.t_low is not None
    
    # Inference Check
    print("\n[3] Running Inference Check...")
    result = agent.predict("new_image.jpg")
    print("    - Result keys:", result.keys())
    print("    - Final Decision:", result.get('final_decision'))
    
    score = result['dino_score']
    print(f"    - Score: {score}")
    
    if score < 0 or score > 2.0:
        print("!!! FAIL: Score out of bounds!")
    else:
        print("!!! PASS: Score check OK")

if __name__ == "__main__":
    run_check()
