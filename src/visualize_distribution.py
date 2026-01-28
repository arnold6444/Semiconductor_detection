"""
Visualize Normal Distance Distribution
======================================
Loads normal images, computes pairwise/centroid distances,
calculates 3-Gate thresholds (95%, 99%), and plots the histogram.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

try:
    from tri_lead_inspector_agent import TriLeadInspectorAgent
except ImportError:
    sys.path.append('src')
    from tri_lead_inspector_agent import TriLeadInspectorAgent

# Configuration
NORMAL_DIR = Path("data/normal") # Change this to your actual normal data path
OUTPUT_PLOT = Path("outputs/normal_distribution.png")

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DistVis")
    
    if not NORMAL_DIR.exists():
        logger.error(f"Normal directory not found: {NORMAL_DIR}")
        logger.info("Please set NORMAL_DIR to your 16 normal images path.")
        return

    # 1. Initialize Agent
    agent = TriLeadInspectorAgent(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Images
    image_paths = sorted([str(p) for p in NORMAL_DIR.glob("*.jpg")]) + \
                  sorted([str(p) for p in NORMAL_DIR.glob("*.png")])
    
    if len(image_paths) == 0:
        logger.error("No images found in normal directory.")
        return
        
    logger.info(f"Loaded {len(image_paths)} normal images.")
    
    # 3. Extract Features
    logger.info("Extracting features...")
    features = agent.feature_extractor.extract_batch_features(image_paths, normalize=True)
    
    # 4. K-Means & Centroids
    logger.info("Fitting Reference Manager (K-Means)...")
    agent.ref_manager.fit(image_paths, [Path(p).name for p in image_paths], [], [], normal_features=features)
    centroids = agent.ref_manager.get_centroids()
    agent.scorer.fit(centroids)
    
    # 5. Compute Distances (Sample vs Centroids)
    logger.info("Computing scores...")
    scores = []
    for i in range(len(features)):
        # Simulate prediction scoring
        score = agent.scorer.compute_score(features[i])
        scores.append(score)
    
    scores = np.array(scores)
    
    # 6. Calculate Thresholds
    t_low = np.percentile(scores, 95)
    t_high = np.percentile(scores, 99)
    
    print(f"\n=== Distribution Statistics ===")
    print(f"Min: {scores.min():.4f}")
    print(f"Max: {scores.max():.4f}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")
    print(f"-------------------------------")
    print(f"T_low (95%): {t_low:.4f}")
    print(f"T_high (99%): {t_high:.4f}")
    print(f"===============================\n")
    
    # 7. Plot
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(t_low, color='green', linestyle='--', label=f'T_low (95%) = {t_low:.4f}')
    plt.axvline(t_high, color='red', linestyle='--', label=f'T_high (99%) = {t_high:.4f}')
    plt.title(f"Normal Score Distribution (N={len(scores)})")
    plt.xlabel("Cosine Distance Score (0~2)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT)
    logger.info(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
