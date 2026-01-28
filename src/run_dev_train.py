"""
Run Training (Auto-Tune) on dev.csv
===================================
Loads dev.csv, filters Normal (label=0) / Abnormal (label=1),
and runs agent.auto_tune() to calculate thresholds.
"""
import pandas as pd
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')
from tri_lead_inspector_agent import TriLeadInspectorAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DevTrain")

CSV_PATH = "dev.csv"

def main():
    if not Path(CSV_PATH).exists():
        logger.error(f"{CSV_PATH} not found!")
        return

    # 1. Load CSV
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded {CSV_PATH}: {len(df)} rows")
    
    # 2. Filter Normal / Abnormal
    # label 0 = Normal, 1 = Abnormal
    normal_df = df[df['label'] == 0]
    abnormal_df = df[df['label'] == 1]
    
    normal_urls = normal_df['img_url'].tolist()
    abnormal_urls = abnormal_df['img_url'].tolist()
    
    logger.info(f"Found {len(normal_urls)} Normal samples")
    logger.info(f"Found {len(abnormal_urls)} Abnormal samples")
    
    if len(normal_urls) == 0:
        logger.error("No normal samples found!")
        return

    # 3. Initialize Agent
    agent = TriLeadInspectorAgent()
    
    # 4. Run Auto-Tune
    logger.info("Starting Auto-Tune...")
    try:
        agent.auto_tune(normal_urls, abnormal_urls)
        
        # 5. Report Results
        print("\n" + "="*40)
        print("          AUTO-TUNE RESULTS          ")
        print("="*40)
        print(f"Normal Samples Used: {len(normal_urls)}")
        print(f"Abnormal Samples Used: {len(abnormal_urls)}")
        print("-" * 40)
        print(f"Threshold T_low (PASS)      : {agent.policy.t_low:.4f} (95%)")
        print(f"Threshold T_high (FAIL_CAND): {agent.policy.t_high:.4f} (99%)")
        print("-" * 40)
        
        # Verify Centroids
        centroids = agent.ref_manager.get_centroids()
        print(f"Score Model Centroids: {centroids.shape}")
        print("="*40 + "\n")
        
    except Exception as e:
        logger.error(f"Auto-Tune Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
