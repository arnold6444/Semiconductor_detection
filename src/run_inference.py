"""
Run Inference on test.csv
=========================
1. Loads dev.csv (Normal samples) to Auto-Tune the Agent.
2. Loads test.csv and runs 2-Stage Inference.
3. Generates sample_submission.csv.
"""
import pandas as pd
import logging
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append('src')
from tri_lead_inspector_agent import TriLeadInspectorAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference.log", mode='w')
    ]
)
logger = logging.getLogger("Inference")

DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
SUBMISSION_CSV = "sample_submission.csv"
DETAILS_CSV = "inference_details.csv"

def main():
    if not Path(DEV_CSV).exists():
        logger.error(f"{DEV_CSV} not found! Needed for auto-tuning.")
        return
    
    if not Path(TEST_CSV).exists():
        logger.error(f"{TEST_CSV} not found!")
        return

    # =========================================================================
    # 1. Tune Agent with DEV data
    # =========================================================================
    logger.info(">>> Step 1: Loading DEV data for Auto-Tuning...")
    dev_df = pd.read_csv(DEV_CSV)
    normal_df = dev_df[dev_df['label'] == 0]
    
    # Optional: Abnormal samples also help if available, but normals are critical for DINO
    abnormal_df = dev_df[dev_df['label'] == 1]
    
    dev_normal_urls = normal_df['img_url'].tolist()
    dev_abnormal_urls = abnormal_df['img_url'].tolist()
    
    logger.info(f"    - Normal samples: {len(dev_normal_urls)}")
    logger.info(f"    - Abnormal samples: {len(dev_abnormal_urls)}")
    
    agent = TriLeadInspectorAgent()
    
    logger.info(">>> Tuning Agent (K-Means & Thresholds)...")
    agent.auto_tune(dev_normal_urls, dev_abnormal_urls)
    logger.info(f"    - T_low: {agent.policy.t_low:.4f}")
    logger.info(f"    - T_high: {agent.policy.t_high:.4f}")

    # =========================================================================
    # 2. Run Inference on TEST data
    # =========================================================================
    logger.info(">>> Step 2: Running Inference on TEST data...")
    test_df = pd.read_csv(TEST_CSV)
    
    predictions = []
    
    # Progress bar for better UX
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        img_id = row['id']
        img_url = row['img_url']
        
        try:
            # Run prediction
            result = agent.predict(img_url)
            
            decision = result.get('final_decision', 'ABNORMAL') # Default conservative
            label = 0 if decision == 'NORMAL' else 1
            
            # Log interesting cases (e.g. LLM called or Abnormal)
            if 'llm_called' in result and result['llm_called']:
                 logger.info(f"[{img_id}] LLM Called -> {decision} (Conf: {result.get('final_conf', 0):.2f}) Reason: {result.get('final_reason')}")
            elif decision == 'ABNORMAL':
                 logger.info(f"[{img_id}] High DINO Score -> ABNORMAL (Score: {result.get('dino_score', 0):.4f})")
            
            # Store detailed result
            predictions.append({
                'id': img_id,
                'label': label,
                'route': result.get('route', 'UNKNOWN'),
                'dino_score': result.get('dino_score', 0),
                'llm_called': result.get('llm_called', False),
                'final_decision': decision,
                'final_reason': result.get('final_reason', 'N/A')
            })
            
        except Exception as e:
            logger.error(f"Failed to predict {img_id}: {e}")
            # Fallback: FAIL (Conservative)
            predictions.append({
                'id': img_id,
                'label': 1,
                'route': 'ERROR',
                'final_reason': str(e)
            })

    # =========================================================================
    # 3. Save Submission & Details
    # =========================================================================
    logger.info(">>> Step 3: Saving Submission...")
    
    # Create DataFrame from all predictions
    full_df = pd.DataFrame(predictions)
    
    # 1. Save Details CSV (For User Verification)
    if 'id' in full_df.columns:
        # Ensure ID alignment just in case
        full_df = test_df[['id']].merge(full_df, on='id', how='left')
    
    full_df.to_csv(DETAILS_CSV, index=False)
    logger.info(f"✅ Saved detailed results to {DETAILS_CSV}")

    # 2. Save Submission CSV (ID, Label only)
    submission_df = full_df[['id', 'label']].fillna(1)
    submission_df['label'] = submission_df['label'].astype(int)
    
    submission_df.to_csv(SUBMISSION_CSV, index=False)
    logger.info(f"✅ Saved submission to {SUBMISSION_CSV}")
    print(submission_df.head(10))

if __name__ == "__main__":
    main()
