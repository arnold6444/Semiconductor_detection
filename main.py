"""
Transistor Anomaly Detection System - Main Entry Point
========================================================
AI Agent 기반 3단계 Funnel 이상탐지 시스템

사용법:
    # 학습
    python main.py --train dev.csv

    # 학습 (캐시 무시)
    python main.py --train dev.csv --force-retrain

    # 평가 모드
    python main.py --train dev.csv --eval

    # 평가 + LLM 사용
    python main.py --train dev.csv --eval --use-llm

    # 테스트 예측
    python main.py --train dev.csv --test test.csv --output outputs/predictions.csv
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def print_banner():
    """배너 출력"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  🔬 Transistor Anomaly Detection System                       ║
    ║  AI Agent-based 3-Stage Funnel Architecture                   ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Stage 1: DINOv2 + k-NN (Statistical Vision Engine)           ║
    ║  Stage 2: EfficientNet (High-Confidence Filter)               ║
    ║  Stage 3: Luxia LLM via MCP (Comparative Reasoning)           ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  "모델이 아닌 Agent가 판단한다"                                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Transistor Anomaly Detection System'
    )
    parser.add_argument(
        '--train', type=str, required=True,
        help='Training CSV file (dev.csv) with id, img_url, label columns'
    )
    parser.add_argument(
        '--test', type=str, default=None,
        help='Test CSV file for prediction'
    )
    parser.add_argument(
        '--output', type=str, default='outputs/predictions.csv',
        help='Output path for predictions'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining, ignore cache'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='Evaluation mode: evaluate on training data'
    )
    parser.add_argument(
        '--validate-all', action='store_true',
        help='Force evaluation of ALL stages (1,2,3) for each sample'
    )
    parser.add_argument(
        '--use-llm', action='store_true',
        help='Enable Luxia LLM for Stage 3'
    )
    parser.add_argument(
        '--submit', action='store_true',
        help='Generate submission CSV'
    )
    
    args = parser.parse_args()
    
    # Load training data
    train_path = Path(args.train)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    train_df = pd.read_csv(train_path)
    train_df.columns = train_df.columns.str.strip()
    
    # Standardize column names
    if 'image_url' in train_df.columns:
        train_df = train_df.rename(columns={'image_url': 'img_url'})
    
    logger.info(f"Training data: {len(train_df)} samples")
    logger.info(f"  - Normal: {len(train_df[train_df['label'] == 0])}")
    logger.info(f"  - Defect: {len(train_df[train_df['label'] == 1])}")
    
    # Create agent
    from src.agent import AnomalyDetectionAgent
    agent = AnomalyDetectionAgent()
    
    # Train
    train_results = agent.train(train_df, force_retrain=args.force_retrain)
    
    # ===== EVALUATION MODE =====
    if args.eval:
        logger.info("\n" + "=" * 60)
        logger.info("🔍 Evaluation Mode")
        logger.info("=" * 60)
        
        eval_results = agent.evaluate(train_df, use_llm=args.use_llm)
        
        logger.info("\n📊 Evaluation Results:")
        logger.info(f"  - F1 Score: {eval_results['f1_score']:.4f}")
        logger.info(f"  - Precision: {eval_results['precision']:.4f}")
        logger.info(f"  - Recall: {eval_results['recall']:.4f}")
        
        logger.info("\n🔀 Funnel Stage Distribution:")
        for stage, count in sorted(eval_results['stage_counts'].items()):
            pct = count / len(train_df) * 100
            logger.info(f"  - Stage {stage}: {count} samples ({pct:.1f}%)")
        
        logger.info("\n📋 Confusion Matrix:")
        cm = eval_results['confusion_matrix']
        logger.info(f"         Pred=0  Pred=1")
        logger.info(f"  Actual=0  {cm[0][0]:>4}    {cm[0][1]:>4}")
        logger.info(f"  Actual=1  {cm[1][0]:>4}    {cm[1][1]:>4}")
        
        # Detailed predictions
        predictions = eval_results['predictions']
        
        logger.info("\n📝 Detailed Predictions:")
        logger.info("-" * 100)
        logger.info(f"{'ID':>8} | {'Actual':>6} | {'Pred':>6} | {'Stage':>5} | {'Conf':>6} | {'DINO':>10} | {'EFF':>8} | Reason")
        logger.info("-" * 100)
        
        for _, row in predictions.iterrows():
            match = '✓' if row['actual'] == row['label'] else '✗'
            eff = f"{row['efficientnet_prob']:.4f}" if row['efficientnet_prob'] else "-"
            logger.info(
                f"{row['id']:>8} | {row['actual']:>6} | {row['label']:>6}{match} | "
                f"{row['funnel_stage']:>5} | {row['confidence']:>6.2f} | "
                f"{row['dino_score']:>10.4f} | {eff:>8} | {row['reason'][:40]}..."
            )
        
        logger.info("-" * 100)
        
        # Save evaluation results
        eval_path = OUTPUT_DIR / 'evaluation.csv'
        predictions.to_csv(eval_path, index=False)
        logger.info(f"\n✓ Evaluation saved to: {eval_path}")
        
        # Misclassified samples
        misclassified = predictions[predictions['actual'] != predictions['label']]
        if len(misclassified) > 0:
            logger.info(f"\n⚠️ Misclassified: {len(misclassified)} samples")
            for _, row in misclassified.iterrows():
                error_type = "FP" if row['actual'] == 0 else "FN"
                logger.info(f"  - {row['id']}: {error_type}, stage={row['funnel_stage']}, score={row['dino_score']:.4f}")
        else:
            logger.info("\n✅ All samples correctly classified!")
    
    # ===== VALIDATE ALL STAGES MODE =====
    elif args.validate_all:
        logger.info("\n" + "=" * 60)
        logger.info("🔬 Validate All Stages Mode")
        logger.info("   모든 샘플에 대해 1차, 2차 (, 3차) 결과 수집")
        logger.info("=" * 60)
        
        validation_df = agent.validate_all_stages(train_df, use_llm=args.use_llm)
        
        # Print detailed results
        logger.info("\n📊 All Stages Results:")
        logger.info("-" * 120)
        header = f"{'ID':>8} | {'Actual':>6} | {'DINO':>10} | {'S1':>10} | {'EFF':>8} | {'S2':>10} | {'Final':>10} | {'Stage':>5}"
        logger.info(header)
        logger.info("-" * 120)
        
        for _, row in validation_df.iterrows():
            actual = row.get('actual', '?')
            dino = row.get('dino_score', 0)
            s1 = row.get('stage1_decision', '?')
            eff = row.get('efficientnet_prob', 0)
            s2 = row.get('stage2_decision', '?')
            final = row.get('final_decision', '?')
            stage = row.get('final_stage', '?')
            
            # Highlight misclassified
            match = '✓' if actual == row.get('final_label', -1) else '✗'
            
            logger.info(
                f"{row['id']:>8} | {actual:>6} | {dino:>10.4f} | {s1:>10} | "
                f"{eff:>8.4f} | {s2:>10} | {final:>10} | {stage:>5} {match}"
            )
        
        logger.info("-" * 120)
        
        # Stage comparison summary
        logger.info("\n📈 Stage Agreement Analysis:")
        
        # Count agreements
        s1_s2_agree = sum(
            1 for _, r in validation_df.iterrows()
            if r.get('stage1_decision') == r.get('stage2_decision')
        )
        
        total = len(validation_df)
        logger.info(f"  - Stage1 == Stage2: {s1_s2_agree}/{total} ({s1_s2_agree/total*100:.1f}%)")
        
        # Accuracy per stage
        if 'actual' in validation_df.columns:
            s1_correct = sum(
                1 for _, r in validation_df.iterrows()
                if (r.get('stage1_decision') == 'normal' and r['actual'] == 0) or
                   (r.get('stage1_decision') != 'normal' and r['actual'] == 1)
            )
            s2_correct = sum(
                1 for _, r in validation_df.iterrows()
                if (r.get('stage2_decision') == 'normal' and r['actual'] == 0) or
                   (r.get('stage2_decision') == 'abnormal' and r['actual'] == 1)
            )
            final_correct = sum(
                1 for _, r in validation_df.iterrows()
                if r.get('final_label') == r['actual']
            )
            
            logger.info(f"\n  Stage-wise Accuracy (if labels available):")
            logger.info(f"    - Stage 1 tendency correct: {s1_correct}/{total}")
            logger.info(f"    - Stage 2 tendency correct: {s2_correct}/{total}")
            logger.info(f"    - Final decision correct:   {final_correct}/{total}")
        
        # Save
        val_path = OUTPUT_DIR / 'validation_all_stages.csv'
        validation_df.to_csv(val_path, index=False)
        logger.info(f"\n✓ Validation saved to: {val_path}")
    
    # ===== TEST PREDICTION MODE =====
    elif args.test:
        test_path = Path(args.test)
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        test_df = pd.read_csv(test_path)
        test_df.columns = test_df.columns.str.strip()
        
        if 'image_url' in test_df.columns:
            test_df = test_df.rename(columns={'image_url': 'img_url'})
        
        logger.info(f"\nTest data: {len(test_df)} samples")
        
        predictions = agent.predict_batch(test_df, use_llm=args.use_llm)
        
        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        predictions.to_csv(output_path, index=False)
        logger.info(f"\n✓ Predictions saved to: {output_path}")
        
        # Submission format
        if args.submit:
            submit_path = OUTPUT_DIR / 'submit.csv'
            submit_df = predictions[['id', 'label']]
            submit_df.to_csv(submit_path, index=False)
            logger.info(f"✓ Submission file saved to: {submit_path}")
        
        # Statistics
        n_normal = len(predictions[predictions['label'] == 0])
        n_defect = len(predictions[predictions['label'] == 1])
        logger.info(f"\n📊 Prediction Statistics:")
        logger.info(f"  - Normal: {n_normal}")
        logger.info(f"  - Defect: {n_defect}")
        
        # Funnel statistics
        stage_counts = predictions['funnel_stage'].value_counts().to_dict()
        logger.info("\n🔀 Funnel Stage Distribution:")
        for stage, count in sorted(stage_counts.items()):
            pct = count / len(test_df) * 100
            logger.info(f"  - Stage {stage}: {count} samples ({pct:.1f}%)")
    
    else:
        logger.info("\n💡 Training completed without test data.")
        logger.info("   Options:")
        logger.info("   - Evaluate: python main.py --train dev.csv --eval")
        logger.info("   - Predict:  python main.py --train dev.csv --test test.csv")


if __name__ == "__main__":
    main()
