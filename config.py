"""
Transistor Anomaly Detection System - Configuration
====================================================
AI Agent 기반 3단계 Funnel 이상탐지 시스템 설정
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# LUXIA CLOUD API (MCP)
# =============================================================================
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_BASE_URL = os.getenv("LUXIA_BASE_URL", "https://bridge.luxiacloud.com/luxia/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini-2024-07-18")

# Vision 전용 (사용자 베이스라인 기반)
LUXIA_VISION_URL = os.getenv("LUXIA_VISION_URL", "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini-2024-07-18")

# =============================================================================
# DINOV2 SETTINGS (1차 필터)
# =============================================================================
DINO_MODEL = os.getenv("DINO_MODEL", "dinov2_vits14")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "518"))

# DINOv2 normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# ANOMALY DETECTION (1차 필터 - DINOv2 + k-NN)
# =============================================================================
# k-NN distance threshold - images with score > threshold are anomalies
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "12.5"))

# Number of nearest neighbors for k-NN scoring
K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", "3"))

# Use Mahalanobis distance instead of k-NN
USE_MAHALANOBIS = os.getenv("USE_MAHALANOBIS", "true").lower() == "true"

# 3-Gate 판정 구간 (Cosine Distance 기반)
# score <= T_LOW (P70): PASS (정상 확정, LLM 차단)
# score >= T_HIGH (P95): FAIL_CAND (비정상 후보)
# 그 사이: REVIEW (LLM 검증)
DINO_NORMAL_PERCENTILE = float(os.getenv("DINO_NORMAL_PERCENTILE", "70"))
DINO_ABNORMAL_PERCENTILE = float(os.getenv("DINO_ABNORMAL_PERCENTILE", "95"))

# =============================================================================
# EFFICIENTNET SETTINGS (2차 필터) - DISABLED in 2-Stage Architecture
# =============================================================================
# EfficientNet은 2-Stage 아키텍처에서 제거됨 (DINOv2 + LLM만 사용)
# EFFICIENTNET_MODEL = os.getenv("EFFICIENTNET_MODEL", "efficientnet_b0")
# EFFICIENTNET_HIGH_CONF = float(os.getenv("EFFICIENTNET_HIGH_CONF", "0.9"))
# EFFICIENTNET_LOW_CONF = float(os.getenv("EFFICIENTNET_LOW_CONF", "0.1"))
# EFFICIENTNET_EPOCHS = int(os.getenv("EFFICIENTNET_EPOCHS", "3"))
# EFFICIENTNET_LR = float(os.getenv("EFFICIENTNET_LR", "0.0001"))
# EFFICIENTNET_DROPOUT = float(os.getenv("EFFICIENTNET_DROPOUT", "0.5"))

# =============================================================================
# LUXIA LLM SETTINGS (3차 필터)
# =============================================================================
# Reference image counts
NORMAL_REF_COUNT = int(os.getenv("NORMAL_REF_COUNT", "3"))
ABNORMAL_REF_COUNT = int(os.getenv("ABNORMAL_REF_COUNT", "2"))

# LLM 요청 시 프롬프트 설정
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# =============================================================================
# CACHE FILES
# =============================================================================
NORMAL_VECTORS_CACHE = CACHE_DIR / "normal_vectors.pt"
PARAMS_CACHE = CACHE_DIR / "params.json"
EFFICIENTNET_CACHE = CACHE_DIR / "efficientnet_weights.pt"
REFERENCE_CACHE = CACHE_DIR / "references.json"

# =============================================================================
# DEFECT CATEGORIES (for LLM observation)
# =============================================================================
OBS_ITEMS = [
    "package_damage",           # 패키지 손상
    "lead_missing_or_broken",   # 리드 누락 또는 파손
    "lead_bent_or_deformed",    # 리드 휨/변형
    "solder_defect",            # 솔더 불량
    "surface_contamination",    # 표면 오염
    "marking_defect",           # 마킹 결함
    "foreign_material",         # 이물질
    "crack_or_chip",            # 크랙 또는 칩핑
]

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
