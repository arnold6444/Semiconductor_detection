# 🔬 Transistor Anomaly Detection System

**AI Agent 기반 3단계 Funnel 이상탐지 시스템**

> "모델이 아닌 Agent가 판단한다"

## Repository Status

This is the canonical portfolio repository for the semiconductor anomaly detection work.

- Active implementation: root `main.py`, `config.py`, and `src/`
- Legacy merged reference: `legacy_hybrid_agent/`
- Archived old GitHub repo: `hy_ngv`

`hy_ngv` and `Semiconductor_detection` are the same project line. Future updates should go into this repository.

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image                               │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: DINOv2 + k-NN                                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • 정상 이미지로 "정상 공간" 학습                              │ │
│  │ • 점수 ≤ normal_upper → 정상 확정 ✓                         │ │
│  │ • 그 외 → Stage 2로 전달                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: EfficientNet (고확신 필터)                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • 이진 분류 (정상/비정상)                                    │ │
│  │ • P(abnormal) ≥ 0.9 → 비정상 확정 ✓                        │ │
│  │ • P(abnormal) ≤ 0.1 → 정상 확정 ✓                          │ │
│  │ • 그 외 → Stage 3로 전달                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ⚠️ DINO Override: DINO > 100 & EFF = normal → Stage 3로 강제    │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Luxia LLM (MCP 기반 비교 추론)                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • 레퍼런스 이미지와 시각적 비교                               │ │
│  │ • 정상/비정상 레퍼런스 각각 제시 후 판단                      │ │
│  │ • 구조화된 JSON 응답 파싱                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Final Decision                               │
│                  (normal=0 / abnormal=1)                         │
└─────────────────────────────────────────────────────────────────┘
```

## 프로젝트 구조

```
hy_ngv3/
├── main.py                      # CLI 엔트리포인트
├── config.py                    # 설정 및 하이퍼파라미터
├── dev.csv                      # 학습 데이터 (정상 16 + 비정상 4)
├── requirements.txt             # 의존성
├── .env                         # 환경 변수 (API 키 등)
│
├── src/
│   ├── __init__.py
│   ├── dino_extractor.py        # Stage 1: DINOv2 특징 추출
│   ├── anomaly_scorer.py        # Stage 1: k-NN 이상 점수 계산
│   ├── efficientnet_filter.py   # Stage 2: EfficientNet 분류기
│   ├── luxia_mcp_client.py      # Stage 3: Luxia LLM 클라이언트
│   ├── reference_manager.py     # LLM용 레퍼런스 이미지 관리
│   ├── policy.py                # 판정 정책 (threshold, 규칙)
│   └── agent.py                 # AI Agent 오케스트레이터
│
├── legacy_hybrid_agent/         # 이전 hy_ngv 구현 참고용
│   ├── agent/                   # Local model + ambiguity + LLM fusion
│   ├── checkpoints/config.example.yaml
│   └── requirements.txt
│
└── outputs/                     # 예측 결과, gitignore 처리
```

## 핵심 설계 결정

### 1. 왜 3단계 Funnel인가?

| 문제 | 해결책 |
|------|--------|
| 비정상 데이터 극히 희소 (4개) | Stage별 역할 분리로 과적합 방지 |
| 단일 모델의 과신 위험 | 다단계 검증으로 신뢰도 향상 |
| False Negative 치명적 | 보수적 정책 + LLM 최종 확인 |

### 2. DINO Override가 필요한 이유

```
문제 상황:
- 새로운 정상 이미지 → DINO 점수 4895 → Stage 2로 전달
- EfficientNet: "정상이다!" (학습 데이터에 없는 패턴)
- 실제로는 비정상

해결:
- DINO > 100이면 EfficientNet 결과 무시
- LLM으로 최종 확인 또는 보수적 abnormal 판정
```

### 3. 데이터 증강 전략

```
원본: 정상 16개, 비정상 4개
      ↓ 오버샘플링
증강: 정상 80개 (5배), 비정상 8개 (2배)
      + ColorJitter, RandomCrop, GaussianBlur, RandomErasing
```

## 사용법

### 설치

```bash
pip install -r requirements.txt
```

### .env 설정

```env
LUXIA_API_KEY=your_api_key
LUXIA_BASE_URL=https://api.luxiacloud.com/v1
```

### 학습

```bash
python main.py --train dev.csv
```

### 평가

```bash
# 기본 평가 (LLM 없이)
python main.py --train dev.csv --eval

# LLM 포함 평가
python main.py --train dev.csv --eval --use-llm

# 모든 단계 강제 검증
python main.py --train dev.csv --validate-all
```

### 예측

```bash
# 테스트 데이터 예측
python main.py --train dev.csv --test test.csv --output outputs/predictions.csv

# LLM 포함 예측
python main.py --train dev.csv --test test.csv --use-llm

# 제출 파일 생성
python main.py --train dev.csv --test test.csv --submit
```

## 성능

| 메트릭 | 값 |
|--------|-----|
| F1 Score | 1.0 |
| Precision | 1.0 |
| Recall | 1.0 |
| 정확도 | 20/20 (100%) |

### 샘플별 결과

| 샘플 | 실제 | DINO | Stage 1 | EfficientNet | Final | 판정 |
|------|------|------|---------|--------------|-------|------|
| 정상 16개 | 0 | ~3.7 | normal | - | normal | ✓ |
| 비정상 4개 | 1 | 3856~5654 | uncertain | 0.04~0.92 | abnormal | ✓ |
