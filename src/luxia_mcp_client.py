"""
Stage 3: Luxia LLM Client via MCP
==================================
MCP(Model Context Protocol) based client for Luxia Cloud LLM.
Performs comparison-based reasoning for ambiguous samples.
"""
import os
import base64
import requests
import json
from typing import Optional, Dict, Any, List, Union
from PIL import Image
from io import BytesIO
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LUXIA_API_KEY, LUXIA_BASE_URL, LLM_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, OBS_ITEMS
)

logger = logging.getLogger(__name__)


class LuxiaMCPClient:
    """
    MCP 기반 Luxia LLM 클라이언트
    
    MCP (Model Context Protocol) 패턴:
    - Agent가 "무엇을 요청할지"만 정의
    - LLM 내부 구현에 의존하지 않음
    - 표준화된 Context 구조로 요청/응답
    
    주요 기능:
    - 비교 기반 추론: Query + Normal Refs + Abnormal Refs
    - 구조화된 응답 파싱
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        self.api_key = api_key or LUXIA_API_KEY
        # Base URL 최적화 (문서 사양: https://bridge.luxiacloud.com/luxia/v1)
        self.base_url = (base_url or LUXIA_BASE_URL).rstrip('/')
        self.model = model or LLM_MODEL
        
        if not self.api_key:
            logger.warning("LUXIA_API_KEY not set. LLM calls will fail.")
        
        self.headers = {
            "Content-Type": "application/json",
            "apikey": self.api_key
        }
        
        logger.info(f"LuxiaMCPClient initialized: model={self.model}")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Luxia Embedding API를 사용하여 텍스트 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터 (1024 dimensions)
        """
        endpoint = f"{self.base_url}/embedding"
        payload = {
            "inputs": [text]
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # data[0].embedding 형식
                return result.get('data', [{}])[0].get('embedding')
            else:
                logger.error(f"Embedding API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def _image_to_base64(self, source: Union[str, Image.Image]) -> str:
        """이미지를 base64로 변환"""
        if isinstance(source, str):
            if source.startswith('http'):
                response = requests.get(source, timeout=30)
                image_data = response.content
            else:
                with open(source, 'rb') as f:
                    image_data = f.read()
        else:
            buffer = BytesIO()
            source.save(buffer, format='PNG')
            image_data = buffer.getvalue()
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def _build_mcp_context(
        self,
        query_image: Union[str, Image.Image],
        normal_refs: List[Union[str, Image.Image]],
        abnormal_refs: List[Union[str, Image.Image]],
        additional_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        MCP Context 구성
        
        Args:
            query_image: 분석할 이미지
            normal_refs: 정상 레퍼런스 이미지들 (3~5장)
            abnormal_refs: 비정상 레퍼런스 이미지들 (2~3장)
            additional_context: 추가 컨텍스트 정보
        
        Returns:
            messages 리스트 (OpenAI-compatible format)
        """
        content = []
        
        # 1. Normal references
        content.append({
            "type": "text",
            "text": "== NORMAL REFERENCE IMAGES (Verified Normal Samples) =="
        })
        for i, ref in enumerate(normal_refs):
            ref_b64 = self._image_to_base64(ref)
            content.append({
                "type": "text",
                "text": f"[Normal Reference {i+1}]"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"}
            })
        
        # 2. Abnormal references
        if abnormal_refs:
            content.append({
                "type": "text",
                "text": "== ABNORMAL REFERENCE IMAGES (Verified Defect Samples) =="
            })
            for i, ref in enumerate(abnormal_refs):
                ref_b64 = self._image_to_base64(ref)
                content.append({
                    "type": "text",
                    "text": f"[Abnormal Reference {i+1}]"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{ref_b64}"}
                })
        
        # 3. Query image
        query_b64 = self._image_to_base64(query_image)
        content.append({
            "type": "text",
            "text": "== QUERY IMAGE (To be analyzed) =="
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{query_b64}"}
        })
        
        # 4. Analysis prompt
        prompt = self._build_analysis_prompt(additional_context)
        content.append({
            "type": "text",
            "text": prompt
        })
        
        return [{"role": "user", "content": content}]
    
    def _build_analysis_prompt(self, additional_context: Dict[str, Any] = None) -> str:
        """분석 요청 프롬프트 생성 (Dynamic for Experts)"""
        mode = "NORMAL"
        expert_type = None
        
        if additional_context:
            mode = additional_context.get("mode", "NORMAL")
            expert_type = additional_context.get("expert_type", None)

        common_json_schema = """
{
  "lead_count": int,
  "missing_lead_count": int,
  "contact_suspected": bool,
  "crack_present": bool,
  "marking_legibility": float,
  "contamination_area_ratio": float,
  "rule_violation": bool,
  "confidence": float,
  "note": string
}
"""

        # 1. Expert Prompts
        if mode == "EXPERT":
            if expert_type == 'A': # Lead Inspector
                return f"""
너는 'Lead Inspector (Expert A)'다.
너의 임무는 오직 '리드(Lead)' 상태만 정밀하게 검증하는 것이다. 다른 결함은 무시하라.

[검증 항목]
1. 리드 개수가 정확히 3개인가? (가장 중요)
2. 리드가 부러지거나(Missing) 잘려나갔는가?
3. 리드끼리 서로 닿아있는가(Contact)? (쇼트 위험)
4. 리드가 심하게 휘었는가?
(참고: 리드 끝이 약간 휜 것은 허용될 수 있으나, 접촉은 절대 금지)

[하드룰 (Rule Violation)]
- 리드 개수 ≠ 3
- 리드 누락/단선 발견
- 리드 간 접촉 의심
위 경우 즉시 rule_violation=true로 설정하라.

[출력 포맷 (JSON Only)]
반드시 아래 필드를 모두 채워라.
{common_json_schema}
"""
            elif expert_type == 'B': # Package Inspector
                return f"""
너는 'Package Inspector (Expert B)'다.
너의 임무는 '패키지 본체 및 표면' 상태만 검증하는 것이다. 리드는 무시하라.

[검증 항목]
1. 패키지 표면에 크랙(Crack), 깨짐, 파손이 있는가? (치명적)
2. 표면에 심한 오염, 변색, 부식이 있는가?
3. 솔더(납땜) 뭉침이나 브리지가 있는가?

[하드룰 (Rule Violation)]
- 패키지 크랙/파손 발견
- 명확한 솔더 브리지/뭉침
위 경우 즉시 rule_violation=true로 설정하라.

[출력 포맷 (JSON Only)]
반드시 아래 필드를 모두 채워라.
{common_json_schema}
"""
            elif expert_type == 'ARBITER': # Arbiter
                return f"""
너는 'Arbiter (중재자)'다.
앞선 전문가들의 의견이 엇갈리고 있다. 이미지를 다시 보고, 어떤 결함이 '확실한지' 판정하라.
너는 통계적 추론을 하지 않으며, 오직 '명확한 시각적 증거'가 있을 때만 위반을 선언한다.

[판단 기준]
- 애매하면 rule_violation=false (정상 추정)
- 명확한 크랙/리드파손/접촉이 보이면 rule_violation=true

[출력 포맷 (JSON Only)]
반드시 아래 필드를 모두 채워라.
{common_json_schema}
"""

        if mode == "TRIAGE":
            return """
너는 반도체 이미지 검사 파이프라인의 오케스트레이터다.
너의 임무는 이 이미지에 대해 어떤 검사 경로가 적절한지 결정하는 것이다.
정상/비정상 라벨은 절대 출력하지 마라.

[선택 가능한 route]
- "DINO_THEN_LLM": 기본 검사 경로. DINO 우선 스캔 후 필요시 LLM 확인 (안정적~경미한 이슈)
- "LLM_STRICT": 이미지 품질/스케일 문제로 DINO가 불안정할 가능성이 높으니 바로 LLM 정밀검증 (중대 이슈)

[판단 기준]
- lead_visibility: 리드(3개)가 또렷이 보이면 "clear"
- chip_size: 소자가 프레임에서 충분히 크면 "large", 너무 작으면 "small"
- blur_or_noise: 흐림/노이즈가 있으면 true
- exposure_issue: 과/저노출이면 true
- background_dominant: 배경이 대부분이면 true

[출력 JSON]
{
  "lead_visibility": "clear|unclear",
  "chip_size": "large|small",
  "blur_or_noise": true,
  "exposure_issue": true,
  "background_dominant": true,
  "route": "DINO_THEN_LLM|LLM_STRICT",
  "confidence": float,
  "reason": "short explanation"
}
"""

        # 2. Legacy / Fallback Prompts (Review mode repurposed as single-shot fallback if needed)
        if mode == "REVIEW":
            return """
이 이미지는 이상 후보로 분류되었다. '정상이라고 보기 어려운 반례'를 찾아라.
[중점 확인] 리드 개수(3개) 위반, 크랙, 접촉.
[출력] JSON 형식 준수. rule_violation 필드 필수.
"""
        
        # Default Normal Prompt
        return """
너는 반도체 외관 검사 AI다. 규격을 기준으로 정량 피처를 추출하라.
[정상 규격] 리드 3개, 크랙 없음, 마킹 식별 가능.
[출력] JSON 형식 준수.
"""
    
    def run_expert_ensemble(
        self,
        query_image: Union[str, Image.Image],
        normal_refs: List[Union[str, Image.Image]],
        abnormal_refs: List[Union[str, Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        3인의 전문가(Expert A, B, C)에게 순차적으로 검증 요청 (Ensemble)
        """
        results = {}
        experts = [
            ('A', 'Lead Inspector'),
            ('B', 'Package Inspector')
        ]
        
        for code, name in experts:
            logger.info(f"Creating request for {name} ({code})...")
            # Reuse compare_and_decide logic but with EXPERT mode
            res = self.compare_and_decide(
                query_image,
                normal_refs,
                abnormal_refs, # Experts behave better with abnormal examples
                additional_context={'mode': 'EXPERT', 'expert_type': code}
            )
            results[code] = res
            
        return results

    def run_arbiter(
        self,
        query_image: Union[str, Image.Image],
        normal_refs: List[Union[str, Image.Image]],
        expert_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Arbiter(중재자) 호출
        """
        logger.info("Calling Arbiter for conflict resolution...")
        # Construct summary of conflict (optional, but prompt mainly relies on image)
        
        res = self.compare_and_decide(
            query_image,
            normal_refs,
            [], # Arbiter focuses on the query image mainly
            additional_context={'mode': 'EXPERT', 'expert_type': 'ARBITER'}
        )
        return res

    def run_triage(
        self,
        query_image: Union[str, Image.Image]
    ) -> Dict[str, Any]:
        """
        Stage 0: LLM Quick Triage (Planning)
        Routes: DINO_ONLY, DINO_THEN_LLM, LLM_STRICT
        """
        logger.info("[TRIAGE] Planning inspection route...")
        
        # Triage only needs the query image, no references needed usually
        # But we can pass empty refs or just query
        res = self.compare_and_decide(
            query_image,
            [], # No refs needed for basic quality check
            [], 
            additional_context={'mode': 'TRIAGE'}
        )
        return res

    def compare_and_decide(
        self,
        query_image: Union[str, Image.Image],
        normal_refs: List[Union[str, Image.Image]],
        abnormal_refs: List[Union[str, Image.Image]] = None,
        additional_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        비교 기반 추론 실행
        (Vision 전용 엔드포인트를 사용하여 멀티모달 분석 수행)
        """
        abnormal_refs = abnormal_refs or []
        
        messages = self._build_mcp_context(
            query_image, normal_refs, abnormal_refs, additional_context
        )
        
        # Vision 전용 설정 사용 (문서 및 사용자 베이스라인 기반)
        from config import LUXIA_VISION_URL, VISION_MODEL
        
        payload = {
            "model": VISION_MODEL,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_completion_tokens": LLM_MAX_TOKENS,
            "stream": False
        }
        
        # Vision 전용 엔드포인트 사용
        endpoint = LUXIA_VISION_URL
        # logger.info(f"Sending Vision request to Luxia Bridge: {endpoint}") # Too verbose for ensemble
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                logger.error(f"Luxia Vision API Error: {response.status_code} - {response.text}")
                return self._fallback_response(f"Vision API error: {response.status_code}")
            
            result = response.json()
            # choices[0].message.content 로직은 동일
            raw_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            return self._parse_llm_response(raw_content)
            
        except requests.RequestException as e:
            logger.error(f"Vision request failed: {e}")
            return self._fallback_response(str(e))
    
    def _parse_llm_response(self, raw_content: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        try:
            # Try to extract JSON from response
            content = raw_content.strip()
            
            # Handle markdown code blocks
            if content.startswith('```'):
                lines = content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith('```') and not in_json:
                        in_json = True
                        continue
                    elif line.startswith('```') and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                content = '\n'.join(json_lines)
            
            parsed = json.loads(content)
            parsed['raw_response'] = raw_content
            
            return parsed
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {raw_content[:200]}")
            
            # Attempt to extract decision from text
            if 'ABNORMAL' in raw_content.upper():
                return {
                    'decision': 'ABNORMAL',
                    'confidence': 0.6,
                    'similar_to': 'unknown',
                    'visual_evidence': [],
                    'defect_type': None,
                    'reasoning': 'Parsed from unstructured response',
                    'raw_response': raw_content
                }
            else:
                return {
                    'decision': 'NORMAL',
                    'confidence': 0.5,
                    'similar_to': 'unknown',
                    'visual_evidence': [],
                    'defect_type': None,
                    'reasoning': 'Parsed from unstructured response',
                    'raw_response': raw_content
                }
    
    def _fallback_response(self, error_msg: str) -> Dict[str, Any]:
        """API 실패 시 폴백 응답"""
        return {
            'decision': 'ABNORMAL',  # Conservative: assume abnormal on error
            'confidence': 0.3,
            'similar_to': 'error',
            'visual_evidence': [],
            'defect_type': None,
            'reasoning': f'Fallback due to error: {error_msg}',
            'raw_response': '',
            'error': error_msg
        }
    
    def test_connection(self) -> bool:
        """API 연결 테스트 (GPT Bridge 사용)"""
        from config import LUXIA_VISION_URL, VISION_MODEL
        try:
            messages = [{"role": "user", "content": "Hello"}]
            payload = {
                "model": VISION_MODEL,
                "messages": messages,
                "max_completion_tokens": 10
            }
            
            response = requests.post(
                LUXIA_VISION_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    client = LuxiaMCPClient()
    
    if client.test_connection():
        print("✓ Luxia API connection successful")
    else:
        print("✗ Luxia API connection failed")
