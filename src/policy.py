"""
Policy - 2-Stage 3-Gate Decision Rules
========================================
Defines the decision rules for the 2-stage funnel:
  Stage 1: DINOv2 Multi-Reference → PASS / REVIEW / FAIL_CAND
  Stage 2: LLM Reviewer (for REVIEW / FAIL_CAND only)
"""
from typing import Tuple, Dict, Any
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANOMALY_THRESHOLD

logger = logging.getLogger(__name__)


class Policy:
    """
    AI Agent 판정 정책 (2-Stage 3-Gate Version)
    
    3구간 게이팅:
    - PASS: score <= T_low → 정상 확정 (LLM 호출 차단)
    - REVIEW: T_low < score < T_high → LLM 검증 필요
    - FAIL_CAND: score >= T_high → 비정상 후보 (LLM으로 근거/유형 기록)
    
    LLM 정책:
    - PASS는 즉시 NORMAL 확정, LLM 호출 금지
    - REVIEW: LLM이 체크리스트로 논리 검증
    - FAIL_CAND: LLM이 결함 근거와 유형 기록 (뒤집기보다 설명용)
    """
    
    def __init__(self):
        # 3-Gate 기준값 (학습 시 동적으로 설정됨)
        self.t_low: float = None   # 이하 = PASS
        self.t_high: float = None  # 이상 = FAIL_CAND
        
        logger.info("Policy initialized (2-Stage 3-Gate mode)")
    
    def set_gate_thresholds(self, t_low: float, t_high: float):
        """3-Gate threshold 설정"""
        self.t_low = t_low
        self.t_high = t_high
        logger.info(f"Gate thresholds set: T_low={t_low:.4f}, T_high={t_high:.4f}")
    
    # Legacy compatibility
    def set_dino_bounds(self, normal_upper: float, abnormal_lower: float):
        """Legacy wrapper for set_gate_thresholds"""
        self.set_gate_thresholds(normal_upper, abnormal_lower)
        # Store with old names for backward compatibility
        self.dino_normal_upper = normal_upper
        self.dino_abnormal_lower = abnormal_lower
    
    def decide_stage1(self, score: float) -> Tuple[str, float, str]:
        """
        Stage 1 (DINOv2) 3-Gate 판정
        
        Returns:
            (gate, confidence, reason)
            gate: 'PASS', 'REVIEW', 'FAIL_CAND'
        """
        if self.t_low is not None and self.t_high is not None:
            if score <= self.t_low:
                conf = max(0.7, 1.0 - (score / self.t_low) * 0.3)
                return 'PASS', conf, f'Score {score:.4f} <= T_low {self.t_low:.4f} → PASS'
            elif score >= self.t_high:
                conf = min(0.95, 0.7 + (score - self.t_high) * 0.1)
                return 'FAIL_CAND', conf, f'Score {score:.4f} >= T_high {self.t_high:.4f} → FAIL_CAND'
            else:
                # REVIEW zone
                position = (score - self.t_low) / (self.t_high - self.t_low)
                return 'REVIEW', 0.5, f'Score {score:.4f} in REVIEW zone (position={position:.1%})'
        else:
            # Fallback: 단순 기준
            if score <= 0.3:
                return 'PASS', 0.8, f'Score {score:.4f} very low → PASS'
            elif score >= 0.7:
                return 'FAIL_CAND', 0.7, f'Score {score:.4f} high → FAIL_CAND'
            else:
                return 'REVIEW', 0.5, f'Score {score:.4f} in ambiguous zone → REVIEW'
    
    def decide_llm(self, llm_result: Dict[str, Any], gate: str = 'REVIEW') -> Tuple[str, float, str]:
        """
        Stage 2 (LLM Reviewer) 판정 (Strict Specification Version)
        
        정상 규격 (Hard Rules):
        - lead_count == 3 (필수)
        - rule_violation == False
        """
        # 1. Parse fields
        lead_count = llm_result.get('lead_count', -1)
        rule_violation = llm_result.get('rule_violation', False)
        contact_suspected = llm_result.get('contact_suspected', False)
        crack_present = llm_result.get('crack_present', False)
        
        confidence = float(llm_result.get('confidence', 0.5))
        reason = llm_result.get('reason', llm_result.get('reasoning', 'No reason provided'))
        
        # 2. Hard Rule Check (Override)
        hard_fail_reasons = []
        
        if isinstance(lead_count, int) and lead_count != 3 and lead_count != -1: # -1 means unknown
            hard_fail_reasons.append(f"Lead Count {lead_count} (Must be 3)")
            
        if rule_violation:
            hard_fail_reasons.append("Rule Violation Flagged")
            
        if contact_suspected:
            hard_fail_reasons.append("Lead Contact Suspected")
            
        if crack_present:
            hard_fail_reasons.append("Crack Present")

        # 3. Decision Logic
        if hard_fail_reasons:
            # 하드룰 위반 -> 무조건 FAIL
            final_decision = 'ABNORMAL'
            final_reason = f"HARD FAIL: {', '.join(hard_fail_reasons)}. {reason}"
            final_conf = max(confidence, 0.9) # 하드룰은 확신 높음
            
        else:
            # 하드룰 통과 -> LLM의 종합 판단 신뢰
            # (JSON에 명시적인 'decision' 필드가 없다면 rule_violation=False를 보고 NORMAL 추정)
            final_conf = confidence

            # fail_cand zone에서는 더 보수적
            if gate == 'FAIL_CAND':
                 # FAIL_CAND인데 특별한 결함 못 찾음 -> 그래도 의심
                 if confidence > 0.8:
                     final_decision = 'NORMAL'
                     final_reason = f"LLM overturned FAIL_CAND (High Conf {confidence:.2f}): {reason}"
                 else:
                     final_decision = 'ABNORMAL'
                     final_reason = f"FAIL_CAND & LLM Low Conf ({confidence:.2f}) -> Conservative FAIL"
            else:
                # REVIEW zone -> LLM 판단 존중
                if confidence > 0.6:
                    final_decision = 'NORMAL'
                    final_reason = f"LLM Confirmed NORMAL ({confidence:.2f}): {reason}"
                else:
                    final_decision = 'ABNORMAL'
                    final_reason = f"LLM Low Conf ({confidence:.2f}) in REVIEW -> Conservative FAIL"

        return final_decision, final_conf, final_reason
        return final_decision, final_conf, final_reason

    def decide_ensemble(self, results: Dict[str, Any], gate: str) -> Tuple[str, float, str]:
        """
        Multi-Expert Ensemble Decision (2-Expert Version)
        
        Logic:
        1. Expert A (Lead) Hard Fail -> Lead Count != 3, Contact, Missing
        2. Expert B (Package) Hard Fail -> Crack, Solder Bridge
        3. If Any Hard Fail -> Immediate ABNORMAL
        4. If Both Clean & Avg Conf >= 0.6 -> NORMAL
        5. Else -> NEEDS_ARBITRATION
        """
        experts = ['A', 'B']
        
        violations = []
        confidences = []
        
        # 1. Check Consensus & Violations
        for code in experts:
            res = results.get(code, {})
            try:
                conf = float(res.get('confidence', 0.5))
            except:
                conf = 0.5
            confidences.append(conf)
            
            # Check Generic Rule Violation
            if res.get('rule_violation', False):
                violations.append(f"Expert {code} Flagged Violation")
                
            # Check Specific Hard Rules
            if code == 'A': # Lead Inspector
                lc = res.get('lead_count', -1)
                if isinstance(lc, int) and lc != 3:
                     violations.append(f"[A] LeadCount {lc}")
                if res.get('contact_suspected'):
                     violations.append(f"[A] Contact")
                if res.get('missing_lead_count', 0) > 0:
                     violations.append(f"[A] Missing Lead")
                     
            if code == 'B': # Package Inspector
                if res.get('crack_present'):
                    violations.append(f"[B] Crack")
                if res.get('solder_bridge'): # If specific field exists
                     violations.append(f"[B] Solder Bridge")
                # Contamination is severe?
                if res.get('contamination_area_ratio', 0.0) > 0.8: # Example threshold
                     violations.append(f"[B] Severe Contamination")

        # 2. Immediate Fail Decision
        if violations:
            # Remove duplicates just in case
            violations = list(set(violations))
            max_conf = max(confidences) if confidences else 0.9
            return 'ABNORMAL', max_conf, f"Ensemble HARD FAIL: {', '.join(violations)}"

        # 3. Consensus Normal Decision
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        if avg_conf >= 0.6:
            return 'NORMAL', avg_conf, f"Ensemble Consensus Normal (Avg Conf {avg_conf:.2f})"
            
        # 4. Conflict / Low Confidence -> Arbitration
        return 'NEEDS_ARBITRATION', avg_conf, f"Ensemble Conflict/LowConf ({avg_conf:.2f}) -> Calling Arbiter"

    def decide_arbiter(self, arbiter_result: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Arbiter Decision (Stage 2.2)
        """
        # 1. Arbiter found clear violation
        if arbiter_result.get('rule_violation', False):
             conf = arbiter_result.get('confidence', 0.7)
             note = arbiter_result.get('note', 'No note')
             return 'ABNORMAL', conf, f"Arbiter Confirmed Violation: {note}"
        
        # 2. Arbiter found NO violation
        conf = arbiter_result.get('confidence', 0.5)
        
        # If Arbiter is confident nothing is wrong -> Normal
        if conf >= 0.7:
            return 'NORMAL', conf, f"Arbiter Ruled Normal (High Conf {conf:.2f})"
        
        # 3. Still Ambiguous -> Conservative Fail (Safety First)
        return 'ABNORMAL', conf, f"Arbiter Uncertain (Conf {conf:.2f}) -> Conservative FAIL"
        """Legacy wrapper for decide_llm"""
        return self.decide_llm(llm_result, gate='REVIEW')
    
    def get_final_label(self, decision: str) -> int:
        """decision 문자열을 레이블로 변환"""
        if decision in ['normal', 'NORMAL', 'PASS']:
            return 0
        else:
            return 1
    
    def get_route(self, gate: str, llm_used: bool = False) -> str:
        """최종 라우트 문자열 반환"""
        if gate == 'PASS':
            return 'PASS'
        elif llm_used:
            return gate  # REVIEW or FAIL_CAND (LLM 사용됨)
        else:
            return 'NEEDS_REVIEW'  # LLM 사용 안 됨


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    policy = Policy()
    policy.set_gate_thresholds(0.15, 0.35)
    
    # Test 3-Gate
    print("\n=== 3-Gate Test ===")
    for score in [0.05, 0.15, 0.25, 0.35, 0.50]:
        gate, conf, reason = policy.decide_stage1(score)
        print(f"Score={score:.2f}: gate={gate}, conf={conf:.2f}")
    
    # Test LLM decision
    print("\n=== LLM Decision Test ===")
    llm_normal = {'decision': 'NORMAL', 'confidence': 0.85, 'reasoning': 'All checks passed'}
    llm_abnormal = {'decision': 'ABNORMAL', 'confidence': 0.9, 'lead_ok': False, 'reasoning': 'Lead bent'}
    
    decision, conf, reason = policy.decide_llm(llm_normal, 'REVIEW')
    print(f"LLM NORMAL from REVIEW: {decision}, {reason[:50]}...")
    
    decision, conf, reason = policy.decide_llm(llm_abnormal, 'FAIL_CAND')
    print(f"LLM ABNORMAL from FAIL_CAND: {decision}, {reason[:50]}...")
