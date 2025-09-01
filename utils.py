"""
Utility functions for workflow agent
워크플로우 에이전트용 유틸리티 함수들
"""

import json
from typing import Dict, Any, Union


def parse_llm_evaluation(evaluation_result: str) -> bool:
    """
    LLM 평가 결과를 파싱하여 boolean으로 변환
    
    Args:
        evaluation_result (str): LLM에서 반환된 평가 결과 문자열
        
    Returns:
        bool: True if evaluation passed, False otherwise
    """
    if not evaluation_result:
        return False
    
    # Clean up the result string
    result = evaluation_result.strip().lower()
    
    # Check for positive indicators
    positive_indicators = ["true", "yes", "pass", "correct", "good", "valid"]
    negative_indicators = ["false", "no", "fail", "incorrect", "bad", "invalid"]
    
    for indicator in positive_indicators:
        if indicator in result:
            return True
    
    for indicator in negative_indicators:
        if indicator in result:
            return False
    
    # Default to False if unclear
    return False


def validate_json_structure(json_data: Union[str, Dict[str, Any]]) -> bool:
    """
    JSON 구조가 올바른지 검증
    
    Args:
        json_data: JSON 문자열 또는 딕셔너리
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # If string, try to parse as JSON
        if isinstance(json_data, str):
            parsed_data = json.loads(json_data)
        else:
            parsed_data = json_data
        
        # Check required fields
        if not isinstance(parsed_data, dict):
            return False
        
        # Check for required fields based on type
        if "type" not in parsed_data:
            return False
        
        workflow_type = parsed_data.get("type")
        
        # Type-specific validation
        if workflow_type == "LLM":
            return "tools" in parsed_data
        elif workflow_type in ["Sequential", "Parallel", "Loop"]:
            return "sub_agents" in parsed_data
        
        return True
        
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def clean_json_output(raw_output: str) -> str:
    """
    LLM 출력에서 JSON 부분만 추출하여 정리
    
    Args:
        raw_output (str): LLM의 원시 출력
        
    Returns:
        str: 정리된 JSON 문자열
    """
    if not raw_output:
        return "{}"
    
    # Remove common prefixes/suffixes
    output = raw_output.strip()
    
    # Remove markdown code blocks
    if output.startswith("```"):
        lines = output.split('\n')
        output = '\n'.join(lines[1:-1]) if len(lines) > 2 else output
    
    # Remove json prefix
    if output.startswith("json"):
        output = output[4:].strip()
    
    # Find JSON object boundaries
    start_idx = output.find('{')
    end_idx = output.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        output = output[start_idx:end_idx + 1]
    
    return output


def extract_agent_names(instruction: str) -> list:
    """
    지시문에서 {AgentName} 형태의 Agent 이름들을 추출
    
    Args:
        instruction (str): 입력 지시문
        
    Returns:
        list: 추출된 Agent 이름들의 리스트
    """
    import re
    
    # Find all {AgentName} patterns
    pattern = r'\{([^}]+Agent)\}'
    matches = re.findall(pattern, instruction)
    
    return matches


def format_workflow_result(
    instruction: str,
    generated_json: Dict[str, Any],
    model_type: str = "unknown",
    retry_attempts: int = 0,
    execution_time: float = 0.0
) -> Dict[str, Any]:
    """
    워크플로우 생성 결과를 표준 형식으로 포맷팅
    
    Args:
        instruction: 원본 지시문
        generated_json: 생성된 JSON 워크플로우
        model_type: 모델 타입
        retry_attempts: 재시도 횟수
        execution_time: 실행 시간
        
    Returns:
        Dict: 표준화된 결과 딕셔너리
    """
    return {
        "instruction": instruction,
        "label_json": generated_json,
        "model_type": model_type,
        "retry_attempts": retry_attempts,
        "execution_time": execution_time,
        "extracted_agents": extract_agent_names(instruction),
        "is_valid": validate_json_structure(generated_json)
    }
