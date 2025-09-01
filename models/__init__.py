"""
Models Package - Workflow Agent Models
베이스라인 모델과 LangGraph 모델을 포함하는 패키지
"""

from .baseline_model import BaselineWorkflowAgent
from .langgraph_model import LangGraphRetryAgent

__all__ = [
    "BaselineWorkflowAgent",
    "LangGraphRetryAgent", 
    "create_model"
]


def create_model(model_type: str = "baseline", **kwargs):
    """
    모델 팩토리 함수
    
    Args:
        model_type: "baseline" 또는 "langgraph"
        **kwargs: 모델별 추가 파라미터
    
    Returns:
        WorkflowAgent instance
    """
    if model_type.lower() == "baseline":
        return BaselineWorkflowAgent(**kwargs)
    elif model_type.lower() == "langgraph":
        return LangGraphRetryAgent(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'baseline' or 'langgraph'")


def get_available_models():
    """사용 가능한 모델 목록 반환"""
    return {
        "baseline": {
            "class": "BaselineWorkflowAgent",
            "description": "Simple JSON generation without retry logic",
            "features": ["json_chain"]
        },
        "langgraph": {
            "class": "LangGraphRetryAgent", 
            "description": "LangGraph conditional edge retry logic",
            "features": ["json_chain", "judge_chain", "conditional_edges"]
        }
    }
