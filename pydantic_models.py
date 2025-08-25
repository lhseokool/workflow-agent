"""
Pydantic Models for Workflow JSON Validation
Defines strict type validation for 4 workflow types
"""

from typing import Dict, Any, List, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator


class AgentModel(BaseModel):
    """Individual agent model"""
    agent_name: str = Field(..., description="Agent name, must end with 'Agent'")
    
    @validator('agent_name')
    def validate_agent_name(cls, v):
        if not v.endswith('Agent'):
            raise ValueError("Agent name must end with 'Agent'")
        if len(v) < 6:  # At least 'XAgent'
            raise ValueError("Agent name too short")
        return v


class ToolModel(BaseModel):
    """Tool agent model for LLM workflows"""
    agent_name: str = Field(..., description="Tool agent name, must end with 'Agent'")
    
    @validator('agent_name')
    def validate_tool_name(cls, v):
        if not v.endswith('Agent'):
            raise ValueError("Tool agent name must end with 'Agent'")
        return v


class LoopWorkflowModel(BaseModel):
    """Loop workflow model (nested)"""
    flow_name: str = Field(..., description="Loop workflow name")
    type: Literal["Loop"] = Field(..., description="Must be 'Loop'")
    sub_agents: List[AgentModel] = Field(..., min_items=2, description="At least 2 agents for loop")


class ParallelWorkflowModel(BaseModel):
    """Parallel workflow model (nested)"""
    flow_name: str = Field(..., description="Parallel workflow name")
    type: Literal["Parallel"] = Field(..., description="Must be 'Parallel'")
    sub_agents: List[AgentModel] = Field(..., min_items=2, description="At least 2 agents for parallel execution")


class FlowWrapper(BaseModel):
    """Wrapper for nested workflows"""
    flow: Union[LoopWorkflowModel, ParallelWorkflowModel] = Field(..., description="Nested workflow")


# Simplified Union handling for sub_agents - direct Union without wrapper


class LLMWorkflowModel(BaseModel):
    """LLM Type Workflow - Q&A and assistance tasks"""
    flow_name: str = Field(..., description="Workflow name")
    type: Literal["LLM"] = Field(..., description="Must be 'LLM'")
    sub_agents: List[AgentModel] = Field(..., min_items=1, description="Main agents")
    tools: List[ToolModel] = Field(..., min_items=1, description="Tool agents for LLM")


class SequentialWorkflowModel(BaseModel):
    """Sequential Type Workflow - Step-by-step execution"""
    flow_name: str = Field(..., description="Workflow name")
    type: Literal["Sequential"] = Field(..., description="Must be 'Sequential'")
    sub_agents: List[Union[AgentModel, FlowWrapper]] = Field(..., min_items=1, description="Sequential agents or nested workflows")


class ParallelOnlyWorkflowModel(BaseModel):
    """Parallel Only Type Workflow - Simultaneous execution"""
    flow_name: str = Field(..., description="Workflow name")
    type: Literal["Parallel"] = Field(..., description="Must be 'Parallel'")
    sub_agents: List[AgentModel] = Field(..., min_items=2, description="At least 2 agents for parallel")


class LoopOnlyWorkflowModel(BaseModel):
    """Loop Only Type Workflow - Repetitive execution"""
    flow_name: str = Field(..., description="Workflow name")
    type: Literal["Loop"] = Field(..., description="Must be 'Loop'")
    sub_agents: List[AgentModel] = Field(..., min_items=2, description="At least 2 agents for loop")


# Union of all workflow types
WorkflowModel = Union[
    LLMWorkflowModel,
    SequentialWorkflowModel,
    ParallelOnlyWorkflowModel,
    LoopOnlyWorkflowModel
]


class WorkflowValidator:
    """Validator class for workflow JSON using Pydantic models"""
    
    @staticmethod
    def validate_workflow_json(workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate workflow JSON using Pydantic models
        Returns: {
            "is_valid": bool,
            "validated_data": dict or None,
            "error_message": str,
            "error_type": str  # "structure", "type", "validation"
        }
        """
        try:
            # Determine workflow type
            workflow_type = workflow_data.get("type")
            
            if not workflow_type:
                return {
                    "is_valid": False,
                    "validated_data": None,
                    "error_message": "Missing 'type' field",
                    "error_type": "structure"
                }
            
            # Validate based on type
            if workflow_type == "LLM":
                validated = LLMWorkflowModel(**workflow_data)
            elif workflow_type == "Sequential":
                validated = SequentialWorkflowModel(**workflow_data)
            elif workflow_type == "Parallel":
                validated = ParallelOnlyWorkflowModel(**workflow_data)
            elif workflow_type == "Loop":
                validated = LoopOnlyWorkflowModel(**workflow_data)
            else:
                return {
                    "is_valid": False,
                    "validated_data": None,
                    "error_message": f"Invalid workflow type: {workflow_type}. Must be one of: LLM, Sequential, Parallel, Loop",
                    "error_type": "type"
                }
            
            return {
                "is_valid": True,
                "validated_data": validated.dict(),
                "error_message": "",
                "error_type": ""
            }
            
        except Exception as e:
            error_message = str(e)
            error_type = "validation"
            
            # Categorize error types
            if "field required" in error_message.lower():
                error_type = "structure"
            elif "must end with" in error_message.lower():
                error_type = "naming"
            elif "min_items" in error_message.lower():
                error_type = "count"
            
            return {
                "is_valid": False,
                "validated_data": None,
                "error_message": error_message,
                "error_type": error_type
            }
    
    @staticmethod
    def get_validation_feedback(error_message: str, error_type: str) -> str:
        """Generate human-readable feedback for validation errors"""
        feedback_map = {
            "structure": "Missing required fields or incorrect JSON structure",
            "type": "Invalid workflow type - must be LLM, Sequential, Parallel, or Loop",
            "naming": "Agent names must end with 'Agent'",
            "count": "Insufficient number of agents for the workflow type",
            "validation": "General validation error"
        }
        
        base_feedback = feedback_map.get(error_type, "Validation error")
        return f"{base_feedback}: {error_message}"


# Export main classes
__all__ = [
    "WorkflowValidator",
    "LLMWorkflowModel", 
    "SequentialWorkflowModel",
    "ParallelOnlyWorkflowModel",
    "LoopOnlyWorkflowModel",
    "AgentModel",
    "ToolModel"
]
