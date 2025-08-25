"""
LangGraph-based Workflow Agent
State Graph implementation for JSON workflow generation with validation and retry logic
"""

import json
from typing import Dict, Any, TypedDict, Literal, List, Union
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from prompts import JSON_PROMPT, JSON_EVALUATION_PROMPT
from utils import parse_llm_evaluation
from pydantic_models import (
    WorkflowValidator, 
    LLMWorkflowModel, 
    SequentialWorkflowModel,
    ParallelOnlyWorkflowModel,
    LoopOnlyWorkflowModel
)


class WorkflowState(TypedDict):
    """State for the workflow generation and validation process"""
    instruction: str
    generated_json: Dict[str, Any]
    is_valid: bool
    validation_feedback: str
    retry_count: int
    max_retries: int
    final_result: Dict[str, Any]
    error_message: str
    # Pydantic validation results
    pydantic_valid: bool
    pydantic_error: str
    pydantic_error_type: str
    llm_validation_score: bool


class LangGraphWorkflowAgent:
    """LangGraph-based workflow agent with state graph for generation and validation"""
    
    def __init__(self, model_name: str = "qwen3:4b", max_retries: int = 3):
        """Initialize the LangGraph workflow agent"""
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            num_predict=512
        )
        self.max_retries = max_retries
        
        # Create union type for all workflow models
        WorkflowUnion = Union[LLMWorkflowModel, SequentialWorkflowModel, ParallelOnlyWorkflowModel, LoopOnlyWorkflowModel]
        
        # Initialize chains with structured output
        try:
            # Try to use with_structured_output (preferred method)
            self.structured_llm = self.llm.with_structured_output(WorkflowUnion)
            self.json_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.structured_llm
            self.use_structured_output = True
        except AttributeError:
            # Fallback to StrOutputParser if with_structured_output not available
            self.parser = StrOutputParser()
            self.json_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.llm | self.parser
            self.use_structured_output = False
        
        # Validation chain (still uses string output for LLM evaluation)
        self.validation_chain = ChatPromptTemplate.from_template(JSON_EVALUATION_PROMPT) | self.llm | StrOutputParser()
        
        # Build the state graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        builder = StateGraph(WorkflowState)
        
        # Add nodes
        builder.add_node("generate_workflow", self._generate_workflow_node)
        builder.add_node("validate_workflow", self._validate_workflow_node)
        builder.add_node("finalize_result", self._finalize_result_node)
        builder.add_node("handle_failure", self._handle_failure_node)
        
        # Add edges
        builder.add_edge(START, "generate_workflow")
        builder.add_edge("generate_workflow", "validate_workflow")
        
        # Conditional edge based on validation result
        builder.add_conditional_edges(
            "validate_workflow",
            self._should_retry,
            {
                "retry": "generate_workflow",
                "finalize": "finalize_result",
                "fail": "handle_failure"
            }
        )
        
        builder.add_edge("finalize_result", END)
        builder.add_edge("handle_failure", END)
        
        return builder.compile()
    
    def _generate_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Generate JSON workflow from instruction with structured output"""
        try:
            # Generate improved prompt based on retry feedback
            instruction = state["instruction"]
            if state["retry_count"] > 0 and state["validation_feedback"]:
                enhanced_instruction = f"{instruction}\n\nPrevious attempt failed with feedback: {state['validation_feedback']}\nPlease address these issues in the new generation."
            else:
                enhanced_instruction = instruction
            
            if self.use_structured_output:
                # Use structured output - returns Pydantic model directly
                try:
                    workflow_model = self.json_chain.invoke({"instruction": enhanced_instruction})
                    # Convert Pydantic model to dict
                    workflow_json = workflow_model.dict() if hasattr(workflow_model, 'dict') else workflow_model.model_dump()
                    
                    return {
                        **state,
                        "generated_json": workflow_json,
                        "error_message": "",
                        "pydantic_valid": True,  # Already validated by structured output
                        "pydantic_error": "",
                        "pydantic_error_type": ""
                    }
                    
                except Exception as e:
                    # If structured output fails, fall back to string parsing
                    return self._fallback_generation(state, enhanced_instruction, str(e))
            else:
                # Fallback to string parsing
                return self._fallback_generation(state, enhanced_instruction, "No structured output support")
                
        except Exception as e:
            return {
                **state,
                "generated_json": {},
                "error_message": f"Generation error: {str(e)}",
                "pydantic_valid": False,
                "pydantic_error": str(e),
                "pydantic_error_type": "generation_error"
            }
    
    def _fallback_generation(self, state: WorkflowState, instruction: str, reason: str) -> WorkflowState:
        """Fallback to string parsing when structured output fails"""
        try:
            # Use string-based chain
            if hasattr(self, 'parser'):
                result = self.json_chain.invoke({"instruction": instruction})
            else:
                # Create temporary string chain
                string_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.llm | StrOutputParser()
                result = string_chain.invoke({"instruction": instruction})
            
            workflow_json = json.loads(result.strip())
            
            # Basic validation for fallback
            required_fields = ["flow_name", "type", "sub_agents"]
            if not all(field in workflow_json for field in required_fields):
                raise ValueError(f"Missing required fields: {[f for f in required_fields if f not in workflow_json]}")
            
            return {
                **state,
                "generated_json": workflow_json,
                "error_message": f"Used fallback parsing: {reason}",
                "pydantic_valid": False,  # Will be validated later
                "pydantic_error": f"Fallback: {reason}",
                "pydantic_error_type": "fallback"
            }
            
        except json.JSONDecodeError as e:
            return {
                **state,
                "generated_json": {},
                "error_message": f"JSON parsing error: {str(e)}",
                "pydantic_valid": False,
                "pydantic_error": str(e),
                "pydantic_error_type": "json_parse"
            }
        except Exception as e:
            return {
                **state,
                "generated_json": {},
                "error_message": f"Fallback generation error: {str(e)}",
                "pydantic_valid": False,
                "pydantic_error": str(e),
                "pydantic_error_type": "fallback_error"
            }
    
    def _validate_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """Validate the generated workflow using both Pydantic and LLM validation"""
        if state["error_message"]:
            return {
                **state,
                "is_valid": False,
                "validation_feedback": state["error_message"],
                "pydantic_valid": False,
                "pydantic_error": state["error_message"],
                "pydantic_error_type": "generation_error",
                "llm_validation_score": False
            }
        
        try:
            # Check if Pydantic validation was already done in generation step
            if state.get("pydantic_valid") is True and self.use_structured_output:
                # Skip Pydantic validation - already validated by structured output
                pydantic_valid = True
                pydantic_result = {
                    "is_valid": True,
                    "error_message": "",
                    "error_type": ""
                }
            else:
                # 1. Pydantic Type Validation (for fallback cases)
                pydantic_result = WorkflowValidator.validate_workflow_json(state["generated_json"])
                pydantic_valid = pydantic_result["is_valid"]
            
            # 2. LLM-based Semantic Validation (always perform)
            semantic_result = self._validate_workflow_semantics(
                state["instruction"], 
                state["generated_json"]
            )
            semantic_valid = semantic_result["is_valid"]
            
            # 3. Combined validation decision
            overall_valid = pydantic_valid and semantic_valid
            
            # Generate comprehensive feedback
            feedback_parts = []
            if not pydantic_valid:
                pydantic_feedback = WorkflowValidator.get_validation_feedback(
                    pydantic_result["error_message"], 
                    pydantic_result["error_type"]
                )
                feedback_parts.append(f"Type validation: {pydantic_feedback}")
            
            if not semantic_valid:
                feedback_parts.append(f"Semantic validation: {semantic_result['feedback']}")
            
            if overall_valid:
                final_feedback = "Workflow passed both type and semantic validation"
            else:
                final_feedback = " | ".join(feedback_parts)
            
            # Preserve existing pydantic state from generation or update it
            current_pydantic_valid = state.get("pydantic_valid", pydantic_valid)
            current_pydantic_error = state.get("pydantic_error", pydantic_result["error_message"])
            current_pydantic_error_type = state.get("pydantic_error_type", pydantic_result["error_type"])
            
            return {
                **state,
                "is_valid": overall_valid,
                "validation_feedback": final_feedback,
                "pydantic_valid": current_pydantic_valid,
                "pydantic_error": current_pydantic_error,
                "pydantic_error_type": current_pydantic_error_type,
                "llm_validation_score": semantic_valid
            }
                
        except Exception as e:
            return {
                **state,
                "is_valid": False,
                "validation_feedback": f"Validation error: {str(e)}",
                "pydantic_valid": False,
                "pydantic_error": f"Exception: {str(e)}",
                "pydantic_error_type": "exception",
                "llm_validation_score": False
            }
    
    def _validate_workflow_semantics(self, instruction: str, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic validation of the workflow"""
        feedback_points = []
        
        # Check workflow type appropriateness
        workflow_type = workflow_json.get("type")
        
        # Type-specific validations
        if workflow_type == "LLM":
            if "tools" not in workflow_json:
                feedback_points.append("LLM type workflows should include 'tools' field")
            elif not isinstance(workflow_json["tools"], list) or len(workflow_json["tools"]) == 0:
                feedback_points.append("LLM type workflows should have at least one tool")
        
        elif workflow_type == "Sequential":
            sub_agents = workflow_json.get("sub_agents", [])
            if len(sub_agents) < 2:
                feedback_points.append("Sequential workflows should have at least 2 sub_agents")
        
        elif workflow_type == "Parallel":
            sub_agents = workflow_json.get("sub_agents", [])
            if len(sub_agents) < 2:
                feedback_points.append("Parallel workflows should have at least 2 sub_agents")
        
        # Check agent naming conventions
        sub_agents = workflow_json.get("sub_agents", [])
        for agent in sub_agents:
            if "agent_name" in agent:
                if not agent["agent_name"].endswith("Agent"):
                    feedback_points.append(f"Agent name '{agent['agent_name']}' should end with 'Agent'")
        
        # Check if workflow matches instruction intent
        if "Q&A" in instruction or "문의" in instruction or "답변" in instruction:
            if workflow_type != "LLM":
                feedback_points.append("Q&A or inquiry instructions should use LLM type workflow")
        
        if "동시" in instruction or "병렬" in instruction or "parallel" in instruction.lower():
            if "Parallel" not in json.dumps(workflow_json):
                feedback_points.append("Instructions mentioning parallel execution should include Parallel type")
        
        if "반복" in instruction or "개선" in instruction or "iterative" in instruction.lower():
            if "Loop" not in json.dumps(workflow_json):
                feedback_points.append("Instructions mentioning iteration should include Loop type")
        
        is_valid = len(feedback_points) == 0
        feedback = "; ".join(feedback_points) if feedback_points else "Valid workflow"
        
        return {
            "is_valid": is_valid,
            "feedback": feedback
        }
    
    def _should_retry(self, state: WorkflowState) -> Literal["retry", "finalize", "fail"]:
        """Decide whether to retry, finalize, or fail based on validation result"""
        if state["is_valid"]:
            return "finalize"
        elif state["retry_count"] < state["max_retries"]:
            # Increment retry count for the next attempt
            state["retry_count"] = state["retry_count"] + 1
            return "retry"
        else:
            return "fail"
    
    def _finalize_result_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the successful result"""
        return {
            **state,
            "final_result": {
                "instruction": state["instruction"],
                "label_json": state["generated_json"],
                "success": True,
                "retry_count": state["retry_count"],
                "validation_feedback": state["validation_feedback"],
                "pydantic_valid": state["pydantic_valid"],
                "pydantic_error_type": state["pydantic_error_type"],
                "llm_validation_score": state["llm_validation_score"]
            }
        }
    
    def _handle_failure_node(self, state: WorkflowState) -> WorkflowState:
        """Handle failure after max retries"""
        return {
            **state,
            "final_result": {
                "instruction": state["instruction"],
                "label_json": {"flow_name": "FailedWorkflow", "type": "Sequential", "sub_agents": []},
                "success": False,
                "retry_count": state["retry_count"],
                "validation_feedback": state["validation_feedback"],
                "pydantic_valid": state.get("pydantic_valid", False),
                "pydantic_error_type": state.get("pydantic_error_type", "unknown"),
                "llm_validation_score": state.get("llm_validation_score", False),
                "error": "Max retries exceeded"
            }
        }
    
    def generate_workflow(self, instruction: str) -> Dict[str, Any]:
        """Generate workflow using the state graph"""
        initial_state = WorkflowState(
            instruction=instruction,
            generated_json={},
            is_valid=False,
            validation_feedback="",
            retry_count=0,
            max_retries=self.max_retries,
            final_result={},
            error_message="",
            pydantic_valid=False,
            pydantic_error="",
            pydantic_error_type="",
            llm_validation_score=False
        )
        
        result = self.graph.invoke(initial_state)
        return result["final_result"]
    
    def evaluate_json_with_llm(self, instruction: str, expected_json: str, generated_json: str) -> bool:
        """Evaluate JSON using LLM (keeping compatibility with existing code)"""
        try:
            result = self.validation_chain.invoke({
                "instruction": instruction,
                "expected_json": expected_json,
                "generated_json": generated_json
            })
            return parse_llm_evaluation(result)
        except Exception as e:
            print(f"JSON LLM evaluation error: {e}")
            return False


# For backward compatibility
WorkflowAgent = LangGraphWorkflowAgent
