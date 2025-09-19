"""
LangGraph Model 2 - 3-Stage Workflow Generation

Three-stage pipeline:
1) Predict workflow type
2) Generate workflow JSON
3) Validate the generated workflow
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .llm_registry import init_llm
from prompts import (
    WORKFLOW_TYPE_PREDICTION_PROMPT,
    WORKFLOW_GENERATION_PROMPT,
    WORKFLOW_VALIDATION_PROMPT,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ThreeStageWorkflowState(TypedDict):
    """State container for the 3-stage workflow.

    Attributes:
        instruction: Original user instruction.
        predicted_type: Predicted workflow type (LLM / Sequential / Loop / Parallel).
        prediction_reason: Reasoning for the predicted type.
        generated_json: Workflow JSON produced by the generation stage.
        json_valid: True if validation judged the JSON as valid.
        type_consistent: True if type matches prediction.
        intent_matched: True if output matches user intent.
        overall_passed: Final validation result.
        validation_reason: Reason for pass/fail from validation.
        error_message: Last error message (if any).
        retry_count: Number of retries attempted.
        max_retries: Retry limit.
        success: Overall success flag (same as overall_passed).
    """

    instruction: str
    predicted_type: str
    prediction_reason: str
    generated_json: Dict[str, Any]
    json_valid: bool
    type_consistent: bool
    intent_matched: bool
    overall_passed: bool
    validation_reason: str
    error_message: str
    retry_count: int
    max_retries: int
    success: bool


class ThreeStageWorkflowAgent:
    """3-Stage Workflow Generation Agent.

    Pipeline:
      1. Predict workflow type (LLM / Sequential / Loop / Parallel)
      2. Generate workflow JSON based on predicted type
      3. Validate the generated workflow and optionally retry generation
    """

    def __init__(self, model_name: str = "gpt-4o-mini", max_retries: int = 3) -> None:
        """Initialize the agent and build the graph.

        Args:
            model_name: LLM name resolved via the registry.
            max_retries: Maximum number of retries upon validation failure.
        """
        self.model_name = model_name
        self.max_retries = max_retries

        self.llm = init_llm(model_name)
        self.parser = StrOutputParser()

        # Chains
        self.prediction_chain = (
            ChatPromptTemplate.from_template(WORKFLOW_TYPE_PREDICTION_PROMPT)
            | self.llm
            | self.parser
        )
        self.generation_chain = (
            ChatPromptTemplate.from_template(WORKFLOW_GENERATION_PROMPT)
            | self.llm
            | self.parser
        )
        self.validation_chain = (
            ChatPromptTemplate.from_template(WORKFLOW_VALIDATION_PROMPT)
            | self.llm
            | self.parser
        )

        # Graph
        self.graph = self._build_graph()

    # ---- Stage Nodes ---------------------------------------------------------
    def _predict_workflow_type_node(self, state: ThreeStageWorkflowState) -> Dict[str, Any]:
        """Stage 1: Predict workflow type.

        Args:
            state: Current workflow state.

        Returns:
            Partial state update with predicted type and reasoning.
        """
        start = time.time()
        try:
            result = self.prediction_chain.invoke({"instruction": state["instruction"]})
            try:
                parsed = json.loads(result.strip())
                predicted_type = parsed.get("type", "Sequential")
                reason = parsed.get("reason", "No reason provided")
            except json.JSONDecodeError:
                valid_types = ["LLM", "Sequential", "Loop", "Parallel"]
                predicted_type = next((t for t in valid_types if t in result), "Sequential")
                reason = "Fallback extraction from text"
            elapsed = time.time() - start
            print(f"Stage 1 - Predicted: {predicted_type} ({elapsed:.2f}s)")
            return {"predicted_type": predicted_type, "prediction_reason": reason, "error_message": ""}
        except Exception as e:
            elapsed = time.time() - start
            print(f"Stage 1 - Error ({elapsed:.2f}s)")
            return {
                "predicted_type": "Sequential",
                "prediction_reason": f"Prediction error: {e}",
                "error_message": f"Type prediction error: {e}",
            }

    def _generate_workflow_node(self, state: ThreeStageWorkflowState) -> Dict[str, Any]:
        """Stage 2: Generate workflow JSON.

        Args:
            state: Current workflow state.

        Returns:
            Partial state update with generated JSON or fallback.
        """
        start = time.time()
        try:
            result = self.generation_chain.invoke(
                {"instruction": state["instruction"], "predicted_type": state["predicted_type"]}
            )
            generated_json = json.loads(result.strip())
            elapsed = time.time() - start
            print(f"Stage 2 - Generated ({elapsed:.2f}s)")
            return {"generated_json": generated_json, "error_message": ""}
        except Exception as e:
            elapsed = time.time() - start
            print(f"Stage 2 - Error ({elapsed:.2f}s)")
            return {
                "generated_json": self._get_fallback_json(state.get("predicted_type", "Sequential")),
                "error_message": f"Workflow generation error: {e}",
            }

    def _validate_workflow_node(self, state: ThreeStageWorkflowState) -> Dict[str, Any]:
        """Stage 3: Validate generated workflow.

        Args:
            state: Current workflow state.

        Returns:
            Partial state update with validation results.
        """
        start = time.time()
        try:
            result = self.validation_chain.invoke(
                {
                    "instruction": state["instruction"],
                    "predicted_type": state["predicted_type"],
                    "generated_json": json.dumps(state["generated_json"], ensure_ascii=False),
                }
            )
            try:
                parsed = json.loads(result.strip())
                overall_passed = parsed.get("passed", False)
                validation_reason = parsed.get("reason", "No reason provided")
            except json.JSONDecodeError:
                text = (result or "").lower()
                overall_passed = ("true" in text) or ("pass" in text)
                validation_reason = f"Fallback textual evaluation: {result[:100]}"

            elapsed = time.time() - start
            status = "PASSED" if overall_passed else "FAILED"
            print(f"Stage 3 - Validation {status} ({elapsed:.2f}s)")

            return {
                "json_valid": True,
                "type_consistent": overall_passed,
                "intent_matched": overall_passed,
                "overall_passed": overall_passed,
                "validation_reason": validation_reason,
                "success": overall_passed,
                "retry_count": state["retry_count"] + (0 if overall_passed else 1),
                "error_message": "" if overall_passed else validation_reason,
            }
        except Exception as e:
            elapsed = time.time() - start
            print(f"Stage 3 - Error ({elapsed:.2f}s)")
            return {
                "json_valid": False,
                "type_consistent": False,
                "intent_matched": False,
                "overall_passed": False,
                "validation_reason": f"API error: {e}",
                "success": False,
                "retry_count": state["retry_count"] + 1,
                "error_message": f"Validation error: {e}",
            }

    # ---- Conditional Edges ---------------------------------------------------
    def _should_retry(self, state: ThreeStageWorkflowState) -> str:
        """Decide whether to retry generation or end.

        Args:
            state: Current workflow state.

        Returns:
            "retry" or "end".
        """
        if state.get("overall_passed", False):
            return "end"
        if state["retry_count"] >= state["max_retries"]:
            print(f"Max retries reached ({state['max_retries']})")
            return "end"
        print(f"Retry {state['retry_count']}/{state['max_retries']}")
        return "retry"

    # ---- Helpers -------------------------------------------------------------
    def _get_fallback_json(self, predicted_type: str) -> Dict[str, Any]:
        """Provide a minimal, type-aware fallback JSON."""
        fallbacks = {
            "LLM": {"flow_name": "DefaultSystem", "type": "LLM", "tools": [{"agent_name": "DefaultAgent"}]},
            "Loop": {
                "flow_name": "DefaultPipeline",
                "type": "Sequential",
                "sub_agents": [
                    {"agent_name": "InitialAgent"},
                    {
                        "flow": {
                            "flow_name": "DefaultLoop",
                            "type": "Loop",
                            "sub_agents": [{"agent_name": "ReviewAgent"}, {"agent_name": "RefineAgent"}],
                        }
                    },
                ],
            },
            "Parallel": {
                "flow_name": "DefaultPipeline",
                "type": "Sequential",
                "sub_agents": [
                    {
                        "flow": {
                            "flow_name": "DefaultParallel",
                            "type": "Parallel",
                            "sub_agents": [{"agent_name": "Agent1"}, {"agent_name": "Agent2"}],
                        }
                    },
                    {"agent_name": "SynthesisAgent"},
                ],
            },
        }
        return fallbacks.get(
            predicted_type,
            {"flow_name": "DefaultPipeline", "type": "Sequential", "sub_agents": [{"agent_name": "DefaultAgent"}]},
        )

    def _build_graph(self) -> StateGraph:
        """Build the 3-stage LangGraph state machine."""
        workflow = StateGraph(ThreeStageWorkflowState)
        workflow.add_node("predict_type", self._predict_workflow_type_node)
        workflow.add_node("generate_workflow", self._generate_workflow_node)
        workflow.add_node("validate_workflow", self._validate_workflow_node)

        workflow.add_edge("predict_type", "generate_workflow")
        workflow.add_edge("generate_workflow", "validate_workflow")
        workflow.add_conditional_edges(
            "validate_workflow",
            self._should_retry,
            {"retry": "generate_workflow", "end": END},
        )
        workflow.set_entry_point("predict_type")
        return workflow.compile()

    # ---- Public API ----------------------------------------------------------
    def generate_workflow(self, instruction: str, save_graph: bool = False) -> Dict[str, Any]:
        """Run the 3-stage workflow and return a result payload.

        Args:
            instruction: Natural-language instruction to transform into a workflow.
            save_graph: If True, attempt to save a PNG diagram of the graph (best-effort).

        Returns:
            Dict with instruction, predicted type, generated JSON, validation flags, retries, and timings.
        """
        total_start = time.time()

        if save_graph:
            # Best effort; ignore failures
            _ = self.save_graph_as_png()

        initial_state: ThreeStageWorkflowState = {
            "instruction": instruction,
            "predicted_type": "",
            "prediction_reason": "",
            "generated_json": {},
            "json_valid": False,
            "type_consistent": False,
            "intent_matched": False,
            "overall_passed": False,
            "validation_reason": "",
            "error_message": "",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "success": False,
        }

        print("3-Stage Workflow Generation")
        print(instruction)
        print("-" * 60)

        try:
            final_state = self.graph.invoke(initial_state)
            total_time = time.time() - total_start
            print("-" * 60)
            print(f"Total Time: {total_time:.2f}s")

            return {
                "instruction": instruction,
                "predicted_type": final_state["predicted_type"],
                "prediction_reason": final_state.get("prediction_reason", ""),
                "label_json": final_state["generated_json"],
                "model_type": "langgraph_3stage",
                "retry_attempts": final_state["retry_count"],
                "success": final_state["success"],
                "overall_passed": final_state["overall_passed"],
                "json_valid": final_state["json_valid"],
                "type_consistent": final_state["type_consistent"],
                "intent_matched": final_state["intent_matched"],
                "validation_reason": final_state.get("validation_reason", ""),
                "error_message": final_state.get("error_message", ""),
                "total_time": total_time,
            }
        except Exception as e:
            total_time = time.time() - total_start
            print(f"Graph Error: {e} ({total_time:.2f}s)")
            return {
                "instruction": instruction,
                "predicted_type": "Unknown",
                "prediction_reason": f"Graph error: {e}",
                "model_type": "langgraph_3stage",
                "label_json": {"type": "Sequential", "sub_agents": [{"agent_name": "DefaultAgent"}]},
                "retry_attempts": 0,
                "success": False,
                "overall_passed": False,
                "json_valid": False,
                "type_consistent": False,
                "intent_matched": False,
                "validation_reason": "Graph execution failed",
                "total_time": total_time,
                "error_message": f"Graph execution error: {e}",
            }

    def save_graph_as_png(self, output_dir: str = "./models") -> str:
        """Save graph as PNG without Mermaid-text fallback.

        Tries `draw_mermaid_png` first, then legacy `draw_png`. On failure, returns "".

        Args:
            output_dir: Directory to save the PNG.

        Returns:
            Path to the PNG, or "" on failure.
        """
        try:
            import os
            from datetime import datetime

            os.makedirs(output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir, f"langgraph_3stage_workflow_{ts}.png")

            try:
                self.graph.get_graph().draw_mermaid_png(output_file_path=path)
                print(f"Graph saved as PNG: {path}")
                return path
            except Exception as e1:
                try:
                    self.graph.get_graph().draw_png(output_file_path=path)
                    print(f"Graph saved as PNG (legacy): {path}")
                    return path
                except Exception as e2:
                    print(f"Failed to save graph as PNG: mermaid_png error={e1}; draw_png error={e2}")
                    return ""
        except Exception as e:
            print(f"Failed to save graph as PNG (setup error): {e}")
            return ""

    def get_model_info(self) -> Dict[str, str]:
        """Return model metadata."""
        return {
            "model_type": "langgraph_3stage",
            "model_name": self.model_name,
            "description": "3-Stage Workflow Generation: Predict -> Generate -> Validate",
            "features": "type_prediction + workflow_generation + validation",
            "max_retries": str(self.max_retries),
        }


# ---- Test Entrypoints --------------------------------------------------------
def test_langgraph_3stage_workflow() -> None:
    """Smoke test for 3-stage workflow generation (no graph saving)."""
    print("Testing 3-Stage Workflow Generation Model")
    print("=" * 60)

    model = ThreeStageWorkflowAgent(max_retries=2)

    instruction = (
        "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. "
        "{텍스트작성Agent}, {이미지생성Agent}, {동영상편집Agent}가 동시에 작업하고 "
        "{콘텐츠통합Agent}가 최종 결과물을 만들도록 해"
    )

    result = model.generate_workflow(instruction)

    print("\nResults:")
    print(
        f"  Type: {result['predicted_type']} | Success: {result['success']} | "
        f"Retries: {result['retry_attempts']} | Time: {result.get('total_time', 0):.2f}s"
    )
    if not result["success"] or not result["overall_passed"]:
        print(f"  Reason: {result.get('validation_reason', 'N/A')}")
    if result.get("error_message"):
        print(f"  Error: {result['error_message']}")

    print("\nGenerated JSON:")
    print(json.dumps(result["label_json"], ensure_ascii=False, indent=2))


def test_graph_saving() -> None:
    """Test only the graph saving helper."""
    print("Testing Graph Saving")
    print("=" * 50)

    model = ThreeStageWorkflowAgent(max_retries=2)
    saved = model.save_graph_as_png("./models")
    if saved:
        print(f"Saved: {saved}")
    else:
        print("Graph saving failed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-graph":
        test_graph_saving()
    else:
        test_langgraph_3stage_workflow()