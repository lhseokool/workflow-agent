"""
LangGraph Model - Conditional Edge Retry Logic

A LangGraph-based agent that uses conditional edges for retry logic.
- No JSON repair: preserves raw output on parse failure
- Skips judge when JSON parsing fails
- Returns a compact result payload
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, TypedDict, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from .llm_registry import init_llm
from prompts import JSON_PROMPT, LLM_JUDGE_WITH_REASEON_PROMPT
from utils import parse_llm_evaluation

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class WorkflowState(TypedDict):
    """State container for the LangGraph workflow."""
    instruction: str
    generated_json: Union[Dict[str, Any], str]
    json_valid: bool
    label_json_parse_error: Optional[str]
    retry_count: int
    max_retries: int
    judge_passed: bool
    judge_reason: str
    success: bool
    error_msg: str


class LangGraphRetryAgent:
    """LangGraph agent with conditional-edge-based retries."""

    def __init__(self, model_name: str = "gpt-4o-mini", max_retries: int = 3) -> None:
        """Initialize the agent and build the graph."""
        self.model_name = model_name
        self.max_retries = max_retries

        self.llm = init_llm(model_name)
        self.parser = StrOutputParser()

        self.json_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.llm | self.parser
        self.judge_chain = (
            ChatPromptTemplate.from_template(LLM_JUDGE_WITH_REASEON_PROMPT) | self.llm | self.parser
        )

        self.graph = self._build_graph()

    # ----- LangGraph nodes -----

    def _generate_json(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: generate JSON (no repair; preserve raw on parse failure)."""
        start = time.time()
        instruction = state["instruction"]
        result: Optional[str] = None

        try:
            result = self.json_chain.invoke({"instruction": instruction})
            parsed = json.loads((result or "").strip())
            print(f"JSON generated in {time.time() - start:.2f}s")
            return {
                "generated_json": parsed,
                "json_valid": True,
                "label_json_parse_error": None,
                "error_msg": "",
            }
        except json.JSONDecodeError as e:
            print(f"JSON parse error in {time.time() - start:.2f}s")
            if result is not None:
                print(f"Raw output: {result}")
            return {
                "generated_json": result if result is not None else "",
                "json_valid": False,
                "label_json_parse_error": str(e),
                "error_msg": f"JSON parsing error: {str(e)}",
            }
        except Exception as e:
            print(f"JSON generation error in {time.time() - start:.2f}s")
            return {
                "generated_json": result if result is not None else "",
                "json_valid": False,
                "label_json_parse_error": f"{type(e).__name__}: {e}",
                "error_msg": f"JSON generation error: {e}",
            }

    def _judge_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Node: judge validation (skips when parsing failed)."""
        if not state.get("json_valid", True):
            reason = f"Skipped judge due to JSON parse error: {state.get('label_json_parse_error')}"
            print(f"Judge skipped: {reason}")
            return {
                "judge_passed": False,
                "judge_reason": reason,
                "success": False,
                "retry_count": state["retry_count"] + 1,
            }

        start = time.time()
        try:
            result = self.judge_chain.invoke(
                {
                    "instruction": state["instruction"],
                    "generated_json": json.dumps(state["generated_json"], ensure_ascii=False),
                }
            )

            try:
                judge_result = json.loads((result or "").strip())
                judge_passed = judge_result.get("passed", False)
                judge_reason = judge_result.get("reason", "No reason provided")
            except json.JSONDecodeError:
                judge_passed = parse_llm_evaluation(result or "")
                judge_reason = f"Simple evaluation result: {(result or '').strip()}"

            print(f"Judge {'PASSED' if judge_passed else 'FAILED'} in {time.time() - start:.2f}s")
            if not judge_passed:
                print(f"Reason: {judge_reason}")

            return {
                "judge_passed": judge_passed,
                "judge_reason": judge_reason,
                "success": judge_passed,
                "retry_count": state["retry_count"] + (0 if judge_passed else 1),
            }
        except Exception as e:
            err = f"Judge error: {str(e)}"
            print(f"Judge error in {time.time() - start:.2f}s")
            print(f"Reason: {err}")
            return {
                "judge_passed": False,
                "judge_reason": err,
                "success": False,
                "retry_count": state["retry_count"] + 1,
                "error_msg": err,
            }

    # ----- Graph wiring -----

    def _should_retry(self, state: WorkflowState) -> str:
        """Edge condition: decide whether to retry or end."""
        if not state.get("json_valid", True):
            if state["retry_count"] >= state["max_retries"]:
                print(f"Max retries reached for JSON parsing ({state['max_retries']})")
                return "end"
            print(f"Retry due to JSON parse failure {state['retry_count']}/{state['max_retries']}")
            return "retry"

        if state["judge_passed"]:
            return "end"

        if state["retry_count"] >= state["max_retries"]:
            print(f"Max retries reached for judge validation ({state['max_retries']})")
            return "end"

        print(f"Retry due to judge failure {state['retry_count']}/{state['max_retries']}")
        return "retry"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(WorkflowState)
        workflow.add_node("generate_json", self._generate_json)
        workflow.add_node("judge", self._judge_node)
        workflow.add_edge("generate_json", "judge")
        workflow.add_conditional_edges("judge", self._should_retry, {"retry": "generate_json", "end": END})
        workflow.set_entry_point("generate_json")
        return workflow.compile()

    # ----- Public API -----

    def generate_workflow(self, instruction: str) -> Dict[str, Any]:
        """Run the graph and return a compact result payload."""
        total_start = time.time()

        initial_state: WorkflowState = {
            "instruction": instruction,
            "generated_json": {},
            "json_valid": True,
            "label_json_parse_error": None,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "judge_passed": False,
            "judge_reason": "",
            "success": False,
            "error_msg": "",
        }

        print("LangGraph Conditional Workflow")
        print(instruction)
        print("-" * 60)

        try:
            final_state = self.graph.invoke(initial_state)
            total_time = time.time() - total_start

            print("-" * 60)
            print(f"Total time: {total_time:.2f}s")

            return {
                "instruction": instruction,
                "label_json": final_state["generated_json"],
                "label_json_parse_error": final_state.get("label_json_parse_error"),
                "model_type": "langgraph_retry",
                "retry_attempts": final_state.get("retry_count", 0),
            }
        except Exception as e:
            total_time = time.time() - total_start
            print(f"Graph error: {str(e)} ({total_time:.2f}s)")
            return {
                "instruction": instruction,
                "label_json": {},
                "label_json_parse_error": f"Graph execution error: {e}",
                "model_type": "langgraph_retry",
                "retry_attempts": 0,
            }

    def save_graph_as_png(self, output_dir: str = "./models") -> str:
        """Save the graph as PNG (no Mermaid text fallback)."""
        try:
            import os
            from datetime import datetime

            os.makedirs(output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir, f"langgraph_retry_workflow_{ts}.png")

            try:
                # Preferred: Mermaid backend (may require internet unless using PYPPETEER method)
                self.graph.get_graph().draw_mermaid_png(output_file_path=path)
                print(f"Graph saved as PNG: {path}")
                return path
            except Exception as e1:
                # Fallback: legacy PNG via pygraphviz
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
        """Return minimal model metadata."""
        return {
            "model_type": "langgraph_retry",
            "model_name": self.model_name,
            "description": "LangGraph-based agent with conditional retry logic",
            "features": "json_generation + llm_judge + conditional_retry",
            "max_retries": str(self.max_retries),
        }


# ----- Simple tests -----

def test_langgraph_retry_workflow() -> None:
    """Smoke test for the conditional-edge model."""
    print("Testing LangGraph Conditional Edge Model")
    print("=" * 60)

    model = LangGraphRetryAgent(max_retries=2)

    instruction = (
        "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. "
        "{텍스트작성Agent}, {이미지생성Agent}, {동영상편집Agent}가 동시에 작업하고 "
        "{콘텐츠통합Agent}가 최종 결과물을 만들도록 해"
    )

    result = model.generate_workflow(instruction)

    print("\nResults:")
    print(
        f"  Model Type: {result['model_type']} | "
        f"Retries: {result['retry_attempts']}"
    )
    if result.get("label_json_parse_error"):
        print(f"  JSON Parse Error: {result['label_json_parse_error']}")

    print("\nGenerated JSON (raw or parsed):")
    lj = result["label_json"]
    if isinstance(lj, dict):
        print(json.dumps(lj, ensure_ascii=False, indent=2))
    else:
        print(str(lj)[:2000])


def test_graph_saving() -> None:
    """Test graph saving helper."""
    print("Testing Graph Saving")
    print("=" * 50)
    model = LangGraphRetryAgent(max_retries=2)
    print("Saving graph...")
    saved_path = model.save_graph_as_png("./models")
    if saved_path:
        print(f"Saved: {saved_path}")
        import os
        if os.path.exists(saved_path):
            print(f"File size: {os.path.getsize(saved_path)} bytes")
        else:
            print("File not found after saving")
    else:
        print("Graph saving failed")


if __name__ == "__main__":
    test_langgraph_retry_workflow()
    print("\n" + "=" * 60)
    test_graph_saving()