"""
Baseline Model - Simple JSON Generation

A minimal baseline model that generates workflow JSON via a single chain.
No retry logic; JSON is either parsed or the raw string is returned.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm_registry import init_llm
from prompts import JSON_PROMPT, JSON_TO_INSTRUCTION_PROMPT

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BaselineWorkflowAgent:
    """Baseline workflow agent (no retries, no repair).

    Attributes:
        model_name: LLM identifier used by the registry.
        llm: Initialized LLM instance from the registry.
        parser: Output parser converting model output to string.
        json_chain: Chain for instruction → JSON generation.
        json_to_instruction_chain: Chain for JSON → instruction conversion.
    """

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        """Initialize chain components.

        Args:
            model_name: LLM name to initialize via the registry.
        """
        self.model_name = model_name
        self.llm = init_llm(model_name)
        self.parser = StrOutputParser()

        # Instruction → JSON
        self.json_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.llm | self.parser
        # JSON → Instruction
        self.json_to_instruction_chain = (
            ChatPromptTemplate.from_template(JSON_TO_INSTRUCTION_PROMPT) | self.llm | self.parser
        )

    def _generate_json(self, instruction: str) -> Tuple[Union[Dict[str, Any], str], Optional[str]]:
        """Generate JSON for the given instruction.

        The method does not repair or auto-correct JSON. If parsing fails,
        the raw model output is returned with an error message.

        Args:
            instruction: Natural-language instruction in Korean.

        Returns:
            A tuple of:
              - dict on success, or raw string on parse failure
              - None on success, or error message string on failure
        """
        raw: str = ""
        # 1) Generation
        try:
            raw = self.json_chain.invoke({"instruction": instruction})
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"

        # 2) Parsing (no light-repair)
        try:
            parsed = json.loads(raw.strip())
            return parsed, None
        except json.JSONDecodeError as e:
            return raw, str(e)

    def generate_workflow(self, instruction: str) -> Dict[str, Any]:
        """Generate a workflow result payload.

        Args:
            instruction: Natural-language instruction.

        Returns:
            Dict containing original instruction, JSON content (dict or raw str),
            parse error (if any), and basic metadata.
        """
        json_result, parse_err = self._generate_json(instruction)
        return {
            "instruction": instruction,
            "label_json": json_result,  # dict or str
            "label_json_parse_error": parse_err,  # str or None
            "model_type": "baseline",
            "retry_attempts": 0,
        }

    def json_to_instruction(self, workflow_json: Dict[str, Any]) -> str:
        """Convert workflow JSON back to a Korean instruction.

        Args:
            workflow_json: Structured workflow JSON.

        Returns:
            Reconstructed instruction text, or an error message on failure.
        """
        try:
            json_str = json.dumps(workflow_json, ensure_ascii=False)
            result = self.json_to_instruction_chain.invoke({"workflow_json": json_str})
            return result.strip()
        except Exception as e:
            return f"Error converting workflow JSON to instruction: {str(e)}"

    def get_model_info(self) -> Dict[str, str]:
        """Return model metadata."""
        return {
            "model_type": "baseline",
            "model_name": self.model_name,
            "description": "Simple JSON generation without retry logic",
            "features": "json_chain + json_to_instruction_chain",
        }


def test_baseline_workflow() -> None:
    """Simple smoke test for the baseline model."""
    print("Testing Baseline Model")
    print("=" * 50)

    model = BaselineWorkflowAgent()

    instruction = (
        "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. "
        "{텍스트작성Agent}, {이미지생성Agent}, {동영상편집Agent}가 동시에 작업하고 "
        "{콘텐츠통합Agent}가 최종 결과물을 만들도록 해"
    )
    print(f"Instruction: {instruction}")
    print("-" * 50)

    result = model.generate_workflow(instruction)

    print(f"Model Type: {result['model_type']}")
    print(f"Retry Attempts: {result['retry_attempts']}")

    parse_err = result.get("label_json_parse_error")
    if parse_err:
        print(f"JSON Parse Error: {parse_err}")

    print("Generated JSON (raw or parsed):")
    lj = result.get("label_json")
    if isinstance(lj, dict):
        print(json.dumps(lj, ensure_ascii=False, indent=2))
    elif isinstance(lj, str):
        print(lj)
    else:
        print(repr(lj))

    info = model.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nTesting JSON → Instruction:")
    print("-" * 50)
    if isinstance(lj, dict):
        converted = model.json_to_instruction(lj)
        print(f"Original: {instruction}")
        print(f"Converted: {converted}")
    else:
        print("Skip: label_json is not a dict (parse failed).")


if __name__ == "__main__":
    test_baseline_workflow()