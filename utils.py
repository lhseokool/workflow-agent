"""
Utility functions for workflow agents.

Provides helpers for parsing LLM outputs, validating/cleaning JSON,
string utilities, and standardized workflow result formatting.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Precompiled regex patterns
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
_AGENT_PATTERN_RE = re.compile(r"\{([^}]+Agent)\}")


def parse_llm_evaluation(evaluation_result: str) -> bool:
    """Parse a textual LLM evaluation into a boolean.

    Args:
        evaluation_result: Raw evaluation string returned by an LLM.

    Returns:
        True if the text indicates a positive/pass result; otherwise False.
    """
    if not evaluation_result:
        return False

    result = evaluation_result.strip().lower()

    positive = ("true", "yes", "pass", "correct", "good", "valid")
    negative = ("false", "no", "fail", "incorrect", "bad", "invalid")

    if any(tok in result for tok in positive):
        return True
    if any(tok in result for tok in negative):
        return False
    return False


def validate_json_structure(json_data: Union[str, Dict[str, Any]]) -> bool:
    """Validate minimal workflow JSON structure.

    Rules:
      - Root must be an object containing "type".
      - type == "LLM"        -> must contain "tools"
      - type in {"Sequential","Parallel","Loop"} -> must contain "sub_agents"

    Args:
        json_data: JSON string or dict.

    Returns:
        True if minimally valid; otherwise False.
    """
    try:
        parsed: Any = json.loads(json_data) if isinstance(json_data, str) else json_data
        if not isinstance(parsed, dict):
            return False

        wtype = parsed.get("type")
        if not wtype:
            return False

        if wtype == "LLM":
            return "tools" in parsed
        if wtype in {"Sequential", "Parallel", "Loop"}:
            return "sub_agents" in parsed
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def clean_json_output(raw_output: str) -> str:
    """Extract and normalize the JSON segment from an LLM raw output.

    Heuristics handled:
      - Markdown code fences (``` or ```json)
      - Leading "json" language tag
      - Trims to outermost {...} block

    Args:
        raw_output: Raw LLM text.

    Returns:
        A best-effort JSON string (may still be invalid JSON).
    """
    if not raw_output:
        return "{}"

    output = raw_output.strip()

    # ```json ... ``` or ``` ... ```
    m = _CODE_FENCE_RE.match(output)
    if m:
        output = m.group(1).strip()

    # Leading "json" label
    if output.lower().startswith("json"):
        output = output[4:].strip()

    # Trim to outermost braces
    start = output.find("{")
    end = output.rfind("}")
    if start != -1 and end != -1 and start <= end:
        output = output[start : end + 1]

    return output


def extract_agent_names(instruction: str) -> List[str]:
    """Extract agent names in the form {AgentName}.

    Args:
        instruction: Input instruction text.

    Returns:
        List of agent names without braces, e.g., ["TextAgent", "ImageAgent"].
    """
    return _AGENT_PATTERN_RE.findall(instruction or "")


def sanitize_filename(s: str) -> str:
    """Return a filesystem-friendly filename.

    Args:
        s: Arbitrary string.

    Returns:
        Lowercased, alphanumeric/._- only string; never empty.
    """
    s = re.sub(r"[^\w.\-]+", "_", (s or "").strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "unknown"


def try_parse_json(x: Any) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Attempt to parse a JSON object.

    Args:
        x: Dict (returned as-is) or JSON string.

    Returns:
        Tuple of (ok, parsed_dict_or_None, error_message).
    """
    if isinstance(x, dict):
        return True, x, ""
    if isinstance(x, str):
        try:
            return True, json.loads(x), ""
        except json.JSONDecodeError as e:
            return False, None, f"jsondecodeerror: {e}"
    return False, None, f"unsupported_type:{type(x).__name__}"


def remove_flow_name(data: Any) -> Any:
    """Recursively drop `flow_name` keys for comparison.

    Args:
        data: Arbitrary JSON-like structure.

    Returns:
        Structure without any `flow_name` keys.
    """
    if isinstance(data, dict):
        return {k: remove_flow_name(v) for k, v in data.items() if k != "flow_name"}
    if isinstance(data, list):
        return [remove_flow_name(v) for v in data]
    return data


def exact_match_eval(generated: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Compare two workflow dicts ignoring `flow_name`.

    Args:
        generated: Generated JSON dict.
        expected: Expected JSON dict.

    Returns:
        True if equal after removing `flow_name`; otherwise False.
    """
    return remove_flow_name(generated) == remove_flow_name(expected)


def format_workflow_result(
    instruction: str,
    generated_json: Dict[str, Any],
    model_type: str = "unknown",
    retry_attempts: int = 0,
    execution_time: float = 0.0,
) -> Dict[str, Any]:
    """Produce a standardized workflow result payload.

    Args:
        instruction: Original instruction text.
        generated_json: Workflow JSON (dict).
        model_type: Logical model identifier.
        retry_attempts: Number of retry attempts performed.
        execution_time: Total execution time in seconds.

    Returns:
        Result dict with normalized fields and quick validation flags.
    """
    return {
        "instruction": instruction,
        "label_json": generated_json,
        "model_type": model_type,
        "retry_attempts": retry_attempts,
        "execution_time": execution_time,
        "extracted_agents": extract_agent_names(instruction),
        "is_valid": validate_json_structure(generated_json),
    }