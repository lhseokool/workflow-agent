#!/usr/bin/env python3
"""
JSON Parsing Performance Test (compact, agent-style output)

Compares multiple JSON parsing approaches for workflow JSON generation:
- StrOutputParser
- JsonOutputParser (with Pydantic schema)
- PydanticOutputParser
- with_structured_output (Pydantic)

Console:
- Parser-level summaries (exact match %, success %, avg time)
- Per-case lines only with --verbose

Excel:
- Detailed Results
- Parser Summary
- Type Analysis (per parser & type)
- Exact by Type (per parser & type, trimmed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Optional env loader
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Dependency check early to fail fast ---
def check_dependencies() -> None:
    try:
        import pandas as pd  # noqa: F401
        from pydantic import BaseModel, Field  # noqa: F401
        from langchain_core.prompts import ChatPromptTemplate  # noqa: F401
        from langchain_core.output_parsers import (  # noqa: F401
            JsonOutputParser, PydanticOutputParser, StrOutputParser
        )
        from langchain_openai import ChatOpenAI  # noqa: F401
        from prompts import JSON_PROMPT  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install:")
        print("  pip install pandas pydantic langchain langchain-openai langchain-core")
        sys.exit(1)

check_dependencies()

# Heavy imports after check
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    JsonOutputParser, PydanticOutputParser, StrOutputParser
)
from langchain_openai import ChatOpenAI
from prompts import JSON_PROMPT

# Optional utils (with fallback)
try:
    from utils import sanitize_filename
except Exception:
    import re
    def sanitize_filename(s: str) -> str:
        s = re.sub(r"[^\w\.\-]+", "_", (s or "").strip())
        s = re.sub(r"_+", "_", s).strip("_").lower()
        return s or "unknown"


# ===== Schema =====
class Flow(BaseModel):
    """Unified workflow schema for all parsers."""
    flow_name: str = Field(description="Flow name")
    type: str = Field(description="One of: LLM, Sequential, Loop, Parallel")
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="LLM only: [{'agent_name': str}, ...]"
    )
    sub_agents: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sequential/Loop/Parallel only: [{'agent_name': str} | {'flow': Flow}, ...]"
    )


# ===== Data & helpers =====
def load_test_data(path: str = "data/test_data.json") -> List[Dict[str, Any]]:
    """Load test cases."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{path} not found. Run from project root or fix path.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path}: {e}")
        sys.exit(1)


def clean_for_comparison(data: Any, remove_flow_name: bool = True, remove_none: bool = True) -> Any:
    """Normalize for exact-match comparison."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if remove_flow_name and k == "flow_name":
                continue
            if remove_none and v is None:
                continue
            out[k] = clean_for_comparison(v, remove_flow_name, remove_none)
        return out
    if isinstance(data, list):
        return [
            clean_for_comparison(x, remove_flow_name, remove_none)
            for x in data
            if not (remove_none and x is None)
        ]
    return data


def exact_match_eval(generated: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Exact-match ignoring 'flow_name' and None values."""
    return clean_for_comparison(generated) == clean_for_comparison(expected)


def validate_api_key() -> None:
    """Require OPENAI_API_KEY for ChatOpenAI."""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing. Set env or .env file.")
        sys.exit(1)


# ===== Chains =====
def setup_parsers() -> Dict[str, Any]:
    """Create prompt|LLM|parser chains for all strategies."""
    validate_api_key()
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception as e:
        print(f"LLM init failed: {e}")
        sys.exit(1)

    chains: Dict[str, Any] = {}

    # 1) StrOutputParser
    chains["StrOutputParser"] = ChatPromptTemplate.from_template(JSON_PROMPT) | llm | StrOutputParser()

    # 2) JsonOutputParser (with Pydantic schema)
    json_parser = JsonOutputParser(pydantic_object=Flow)
    prompt_json = ChatPromptTemplate.from_template(JSON_PROMPT).partial(
        format_instructions=json_parser.get_format_instructions()
    )
    chains["JsonOutputParser"] = prompt_json | llm | json_parser

    # 3) PydanticOutputParser
    pyd_parser = PydanticOutputParser(pydantic_object=Flow)
    prompt_pyd = ChatPromptTemplate.from_template(JSON_PROMPT).partial(
        format_instructions=pyd_parser.get_format_instructions()
    )
    chains["PydanticOutputParser"] = prompt_pyd | llm | pyd_parser

    # 4) with_structured_output
    llm_struct = llm.with_structured_output(Flow)
    chains["StructuredOutput"] = ChatPromptTemplate.from_template(JSON_PROMPT) | llm_struct

    return chains


def process_chain_result(result: Any, start_time: float) -> Tuple[Any, float, Optional[str]]:
    """Normalize outputs to dict and capture latency."""
    elapsed = time.time() - start_time
    if isinstance(result, BaseModel):
        return result.dict(), elapsed, None
    if isinstance(result, str):
        try:
            return json.loads(result), elapsed, None
        except json.JSONDecodeError as e:
            return result, elapsed, f"JSON parsing failed: {e}"
    return result, elapsed, None


def test_single_case(
    chain: Any,
    instruction: str,
    expected_output: Dict[str, Any],
    parser_name: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute one case on one chain."""
    start = time.time()
    try:
        raw = chain.invoke({"instruction": instruction})
        normalized, exec_time, err = process_chain_result(raw, start)

        if err:
            if verbose:
                print(f"    -> FAIL ({parser_name}) error: {err}")
            return {"success": False, "error": err, "execution_time": exec_time, "generated": normalized, "exact_match": False}

        exact = exact_match_eval(normalized, expected_output)
        if verbose:
            print(f"    -> {'OK' if exact else 'MISS'} ({parser_name}) time={exec_time:.2f}s")
        return {"success": True, "error": None, "execution_time": exec_time, "generated": normalized, "exact_match": exact}

    except Exception as e:
        if verbose:
            print(f"    -> FAIL ({parser_name}) exception: {e}")
        return {"success": False, "error": str(e), "execution_time": time.time() - start, "generated": None, "exact_match": False}


def initialize_parser_stats() -> Dict[str, Any]:
    return {
        "total_cases": 0,
        "successful_cases": 0,
        "exact_matches": 0,
        "total_time": 0.0,
        "avg_time": 0.0,
        "success_rate": 0.0,
        "exact_match_rate": 0.0,
        "errors": [],
    }


def run_case_multiple_times(
    chain: Any, case: Dict[str, Any], parser_name: str, n_runs: int, verbose: bool
) -> Tuple[List[Dict[str, Any]], int, int, float]:
    results: List[Dict[str, Any]] = []
    for _ in range(n_runs):
        r = test_single_case(chain, case["input"], case["expected_output"], parser_name, verbose)
        results.append(r)
    succ = sum(1 for r in results if r["success"])
    exact = sum(1 for r in results if r["exact_match"])
    avg_t = sum(r["execution_time"] for r in results) / len(results)
    return results, succ, exact, avg_t


def create_detailed_row(
    parser_name: str,
    case: Dict[str, Any],
    successful_runs: int,
    exact_match_runs: int,
    avg_time: float,
    best_result: Dict[str, Any],
    n_runs: int,
) -> Dict[str, Any]:
    instr = case["input"]
    short = instr[:100] + "..." if len(instr) > 100 else instr
    return {
        "Parser": parser_name,
        "Case_ID": case["id"],
        "Type": case["type"],
        "Instruction": short,
        "Success_Rate": f"{successful_runs}/{n_runs}",
        "Exact_Match_Rate": f"{exact_match_runs}/{n_runs}",
        "Avg_Time": round(avg_time, 3),
        "Best_Exact_Match": best_result["exact_match"],
        "Best_Success": best_result["success"],
        "Error": best_result["error"] or "",
        "Generated_JSON": json.dumps(best_result["generated"], ensure_ascii=False) if best_result["generated"] is not None else "",
        "Expected_JSON": json.dumps(case["expected_output"], ensure_ascii=False),
    }


def finalize_parser_stats(stats: Dict[str, Any], n_runs: int) -> Dict[str, Any]:
    total_possible = stats["total_cases"] * n_runs
    stats["avg_time"] = stats["total_time"] / max(stats["total_cases"], 1)
    if total_possible > 0:
        stats["success_rate"] = (stats["successful_cases"] / total_possible) * 100.0
        stats["exact_match_rate"] = (stats["exact_matches"] / total_possible) * 100.0
    return stats


def run_comprehensive_test(n_cases: Optional[int] = None, n_runs: int = 3, verbose: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print("JSON Parsing Performance Test")
    print("=" * 50)

    test_data = load_test_data()
    chains = setup_parsers()

    if n_cases is None:
        n_cases = len(test_data)
    else:
        n_cases = min(n_cases, len(test_data))
    cases = test_data[:n_cases]

    print(f"Testing {len(cases)} cases with {len(chains)} parsers ({n_runs} runs each)")
    print(f"Total tests: {len(cases) * len(chains) * n_runs}")

    detailed_rows: List[Dict[str, Any]] = []
    parser_summary: Dict[str, Any] = {}

    for parser_name, chain in chains.items():
        print(f"\nTesting {parser_name}...")
        stats = initialize_parser_stats()

        for case in cases:
            if verbose:
                print(f"  Case {case['id']} ({case['type']})")
            case_results, succ_runs, exact_runs, avg_t = run_case_multiple_times(chain, case, parser_name, n_runs, verbose)
            best = max(case_results, key=lambda x: (x["exact_match"], x["success"]))

            detailed_rows.append(create_detailed_row(parser_name, case, succ_runs, exact_runs, avg_t, best, n_runs))

            stats["total_cases"] += 1
            stats["successful_cases"] += succ_runs
            stats["exact_matches"] += exact_runs
            stats["total_time"] += avg_t
            if not best["success"] and best["error"]:
                stats["errors"].append(f"Case {case['id']}: {best['error']}")

        stats = finalize_parser_stats(stats, n_runs)
        parser_summary[parser_name] = stats
        print(
            f"  Done: {stats['exact_match_rate']:.1f}% exact match, "
            f"{stats['success_rate']:.1f}% success, "
            f"{stats['avg_time']:.3f}s avg"
        )

    return detailed_rows, parser_summary


def save_results(
    detailed_results: List[Dict[str, Any]],
    parser_summary: Dict[str, Any],
    output_dir: str = "results",
) -> str:
    """Write Excel with 4 sheets: Detailed, Parser Summary, Type Analysis, Exact by Type."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"json_parsing_test_results_{ts}.xlsx")

    # Prepare frames
    detailed_df = pd.DataFrame(detailed_results)

    summary_rows = []
    for p, s in parser_summary.items():
        summary_rows.append({
            "Parser": p,
            "Total_Cases": s["total_cases"],
            "Success_Rate(%)": round(s["success_rate"], 1),
            "Exact_Match_Rate(%)": round(s["exact_match_rate"], 1),
            "Avg_Time(s)": round(s["avg_time"], 3),
            "Error_Count": len(s["errors"]),
        })
    summary_df = pd.DataFrame(summary_rows)

    # Type Analysis (per parser & type)
    type_rows: List[Dict[str, Any]] = []
    if not detailed_df.empty:
        for case_type in ["LLM", "Sequential", "Loop", "Parallel"]:
            type_subset = detailed_df[detailed_df["Type"] == case_type]
            if type_subset.empty:
                continue
            for parser in parser_summary.keys():
                pt = type_subset[type_subset["Parser"] == parser]
                if pt.empty:
                    continue
                exact = int(pt["Best_Exact_Match"].sum())
                cases = len(pt)
                type_rows.append({
                    "Type": case_type,
                    "Parser": parser,
                    "Cases": cases,
                    "Exact_Matches": exact,
                    "Exact_Match_Rate(%)": round((exact / cases) * 100, 1) if cases else 0.0,
                    "Avg_Time(s)": round(pt["Avg_Time"].mean(), 3),
                })
    type_df = pd.DataFrame(type_rows)

    # Exact by Type (lean view per parser & type, without Avg_Time)
    exact_by_type_rows: List[Dict[str, Any]] = []
    for row in type_rows:
        exact_by_type_rows.append({
            "Type": row["Type"],
            "Parser": row["Parser"],
            "Total_Cases": row["Cases"],
            "Exact_Match_Count": row["Exact_Matches"],
            "Exact_Match_Accuracy(%)": row["Exact_Match_Rate(%)"],
        })
    exact_by_type_df = pd.DataFrame(exact_by_type_rows)

    with pd.ExcelWriter(filename, engine="openpyxl") as w:
        detailed_df.to_excel(w, sheet_name="Detailed Results", index=False)
        summary_df.to_excel(w, sheet_name="Parser Summary", index=False)
        if not type_df.empty:
            type_df.to_excel(w, sheet_name="Type Analysis", index=False)
        if not exact_by_type_df.empty:
            exact_by_type_df.to_excel(w, sheet_name="Exact by Type", index=False)

    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON Parsing Performance Test")
    parser.add_argument("--cases", type=int, default=None, help="Number of cases to run (default: all)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per case (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Print per-case lines")
    parser.add_argument("--output", default="results", help="Directory for Excel output")
    args = parser.parse_args()

    print("Starting JSON parsing performance test...")
    try:
        detailed, summary = run_comprehensive_test(n_cases=args.cases, n_runs=args.runs, verbose=args.verbose)

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        # Rank parsers by exact match rate, tie-breaker avg_time asc
        ranked = sorted(summary.items(), key=lambda x: (x[1]["exact_match_rate"], -x[1]["avg_time"]), reverse=True)

        print("\nParser Ranking (by Exact Match Rate):")
        for i, (name, s) in enumerate(ranked, 1):
            print(f"  {i}. {name}")
            print(f"     - Exact Match: {s['exact_match_rate']:.1f}%")
            print(f"     - Success:     {s['success_rate']:.1f}%")
            print(f"     - Avg Time:    {s['avg_time']:.3f}s")
            if s["errors"]:
                print(f"     - Errors:      {len(s['errors'])}")
            print()

        # Type-level overview across parsers (how many exact matches per type by best run)
        print("Performance by Workflow Type (best per parser-case row basis):")
        totals: Dict[str, Dict[str, int]] = {}
        for row in detailed:
            t = row["Type"]
            totals.setdefault(t, {"total": 0, "exact": 0})
            totals[t]["total"] += 1
            if row["Best_Exact_Match"]:
                totals[t]["exact"] += 1
        for t, s in totals.items():
            rate = (s["exact"] / s["total"]) * 100 if s["total"] else 0.0
            print(f"  {t}: {rate:.1f}% exact match ({s['exact']}/{s['total']})")

        # Save Excel (fallback to JSON)
        try:
            path = save_results(detailed, summary, args.output)
            print(f"\nResults saved to: {path}")
        except Exception as e:
            print(f"\nFailed to save Excel: {e}")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = os.path.join(args.output, f"json_parsing_test_backup_{ts}.json")
            os.makedirs(args.output, exist_ok=True)
            with open(backup, "w", encoding="utf-8") as f:
                json.dump({"detailed_results": detailed, "parser_summary": summary}, f, ensure_ascii=False, indent=2)
            print(f"Backup saved to: {backup}")

        print("\nTest completed.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()