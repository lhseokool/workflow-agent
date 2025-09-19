#!/usr/bin/env python3
"""
Workflow Agent Performance Test (compact console output)

Metrics
- FINAL JSON Parse (based on `label_json`)
- Exact Match (ignores `flow_name`, final JSON only)
- Type-wise analysis + "Exact by Type" sheet

Console
- Prints only FINAL JSON Parse + Exact Match per case
- If the agent returns `label_json_parse_error`, the case is marked failed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

# Agents
from models.baseline_model import BaselineWorkflowAgent
from utils import sanitize_filename, try_parse_json, remove_flow_name, exact_match_eval


# -----------------------
# Utilities
# -----------------------
def load_test_data(path: str = "data/test_data.json") -> List[Dict[str, Any]]:
    """Load test dataset from a json file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{path} not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path}: {e}")
        raise


def save_to_excel(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    type_rows: List[Dict[str, Any]],
    exact_by_type_rows: List[Dict[str, Any]],
    model_type: str,
    llm_name: str,
    outdir: str = "results",
) -> str:
    """Save detailed results and summaries to an Excel workbook."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(
        outdir,
        f"{sanitize_filename(model_type)}__{sanitize_filename(llm_name)}__test_results_{ts}.xlsx",
    )
    with pd.ExcelWriter(filename, engine="openpyxl") as w:
        pd.DataFrame(results).to_excel(w, sheet_name="Detailed Results", index=False)
        pd.DataFrame([summary]).to_excel(w, sheet_name="Summary", index=False)
        if type_rows:
            pd.DataFrame(type_rows).to_excel(w, sheet_name="Type Analysis", index=False)
        if exact_by_type_rows:
            pd.DataFrame(exact_by_type_rows).to_excel(
                w, sheet_name="Exact by Type", index=False
            )
    return filename


# -----------------------
# Main
# -----------------------
def main() -> None:
    """CLI entrypoint for compact workflow agent benchmarking."""
    parser = argparse.ArgumentParser(description="Workflow Agent Performance Tester (Compact)")
    parser.add_argument(
        "--model",
        choices=["baseline", "langgraph", "3stage"],
        default="baseline",
        help="Model to test",
    )
    parser.add_argument("--outdir", default="results", help="Directory to save results")
    parser.add_argument(
        "--llm-name",
        choices=["gpt-4o-mini", "gpt-oss-120b", "Llama33", "gauss2-3-37b", "gpt-oss-20b"],
        default="gpt-4o-mini",
        help="LLM name",
    )
    args = parser.parse_args()

    model_type, llm_name = args.model, args.llm_name
    print(f"{model_type.capitalize()} Model | LLM: {llm_name}\n" + "=" * 50)

    # Instantiate agent lazily based on choice
    if model_type == "baseline":
        agent = BaselineWorkflowAgent(llm_name)
    elif model_type == "langgraph":
        from models.langgraph_retry_model import LangGraphRetryAgent

        agent = LangGraphRetryAgent(llm_name)
    elif model_type == "3stage":
        from models.langgraph_3stage_model import ThreeStageWorkflowAgent

        agent = ThreeStageWorkflowAgent(llm_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    data = load_test_data()
    total = len(data)
    final_ok = exact_ok_total = 0
    total_time = 0.0

    # Aggregates
    type_stats: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    print(f"Testing {total} cases...")

    for case in data:
        cid = case.get("id")
        ctype = case.get("type", "Unknown")
        instr = case.get("input", "")
        exp_raw = case.get("expected_output")

        # Ensure expected JSON is a dict when possible
        exp_ok, exp_parsed, _ = try_parse_json(exp_raw)
        if not (exp_ok and isinstance(exp_parsed, dict)):
            exp_parsed = None

        print(f"\nCase {cid}: {ctype}")
        t0 = time.perf_counter()
        case_err = ""

        try:
            r = agent.generate_workflow(instr)
        except Exception as e:
            case_err = f"generate_error:{type(e).__name__}:{e}"
            dt = time.perf_counter() - t0
            total_time += dt
            bucket = type_stats.setdefault(
                ctype, {"n": 0, "final_ok": 0, "exact_ok": 0, "time_sum": 0.0}
            )
            bucket["n"] += 1
            bucket["time_sum"] += dt
            results.append(
                {
                    "Case_ID": cid,
                    "Type": ctype,
                    "FINAL_JSON_Parse": "X (generate_failed)",
                    "Exact_Match": "X",
                    "Execution_Time": round(dt, 2),
                    "Final_JSON": "",
                    "Expected_JSON": (
                        json.dumps(exp_parsed, ensure_ascii=False)
                        if exp_parsed is not None
                        else ""
                    ),
                    "Generated_wo_flow_name": "",
                    "Expected_wo_flow_name": (
                        json.dumps(remove_flow_name(exp_parsed), ensure_ascii=False)
                        if isinstance(exp_parsed, dict)
                        else ""
                    ),
                    "Case_Error": case_err,
                }
            )
            print(f"  JSON Parse: FAIL | Exact Match: FAIL | Time: {dt:.2f}s")
            continue

        generated = r.get("label_json")                 # dict or str
        parse_err_msg = r.get("label_json_parse_error") # str or None

        # Final JSON considered OK only if it's a dict and no parse error flag
        final_ok_i = isinstance(generated, dict)
        final_json_error = parse_err_msg is not None
        if final_json_error:
            print(f"  JSON parsing error reported by agent: {parse_err_msg}")

        dt = time.perf_counter() - t0
        total_time += dt

        # Update aggregates
        bucket = type_stats.setdefault(
            ctype, {"n": 0, "final_ok": 0, "exact_ok": 0, "time_sum": 0.0}
        )
        bucket["n"] += 1
        bucket["time_sum"] += dt
        if final_ok_i and not final_json_error:
            final_ok += 1
            bucket["final_ok"] += 1

        # Exact match requires final OK and a valid expected dict
        exact_i = False
        if final_ok_i and not final_json_error and isinstance(exp_parsed, dict):
            try:
                exact_i = exact_match_eval(generated, exp_parsed)
            except Exception:
                exact_i = False
            if exact_i:
                exact_ok_total += 1
                bucket["exact_ok"] += 1

        # Console line
        console_json_ok = bool(final_ok_i and not final_json_error)
        print(
            f"  JSON Parse: {'OK' if console_json_ok else 'FAIL'} | "
            f"Exact Match: {'OK' if exact_i else 'FAIL'} | Time: {dt:.2f}s"
        )

        # Persist detailed row
        results.append(
            {
                "Case_ID": cid,
                "Type": ctype,
                "FINAL_JSON_Parse": (
                    "O"
                    if (final_ok_i and not final_json_error)
                    else ("X (jsondecodeerror)" if final_json_error else "X (not_dict)")
                ),
                "Exact_Match": "O" if exact_i else "X",
                "Execution_Time": round(dt, 2),
                "Final_JSON": (
                    json.dumps(generated, ensure_ascii=False)
                    if isinstance(generated, dict)
                    else (str(generated)[:2000] if generated is not None else "")
                ),
                "Expected_JSON": (
                    json.dumps(exp_parsed, ensure_ascii=False)
                    if exp_parsed is not None
                    else ""
                ),
                "Generated_wo_flow_name": (
                    json.dumps(remove_flow_name(generated), ensure_ascii=False)
                    if (final_ok_i and not final_json_error)
                    else ""
                ),
                "Expected_wo_flow_name": (
                    json.dumps(remove_flow_name(exp_parsed), ensure_ascii=False)
                    if isinstance(exp_parsed, dict)
                    else ""
                ),
                "Case_Error": case_err,
            }
        )

    # Totals
    final_acc = (final_ok / total * 100.0) if total else 0.0
    exact_acc = (exact_ok_total / total * 100.0) if total else 0.0
    avg_time = (total_time / total) if total else 0.0

    print("\nFinal Results")
    print(f"  JSON Parse (FINAL): {final_ok}/{total} ({final_acc:.1f}%)")
    print(f"  Exact Match       : {exact_ok_total}/{total} ({exact_acc:.1f}%)")
    print(f"  Total Time        : {total_time:.2f}s")
    print(f"  Average Time      : {avg_time:.2f}s")

    # Type-wise table
    type_rows: List[Dict[str, Any]] = []
    for t, b in type_stats.items():
        n = b["n"]
        type_rows.append(
            {
                "Workflow_Type": t,
                "Total_Cases": n,
                "FINAL_JSON_Parse_Success": b["final_ok"],
                "FINAL_JSON_Parse_Accuracy(%)": (round(b["final_ok"] / n * 100.0, 1) if n else 0.0),
                "Exact_Match_Count": b["exact_ok"],
                "Exact_Match_Accuracy(%)": (round(b["exact_ok"] / n * 100.0, 1) if n else 0.0),
                "Avg_Time(s)": round(b["time_sum"] / n, 2) if n else 0.0,
            }
        )

    # Exact by Type (fixed order)
    ordered = ["LLM", "Sequential", "Loop", "Parallel"]
    exact_by_type_rows: List[Dict[str, Any]] = []
    exact_by_type_summary: Dict[str, float] = {}
    for ot in ordered:
        b = type_stats.get(ot)
        if b and b["n"] > 0:
            n, ex = b["n"], b["exact_ok"]
            acc = round(ex / n * 100.0, 1)
            exact_by_type_rows.append(
                {"Workflow_Type": ot, "Total_Cases": n, "Exact_Match_Count": ex, "Exact_Match_Accuracy(%)": acc}
            )
            exact_by_type_summary[f"Exact_Acc_{ot}(%)"] = acc
        else:
            exact_by_type_rows.append(
                {"Workflow_Type": ot, "Total_Cases": 0, "Exact_Match_Count": 0, "Exact_Match_Accuracy(%)": 0.0}
            )
            exact_by_type_summary[f"Exact_Acc_{ot}(%)"] = 0.0

    summary = {
        "Test_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model_Type": model_type,
        "LLM_Name": llm_name,
        "Total_Cases": total,
        "FINAL_JSON_Parse_Success": final_ok,
        "FINAL_JSON_Parse_Accuracy(%)": round(final_acc, 1),
        "Exact_Match_Count": exact_ok_total,
        "Exact_Match_Accuracy(%)": round(exact_acc, 1),
        "Total_Time(s)": round(total_time, 2),
        "Average_Time(s)": round(avg_time, 2),
        **exact_by_type_summary,
    }

    try:
        fn = save_to_excel(
            results, summary, type_rows, exact_by_type_rows, model_type, llm_name, args.outdir
        )
        print(f"\nSaved: {fn}")
    except Exception as e:
        print(f"Excel save failed: {e}\nSaving JSON fallback...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(
            args.outdir,
            f"{sanitize_filename(model_type)}__{sanitize_filename(llm_name)}__test_results_{ts}.json",
        )
        os.makedirs(args.outdir, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "type_analysis": type_rows,
                    "exact_by_type": exact_by_type_rows,
                    "detailed_results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Backup saved: {out}")


if __name__ == "__main__":
    main()