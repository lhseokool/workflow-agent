"""
Workflow Agent Utilities
Functions for loading test data, comparing results, and saving to Excel
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List


def load_test_data(file_path: str = "test_data.json") -> List[Dict[str, Any]]:
    """Load test data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return []


def compare_json_workflows(expected_json: Dict[str, Any], actual_json: Dict[str, Any]) -> bool:
    """Compare two JSON workflow structures for semantic equality"""
    # Check flow_name
    if expected_json.get("flow_name") != actual_json.get("flow_name"):
        return False
    
    # Check type
    if expected_json.get("type") != actual_json.get("type"):
        return False
    
    # Check sub_agents
    expected_agents = expected_json.get("sub_agents", [])
    actual_agents = actual_json.get("sub_agents", [])
    
    if len(expected_agents) != len(actual_agents):
        return False
    
    for exp_agent, act_agent in zip(expected_agents, actual_agents):
        if "agent_name" in exp_agent and "agent_name" in act_agent:
            if exp_agent["agent_name"] != act_agent["agent_name"]:
                return False
        elif "flow" in exp_agent and "flow" in act_agent:
            if not compare_json_workflows(exp_agent["flow"], act_agent["flow"]):
                return False
        else:
            return False
    
    # Check tools (for LLM type)
    if expected_json.get("type") == "LLM":
        expected_tools = expected_json.get("tools", [])
        actual_tools = actual_json.get("tools", [])
        
        if len(expected_tools) != len(actual_tools):
            return False
        
        expected_tool_names = {tool["agent_name"] for tool in expected_tools}
        actual_tool_names = {tool["agent_name"] for tool in actual_tools}
        
        if expected_tool_names != actual_tool_names:
            return False
    
    return True


def compare_results_exact(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, bool]:
    """Compare results using exact matching for JSON workflows"""
    expected_json = expected.get("label_json", {})
    actual_json = actual.get("label_json", {})
    
    json_match = compare_json_workflows(expected_json, actual_json)
    
    return {
        "json_match": json_match,
        "exact_match": json_match
    }


def parse_llm_evaluation(eval_result: str) -> bool:
    """Parse LLM evaluation result (True/False only)"""
    result = eval_result.strip().lower()
    return result == "true"


def save_to_excel(results: List[Dict[str, Any]], prefix: str = "json_workflow") -> str:
    """Save results to Excel file with JSON workflow evaluation results"""
    try:
        excel_data = []
        for r in results:
            row = {
                "Test_ID": r["test_id"],
                "Instruction": r["instruction"],
                "Expected_JSON": r["expected_json"],
                "Generated_JSON": r["actual_json"],
                "JSON_Exact_Match": "O" if r["json_match"] else "X",
                "JSON_LLM_Correct": "O" if r.get("json_llm_correct", False) else "X",
                "Pydantic_Type_Valid": "O" if r.get("pydantic_valid", False) else "X",
                "LLM_Semantic_Valid": "O" if r.get("llm_validation_score", False) else "X",
                "Generation_Success": "O" if r.get("success", True) else "X",
                "Retry_Count": r.get("retry_count", 0),
                "Parsing_Method": r.get("parsing_method", "unknown"),
                "Pydantic_Error_Type": r.get("pydantic_error_type", ""),
                "Validation_Feedback": r.get("validation_feedback", ""),
                "Total_Time": f"{r['elapsed_time']:.2f}s",
                "LLM_Eval_Time": f"{r.get('llm_eval_time', 0):.2f}s"
            }
            excel_data.append(row)
        
        # Save to Excel
        df = pd.DataFrame(excel_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_results_{prefix}_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Results']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return filename
    except Exception as e:
        print(f"❌ Excel save failed: {e}")
        return ""


def print_results_summary(results: List[Dict[str, Any]]):
    """Print test results summary with JSON workflow evaluation metrics"""
    total_tests = len(results)
    if total_tests == 0:
        print("❌ No test results available.")
        return
    
    avg_time = sum(r["elapsed_time"] for r in results) / total_tests
    avg_llm_time = sum(r.get("llm_eval_time", 0) for r in results) / total_tests
    
    # Calculate metrics
    json_exact = sum(1 for r in results if r["json_match"])
    json_llm = sum(1 for r in results if r.get("json_llm_correct", False))
    generation_success = sum(1 for r in results if r.get("success", True))
    pydantic_valid = sum(1 for r in results if r.get("pydantic_valid", False))
    llm_semantic_valid = sum(1 for r in results if r.get("llm_validation_score", False))
    total_retries = sum(r.get("retry_count", 0) for r in results)
    avg_retries = total_retries / total_tests if total_tests > 0 else 0
    
    # Error type breakdown
    error_types = {}
    parsing_methods = {}
    for r in results:
        error_type = r.get("pydantic_error_type", "")
        if error_type and not r.get("pydantic_valid", False):
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Parsing method breakdown
        parsing_method = r.get("parsing_method", "unknown")
        parsing_methods[parsing_method] = parsing_methods.get(parsing_method, 0) + 1
    
    print(f"\n{'='*80}")
    print(f"📊 Test Results Summary (Pydantic + LLM + LangGraph Evaluation)")
    print(f"{'='*80}")
    print(f"Total Tests: {total_tests}")
    print(f"\nGeneration Results:")
    print(f"  Successful Generation: {generation_success}/{total_tests} ({generation_success/total_tests*100:.1f}%)")
    print(f"  Total Retries: {total_retries}")
    print(f"  Average Retries per Test: {avg_retries:.1f}")
    
    print(f"\nValidation Results:")
    print(f"  1️⃣ Pydantic Type Valid: {pydantic_valid}/{total_tests} ({pydantic_valid/total_tests*100:.1f}%)")
    print(f"  2️⃣ LLM Semantic Valid: {llm_semantic_valid}/{total_tests} ({llm_semantic_valid/total_tests*100:.1f}%)")
    
    print(f"\nComparison Results:")
    print(f"  Exact Match: {json_exact}/{total_tests} ({json_exact/total_tests*100:.1f}%)")
    print(f"  LLM Correct: {json_llm}/{total_tests} ({json_llm/total_tests*100:.1f}%)")
    
    if error_types:
        print(f"\nPydantic Error Breakdown:")
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count} cases")
    
    print(f"\nParsing Method Usage:")
    for method, count in parsing_methods.items():
        percentage = count/total_tests*100 if total_tests > 0 else 0
        icon = "🏗️" if method == "structured" else "📝" if method == "string" else "❌"
        print(f"  {icon} {method}: {count}/{total_tests} ({percentage:.1f}%)")
    
    print(f"\nTiming:")
    print(f"  Average Total Time: {avg_time:.2f}s")
    print(f"  Average LLM Eval Time: {avg_llm_time:.2f}s")