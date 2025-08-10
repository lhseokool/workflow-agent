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


def compare_results_exact(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, bool]:
    """Compare results using exact matching"""
    # ADK comparison
    adk_match = expected.get("label_adk") == actual.get("label_adk")
    
    # JSON type comparison
    expected_json = expected.get("label_json", {})
    actual_json = actual.get("label_json", {})
    type_match = expected_json.get("type") == actual_json.get("type")
    
    # Agent comparison
    expected_agents = set()
    actual_agents = set()
    
    if "sub_agents" in expected_json:
        expected_agents = {agent["name"] for agent in expected_json["sub_agents"]}
    if "sub_agents" in actual_json:
        actual_agents = {agent["name"] for agent in actual_json["sub_agents"]}
    
    agents_match = expected_agents == actual_agents
    json_match = type_match and agents_match
    
    return {
        "adk_match": adk_match,
        "json_match": json_match,
        "exact_match": adk_match and json_match
    }


def parse_llm_evaluation(eval_result: str) -> bool:
    """Parse LLM evaluation result (True/False only)"""
    result = eval_result.strip().lower()
    return result == "true"


def save_to_excel(results: List[Dict[str, Any]], prefix: str = "combined") -> str:
    """Save results to Excel file with both exact and LLM evaluation results"""
    try:
        excel_data = []
        for r in results:
            row = {
                "Test_ID": r["test_id"],
                "Instruction": r["instruction"],
                "Expected_ADK": r["expected_adk"],
                "Generated_ADK": r["actual_adk"],
                "Expected_JSON": r["expected_json"],
                "Generated_JSON": r["actual_json"],
                "ADK_Exact_Match": "O" if r["adk_match"] else "X",
                "ADK_LLM_Correct": "O" if r.get("adk_llm_correct", False) else "X",
                "JSON_Exact_Match": "O" if r["json_match"] else "X", 
                "JSON_LLM_Correct": "O" if r.get("json_llm_correct", False) else "X",
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
    """Print test results summary with both exact and LLM evaluation metrics"""
    total_tests = len(results)
    if total_tests == 0:
        print("❌ No test results available.")
        return
    
    avg_time = sum(r["elapsed_time"] for r in results) / total_tests
    avg_llm_time = sum(r.get("llm_eval_time", 0) for r in results) / total_tests
    
    # Calculate metrics
    adk_exact = sum(1 for r in results if r["adk_match"])
    adk_llm = sum(1 for r in results if r.get("adk_llm_correct", False))
    json_exact = sum(1 for r in results if r["json_match"])
    json_llm = sum(1 for r in results if r.get("json_llm_correct", False))
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results Summary (Combined Evaluation)")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"\nADK Results:")
    print(f"  Exact Match: {adk_exact}/{total_tests} ({adk_exact/total_tests*100:.1f}%)")
    print(f"  LLM Correct: {adk_llm}/{total_tests} ({adk_llm/total_tests*100:.1f}%)")
    print(f"\nJSON Results:")
    print(f"  Exact Match: {json_exact}/{total_tests} ({json_exact/total_tests*100:.1f}%)")
    print(f"  LLM Correct: {json_llm}/{total_tests} ({json_llm/total_tests*100:.1f}%)")
    print(f"\nTiming:")
    print(f"  Average Total Time: {avg_time:.2f}s")
    print(f"  Average LLM Eval Time: {avg_llm_time:.2f}s")
