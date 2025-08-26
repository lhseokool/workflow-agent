"""
Workflow Agent Utilities
Simple functions for loading test data, comparing results, and saving to Excel
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
    """Compare results using exact matching for JSON workflows"""
    expected_json = expected.get("label_json", {})
    actual_json = actual.get("label_json", {})
    
    # Simple exact match comparison
    json_match = expected_json == actual_json
    
    return {
        "json_match": json_match,
        "exact_match": json_match
    }


def parse_llm_evaluation(eval_result: str) -> bool:
    """Parse LLM evaluation result (True/False only)"""
    result = eval_result.strip().lower()
    return result == "true"


def save_to_excel(results: List[Dict[str, Any]], prefix: str = "evaluation") -> str:
    """Save results to Excel file with exact match, LLM+GT, and LLM judge (no GT) results"""
    try:
        excel_data = []
        for r in results:
            row = {
                "Test_ID": r["test_id"],
                "Instruction": r["instruction"],
                "Expected_JSON": r["expected_json"],
                "Generated_JSON": r["actual_json"],
                "JSON_Exact_Match": "O" if r.get("json_exact_match", False) else "X",
                "JSON_LLM_with_GT": "O" if r.get("json_llm_with_gt", False) else "X",
                "LLM_Judge_no_GT": "O" if r.get("llm_judge_no_gt", False) else "X",
                "Total_Time": f"{r['elapsed_time']:.2f}s",
                "Judge_Eval_Time": f"{r.get('judge_eval_time', 0):.2f}s"
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
    """Print test results summary with exact match, LLM+GT, and LLM judge metrics"""
    total_tests = len(results)
    if total_tests == 0:
        print("❌ No test results available.")
        return
    
    avg_time = sum(r["elapsed_time"] for r in results) / total_tests
    avg_judge_time = sum(r.get("judge_eval_time", 0) for r in results) / total_tests
    
    # Calculate metrics
    json_exact = sum(1 for r in results if r.get("json_exact_match", False))
    json_llm_gt = sum(1 for r in results if r.get("json_llm_with_gt", False))
    llm_judge = sum(1 for r in results if r.get("llm_judge_no_gt", False))
    
    print(f"\n{'='*70}")
    print(f"📊 Test Results Summary")
    print(f"{'='*70}")
    print(f"Total Tests: {total_tests}")
    print(f"\nEvaluation Results:")
    print(f"  Exact Match (with GT):     {json_exact}/{total_tests} ({json_exact/total_tests*100:.1f}%)")
    print(f"  LLM Eval (with GT):        {json_llm_gt}/{total_tests} ({json_llm_gt/total_tests*100:.1f}%)")
    print(f"  LLM Judge (no GT needed):  {llm_judge}/{total_tests} ({llm_judge/total_tests*100:.1f}%)")
    print(f"\nTiming:")
    print(f"  Average Total Time:   {avg_time:.2f}s")
    print(f"  Average Judge Time:   {avg_judge_time:.2f}s")