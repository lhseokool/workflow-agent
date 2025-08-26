"""
Workflow Agent - Clean and Simple Implementation
Runs both exact match evaluation (with GT) and LLM-as-Judge evaluation (no GT)
"""

import json
import time
from models import WorkflowAgent
from utils import load_test_data, compare_results_exact, save_to_excel, print_results_summary


def run_workflow_evaluation():
    """Run combined evaluation: exact match (GT) + LLM judge (no GT)"""
    print("🚀 Starting Workflow Agent Evaluation")
    print("Running exact match (with GT) and LLM-as-Judge (no GT) evaluation...")
    
    # Load test data
    test_data = load_test_data()
    if not test_data:
        print("❌ No test data found")
        return
    
    print(f"✅ Loaded {len(test_data)} test cases")
    
    # Initialize agent
    agent = WorkflowAgent()
    results = []
    
    # Process each test case
    for i, test_case in enumerate(test_data):
        instruction = test_case["instruction"]
        print(f"\n[{i+1}/{len(test_data)}] Processing: {instruction[:50]}...")
        
        start_time = time.time()
        
        try:
            # Generate workflow
            workflow_result = agent.generate_workflow(instruction)
            generation_time = time.time() - start_time
            
            # 1. Exact match evaluation (with ground truth)
            exact_comparison = compare_results_exact(test_case, workflow_result)
            
            # 2. LLM-as-Judge evaluation (no ground truth needed)
            judge_start_time = time.time()
            llm_judge_correct = agent.judge_instruction_result(
                instruction,
                json.dumps(workflow_result.get("label_json", {}), ensure_ascii=False)
            )
            
            # 3. LLM evaluation with ground truth (if available)
            json_llm_correct = False
            if "label_json" in test_case:
                json_llm_correct = agent.evaluate_json_with_llm(
                    instruction,
                    json.dumps(test_case.get("label_json", {}), ensure_ascii=False),
                    json.dumps(workflow_result.get("label_json", {}), ensure_ascii=False)
                )
            
            judge_eval_time = time.time() - judge_start_time
            total_time = generation_time + judge_eval_time
            
            # Store results
            result = {
                "test_id": i + 1,
                "instruction": instruction,
                "expected_json": json.dumps(test_case.get("label_json", {}), ensure_ascii=False),
                "actual_json": json.dumps(workflow_result.get("label_json", {}), ensure_ascii=False),
                "json_exact_match": exact_comparison.get("json_match", False),
                "json_llm_with_gt": json_llm_correct,
                "llm_judge_no_gt": llm_judge_correct,
                "elapsed_time": total_time,
                "judge_eval_time": judge_eval_time
            }
            results.append(result)
            
            # Print progress
            exact_json = "✅" if exact_comparison.get("json_match", False) else "❌"
            llm_gt = "✅" if json_llm_correct else "❌"
            judge_no_gt = "✅" if llm_judge_correct else "❌"
            
            print(f"Results - Exact: {exact_json} | LLM+GT: {llm_gt} | Judge(no GT): {judge_no_gt}")
            print(f"Time: {total_time:.2f}s (Generation: {generation_time:.2f}s, Judge: {judge_eval_time:.2f}s)")
            
        except Exception as e:
            print(f"❌ Error processing test case: {e}")
            # Add failed result
            results.append({
                "test_id": i + 1,
                "instruction": instruction,
                "expected_json": json.dumps(test_case.get("label_json", {}), ensure_ascii=False),
                "actual_json": "",
                "json_exact_match": False,
                "json_llm_with_gt": False,
                "llm_judge_no_gt": False,
                "elapsed_time": 0,
                "judge_eval_time": 0
            })
    
    # Print summary and save results
    print_results_summary(results)
    filename = save_to_excel(results)
    if filename:
        print(f"✅ Results saved to: {filename}")
    else:
        print("❌ Failed to save results")


def main():
    """Main entry point"""
    try:
        run_workflow_evaluation()
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()