"""
Workflow Agent - Clean and Simple Implementation
Automatically runs both exact and LLM evaluations for incoming data
"""

import json
import time
from models import WorkflowAgent
from utils import load_test_data, compare_results_exact, save_to_excel, print_results_summary


def run_workflow_evaluation():
    """Run combined exact and LLM evaluation on test data"""
    print("🚀 Starting Workflow Agent Evaluation")
    print("Running both exact match and LLM evaluation...")
    
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
            
            # Exact match evaluation
            exact_comparison = compare_results_exact(test_case, workflow_result)
            
            # LLM evaluation
            llm_start_time = time.time()
            adk_llm_correct = agent.evaluate_adk_with_llm(
                instruction,
                test_case.get("label_adk", ""),
                workflow_result.get("label_adk", "")
            )
            json_llm_correct = agent.evaluate_json_with_llm(
                instruction,
                json.dumps(test_case.get("label_json", {}), ensure_ascii=False),
                json.dumps(workflow_result.get("label_json", {}), ensure_ascii=False)
            )
            llm_eval_time = time.time() - llm_start_time
            total_time = generation_time + llm_eval_time
            
            # Store results
            result = {
                "test_id": i + 1,
                "instruction": instruction,
                "expected_adk": test_case.get("label_adk", ""),
                "actual_adk": workflow_result.get("label_adk", ""),
                "expected_json": json.dumps(test_case.get("label_json", {}), ensure_ascii=False),
                "actual_json": json.dumps(workflow_result.get("label_json", {}), ensure_ascii=False),
                "adk_match": exact_comparison["adk_match"],
                "json_match": exact_comparison["json_match"],
                "adk_llm_correct": adk_llm_correct,
                "json_llm_correct": json_llm_correct,
                "elapsed_time": total_time,
                "llm_eval_time": llm_eval_time
            }
            results.append(result)
            
            # Print progress
            exact_adk = "✅" if exact_comparison["adk_match"] else "❌"
            exact_json = "✅" if exact_comparison["json_match"] else "❌"
            llm_adk = "✅" if adk_llm_correct else "❌"
            llm_json = "✅" if json_llm_correct else "❌"
            
            print(f"Results - ADK: {exact_adk}(exact) {llm_adk}(LLM) | JSON: {exact_json}(exact) {llm_json}(LLM)")
            print(f"Time: {total_time:.2f}s (Generation: {generation_time:.2f}s, Evaluation: {llm_eval_time:.2f}s)")
            
        except Exception as e:
            print(f"❌ Error processing test case: {e}")
            # Add failed result
            results.append({
                "test_id": i + 1,
                "instruction": instruction,
                "expected_adk": test_case.get("label_adk", ""),
                "actual_adk": "",
                "expected_json": json.dumps(test_case.get("label_json", {}), ensure_ascii=False),
                "actual_json": "",
                "adk_match": False,
                "json_match": False,
                "adk_llm_correct": False,
                "json_llm_correct": False,
                "elapsed_time": 0,
                "llm_eval_time": 0
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