"""
Workflow Agent - Clean and Simple Implementation
Automatically runs JSON workflow generation and evaluation
"""

import json
import time
from models import WorkflowAgent
from utils import load_test_data, compare_results_exact, save_to_excel, print_results_summary


def run_workflow_evaluation():
    """Run JSON workflow evaluation on test data"""
    print("🚀 Starting Workflow Agent Evaluation")
    print("Running JSON workflow generation and evaluation...")
    
    # Load test data
    test_data = load_test_data()
    if not test_data:
        print("❌ No test data found")
        return
    
    print(f"✅ Loaded {len(test_data)} test cases")
    
    # Initialize agent
    agent = WorkflowAgent()
    print(f"💡 Using {'structured output' if agent.use_structured_output else 'string parsing'} for JSON generation")
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
            
            # LLM evaluation (JSON only)
            llm_start_time = time.time()
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
                "expected_json": json.dumps(test_case.get("label_json", {}), ensure_ascii=False),
                "actual_json": json.dumps(workflow_result.get("label_json", {}), ensure_ascii=False),
                "json_match": exact_comparison["json_match"],
                "json_llm_correct": json_llm_correct,
                "elapsed_time": total_time,
                "llm_eval_time": llm_eval_time,
                "retry_count": workflow_result.get("retry_count", 0),
                "success": workflow_result.get("success", True),
                "validation_feedback": workflow_result.get("validation_feedback", ""),
                # Pydantic validation results
                "pydantic_valid": workflow_result.get("pydantic_valid", False),
                "pydantic_error_type": workflow_result.get("pydantic_error_type", ""),
                "llm_validation_score": workflow_result.get("llm_validation_score", False),
                "parsing_method": "structured" if agent.use_structured_output and workflow_result.get("pydantic_valid", False) else "string"
            }
            results.append(result)
            
            # Print progress
            exact_json = "✅" if exact_comparison["json_match"] else "❌"
            llm_json = "✅" if json_llm_correct else "❌"
            success_icon = "✅" if workflow_result.get("success", True) else "❌"
            pydantic_icon = "✅" if workflow_result.get("pydantic_valid", False) else "❌"
            llm_semantic_icon = "✅" if workflow_result.get("llm_validation_score", False) else "❌"
            retry_info = f" (Retries: {workflow_result.get('retry_count', 0)})" if workflow_result.get('retry_count', 0) > 0 else ""
            
            # Show parsing method used
            parsing_method = "🏗️(structured)" if agent.use_structured_output and workflow_result.get("pydantic_valid", False) else "📝(string)"
            
            print(f"Results - JSON: {exact_json}(exact) {llm_json}(LLM) | Validation: {pydantic_icon}(type) {llm_semantic_icon}(semantic) {success_icon}(overall) {parsing_method}{retry_info}")
            print(f"Time: {total_time:.2f}s (Generation: {generation_time:.2f}s, Evaluation: {llm_eval_time:.2f}s)")
            if workflow_result.get("validation_feedback"):
                print(f"Feedback: {workflow_result['validation_feedback']}")
            if workflow_result.get("pydantic_error_type") and workflow_result.get("pydantic_error_type") != "":
                print(f"Type Error: {workflow_result['pydantic_error_type']}")
            
        except Exception as e:
            print(f"❌ Error processing test case: {e}")
            # Add failed result
            results.append({
                "test_id": i + 1,
                "instruction": instruction,
                "expected_json": json.dumps(test_case.get("label_json", {}), ensure_ascii=False),
                "actual_json": "",
                "json_match": False,
                "json_llm_correct": False,
                "elapsed_time": 0,
                "llm_eval_time": 0,
                "retry_count": 0,
                "success": False,
                "validation_feedback": f"Exception occurred: {str(e)}",
                "pydantic_valid": False,
                "pydantic_error_type": "exception",
                "llm_validation_score": False,
                "parsing_method": "error"
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