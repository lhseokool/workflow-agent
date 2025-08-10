"""
Quick Test - Fast testing script
Single instruction test to quickly check ADK/JSON generation results
"""

import json
import time
from models import WorkflowAgent
from utils import compare_results_exact


def quick_test_combined():
    """Quick test with combined exact and LLM evaluation"""
    print("🚀 Quick Test - Workflow Agent (Combined Evaluation)")
    print("="*60)
    
    # Initialize workflow agent
    try:
        agent = WorkflowAgent()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return
    
    # Test instruction
    test_instruction = "(CodeWriterAgent)로 코드를 작성한 후, (CodeReviewerAgent)로 점검하고 (CodeRefactorAgent)로 리팩토링하는 멀티 에이전트를 구성하세요."
    
    print(f"\n📝 Test Instruction:")
    print(f"{test_instruction}")
    print("-" * 60)
    
    # Generate workflow
    print("⏳ Generating workflow...")
    start_time = time.time()
    
    try:
        result = agent.generate_workflow(test_instruction)
        generation_time = time.time() - start_time
        
        print(f"\n✅ Generation completed! (Time: {generation_time:.2f}s)")
        
        # Display results
        print(f"\n🔹 ADK Result:")
        print(f"{result['label_adk']}")
        
        print(f"\n🔹 JSON Result:")
        print(json.dumps(result['label_json'], ensure_ascii=False, indent=2))
        
        # Expected results
        expected_data = {
            "label_adk": "Sequential([CodeWriterAgent, CodeReviewerAgent, CodeRefactorAgent])",
            "label_json": {
                "type": "Sequential",
                "sub_agents": [
                    {"name": "CodeWriterAgent"},
                    {"name": "CodeReviewerAgent"},
                    {"name": "CodeRefactorAgent"}
                ]
            }
        }
        
        # Exact match comparison
        exact_comparison = compare_results_exact(expected_data, result)
        
        # LLM evaluation
        print("\n🔍 Running LLM evaluation...")
        llm_start_time = time.time()
        
        adk_llm_correct = agent.evaluate_adk_with_llm(
            test_instruction,
            expected_data["label_adk"],
            result["label_adk"]
        )
        
        json_llm_correct = agent.evaluate_json_with_llm(
            test_instruction,
            json.dumps(expected_data["label_json"], ensure_ascii=False),
            json.dumps(result["label_json"], ensure_ascii=False)
        )
        
        llm_eval_time = time.time() - llm_start_time
        total_time = generation_time + llm_eval_time
        
        print(f"✅ LLM evaluation completed! (Time: {llm_eval_time:.2f}s)")
        
        # Results comparison
        print(f"\n📊 Evaluation Results:")
        print("-" * 50)
        
        exact_adk = "✅ PASS" if exact_comparison["adk_match"] else "❌ FAIL"
        exact_json = "✅ PASS" if exact_comparison["json_match"] else "❌ FAIL"
        llm_adk = "✅ PASS" if adk_llm_correct else "❌ FAIL"
        llm_json = "✅ PASS" if json_llm_correct else "❌ FAIL"
        
        print(f"ADK Results:")
        print(f"  Exact Match: {exact_adk}")
        print(f"  LLM Evaluation: {llm_adk}")
        if not exact_comparison["adk_match"]:
            print(f"  Expected: {expected_data['label_adk']}")
            print(f"  Actual:   {result['label_adk']}")
        
        print(f"\nJSON Results:")
        print(f"  Exact Match: {exact_json}")
        print(f"  LLM Evaluation: {llm_json}")
        if not exact_comparison["json_match"]:
            print(f"  Expected: {json.dumps(expected_data['label_json'], ensure_ascii=False)}")
            print(f"  Actual:   {json.dumps(result['label_json'], ensure_ascii=False)}")
        
        print(f"\n⏱️ Timing Results:")
        print(f"  Generation Time: {generation_time:.2f}s")
        print(f"  Evaluation Time: {llm_eval_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")
        
        # Overall results
        exact_success = exact_comparison["exact_match"]
        llm_success = adk_llm_correct and json_llm_correct
        
        print(f"\n🎯 Overall Results:")
        if exact_success and llm_success:
            print(f"🎉 ALL TESTS PASSED in both evaluation methods!")
        elif exact_success or llm_success:
            print(f"🔄 PARTIAL SUCCESS")
            print(f"  Exact Match: {'PASS' if exact_success else 'FAIL'}")
            print(f"  LLM Evaluation: {'PASS' if llm_success else 'FAIL'}")
        else:
            print(f"⚠️ ALL TESTS FAILED in both evaluation methods")
        
        # Summary table
        print(f"\n📋 Summary Table:")
        print("-" * 50)
        print(f"{'Metric':<20} {'Exact':<10} {'LLM':<10}")
        print("-" * 50)
        print(f"{'ADK':<20} {'PASS' if exact_comparison['adk_match'] else 'FAIL':<10} {'PASS' if adk_llm_correct else 'FAIL':<10}")
        print(f"{'JSON':<20} {'PASS' if exact_comparison['json_match'] else 'FAIL':<10} {'PASS' if json_llm_correct else 'FAIL':<10}")
        print("-" * 50)
            
    except Exception as e:
        print(f"❌ Error during test execution: {e}")


def main():
    """Main entry point - runs combined evaluation automatically"""
    try:
        quick_test_combined()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()