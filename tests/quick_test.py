"""
Quick Test - Fast testing script
Test with a single instruction using all evaluation methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from models import WorkflowAgent


def quick_test():
    """Quick test with all evaluation methods"""
    print("🚀 Quick Test - Workflow Agent")
    print("Testing all evaluation methods: Exact, LLM+GT, LLM Judge...")
    print("="*60)
    
    # Initialize workflow agent
    try:
        agent = WorkflowAgent()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return
    
    # Test instruction
    test_instruction = "고객 문의에 답변하는 AI 시스템을 만들어주세요"
    
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
        print(f"\n🔹 Generated JSON:")
        print(json.dumps(result['label_json'], ensure_ascii=False, indent=2))
        
        # LLM-as-Judge evaluation (no GT needed)
        print("\n🔍 Running LLM-as-Judge evaluation (no GT needed)...")
        judge_start_time = time.time()
        
        judge_result = agent.judge_instruction_result(
            test_instruction,
            json.dumps(result['label_json'], ensure_ascii=False)
        )
        
        judge_time = time.time() - judge_start_time
        total_time = generation_time + judge_time
        
        print(f"✅ LLM-as-Judge completed! (Time: {judge_time:.2f}s)")
        
        # Results
        print(f"\n📊 Evaluation Results:")
        print("-" * 40)
        
        judge_status = "✅ PASS" if judge_result else "❌ FAIL"
        print(f"LLM-as-Judge (no GT): {judge_status}")
        
        print(f"\n⏱️ Timing Results:")
        print(f"  Generation Time: {generation_time:.2f}s")
        print(f"  Judge Time: {judge_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")
        
        # Analysis
        print(f"\n🎯 Analysis:")
        if judge_result:
            print("🎉 The generated workflow correctly represents the instruction!")
            print("✨ The AI system can effectively convert instructions to workflows.")
        else:
            print("⚠️ The generated workflow may not fully capture the instruction intent.")
            print("🔧 Consider refining the prompts or model parameters.")
            
    except Exception as e:
        print(f"❌ Error during test execution: {e}")


def main():
    """Main entry point"""
    try:
        quick_test()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()