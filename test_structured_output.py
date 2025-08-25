"""
Test Structured Output Parser
Simple test to verify the with_structured_output implementation
"""

from typing import Union
from langgraph_models import LangGraphWorkflowAgent
from pydantic_models import LLMWorkflowModel, SequentialWorkflowModel


def test_structured_output_parser():
    """Test the structured output parser functionality"""
    print("🧪 Testing Structured Output Parser...")
    
    # Test instructions for each workflow type
    test_cases = [
        {
            "instruction": "HR 관련 문의사항에 답변하는 에이전트를 만드세요. 직원정보와 급여정보를 참조할 수 있어야 합니다.",
            "expected_type": "LLM",
            "description": "LLM Type Workflow"
        },
        {
            "instruction": "문서를 작성하고, 검토한 후, 수정하는 단계별 파이프라인을 구성하세요.",
            "expected_type": "Sequential", 
            "description": "Sequential Type Workflow"
        },
        {
            "instruction": "데이터를 수집하고 처리하는 간단한 워크플로우를 만드세요.",
            "expected_type": "Sequential",
            "description": "Simple Sequential Workflow"
        }
    ]
    
    print(f"\nTesting {len(test_cases)} structured output cases...\n")
    
    # Test agent initialization
    agent = LangGraphWorkflowAgent(max_retries=1)
    print(f"Agent initialized - Structured Output: {agent.use_structured_output}")
    
    if not agent.use_structured_output:
        print("⚠️  Structured output not available, using string parsing fallback")
    else:
        print("✅ Structured output parser available")
    
    # Test each case
    results = []
    for i, test_case in enumerate(test_cases, 1):
        instruction = test_case["instruction"]
        expected_type = test_case["expected_type"]
        description = test_case["description"]
        
        print(f"\n[{i}] {description}")
        print(f"Instruction: {instruction[:50]}...")
        
        try:
            # Generate workflow
            result = agent.generate_workflow(instruction)
            
            # Check results
            success = result.get("success", False)
            generated_json = result.get("label_json", {})
            pydantic_valid = result.get("pydantic_valid", False)
            parsing_method = "structured" if agent.use_structured_output and pydantic_valid else "string"
            
            print(f"Success: {'✅' if success else '❌'}")
            print(f"Parsing Method: {'🏗️' if parsing_method == 'structured' else '📝'} {parsing_method}")
            print(f"Pydantic Valid: {'✅' if pydantic_valid else '❌'}")
            
            if generated_json:
                actual_type = generated_json.get("type", "Unknown")
                type_match = actual_type == expected_type
                print(f"Type: {actual_type} {'✅' if type_match else '❌'}")
                print(f"Flow Name: {generated_json.get('flow_name', 'N/A')}")
                
                # Show structure
                if "sub_agents" in generated_json:
                    agent_count = len(generated_json["sub_agents"])
                    print(f"Sub Agents: {agent_count} agents")
                
                if "tools" in generated_json:
                    tool_count = len(generated_json["tools"])
                    print(f"Tools: {tool_count} tools")
            else:
                print("❌ No JSON generated")
                type_match = False
                actual_type = "None"
            
            # Store result
            results.append({
                "case": i,
                "description": description,
                "success": success,
                "parsing_method": parsing_method,
                "pydantic_valid": pydantic_valid,
                "expected_type": expected_type,
                "actual_type": actual_type,
                "type_match": type_match,
                "retry_count": result.get("retry_count", 0)
            })
            
            if result.get("validation_feedback"):
                print(f"Feedback: {result['validation_feedback']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "case": i,
                "description": description,
                "success": False,
                "parsing_method": "error",
                "pydantic_valid": False,
                "expected_type": expected_type,
                "actual_type": "Error",
                "type_match": False,
                "retry_count": 0
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Structured Output Test Summary")
    print(f"{'='*60}")
    
    total_cases = len(results)
    successful = sum(1 for r in results if r["success"])
    structured_used = sum(1 for r in results if r["parsing_method"] == "structured")
    pydantic_valid = sum(1 for r in results if r["pydantic_valid"])
    type_matches = sum(1 for r in results if r["type_match"])
    
    print(f"Total Tests: {total_cases}")
    print(f"Successful Generation: {successful}/{total_cases} ({successful/total_cases*100:.1f}%)")
    print(f"Structured Output Used: {structured_used}/{total_cases} ({structured_used/total_cases*100:.1f}%)")
    print(f"Pydantic Valid: {pydantic_valid}/{total_cases} ({pydantic_valid/total_cases*100:.1f}%)")
    print(f"Type Matches: {type_matches}/{total_cases} ({type_matches/total_cases*100:.1f}%)")
    
    # Parsing method breakdown
    parsing_methods = {}
    for r in results:
        method = r["parsing_method"]
        parsing_methods[method] = parsing_methods.get(method, 0) + 1
    
    print(f"\nParsing Methods:")
    for method, count in parsing_methods.items():
        icon = "🏗️" if method == "structured" else "📝" if method == "string" else "❌"
        print(f"  {icon} {method}: {count} cases")
    
    if structured_used > 0:
        print(f"\n🎉 Structured output parser working correctly!")
    else:
        print(f"\n⚠️  No structured output used - check LLM compatibility")
    
    print(f"✅ Structured output test completed!")


if __name__ == "__main__":
    test_structured_output_parser()
