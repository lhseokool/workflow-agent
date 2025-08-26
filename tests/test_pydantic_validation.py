"""
Pydantic Validation Test
Test the Pydantic models for JSON workflow validation
"""

import json
from pydantic_models import WorkflowValidator


def test_pydantic_validation():
    """Test Pydantic validation for all 4 workflow types"""
    print("🧪 Testing Pydantic Validation...")
    
    # Test cases: valid and invalid workflows
    test_cases = [
        # 1. Valid LLM workflow
        {
            "name": "Valid LLM Workflow",
            "data": {
                "flow_name": "HRQNAAgent",
                "type": "LLM",
                "sub_agents": [{"agent_name": "답변작성Agent"}],
                "tools": [{"agent_name": "직원정보Agent"}, {"agent_name": "급여정보Agent"}]
            },
            "should_pass": True
        },
        
        # 2. Invalid LLM workflow (missing tools)
        {
            "name": "Invalid LLM Workflow (missing tools)",
            "data": {
                "flow_name": "BadLLMAgent",
                "type": "LLM",
                "sub_agents": [{"agent_name": "답변Agent"}]
            },
            "should_pass": False
        },
        
        # 3. Invalid LLM workflow (bad agent name)
        {
            "name": "Invalid LLM Workflow (bad agent name)",
            "data": {
                "flow_name": "BadLLMAgent",
                "type": "LLM",
                "sub_agents": [{"agent_name": "BadName"}],  # doesn't end with Agent
                "tools": [{"agent_name": "ToolAgent"}]
            },
            "should_pass": False
        },
        
        # 4. Valid Sequential workflow
        {
            "name": "Valid Sequential Workflow",
            "data": {
                "flow_name": "DocumentPipelineAgent",
                "type": "Sequential",
                "sub_agents": [
                    {"agent_name": "문서작성Agent"},
                    {"agent_name": "문서검토Agent"},
                    {"agent_name": "문서수정Agent"}
                ]
            },
            "should_pass": True
        },
        
        # 5. Valid Sequential + Loop workflow
        {
            "name": "Valid Sequential + Loop Workflow",
            "data": {
                "flow_name": "IterativeReportPipeline",
                "type": "Sequential",
                "sub_agents": [
                    {"agent_name": "초기작성Agent"},
                    {
                        "flow": {
                            "flow_name": "RefinementLoop",
                            "type": "Loop",
                            "sub_agents": [
                                {"agent_name": "비평Agent"},
                                {"agent_name": "수정Agent"}
                            ]
                        }
                    }
                ]
            },
            "should_pass": True
        },
        
        # 6. Invalid workflow (missing required field)
        {
            "name": "Invalid Workflow (missing flow_name)",
            "data": {
                "type": "Sequential",
                "sub_agents": [{"agent_name": "TestAgent"}]
            },
            "should_pass": False
        },
        
        # 7. Invalid workflow (wrong type)
        {
            "name": "Invalid Workflow (wrong type)",
            "data": {
                "flow_name": "TestWorkflow",
                "type": "InvalidType",
                "sub_agents": [{"agent_name": "TestAgent"}]
            },
            "should_pass": False
        },
        
        # 8. Valid Parallel workflow
        {
            "name": "Valid Parallel Workflow",
            "data": {
                "flow_name": "ParallelProcess",
                "type": "Parallel",
                "sub_agents": [
                    {"agent_name": "Agent1"},
                    {"agent_name": "Agent2"},
                    {"agent_name": "Agent3"}
                ]
            },
            "should_pass": True
        }
    ]
    
    print(f"\nTesting {len(test_cases)} validation cases...\n")
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        data = test_case["data"]
        should_pass = test_case["should_pass"]
        
        print(f"[{i}] {name}")
        
        # Run validation
        result = WorkflowValidator.validate_workflow_json(data)
        is_valid = result["is_valid"]
        error_message = result["error_message"]
        error_type = result["error_type"]
        
        # Check if result matches expectation
        if is_valid == should_pass:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"    Expected: {'Pass' if should_pass else 'Fail'}")
        print(f"    Result: {'Valid' if is_valid else 'Invalid'}")
        print(f"    Status: {status}")
        
        if not is_valid:
            print(f"    Error Type: {error_type}")
            print(f"    Error: {error_message}")
        
        if is_valid and result["validated_data"]:
            print(f"    Validated Type: {result['validated_data']['type']}")
        
        print()
    
    print(f"{'='*60}")
    print(f"📊 Pydantic Validation Test Results")
    print(f"{'='*60}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(test_cases)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All Pydantic validation tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    
    # Test error feedback generation
    print(f"\n🔍 Testing error feedback generation...")
    test_errors = [
        ("Missing required field", "structure"),
        ("Invalid workflow type", "type"),
        ("Agent name validation failed", "naming"),
        ("Insufficient agents", "count")
    ]
    
    for error_msg, error_type in test_errors:
        feedback = WorkflowValidator.get_validation_feedback(error_msg, error_type)
        print(f"  {error_type}: {feedback}")


if __name__ == "__main__":
    test_pydantic_validation()
