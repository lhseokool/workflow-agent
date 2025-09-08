"""
Baseline Model - Simple JSON Generation
기본 베이스라인 모델: json_chain만 사용하는 단순한 구현
"""

import json
import os
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")


# Import prompts from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import JSON_PROMPT


class BaselineWorkflowAgent:
    """
    Baseline Workflow Agent - 기본 베이스라인 모델
    json_chain만 사용하는 단순한 구현 (retry 로직 없음)
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the baseline workflow agent with OpenAI model"""
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=512
        )
        self.parser = StrOutputParser()
        
        # Simple JSON generation chain - 기본 체인만 사용
        self.json_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.llm | self.parser
    
    def generate_json(self, instruction: str) -> Dict[str, Any]:
        """
        Generate JSON structure from instruction
        단순한 JSON 생성 (retry 로직 없음)
        """
        try:
            result = self.json_chain.invoke({"instruction": instruction})
            return json.loads(result.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw output: {result}")
            # Return fallback JSON structure
            return {
                "type": "LLM", 
                "sub_agents": [{"name": "DefaultAgent"}]
            }
        except Exception as e:
            print(f"JSON generation error: {e}")
            # Return fallback JSON structure
            return {
                "type": "LLM", 
                "sub_agents": [{"name": "DefaultAgent"}]
            }
    
    def generate_workflow(self, instruction: str) -> Dict[str, Any]:
        """
        Generate complete workflow from instruction
        단순한 워크플로우 생성 (retry 없음)
        """
        json_result = self.generate_json(instruction)
        return {
            "instruction": instruction,
            "label_json": json_result,
            "model_type": "baseline",
            "retry_attempts": 0  # 베이스라인은 retry 없음
        }
    
    def get_model_info(self) -> Dict[str, str]:
        """모델 정보 반환"""
        return {
            "model_type": "baseline",
            "model_name": self.model_name,
            "description": "Simple JSON generation without retry logic",
            "features": "json_chain only"
        }


# 테스트 함수
def test_baseline_model():
    """베이스라인 모델 테스트"""
    print("🧪 Testing Baseline Model")
    print("="*50)
    
    # Initialize model
    model = BaselineWorkflowAgent()
    
    # Test instruction
    test_instruction = "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. {텍스트작성Agent}, {이미지생성Agent}, {동영상편집Agent}가 동시에 작업하고 {콘텐츠통합Agent}가 최종 결과물을 만들도록 해"
    
    print(f"📝 Test Instruction: {test_instruction}")
    print("-"*50)
    
    # Generate workflow
    result = model.generate_workflow(test_instruction)
    
    # Display results
    print(f"✅ Model Type: {result['model_type']}")
    print(f"🔢 Retry Attempts: {result['retry_attempts']}")
    print(f"📄 Generated JSON:")
    print(json.dumps(result['label_json'], ensure_ascii=False, indent=2))
    
    # Model info
    info = model.get_model_info()
    print(f"\n📋 Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_baseline_model()
