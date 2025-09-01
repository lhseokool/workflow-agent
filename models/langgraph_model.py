"""
LangGraph Model - Conditional Edge Retry Logic
LangGraph의 conditional edge를 이용한 retry 로직
"""

import json
import os
from typing import Dict, Any, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

# Import prompts and utils from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import JSON_PROMPT, INSTRUCTION_JUDGE_PROMPT
from utils import parse_llm_evaluation


class WorkflowState(TypedDict):
    """워크플로우 상태"""
    instruction: str
    generated_json: Dict[str, Any]
    retry_count: int
    max_retries: int
    judge_passed: bool
    success: bool
    error_message: str


class LangGraphRetryAgent:
    """
    LangGraph Conditional Edge Retry Agent
    conditional edge를 이용한 우아한 retry 로직
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", max_retries: int = 3):
        """Initialize the LangGraph retry agent"""
        self.model_name = model_name
        self.max_retries = max_retries
        
        # LLM 초기화 (baseline_model과 동일한 설정으로 통일)
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # baseline과 동일하게 0.0으로 변경
            max_tokens=512
        )
        self.parser = StrOutputParser()
        
        # Chains
        self.json_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.llm | self.parser
        self.judge_chain = ChatPromptTemplate.from_template(INSTRUCTION_JUDGE_PROMPT) | self.llm | self.parser
        
        # Build LangGraph
        self.graph = self._build_graph()
    
    def _generate_json_node(self, state: WorkflowState) -> Dict[str, Any]:
        """JSON 생성 노드"""
        try:
            instruction = state["instruction"]
            result = self.json_chain.invoke({"instruction": instruction})
            generated_json = json.loads(result.strip())
            
            return {
                "generated_json": generated_json,
                "error_message": ""
            }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw output: {result}")
            # Return fallback JSON structure
            fallback_json = {"type": "LLM", "sub_agents": [{"name": "DefaultAgent"}]}
            return {
                "generated_json": fallback_json,
                "error_message": f"JSON generation error: {str(e)}"
            }
        except Exception as e:
            print(f"JSON generation error: {e}")
            # Return fallback JSON structure
            fallback_json = {"type": "LLM", "sub_agents": [{"name": "DefaultAgent"}]}
            return {
                "generated_json": fallback_json,
                "error_message": f"JSON generation error: {str(e)}"
            }
    
    def _judge_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Judge 평가 노드"""
        try:
            instruction = state["instruction"]
            generated_json = state["generated_json"]
            json_string = json.dumps(generated_json, ensure_ascii=False)
            
            result = self.judge_chain.invoke({
                "instruction": instruction,
                "generated_json": json_string
            })
            
            judge_passed = parse_llm_evaluation(result)
            
            if judge_passed:
                return {
                    "judge_passed": True,
                    "success": True
                }
            else:
                return {
                    "judge_passed": False,
                    "success": False,
                    "retry_count": state["retry_count"] + 1
                }
        except Exception as e:
            return {
                "judge_passed": False,
                "success": False,
                "error_message": f"Judge error: {str(e)}"
            }
    
    def _should_retry(self, state: WorkflowState) -> str:
        """Conditional edge: retry 여부 결정"""
        # Judge 통과했으면 종료
        if state["judge_passed"]:
            return "end"
        
        # 최대 retry 도달했으면 종료
        if state["retry_count"] >= state["max_retries"]:
            return "end"
        
        # retry 계속
        return "retry"
    
    # increment_retry 노드 제거: judge 노드에서 실패 시 retry_count를 증가시킵니다.
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 구성"""
        # StateGraph 생성
        workflow = StateGraph(WorkflowState)
        
        # 노드 추가
        workflow.add_node("generate_json", self._generate_json_node)
        workflow.add_node("judge", self._judge_node)
        
        # 엣지 추가
        workflow.add_edge("generate_json", "judge")
        workflow.add_conditional_edges(
            "judge",
            self._should_retry,
            {
                "retry": "generate_json",
                "end": END
            }
        )
        
        # 시작점 설정
        workflow.set_entry_point("generate_json")
        
        return workflow.compile()
    
    def generate_workflow(self, instruction: str) -> Dict[str, Any]:
        """
        Generate workflow using LangGraph conditional edges
        """
        # 초기 상태
        initial_state = {
            "instruction": instruction,
            "generated_json": {},
            "retry_count": 0,
            "max_retries": self.max_retries,
            "judge_passed": False,
            "success": False,
            "error_message": ""
        }
        
        # LangGraph 실행
        try:
            result = self.graph.invoke(initial_state)
            
            # 최종 상태에서 결과 추출
            final_state = result["judge"] if "judge" in result else result
            
            return {
                "instruction": instruction,
                "label_json": final_state["generated_json"],
                "model_type": "langgraph_conditional",
                "retry_attempts": final_state["retry_count"],
                "success": final_state["success"],
                "judge_passed": final_state["judge_passed"],
                "error_message": final_state.get("error_message", "")
            }
            
        except Exception as e:
            return {
                "instruction": instruction,
                "label_json": {"type": "LLM", "sub_agents": [{"name": "DefaultAgent"}]},
                "model_type": "langgraph_conditional",
                "retry_attempts": 0,
                "success": False,
                "judge_passed": False,
                "error_message": f"Graph execution error: {str(e)}"
            }
    
    def save_graph_as_png(self, output_dir: str = "./models") -> str:
        """LangGraph를 PNG 이미지로 저장"""
        try:
            import os
            from datetime import datetime
            
            # 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일명 생성 (타임스탬프 포함)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_filename = f"langgraph_workflow_{timestamp}.png"
            png_path = os.path.join(output_dir, png_filename)
            
            # LangGraph의 내장 기능으로 PNG 저장
            self.graph.get_graph().draw_mermaid_png(output_file_path=png_path)
            
            return png_path
            
        except Exception as e:
            print(f"⚠️ Failed to save graph as PNG: {e}")
            print("💡 Make sure you have the required dependencies installed:")
            print("   pip install pygraphviz or pip install pydot")
            return ""
    
    def display_graph(self, output_dir: str = "./modelss") -> None:
        """LangGraph를 PNG로 저장하고 표시"""
        try:
            from PIL import Image
            
            # PNG 저장
            png_path = self.save_graph_as_png(output_dir)
            if png_path:
                # 이미지 열기 및 표시
                Image.open(png_path).show()
                print(f"📊 Graph displayed and saved: {png_path}")
            
        except ImportError:
            print("⚠️ PIL (Pillow) not installed. Install with: pip install Pillow")
        except Exception as e:
            print(f"⚠️ Failed to display graph: {e}")

    def get_model_info(self) -> Dict[str, str]:
        """모델 정보 반환"""
        return {
            "model_type": "langgraph_conditional",
            "model_name": self.model_name,
            "description": "LangGraph conditional edge retry logic",
            "features": "json_chain + judge_chain + conditional_edges",
            "max_retries": str(self.max_retries)
        }


# 테스트 함수
def test_langgraph_conditional():
    """LangGraph Conditional Edge 모델 테스트"""
    print("🧪 Testing LangGraph Conditional Edge Model")
    print("="*60)
    
    # Initialize model
    model = LangGraphRetryAgent(max_retries=2)
    
    # Save LangGraph as PNG image
    print("📊 Saving LangGraph workflow as PNG image...")
    saved_png = model.save_graph_as_png()
    if saved_png:
        print(f"✅ Graph saved as PNG: {saved_png}")
    
    # Optional: Display the graph (uncomment if you want to show image)
    # model.display_graph()
    
    # Test instruction (baseline과 동일한 테스트 케이스로 변경)
    test_instruction = "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. {텍스트작성Agent}, {이미지생성Agent}, {동영상편집Agent}가 동시에 작업하고 {콘텐츠통합Agent}가 최종 결과물을 만들도록 해"
    
    print(f"📝 Test Instruction: {test_instruction}")
    print("-"*60)
    
    # Generate workflow
    result = model.generate_workflow(test_instruction)
    
    # Display results (baseline과 유사한 형식으로 변경)
    print(f"✅ Model Type: {result['model_type']}")
    print(f"🔢 Retry Attempts: {result['retry_attempts']}")
    print(f"🎯 Success: {result['success']}")
    print(f"⚖️ Judge Passed: {result['judge_passed']}")
    
    if result.get('error_message'):
        print(f"⚠️ Error Message: {result['error_message']}")
    
    print(f"📄 Generated JSON:")
    print(json.dumps(result['label_json'], ensure_ascii=False, indent=2))
    
    # Model info
    info = model.get_model_info()
    print(f"\n📋 Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 성능 분석
    print(f"\n🎯 Performance Analysis:")
    if result['success'] and result['judge_passed']:
        print("🎉 Perfect! LangGraph conditional edge worked successfully!")
        if result['retry_attempts'] == 0:
            print("✨ Generated correct result on first attempt!")
        else:
            print(f"🔄 Required {result['retry_attempts']} retries to pass judge evaluation")
    elif result['retry_attempts'] >= model.max_retries:
        print("⚠️ Reached maximum retries without passing judge evaluation")
        print("🔧 Consider adjusting prompts or increasing max_retries")
    else:
        print("❌ Workflow failed before reaching max retries")


if __name__ == "__main__":
    test_langgraph_conditional()
