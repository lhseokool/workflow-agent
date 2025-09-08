"""
LangGraph Model - Conditional Edge Retry Logic
LangGraph의 conditional edge를 이용한 retry 로직
"""

import json
import os
import time
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
from prompts import JSON_PROMPT, LLM_JUDGE_WITH_REASEON_PROMPT
from utils import parse_llm_evaluation


class WorkflowState(TypedDict):
    """워크플로우 상태"""
    instruction: str
    generated_json: Dict[str, Any]
    retry_count: int
    max_retries: int
    judge_passed: bool
    judge_reason: str
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
        self.judge_chain = ChatPromptTemplate.from_template(LLM_JUDGE_WITH_REASEON_PROMPT) | self.llm | self.parser
        
        # Build LangGraph
        self.graph = self._build_graph()
    
    def _generate_json_node(self, state: WorkflowState) -> Dict[str, Any]:
        """JSON 생성 노드"""
        start_time = time.time()
        try:
            instruction = state["instruction"]
            result = self.json_chain.invoke({"instruction": instruction})
            generated_json = json.loads(result.strip())
            
            elapsed_time = time.time() - start_time
            print(f"⚙️ JSON Generated ({elapsed_time:.2f}s)")
            
            return {
                "generated_json": generated_json,
                "error_message": ""
            }
        except json.JSONDecodeError as e:
            elapsed_time = time.time() - start_time
            print(f"❌ JSON Parse Error ({elapsed_time:.2f}s)")
            # Return fallback JSON structure
            fallback_json = {"type": "LLM", "sub_agents": [{"name": "DefaultAgent"}]}
            return {
                "generated_json": fallback_json,
                "error_message": f"JSON generation error: {str(e)}"
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ JSON Generation Error ({elapsed_time:.2f}s)")
            # Return fallback JSON structure
            fallback_json = {"type": "LLM", "sub_agents": [{"name": "DefaultAgent"}]}
            return {
                "generated_json": fallback_json,
                "error_message": f"JSON generation error: {str(e)}"
            }
    
    def _judge_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Judge 평가 노드"""
        start_time = time.time()
        try:
            # Judge 체인 실행
            result = self.judge_chain.invoke({
                "instruction": state["instruction"],
                "generated_json": json.dumps(state["generated_json"], ensure_ascii=False)
            })
            
            # Judge 결과 파싱
            try:
                judge_result = json.loads(result.strip())
                judge_passed = judge_result.get("passed", False)
                judge_reason = judge_result.get("reason", "No reason provided")
            except json.JSONDecodeError:
                # Fallback parsing
                judge_passed = parse_llm_evaluation(result)
                judge_reason = f"Simple evaluation result: {result.strip()}"
            
            elapsed_time = time.time() - start_time
            status = "✅ PASSED" if judge_passed else "❌ FAILED"
            print(f"{status} Judge Evaluation ({elapsed_time:.2f}s)")
            
            # 실패 시 이유 출력
            if not judge_passed:
                print(f"   Reason: {judge_reason}")
            
            return {
                "judge_passed": judge_passed,
                "judge_reason": judge_reason,
                "success": judge_passed,
                "retry_count": state["retry_count"] + (0 if judge_passed else 1)
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_reason = f"Judge error: {str(e)}"
            print(f"❌ Judge Error ({elapsed_time:.2f}s)")
            print(f"   Reason: {error_reason}")
            return {
                "judge_passed": False,
                "judge_reason": error_reason,
                "success": False,
                "retry_count": state["retry_count"] + 1,
                "error_message": error_reason
            }
    
    def _should_retry(self, state: WorkflowState) -> str:
        """Conditional edge: retry 여부 결정"""
        # Judge 통과했으면 종료
        if state["judge_passed"]:
            return "end"
        
        # 최대 retry 도달했으면 종료
        if state["retry_count"] >= state["max_retries"]:
            print(f"⚠️ Max retries reached ({state['max_retries']})")
            return "end"
        
        # retry 계속
        print(f"🔄 Retry {state['retry_count']}/{state['max_retries']}")
        return "retry"
    
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
        """LangGraph conditional edges를 사용한 워크플로우 생성"""
        total_start_time = time.time()
        
        # 초기 상태 설정
        initial_state = {
            "instruction": instruction,
            "generated_json": {},
            "retry_count": 0,
            "max_retries": self.max_retries,
            "judge_passed": False,
            "judge_reason": "",
            "success": False,
            "error_message": ""
        }
        
        print(f"🚀 LangGraph Conditional Workflow")
        print(f"📝 {instruction}")
        print("-" * 60)
        
        try:
            # LangGraph 실행
            final_state = self.graph.invoke(initial_state)
            total_time = time.time() - total_start_time
            
            print("-" * 60)
            print(f"🎯 Total Time: {total_time:.2f}s")
            
            return {
                "instruction": instruction,
                "label_json": final_state["generated_json"],
                "model_type": "langgraph_conditional",
                "retry_attempts": final_state["retry_count"],
                "success": final_state["success"],
                "judge_passed": final_state["judge_passed"],
                "judge_reason": final_state.get("judge_reason", ""),
                "error_message": final_state.get("error_message", ""),
                "total_time": total_time
            }
            
        except Exception as e:
            total_time = time.time() - total_start_time
            print(f"❌ Graph Error: {str(e)} ({total_time:.2f}s)")
            return {
                "instruction": instruction,
                "label_json": {"type": "LLM", "sub_agents": [{"name": "DefaultAgent"}]},
                "model_type": "langgraph_conditional",
                "retry_attempts": 0,
                "success": False,
                "judge_passed": False,
                "judge_reason": "Graph execution failed",
                "error_message": f"Graph execution error: {str(e)}",
                "total_time": total_time
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
    
    # 모델 초기화
    model = LangGraphRetryAgent(max_retries=2)
    
    # 테스트 지시사항
    test_instruction = "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. {텍스트작성Agent}, {이미지생성Agent}, {동영상편집Agent}가 동시에 작업하고 {콘텐츠통합Agent}가 최종 결과물을 만들도록 해"
    
    # 워크플로우 생성
    result = model.generate_workflow(test_instruction)
    
    # 결과 출력
    print(f"\n🎯 Results:")
    print(f"   Success: {result['success']} | Judge: {result['judge_passed']} | Retries: {result['retry_attempts']} | Time: {result.get('total_time', 0):.2f}s")
    
    # 실패 시 이유 출력
    if not result['success'] or not result['judge_passed']:
        print(f"   ❌ Reason: {result.get('judge_reason', 'N/A')}")
    
    if result.get('error_message'):
        print(f"   ⚠️ Error: {result['error_message']}")
    
    print(f"\n📄 Generated JSON:")
    print(json.dumps(result['label_json'], ensure_ascii=False, indent=2))
    
    # 성능 요약
    if result['success'] and result['judge_passed']:
        print(f"\n✅ SUCCESS - Judge validation passed!")
    else:
        print(f"\n❌ FAILED - Check judge validation or increase retries")


if __name__ == "__main__":
    test_langgraph_conditional()
