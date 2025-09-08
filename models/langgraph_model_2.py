"""
LangGraph Model 2 - 3-Stage Workflow Generation
워크플로우 타입 예측 -> 워크플로우 생성 -> 검증의 3단계 프로세스
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
from prompts import (
    WORKFLOW_TYPE_PREDICTION_PROMPT, 
    WORKFLOW_GENERATION_PROMPT, 
    WORKFLOW_VALIDATION_PROMPT
)


class ThreeStageWorkflowState(TypedDict):
    """3단계 워크플로우 상태"""
    instruction: str
    predicted_type: str
    prediction_reason: str
    generated_json: Dict[str, Any]
    json_valid: bool
    type_consistent: bool
    intent_matched: bool
    overall_passed: bool
    validation_reason: str
    error_message: str
    retry_count: int
    max_retries: int
    success: bool


class ThreeStageWorkflowAgent:
    """
    3-Stage Workflow Generation Agent
    1. Predict workflow type (LLM/Sequential/Loop/Parallel)
    2. Generate workflow JSON based on predicted type
    3. Validate the generated workflow
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", max_retries: int = 3):
        """Initialize the 3-stage workflow agent"""
        self.model_name = model_name
        self.max_retries = max_retries
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=512
        )
        self.parser = StrOutputParser()
        
        # Chains for each stage
        self.prediction_chain = ChatPromptTemplate.from_template(WORKFLOW_TYPE_PREDICTION_PROMPT) | self.llm | self.parser
        self.generation_chain = ChatPromptTemplate.from_template(WORKFLOW_GENERATION_PROMPT) | self.llm | self.parser
        self.validation_chain = ChatPromptTemplate.from_template(WORKFLOW_VALIDATION_PROMPT) | self.llm | self.parser
        
        # Build LangGraph
        self.graph = self._build_graph()
    
    def _predict_workflow_type_node(self, state: ThreeStageWorkflowState) -> Dict[str, Any]:
        """Stage 1: 워크플로우 타입 예측"""
        start_time = time.time()
        try:
            # 타입 예측 체인 실행
            result = self.prediction_chain.invoke({"instruction": state["instruction"]})
            
            # JSON 응답 파싱
            try:
                prediction_result = json.loads(result.strip())
                predicted_type = prediction_result.get("type", "Sequential")
                reason = prediction_result.get("reason", "No reason provided")
            except json.JSONDecodeError:
                # Fallback: 텍스트에서 타입 추출
                valid_types = ["LLM", "Sequential", "Loop", "Parallel"]
                predicted_type = next((t for t in valid_types if t in result), "Sequential")
                reason = "Fallback extraction from text"
            
            elapsed_time = time.time() - start_time
            print(f"🔮 Stage 1 - {predicted_type} ({elapsed_time:.2f}s)")
            
            return {
                "predicted_type": predicted_type,
                "prediction_reason": reason,
                "error_message": ""
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Stage 1 - Error ({elapsed_time:.2f}s)")
            return {
                "predicted_type": "Sequential",  # 기본값
                "prediction_reason": f"Prediction error: {str(e)}",
                "error_message": f"Type prediction error: {str(e)}"
            }
    
    def _generate_workflow_node(self, state: ThreeStageWorkflowState) -> Dict[str, Any]:
        """Stage 2: 워크플로우 생성"""
        start_time = time.time()
        try:
            # 워크플로우 생성 체인 실행
            result = self.generation_chain.invoke({
                "instruction": state["instruction"],
                "predicted_type": state["predicted_type"]
            })
            
            # JSON 파싱
            generated_json = json.loads(result.strip())
            elapsed_time = time.time() - start_time
            print(f"⚙️ Stage 2 - Generated ({elapsed_time:.2f}s)")
            
            return {
                "generated_json": generated_json, 
                "error_message": ""
            }
            
        except (json.JSONDecodeError, Exception) as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Stage 2 - Error ({elapsed_time:.2f}s)")
            return {
                "generated_json": self._get_fallback_json(state.get("predicted_type", "Sequential")),
                "error_message": f"Workflow generation error: {str(e)}"
            }
    
    def _validate_workflow_node(self, state: ThreeStageWorkflowState) -> Dict[str, Any]:
        """Stage 3: 워크플로우 검증"""
        start_time = time.time()
        try:
            # 검증 체인 실행 (타임아웃 설정)
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Validation chain timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10초 타임아웃
            
            try:
                result = self.validation_chain.invoke({
                    "instruction": state["instruction"],
                    "predicted_type": state["predicted_type"],
                    "generated_json": json.dumps(state["generated_json"], ensure_ascii=False)
                })
            finally:
                signal.alarm(0)  # 타임아웃 해제
            
            # 검증 결과 파싱
            try:
                validation_result = json.loads(result.strip())
                overall_passed = validation_result.get("passed", False)
                validation_reason = validation_result.get("reason", "No reason provided")
            except json.JSONDecodeError as json_err:
                print(f"   JSON Parse Error in validation result: {json_err}")
                print(f"   Raw validation result: {result}")
                # Fallback: 간단한 텍스트 분석
                overall_passed = "true" in result.lower() or "pass" in result.lower()
                validation_reason = f"JSON parse failed, fallback analysis: {result[:100]}"
            
            elapsed_time = time.time() - start_time
            status = "✅ PASSED" if overall_passed else "❌ FAILED"
            print(f"{status} Stage 3 - Validation ({elapsed_time:.2f}s)")
            
            # 실패 시 이유 출력
            if not overall_passed:
                print(f"   Reason: {validation_reason}")
            
            return {
                "json_valid": True,
                "type_consistent": overall_passed,
                "intent_matched": overall_passed,
                "overall_passed": overall_passed,
                "validation_reason": validation_reason,
                "success": overall_passed,
                "retry_count": state["retry_count"] + (0 if overall_passed else 1),
                "error_message": "" if overall_passed else validation_reason
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Stage 3 - Error ({elapsed_time:.2f}s)")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error details: {str(e)}")
            
            # API 에러 시 FAILED 처리 (안전한 선택)
            validation_reason = f"API error: {str(e)[:100]} - Cannot validate reliably"
            print(f"   Fallback result: FAILED (API error - retry needed)")
            
            return {
                "json_valid": False,
                "type_consistent": False,
                "intent_matched": False,
                "overall_passed": False,
                "validation_reason": validation_reason,
                "success": False,
                "retry_count": state["retry_count"] + 1,
                "error_message": f"Validation error: {str(e)}"
            }
    
    def _should_retry(self, state: ThreeStageWorkflowState) -> str:
        """Conditional edge: retry 여부 결정"""
        # 검증 통과했으면 종료
        if state.get("overall_passed", False):
            return "end"
        
        # 최대 retry 도달했으면 종료
        if state["retry_count"] >= state["max_retries"]:
            print(f"⚠️ Max retries reached ({state['max_retries']})")
            return "end"
        
        # retry 계속 (타입 예측부터 다시 시작)
        print(f"🔄 Retry {state['retry_count']}/{state['max_retries']}")
        return "retry"
    
    def _get_fallback_json(self, predicted_type: str) -> Dict[str, Any]:
        """타입별 fallback JSON 구조"""
        fallbacks = {
            "LLM": {"flow_name": "DefaultSystem", "type": "LLM", "tools": [{"agent_name": "DefaultAgent"}]},
            "Loop": {
                "flow_name": "DefaultPipeline", "type": "Sequential", 
                "sub_agents": [
                    {"agent_name": "InitialAgent"}, 
                    {"flow": {"flow_name": "DefaultLoop", "type": "Loop", "sub_agents": [{"agent_name": "ReviewAgent"}, {"agent_name": "RefineAgent"}]}}
                ]
            },
            "Parallel": {
                "flow_name": "DefaultPipeline", "type": "Sequential", 
                "sub_agents": [
                    {"flow": {"flow_name": "DefaultParallel", "type": "Parallel", "sub_agents": [{"agent_name": "Agent1"}, {"agent_name": "Agent2"}]}}, 
                    {"agent_name": "SynthesisAgent"}
                ]
            }
        }
        return fallbacks.get(predicted_type, {"flow_name": "DefaultPipeline", "type": "Sequential", "sub_agents": [{"agent_name": "DefaultAgent"}]})
    
    def _build_graph(self) -> StateGraph:
        """3-Stage LangGraph 구성"""
        # StateGraph 생성
        workflow = StateGraph(ThreeStageWorkflowState)
        
        # 노드 추가
        workflow.add_node("predict_type", self._predict_workflow_type_node)
        workflow.add_node("generate_workflow", self._generate_workflow_node)
        workflow.add_node("validate_workflow", self._validate_workflow_node)
        
        # 엣지 추가
        workflow.add_edge("predict_type", "generate_workflow")
        workflow.add_edge("generate_workflow", "validate_workflow")
        workflow.add_conditional_edges(
            "validate_workflow",
            self._should_retry,
            {
                "retry": "predict_type",  # 실패 시 처음부터 다시 시작
                "end": END
            }
        )
        
        # 시작점 설정
        workflow.set_entry_point("predict_type")
        
        return workflow.compile()
    
    def generate_workflow(self, instruction: str) -> Dict[str, Any]:
        """3단계 프로세스를 사용한 워크플로우 생성"""
        total_start_time = time.time()
        
        # 초기 상태 설정
        initial_state = {
            "instruction": instruction, 
            "predicted_type": "", 
            "prediction_reason": "",
            "generated_json": {}, 
            "json_valid": False, 
            "type_consistent": False,
            "intent_matched": False, 
            "overall_passed": False, 
            "validation_reason": "",
            "error_message": "", 
            "retry_count": 0, 
            "max_retries": self.max_retries,
            "success": False
        }
        
        print(f"🚀 3-Stage Workflow Generation")
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
                "predicted_type": final_state["predicted_type"],
                "prediction_reason": final_state.get("prediction_reason", ""),
                "label_json": final_state["generated_json"],
                "model_type": "langgraph_3stage",
                "retry_attempts": final_state["retry_count"],
                "success": final_state["success"],
                "overall_passed": final_state["overall_passed"],
                "json_valid": final_state["json_valid"],
                "type_consistent": final_state["type_consistent"],
                "intent_matched": final_state["intent_matched"],
                "validation_reason": final_state.get("validation_reason", ""),
                "error_message": final_state.get("error_message", ""),
                "total_time": total_time
            }
            
        except Exception as e:
            total_time = time.time() - total_start_time
            print(f"❌ Graph Error: {str(e)} ({total_time:.2f}s)")
            return {
                "instruction": instruction, 
                "predicted_type": "Unknown",
                "prediction_reason": f"Graph error: {str(e)}", 
                "model_type": "langgraph_3stage",
                "label_json": {"type": "Sequential", "sub_agents": [{"agent_name": "DefaultAgent"}]},
                "retry_attempts": 0, 
                "success": False, 
                "overall_passed": False,
                "json_valid": False, 
                "type_consistent": False, 
                "intent_matched": False,
                "validation_reason": "Graph execution failed", 
                "total_time": total_time,
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
            png_filename = f"langgraph_3stage_workflow_{timestamp}.png"
            png_path = os.path.join(output_dir, png_filename)
            
            # LangGraph의 내장 기능으로 PNG 저장
            self.graph.get_graph().draw_mermaid_png(output_file_path=png_path)
            
            return png_path
            
        except Exception as e:
            print(f"⚠️ Failed to save graph as PNG: {e}")
            print("💡 Make sure you have the required dependencies installed:")
            print("   pip install pygraphviz or pip install pydot")
            return ""
    
    def get_model_info(self) -> Dict[str, str]:
        """모델 정보 반환"""
        return {
            "model_type": "langgraph_3stage",
            "model_name": self.model_name,
            "description": "3-Stage Workflow Generation: Predict -> Generate -> Validate",
            "features": "type_prediction + workflow_generation + validation",
            "max_retries": str(self.max_retries)
        }


# 테스트 함수
def test_three_stage_workflow():
    """3-Stage Workflow Model 테스트"""
    print("🧪 Testing 3-Stage Workflow Generation Model")
    print("="*60)
    
    # 모델 초기화
    model = ThreeStageWorkflowAgent(max_retries=2)
    
    # 테스트 지시사항
    test_instruction = "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. {텍스트작성Agent}, {이미지생성Agent}, {동영상편집Agent}가 동시에 작업하고 {콘텐츠통합Agent}가 최종 결과물을 만들도록 해"
    
    # 워크플로우 생성
    result = model.generate_workflow(test_instruction)
    
    # 결과 출력
    print(f"\n🎯 Results:")
    print(f"   Type: {result['predicted_type']} | Success: {result['success']} | Retries: {result['retry_attempts']} | Time: {result.get('total_time', 0):.2f}s")
    
    # 실패 시 이유 출력
    if not result['success'] or not result['overall_passed']:
        print(f"   ❌ Reason: {result.get('validation_reason', 'N/A')}")
    
    if result.get('error_message'):
        print(f"   ⚠️ Error: {result['error_message']}")
    
    print(f"\n📄 Generated JSON:")
    print(json.dumps(result['label_json'], ensure_ascii=False, indent=2))
    
    # 성능 요약
    if result['success'] and result['overall_passed']:
        print(f"\n✅ SUCCESS - Predicted '{result['predicted_type']}' correctly!")
    else:
        print(f"\n❌ FAILED - Check validation logic or increase retries")


if __name__ == "__main__":
    test_three_stage_workflow()