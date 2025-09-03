"""
Gradio Web Demo for Workflow Agent
사용자가 지시사항을 입력하면 baseline 또는 langgraph 모델로 JSON을 생성하는 웹데모
"""

import gradio as gr
import json
import time
from datetime import datetime
from models.baseline_model import BaselineWorkflowAgent
from models.langgraph_model import LangGraphRetryAgent


def create_workflow_agent(model_type: str):
    """모델 타입에 따라 워크플로우 에이전트 생성"""
    if model_type == "Baseline Model":
        return BaselineWorkflowAgent()
    else:  # LangGraph Model
        return LangGraphRetryAgent(max_retries=3)


def generate_workflow(instruction: str, model_type: str):
    """워크플로우 생성 및 결과 반환"""
    if not instruction.strip():
        return "❌ 지시사항을 입력해주세요.", "", "", "", ""
    
    try:
        # 모델 초기화
        agent = create_workflow_agent(model_type)
        
        # 실행 시간 측정
        start_time = time.time()
        
        # 워크플로우 생성
        result = agent.generate_workflow(instruction)
        
        execution_time = time.time() - start_time
        
        # 결과 JSON 포맷팅
        generated_json = json.dumps(result['label_json'], ensure_ascii=False, indent=2)
        
        # 모델별 상세 정보
        if model_type == "Baseline Model":
            model_info = f"""
📊 **Baseline Model Results**
- Model Type: {result['model_type']}
- Execution Time: {execution_time:.2f}s
- Retry Attempts: {result['retry_attempts']}
- Status: ✅ Completed
            """
            
            # Baseline은 judge 정보가 없음
            judge_info = "Baseline 모델은 Judge 평가를 수행하지 않습니다."
            retry_info = "Baseline 모델은 재시도 로직이 없습니다."
            
        else:  # LangGraph Model
            model_info = f"""
🧠 **LangGraph Model Results**
- Model Type: {result['model_type']}
- Execution Time: {execution_time:.2f}s
- Retry Attempts: {result['retry_attempts']}
- Success: {'✅' if result['success'] else '❌'}
- Judge Passed: {'✅' if result['judge_passed'] else '❌'}
            """
            
            # Judge 결과 상세 정보
            if result['judge_passed']:
                judge_info = f"""
⚖️ **Judge Evaluation: PASSED** ✅
- 생성된 JSON이 지시사항의 의도를 올바르게 반영합니다.
- 품질 기준을 충족하여 추가 검토가 필요하지 않습니다.
                """
            else:
                judge_info = f"""
⚖️ **Judge Evaluation: FAILED** ❌
- 생성된 JSON이 지시사항의 의도를 충족하지 못했습니다.
- {result['retry_attempts']}회 재시도 후에도 품질 기준을 달성하지 못했습니다.
                """
            
            # 재시도 과정 정보
            if result['retry_attempts'] == 0:
                retry_info = """
🔄 **Retry Process: First Attempt Success** ✨
- 첫 번째 시도에서 Judge 평가를 통과했습니다.
- 재시도가 필요하지 않았습니다.
                """
            else:
                retry_info = f"""
🔄 **Retry Process: {result['retry_attempts']} Attempts Made**
- Judge 평가 실패로 인해 {result['retry_attempts']}회 재시도했습니다.
- 각 재시도마다 새로운 JSON을 생성하고 Judge 평가를 수행했습니다.
- 최종적으로 {'성공' if result['success'] else '실패'}했습니다.
                """
        
        # 에러 메시지가 있는 경우
        if result.get('error_message'):
            error_info = f"""
⚠️ **Error Information**
{result['error_message']}
            """
        else:
            error_info = "✅ 에러가 발생하지 않았습니다."
        
        return (
            generated_json,
            model_info,
            judge_info,
            retry_info,
            error_info
        )
        
    except Exception as e:
        error_msg = f"❌ 워크플로우 생성 중 오류가 발생했습니다: {str(e)}"
        return error_msg, "", "", "", ""


def create_demo():
    """Gradio 데모 인터페이스 생성"""
    
    # CSS 스타일링
    css = """
    .main-header {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .model-selector {
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .result-box {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .error-box {
        background: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
    }
    """
    
    with gr.Blocks(css=css, title="Workflow Agent Demo") as demo:
        # 헤더
        gr.HTML('<div class="main-header">🚀 Workflow Agent Demo</div>')
        
        with gr.Row():
            # 왼쪽: 입력 패널
            with gr.Column(scale=1):
                gr.HTML('<div class="model-selector">🎯 <strong>Model Selection</strong></div>')
                model_type = gr.Dropdown(
                    choices=["Baseline Model", "LangGraph Model"],
                    value="Baseline Model",
                    label="Select Model",
                    info="Baseline: 빠르고 단순, LangGraph: 품질 중심 + 재시도"
                )
                
                gr.HTML('<div class="model-selector">📝 <strong>Instruction Input</strong></div>')
                instruction_input = gr.Textbox(
                    label="한국어 지시사항을 입력하세요",
                    placeholder="예: 콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. {텍스트작성Agent}, {이미지생성Agent}, {동영상편집Agent}가 동시에 작업하고 {콘텐츠통합Agent}가 최종 결과물을 만들도록 해",
                    lines=5,
                    max_lines=10
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate Workflow",
                    variant="primary",
                    size="lg"
                )
                
                # 모델별 설명
                with gr.Accordion("ℹ️ Model Information", open=False):
                    gr.Markdown("""
                    ### 📊 Baseline Model
                    - **특징**: 빠르고 단순한 JSON 생성
                    - **장점**: 빠른 실행, 낮은 리소스 사용
                    - **단점**: 재시도 로직 없음, 품질 보장 어려움
                    - **사용처**: 프로토타이핑, 대량 처리
                    
                    ### 🧠 LangGraph Model
                    - **특징**: 조건부 엣지와 재시도 로직
                    - **장점**: 높은 품질, 자동 재시도, Judge 평가
                    - **단점**: 느린 실행, 높은 리소스 사용
                    - **사용처**: 프로덕션, 품질이 중요한 경우
                    """)
            
            # 오른쪽: 결과 패널
            with gr.Column(scale=1):
                gr.HTML('<div class="model-selector">📊 <strong>Generated Results</strong></div>')
                
                # 생성된 JSON
                json_output = gr.Code(
                    label="Generated JSON",
                    language="json",
                    lines=15,
                    interactive=False
                )
                
                # 모델 정보
                model_info_output = gr.Markdown(
                    label="Model Information",
                    value="모델을 선택하고 지시사항을 입력한 후 Generate 버튼을 클릭하세요."
                )
                
                # Judge 결과 (LangGraph만)
                judge_output = gr.Markdown(
                    label="Judge Evaluation",
                    value=""
                )
                
                # 재시도 정보 (LangGraph만)
                retry_output = gr.Markdown(
                    label="Retry Process",
                    value=""
                )
                
                # 에러 정보
                error_output = gr.Markdown(
                    label="Error Information",
                    value=""
                )
        
        # 이벤트 연결
        generate_btn.click(
            fn=generate_workflow,
            inputs=[instruction_input, model_type],
            outputs=[json_output, model_info_output, judge_output, retry_output, error_output]
        )
        
        # 예시 지시사항들
        gr.HTML('<div class="model-selector">💡 <strong>Example Instructions</strong></div>')
        
        examples = [
            "인사 부문 문의는 {직원정보Agent}에서, 급여 관련 요청은 {급여정보Agent}에서 데이터를 불러와서 응답을 생성하는 시스템을 만들어줘",
            "{고객문의Agent}가 고객의 문의를 받고 {답변작성Agent}가 답변을 작성한 후 {답변검토Agent}가 검토하는 플로우로 구성해줘",
            "{초기작성Agent}를 이용해서 짧은 보고서를 작성하는 에이전트를 만들어줘. 초고를 작성하고 {비평Agent}와 {수정Agent}로 몇번의 교정을 거쳐서 완성도를 높혀줄 수 있었으면 좋겠어.",
            "시장조사와 제품 개발, 마케팅 전략을 한 흐름에서 수행하는 에이전트를 설계해줘. {MarketResearchAgent}가 조사한 뒤 {ProductDevelopmentAgent}가 아이디어를 내고 {MarketingStrategyAgent}가 전략을 짠 다음 {SynthesisAgent}가 이를 종합하도록 해."
        ]
        
        gr.Examples(
            examples=examples,
            inputs=instruction_input,
            label="Click to try example instructions"
        )
    
    return demo


if __name__ == "__main__":
    # Gradio 데모 실행
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # 모든 IP에서 접근 가능
        server_port=7860,        # 기본 Gradio 포트
        share=False,             # 공개 링크 생성 여부
        show_error=True,         # 에러 상세 표시
        debug=True               # 디버그 모드
    )
