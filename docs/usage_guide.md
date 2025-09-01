# Workflow Agent 사용 가이드

## 1. 빠른 시작

### 1.1 환경 설정

#### 1.1.1 의존성 설치
```bash
# 프로젝트 디렉토리로 이동
cd workflow_agent

# 필요한 패키지 설치
pip install -r requirements.txt
```

#### 1.1.2 OpenAI API 키 설정
```bash
# 방법 1: .env 파일 생성 (권장)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 방법 2: 환경 변수 설정
export OPENAI_API_KEY="your-api-key-here"

# 방법 3: 대화형 설정
python setup_api_key.py
```

### 1.2 기본 실행

#### 1.2.1 Baseline 모델 테스트
```bash
# Baseline 모델 직접 테스트
python models/baseline_model.py
```

**예상 출력:**
```
🧪 Testing Baseline Model
==================================================
📝 Test Instruction: 콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘...
--------------------------------------------------
✅ Model Type: baseline
🔢 Retry Attempts: 0
📄 Generated JSON:
{
  "flow_name": "ContentCreationPipeline",
  "type": "Sequential",
  "sub_agents": [
    {
      "flow": {
        "flow_name": "ContentProductionParallel",
        "type": "Parallel",
        "sub_agents": [
          {"agent_name": "텍스트작성Agent"},
          {"agent_name": "이미지생성Agent"},
          {"agent_name": "동영상편집Agent"}
        ]
      }
    },
    {"agent_name": "콘텐츠통합Agent"}
  ]
}
```

#### 1.2.2 LangGraph 모델 테스트
```bash
# LangGraph 모델 테스트 (PNG 그래프 생성 포함)
python models/langgraph_model.py
```

**예상 출력:**
```
🧪 Testing LangGraph Conditional Edge Model
============================================================
📊 Saving LangGraph workflow as PNG image...
✅ Graph saved as PNG: ./models/langgraph_workflow_20250901_164255.png
📝 Test Instruction: 콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘...
------------------------------------------------------------
✅ Model Type: langgraph_conditional
🔢 Retry Attempts: 1
🎯 Success: True
⚖️ Judge Passed: True
📄 Generated JSON: {...}
```

## 2. 상세 사용법

### 2.1 Python 코드에서 직접 사용

#### 2.1.1 Baseline 모델 사용
```python
from models.baseline_model import BaselineWorkflowAgent

# 모델 초기화
agent = BaselineWorkflowAgent(model_name="gpt-4o-mini")

# 워크플로우 생성
instruction = "고객 문의를 받고 답변을 작성하는 시스템을 만들어줘"
result = agent.generate_workflow(instruction)

# 결과 확인
print(f"모델 타입: {result['model_type']}")
print(f"재시도 횟수: {result['retry_attempts']}")
print(f"생성된 JSON: {result['label_json']}")

# 모델 정보
info = agent.get_model_info()
print(f"특징: {info['features']}")
```

#### 2.1.2 LangGraph 모델 사용
```python
from models.langgraph_model import LangGraphRetryAgent

# 모델 초기화 (최대 3회 재시도)
agent = LangGraphRetryAgent(model_name="gpt-4o-mini", max_retries=3)

# 그래프 시각화 저장
png_path = agent.save_graph_as_png()
print(f"그래프 저장됨: {png_path}")

# 워크플로우 생성 (재시도 로직 포함)
instruction = "복잡한 워크플로우를 만들어줘"
result = agent.generate_workflow(instruction)

# 결과 확인
print(f"모델 타입: {result['model_type']}")
print(f"재시도 횟수: {result['retry_attempts']}")
print(f"성공 여부: {result['success']}")
print(f"Judge 통과: {result['judge_passed']}")
print(f"에러 메시지: {result.get('error_message', '없음')}")
```

### 2.2 테스트 실행

#### 2.2.1 전체 테스트 데이터로 테스트
```bash
# Baseline 모델로 테스트
python test.py --model baseline --outdir results

# LangGraph 모델로 테스트
python test.py --model langgraph --outdir results

# 기본값 (Baseline 모델)
python test.py
```

#### 2.2.2 테스트 결과 확인
테스트 실행 후 `results/` 디렉토리에 Excel 파일이 생성됩니다:

- **baseline_test_results_YYYYMMDD_HHMMSS.xlsx**
- **langgraph_test_results_YYYYMMDD_HHMMSS.xlsx**

Excel 파일에는 다음 시트들이 포함됩니다:
1. **Detailed Results**: 각 테스트 케이스의 상세 결과
2. **Summary**: 전체 테스트 요약 통계
3. **Type Analysis**: 워크플로우 타입별 분석

## 3. 워크플로우 타입별 예시

### 3.1 LLM 타입 예시

#### 3.1.1 입력 지시문
```
"인사 부문 문의는 {직원정보Agent}에서, 급여 관련 요청은 {급여정보Agent}에서 데이터를 불러와서 응답을 생성하는 시스템을 만들어줘"
```

#### 3.1.2 생성된 JSON
```json
{
  "flow_name": "HRQNAagent",
  "type": "LLM",
  "tools": [
    {"agent_name": "직원정보Agent"},
    {"agent_name": "급여정보Agent"}
  ]
}
```

#### 3.1.3 사용 사례
- 고객 지원 시스템
- FAQ 시스템
- 정보 조회 시스템
- 도움말 시스템

### 3.2 Sequential 타입 예시

#### 3.2.1 입력 지시문
```
"{고객문의Agent}가 고객의 문의를 받고 {답변작성Agent}가 답변을 작성한 후 {답변검토Agent}가 검토하는 플로우로 구성해줘"
```

#### 3.2.2 생성된 JSON
```json
{
  "flow_name": "CustomerSupportPipeline",
  "type": "Sequential",
  "sub_agents": [
    {"agent_name": "고객문의Agent"},
    {"agent_name": "답변작성Agent"},
    {"agent_name": "답변검토Agent"}
  ]
}
```

#### 3.2.3 사용 사례
- 주문 처리 시스템
- 문서 승인 워크플로우
- 품질 검사 프로세스
- 채용 프로세스

### 3.3 Loop 타입 예시

#### 3.3.1 입력 지시문
```
"{초기작성Agent}를 이용해서 짧은 보고서를 작성하는 에이전트를 만들어줘. 초고를 작성하고 {비평Agent}와 {수정Agent}로 몇번의 교정을 거쳐서 완성도를 높혀줄 수 있었으면 좋겠어."
```

#### 3.3.2 생성된 JSON
```json
{
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
}
```

#### 3.3.3 사용 사례
- 문서 작성 및 교정
- 코드 리뷰 프로세스
- 디자인 반복 개선
- 품질 검사 및 수정

### 3.4 Parallel 타입 예시

#### 3.4.1 입력 지시문
```
"시장조사와 제품 개발, 마케팅 전략을 한 흐름에서 수행하는 에이전트를 설계해줘. {MarketResearchAgent}가 조사한 뒤 {ProductDevelopmentAgent}가 아이디어를 내고 {MarketingStrategyAgent}가 전략을 짠 다음 {SynthesisAgent}가 이를 종합하도록 해."
```

#### 3.4.2 생성된 JSON
```json
{
  "flow_name": "BusinessDevelopmentPipeline",
  "type": "Sequential",
  "sub_agents": [
    {
      "flow": {
        "flow_name": "ResearchAndDevelopmentParallel",
        "type": "Parallel",
        "sub_agents": [
          {"agent_name": "MarketResearchAgent"},
          {"agent_name": "ProductDevelopmentAgent"},
          {"agent_name": "MarketingStrategyAgent"}
        ]
      }
    },
    {"agent_name": "SynthesisAgent"}
  ]
}
```

#### 3.4.3 사용 사례
- 프로젝트 계획 수립
- 제품 개발 프로세스
- 이벤트 기획 및 실행
- 연구 및 분석 프로젝트

## 4. 고급 사용법

### 4.1 커스텀 프롬프트 사용

#### 4.1.1 프롬프트 수정
```python
# prompts.py 파일에서 프롬프트 수정
JSON_PROMPT = """
당신은 워크플로우 설계 전문가입니다.
한국어 지시문을 분석하여 적절한 워크플로우 JSON을 생성하세요.

[커스텀 지침 추가...]

Instruction: {instruction}
"""
```

#### 4.1.2 새로운 프롬프트 추가
```python
# prompts.py에 새로운 프롬프트 추가
CUSTOM_PROMPT = """
특정 도메인에 특화된 워크플로우 생성 프롬프트
"""

# 모델에서 사용
self.custom_chain = ChatPromptTemplate.from_template(CUSTOM_PROMPT) | self.llm | self.parser
```

### 4.2 모델 설정 커스터마이징

#### 4.2.1 LLM 파라미터 조정
```python
# Baseline 모델에서 LLM 설정 조정
self.llm = ChatOpenAI(
    model="gpt-4o",  # 더 강력한 모델 사용
    temperature=0.1,  # 약간의 창의성 허용
    max_tokens=1024,  # 더 긴 출력 허용
    timeout=30        # 타임아웃 설정
)
```

#### 4.2.2 재시도 전략 조정
```python
# LangGraph 모델에서 재시도 전략 조정
agent = LangGraphRetryAgent(
    model_name="gpt-4o-mini",
    max_retries=5,  # 더 많은 재시도 허용
)
```

### 4.3 에러 처리 및 로깅

#### 4.3.1 상세한 에러 로깅
```python
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 에러 발생 시 상세 로깅
try:
    result = agent.generate_workflow(instruction)
except Exception as e:
    logger.error(f"워크플로우 생성 실패: {e}")
    logger.error(f"지시문: {instruction}")
    # 에러 복구 로직
```

#### 4.3.2 성능 모니터링
```python
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper

# 함수에 데코레이터 적용
@measure_time
def generate_workflow(self, instruction):
    # 기존 로직
    pass
```

## 5. 문제 해결

### 5.1 일반적인 문제들

#### 5.1.1 API 키 관련 문제
```bash
# API 키 확인
echo $OPENAI_API_KEY

# .env 파일 확인
cat .env

# API 키 재설정
export OPENAI_API_KEY="your-new-key-here"
```

#### 5.1.2 의존성 문제
```bash
# 패키지 재설치
pip uninstall -r requirements.txt
pip install -r requirements.txt

# 가상환경 사용 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 5.1.3 메모리 부족 문제
```python
# LangGraph 모델에서 메모리 사용량 줄이기
agent = LangGraphRetryAgent(
    model_name="gpt-4o-mini",  # 더 작은 모델 사용
    max_retries=2               # 재시도 횟수 줄이기
)
```

### 5.2 성능 최적화

#### 5.2.1 배치 처리
```python
# 여러 지시문을 동시에 처리
instructions = [
    "첫 번째 워크플로우",
    "두 번째 워크플로우",
    "세 번째 워크플로우"
]

results = []
for instruction in instructions:
    result = agent.generate_workflow(instruction)
    results.append(result)
```

#### 5.2.2 캐싱 구현
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate_workflow(instruction):
    return agent.generate_workflow(instruction)
```

### 5.3 디버깅

#### 5.3.1 상세한 디버그 정보
```python
# LangGraph 모델에서 상태 추적
def _generate_json_node(self, state: WorkflowState) -> Dict[str, Any]:
    print(f"현재 상태: {state}")
    print(f"재시도 횟수: {state['retry_count']}")
    
    # 기존 로직...
    
    print(f"생성된 JSON: {generated_json}")
    return result
```

#### 5.3.2 단계별 실행
```python
# LangGraph의 단계별 실행
workflow = agent.graph

# 초기 상태
initial_state = {
    "instruction": "테스트 지시문",
    "generated_json": {},
    "retry_count": 0,
    "max_retries": 3,
    "judge_passed": False,
    "success": False,
    "error_message": ""
}

# 단계별 실행
result1 = workflow.invoke(initial_state, config={"configurable": {"thread_id": "1"}})
print(f"첫 번째 단계 결과: {result1}")

# 다음 단계로 진행
result2 = workflow.invoke(result1, config={"configurable": {"thread_id": "1"}})
print(f"두 번째 단계 결과: {result2}")
```

## 6. 모니터링 및 분석

### 6.1 성능 지표 수집

#### 6.1.1 기본 지표
```python
# 성능 지표 수집
performance_metrics = {
    "total_instructions": 0,
    "successful_generations": 0,
    "failed_generations": 0,
    "average_time": 0.0,
    "total_retries": 0,
    "model_type": "unknown"
}
```

#### 6.1.2 상세 분석
```python
# 워크플로우 타입별 성능 분석
type_performance = {
    "LLM": {"count": 0, "success_rate": 0.0, "avg_time": 0.0},
    "Sequential": {"count": 0, "success_rate": 0.0, "avg_time": 0.0},
    "Loop": {"count": 0, "success_rate": 0.0, "avg_time": 0.0},
    "Parallel": {"count": 0, "success_rate": 0.0, "avg_time": 0.0}
}
```

### 6.2 결과 시각화

#### 6.2.1 차트 생성
```python
import matplotlib.pyplot as plt
import pandas as pd

# 성능 데이터를 DataFrame으로 변환
df = pd.DataFrame(results_data)

# 성공률 차트
plt.figure(figsize=(10, 6))
df.groupby('Type')['Exact_Match'].value_counts().unstack().plot(kind='bar')
plt.title('워크플로우 타입별 성공률')
plt.xlabel('워크플로우 타입')
plt.ylabel('성공률')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

#### 6.2.2 실행 시간 분석
```python
# 실행 시간 분포
plt.figure(figsize=(10, 6))
df['Execution_Time'].hist(bins=20, alpha=0.7)
plt.title('실행 시간 분포')
plt.xlabel('실행 시간 (초)')
plt.ylabel('빈도')
plt.show()
```

## 7. 프로덕션 배포

### 7.1 환경 설정

#### 7.1.1 프로덕션 설정
```python
# config.py
PRODUCTION_CONFIG = {
    "model_name": "gpt-4o",  # 프로덕션용 강력한 모델
    "max_retries": 3,        # 적절한 재시도 횟수
    "timeout": 30,           # 타임아웃 설정
    "log_level": "INFO",     # 로깅 레벨
    "cache_enabled": True,   # 캐싱 활성화
    "monitoring_enabled": True  # 모니터링 활성화
}
```

#### 7.1.2 보안 설정
```python
# 환경 변수에서 민감한 정보 로드
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
```

### 7.2 로드 밸런싱

#### 7.2.1 여러 인스턴스 실행
```python
# 여러 모델 인스턴스 생성
agents = [
    BaselineWorkflowAgent(model_name="gpt-4o-mini"),
    BaselineWorkflowAgent(model_name="gpt-4o-mini"),
    BaselineWorkflowAgent(model_name="gpt-4o-mini")
]

# 라운드 로빈 방식으로 요청 분산
current_agent = 0
def get_next_agent():
    global current_agent
    agent = agents[current_agent]
    current_agent = (current_agent + 1) % len(agents)
    return agent
```

#### 7.2.2 비동기 처리
```python
import asyncio
import aiohttp

async def async_generate_workflows(instructions):
    async def generate_single(instruction):
        # 비동기로 워크플로우 생성
        return await agent.generate_workflow_async(instruction)
    
    # 모든 지시문을 동시에 처리
    tasks = [generate_single(instruction) for instruction in instructions]
    results = await asyncio.gather(*tasks)
    return results
```

이 가이드를 통해 Workflow Agent를 효과적으로 사용하고, 필요에 따라 커스터마이징할 수 있을 것입니다.
