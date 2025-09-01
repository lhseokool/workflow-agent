# Workflow Agent 프로젝트 아키텍처

## 1. 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "User Interface"
        A[한국어 지시문 입력]
        B[모델 선택<br/>Baseline vs LangGraph]
    end
    
    subgraph "Core Models"
        C[Baseline Model<br/>BaselineWorkflowAgent]
        D[LangGraph Model<br/>LangGraphRetryAgent]
    end
    
    subgraph "Processing Pipeline"
        E[JSON 생성]
        F[LLM-as-Judge 평가]
        G{품질 검증}
        H[재시도 로직]
    end
    
    subgraph "Output & Evaluation"
        I[워크플로우 JSON]
        J[테스트 결과]
        K[Excel 리포트]
        L[PNG 그래프]
    end
    
    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    G -->|통과| I
    G -->|실패| H
    H --> E
    I --> J
    J --> K
    D --> L
```

## 2. Baseline Model 아키텍처

```mermaid
graph LR
    A[한국어 지시문] --> B[ChatPromptTemplate]
    B --> C[ChatOpenAI<br/>GPT-4o-mini]
    C --> D[StrOutputParser]
    D --> E[JSON 결과]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
```

### 2.1 Baseline Model 특징
- **단순한 체인 구조**: Input → JSON Chain → Output
- **재시도 로직 없음**: 단일 시도로 빠른 결과 생성
- **낮은 리소스 사용**: 최소한의 API 호출
- **일관된 출력**: Temperature 0.0으로 결정적 결과

## 3. LangGraph Model 아키텍처

```mermaid
graph TD
    A[한국어 지시문] --> B[WorkflowState 초기화]
    B --> C[generate_json_node]
    C --> D[JSON 생성]
    D --> E[judge_node]
    E --> F[LLM-as-Judge 평가]
    F --> G{품질 검증}
    
    G -->|통과| H[성공 종료]
    G -->|실패| I{재시도 가능?}
    
    I -->|Yes| J[retry_count 증가]
    I -->|No| K[최대 재시도 도달]
    
    J --> C
    K --> L[실패 종료]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
    style F fill:#f1f8e9
    style G fill:#ffebee
    style H fill:#e8f5e8
    style I fill:#fff3e0
    style J fill:#f1f8e9
    style K fill:#ffebee
    style L fill:#ffebee
```

### 3.1 LangGraph Model 특징
- **Conditional Edge**: Judge 결과에 따른 조건부 재시도
- **상태 관리**: 완전한 워크플로우 상태 추적
- **품질 보장**: LLM-as-Judge를 통한 지속적인 품질 검증
- **시각화**: PNG 그래프 생성으로 문서화 지원

## 4. 데이터 흐름

```mermaid
flowchart TD
    A[test_data.json] --> B[테스트 케이스 로드]
    B --> C[모델 선택]
    
    C --> D[Baseline Model]
    C --> E[LangGraph Model]
    
    D --> F[단일 JSON 생성]
    E --> G[JSON 생성 + Judge 평가]
    
    G --> H{Judge 통과?}
    H -->|No| I[재시도]
    H -->|Yes| J[결과 저장]
    
    I --> G
    F --> J
    
    J --> K[Exact Match 평가]
    K --> L[Excel 결과 저장]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
    style F fill:#f1f8e9
    style G fill:#fff3e0
    style H fill:#ffebee
    style I fill:#f1f8e9
    style J fill:#e8f5e8
    style K fill:#fff3e0
    style L fill:#e8f5e8
```

## 5. 워크플로우 타입 구조

### 5.1 LLM 타입
```mermaid
graph LR
    A[LLM Type] --> B[tools 배열]
    B --> C[에이전트1]
    B --> D[에이전트2]
    B --> E[에이전트N]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
```

### 5.2 Sequential 타입
```mermaid
graph LR
    A[Sequential Type] --> B[sub_agents 배열]
    B --> C[에이전트1] --> D[에이전트2] --> E[에이전트N]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
```

### 5.3 Loop 타입
```mermaid
graph TD
    A[Sequential Type] --> B[초기 에이전트]
    A --> C[Loop Flow]
    C --> D[Loop Type]
    D --> E[검토 에이전트]
    D --> F[수정 에이전트]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
    style F fill:#f1f8e9
```

### 5.4 Parallel 타입
```mermaid
graph TD
    A[Sequential Type] --> B[Parallel Flow]
    A --> C[통합 에이전트]
    
    B --> D[Parallel Type]
    D --> E[에이전트1]
    D --> F[에이전트2]
    D --> G[에이전트3]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
    style F fill:#f1f8e9
    style G fill:#fff3e0
```

## 6. 테스트 시스템 구조

```mermaid
graph TD
    A[test.py 실행] --> B[모델 선택]
    B --> C[테스트 데이터 로드]
    C --> D[테스트 케이스 순회]
    
    D --> E[워크플로우 생성]
    E --> F[Exact Match 평가]
    F --> G[결과 데이터 수집]
    
    G --> H[통계 계산]
    H --> I[Excel 저장]
    
    subgraph "평가 지표"
        J[Exact Match Rate]
        K[Average Time]
        L[Type별 분석]
    end
    
    I --> J
    I --> K
    I --> L
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
    style F fill:#f1f8e9
    style G fill:#fff3e0
    style H fill:#ffebee
    style I fill:#e8f5e8
    style J fill:#f1f8e9
    style K fill:#fff3e0
    style L fill:#ffebee
```

## 7. 파일 구조 및 의존성

```mermaid
graph TD
    A[main entry points] --> B[test.py]
    A --> C[models/baseline_model.py]
    A --> D[models/langgraph_model.py]
    
    B --> E[prompts.py]
    B --> F[utils.py]
    B --> G[data/test_data.json]
    
    C --> E
    C --> F
    
    D --> E
    D --> F
    D --> H[langgraph]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#e8f5e8
    style F fill:#f1f8e9
    style G fill:#fff3e0
    style H fill:#ffebee
```

## 8. 성능 비교 매트릭스

| 측정 지표 | Baseline Model | LangGraph Model |
|-----------|----------------|-----------------|
| **실행 속도** | ⚡ 매우 빠름 (1-2초) | 🐌 느림 (3-10초) |
| **정확도** | 📊 보통 | 🎯 높음 |
| **재시도 로직** | ❌ 없음 | ✅ 조건부 재시도 |
| **리소스 사용** | 💾 낮음 | 💾 높음 |
| **복잡도** | 🔧 단순 | 🧠 복잡 |
| **시각화** | ❌ 없음 | ✅ PNG 그래프 |
| **적합한 용도** | 프로토타이핑 | 프로덕션 |

## 9. 에러 처리 및 복구 전략

```mermaid
graph TD
    A[에러 발생] --> B{에러 타입}
    
    B -->|JSON 파싱 에러| C[Fallback JSON 반환]
    B -->|API 호출 에러| D[재시도 또는 종료]
    B -->|Judge 평가 에러| E[기본값으로 처리]
    
    C --> F[결과 반환]
    D --> G{재시도 가능?}
    E --> F
    
    G -->|Yes| H[지연 후 재시도]
    G -->|No| I[에러 메시지와 함께 종료]
    
    H --> J[원래 작업 재시도]
    J --> F
    
    style A fill:#ffebee
    style B fill:#fff3e0
    style C fill:#f1f8e9
    style D fill:#ffebee
    style E fill:#f1f8e9
    style F fill:#e8f5e8
    style G fill:#fff3e0
    style H fill:#f1f8e9
    style I fill:#ffebee
    style J fill:#fff3e0
```

## 10. 확장성 고려사항

### 10.1 수평적 확장
- **로드 밸런싱**: 여러 모델 인스턴스 분산 처리
- **마이크로서비스**: 각 모델을 독립적인 서비스로 분리
- **API 게이트웨이**: 통합된 엔드포인트 제공

### 10.2 수직적 확장
- **모델 업그레이드**: 더 강력한 LLM 모델 사용
- **메모리 최적화**: 대용량 데이터 처리 최적화
- **병렬 처리**: 여러 워크플로우 동시 생성

### 10.3 기능적 확장
- **새로운 워크플로우 타입**: Conditional, Event-driven 등
- **다국어 지원**: 영어, 일본어 등 추가 언어
- **도메인 특화**: 업계별 맞춤형 프롬프트
