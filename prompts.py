"""
Workflow Agent Prompts
Simple prompt templates for JSON generation and evaluation
"""



# JSON Generation Prompt
JSON_PROMPT = """
Analyze the Korean instruction and generate a workflow JSON. Return ONLY the JSON object, no explanations or additional text.

You must create one of these 4 workflow types:

1. LLM type (for Q&A, assistance, help systems that fetch data from multiple agents):
{{"flow_name": "SystemName", "type": "LLM", "tools": [{{"agent_name": "ToolAgent1"}}, {{"agent_name": "ToolAgent2"}}]}}

2. Sequential type (for step-by-step processes):
{{"flow_name": "PipelineName", "type": "Sequential", "sub_agents": [{{"agent_name": "Agent1"}}, {{"agent_name": "Agent2"}}]}}

3. Loop type (for iterative refinement processes):
{{"flow_name": "PipelineName", "type": "Sequential", "sub_agents": [{{"agent_name": "InitialAgent"}}, {{"flow": {{"flow_name": "LoopName", "type": "Loop", "sub_agents": [{{"agent_name": "ReviewAgent"}}, {{"agent_name": "RefineAgent"}}]}}}}]}}

4. Parallel type (for simultaneous execution followed by synthesis):
{{"flow_name": "PipelineName", "type": "Sequential", "sub_agents": [{{"flow": {{"flow_name": "ParallelName", "type": "Parallel", "sub_agents": [{{"agent_name": "Agent1"}}, {{"agent_name": "Agent2"}}, {{"agent_name": "Agent3"}}]}}}}, {{"agent_name": "SynthesisAgent"}}]}}

Guidelines:
- LLM type: Use for Q&A/support systems that fetch data from multiple sources
- Sequential type: Use for step-by-step processes
- Loop type: Use for tasks requiring initial creation followed by iterative review/refinement
- Parallel type: Use for complex workflows with simultaneous work followed by integration
- Agent names must end with "Agent"
- flow_name should be meaningful English names
- Use exact agent names specified in the instruction

Examples:

Input: "인사 부문 문의는 ${{직원정보Agent}}에서, 급여 관련 요청은 ${{급여정보Agent}}에서 데이터를 불러와서 응답을 생성하는 시스템을 만들어줘"
Output: {{"flow_name": "HRQNAagent", "type": "LLM", "tools": [{{"agent_name": "직원정보Agent"}}, {{"agent_name": "급여정보Agent"}}]}}

Input: "${{고객문의Agent}}가 고객의 문의를 받고 ${{답변작성Agent}}가 답변을 작성한 후 ${{답변검토Agent}}가 검토하는 플로우로 구성해줘"
Output: {{"flow_name": "CustomerSupportPipeline", "type": "Sequential", "sub_agents": [{{"agent_name": "고객문의Agent"}}, {{"agent_name": "답변작성Agent"}}, {{"agent_name": "답변검토Agent"}}]}}

Input: "${{초기작성Agent}}를 이용해서 짧은 보고서를 작성하는 에이전트를 만들어줘. 초고를 작성하고 ${{비평Agent}}와 ${{수정Agent}}로 몇번의 교정을 거쳐서 완성도를 높혀줄 수 있었으면 좋겠어."
Output: {{"flow_name": "IterativeReportPipeline", "type": "Sequential", "sub_agents": [{{"agent_name": "초기작성Agent"}}, {{"flow": {{"flow_name": "RefinementLoop", "type": "Loop", "sub_agents": [{{"agent_name": "비평Agent"}}, {{"agent_name": "수정Agent"}}]}}}}]}}

Input: "시장조사와 제품 개발, 마케팅 전략을 한 흐름에서 수행하는 에이전트를 설계해줘. ${{MarketResearchAgent}}가 조사한 뒤 ${{ProductDevelopmentAgent}}가 아이디어를 내고 ${{MarketingStrategyAgent}}가 전략을 짠 다음 ${{SynthesisAgent}}가 이를 종합하도록 해."
Output: {{"flow_name": "BusinessDevelopmentPipeline", "type": "Sequential", "sub_agents": [{{"flow": {{"flow_name": "ResearchAndDevelopmentParallel", "type": "Parallel", "sub_agents": [{{"agent_name": "MarketResearchAgent"}}, {{"agent_name": "ProductDevelopmentAgent"}}, {{"agent_name": "MarketingStrategyAgent"}}]}}}}, {{"agent_name": "SynthesisAgent"}}]}}

Instruction: {instruction}"""

# LLM-as-Judge Prompt (No Ground Truth needed)
LLM_JUDGE_PROMPT = """
Evaluate if the generated JSON workflow correctly represents the given Korean instruction.

Original Instruction: {instruction}
Generated JSON: {generated_json}

Does the generated JSON correctly capture the intent and requirements of the instruction?
Consider:
- Workflow type appropriateness (LLM/Sequential/Loop/Parallel)
- Agent relevance and naming convention compliance
- Overall structure and completeness
- Logical flow matching the instruction
- Appropriateness of flow_name
- JSON structure correctness (tools vs sub_agents)
- Proper handling of nested flows for Loop/Parallel types
- Exact usage of agent names specified in the instruction

Return ONLY: True or False"""



LLM_JUDGE_WITH_REASEON_PROMPT = """
Evaluate whether the generated JSON workflow correctly represents the given Korean instruction.

Original Instruction:
{instruction}

Generated JSON:
{generated_json}

Judging criteria (apply all):
- Workflow type appropriateness (LLM/Sequential/Loop/Parallel). An LLM + tools design is acceptable if it can fetch data from the specified agents without explicit routing, unless the instruction clearly demands sequencing, loops, or parallelism.
- Agent relevance and naming compliance. When comparing agent names, treat agent names with or without braces as equivalent.
- Overall structure and completeness; logical flow matching the instruction; appropriateness of flow_name.
- JSON structure correctness (tools vs sub_agents); proper handling of nested flows for Loop/Parallel types.

Normalization rules:
- Normalize agent names by trimming whitespace and removing any surrounding braces '{{' and '}}' in both the instruction and the JSON before comparison.

Output format:
- Return ONLY a single-line minified JSON object with keys "passed" and "reason".
- "passed" must be true or false.
- "reason" must be a concise explanation (<= 120 chars) of the most important factor for the decision.

Example (format only; do NOT include as explanation):
{{"passed": true, "reason": "Agent names and structure match; LLM+tools is sufficient"}}
"""


# 3-Stage Workflow Generation Prompts

# Stage 1: Workflow Type Prediction Prompt
WORKFLOW_TYPE_PREDICTION_PROMPT = """
Analyze the Korean instruction and predict which workflow type is most appropriate.

Workflow Types:
1. LLM - For Q&A, assistance, help systems that fetch data from multiple agents
2. Sequential - For step-by-step processes where one agent follows another
3. Loop - For iterative refinement processes (initial creation + review/refinement cycles)
4. Parallel - For simultaneous execution followed by synthesis/integration

Analysis Guidelines:
- Look for keywords indicating simultaneity: "동시에", "함께", "병렬로", "한번에"
- Look for keywords indicating sequence: "다음에", "그 후", "순서대로", "단계별로"
- Look for keywords indicating iteration: "반복", "수정", "개선", "교정", "피드백"
- Look for keywords indicating data fetching: "조회", "검색", "불러오기", "데이터"

Examples:
Input: "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. {{텍스트작성Agent}}, {{이미지생성Agent}}, {{동영상편집Agent}}가 동시에 작업하고 {{콘텐츠통합Agent}}가 최종 결과물을 만들도록 해"
Output: {{"type": "Parallel", "reason": "Multiple agents work simultaneously ('동시에') with final integration"}}

Input: "{{고객문의Agent}}가 고객의 문의를 받고 {{답변작성Agent}}가 답변을 작성한 후 {{답변검토Agent}}가 검토하는 플로우로 구성해줘"
Output: {{"type": "Sequential", "reason": "Clear step-by-step process with '받고...후' indicating sequence"}}

Input: "인사 부문 문의는 {{직원정보Agent}}에서, 급여 관련 요청은 {{급여정보Agent}}에서 데이터를 불러와서 응답을 생성하는 시스템을 만들어줘"
Output: {{"type": "LLM", "reason": "Data fetching system ('데이터를 불러와서') for Q&A responses"}}

Input: "{{초기작성Agent}}를 이용해서 짧은 보고서를 작성하는 에이전트를 만들어줘. 초고를 작성하고 {{비평Agent}}와 {{수정Agent}}로 몇번의 교정을 거쳐서 완성도를 높혀줄 수 있었으면 좋겠어."
Output: {{"type": "Loop", "reason": "Iterative refinement process with initial creation and multiple rounds of review/editing"}}

Return ONLY a JSON object with 'type' and 'reason' (no markdown, no code blocks, no explanations):

Instruction: {instruction}"""

# Stage 2: Workflow Generation Prompt (based on predicted type)
WORKFLOW_GENERATION_PROMPT = """
Generate a workflow JSON based on the Korean instruction and the predicted workflow type. Return ONLY the JSON object, no explanations.

Predicted Workflow Type: {predicted_type}
Instruction: {instruction}

Generate the appropriate JSON structure:

If predicted_type is "LLM":
{{"flow_name": "SystemName", "type": "LLM", "tools": [{{"agent_name": "ToolAgent1"}}, {{"agent_name": "ToolAgent2"}}]}}

If predicted_type is "Sequential":
{{"flow_name": "PipelineName", "type": "Sequential", "sub_agents": [{{"agent_name": "Agent1"}}, {{"agent_name": "Agent2"}}]}}

If predicted_type is "Loop":
{{"flow_name": "PipelineName", "type": "Sequential", "sub_agents": [{{"agent_name": "InitialAgent"}}, {{"flow": {{"flow_name": "LoopName", "type": "Loop", "sub_agents": [{{"agent_name": "ReviewAgent"}}, {{"agent_name": "RefineAgent"}}]}}}}]}}

If predicted_type is "Parallel":
{{"flow_name": "PipelineName", "type": "Sequential", "sub_agents": [{{"flow": {{"flow_name": "ParallelName", "type": "Parallel", "sub_agents": [{{"agent_name": "Agent1"}}, {{"agent_name": "Agent2"}}, {{"agent_name": "Agent3"}}]}}}}, {{"agent_name": "SynthesisAgent"}}]}}

Examples:

LLM Type Example:
Instruction: "인사 부문 문의는 {{직원정보Agent}}에서, 급여 관련 요청은 {{급여정보Agent}}에서 데이터를 불러와서 응답을 생성하는 시스템을 만들어줘"
Output: {{"flow_name": "HRQNASystem", "type": "LLM", "tools": [{{"agent_name": "직원정보Agent"}}, {{"agent_name": "급여정보Agent"}}]}}

Sequential Type Example:
Instruction: "{{고객문의Agent}}가 고객의 문의를 받고 {{답변작성Agent}}가 답변을 작성한 후 {{답변검토Agent}}가 검토하는 플로우로 구성해줘"
Output: {{"flow_name": "CustomerSupportPipeline", "type": "Sequential", "sub_agents": [{{"agent_name": "고객문의Agent"}}, {{"agent_name": "답변작성Agent"}}, {{"agent_name": "답변검토Agent"}}]}}

Loop Type Example:
Instruction: "{{초기작성Agent}}를 이용해서 짧은 보고서를 작성하는 에이전트를 만들어줘. 초고를 작성하고 {{비평Agent}}와 {{수정Agent}}로 몇번의 교정을 거쳐서 완성도를 높혀줄 수 있었으면 좋겠어."
Output: {{"flow_name": "IterativeReportPipeline", "type": "Sequential", "sub_agents": [{{"agent_name": "초기작성Agent"}}, {{"flow": {{"flow_name": "RefinementLoop", "type": "Loop", "sub_agents": [{{"agent_name": "비평Agent"}}, {{"agent_name": "수정Agent"}}]}}}}]}}

Parallel Type Example:
Instruction: "콘텐츠 제작을 효율적으로 하는 시스템을 구축해줘. {{텍스트작성Agent}}, {{이미지생성Agent}}, {{동영상편집Agent}}가 동시에 작업하고 {{콘텐츠통합Agent}}가 최종 결과물을 만들도록 해"
Output: {{"flow_name": "ContentCreationPipeline", "type": "Sequential", "sub_agents": [{{"flow": {{"flow_name": "ConcurrentContentCreation", "type": "Parallel", "sub_agents": [{{"agent_name": "텍스트작성Agent"}}, {{"agent_name": "이미지생성Agent"}}, {{"agent_name": "동영상편집Agent"}}]}}}}, {{"agent_name": "콘텐츠통합Agent"}}]}}

Guidelines:
- Use exact agent names specified in the instruction
- Agent names must end with "Agent"
- flow_name should be meaningful English names
- For Parallel type, include a synthesis/integration agent at the end"""

# Stage 3: Workflow Validation Prompt (통합된 버전)
WORKFLOW_VALIDATION_PROMPT = """
Evaluate whether the generated JSON workflow correctly represents the given Korean instruction and matches the predicted workflow type.

Original Instruction:
{instruction}

Predicted Type: {predicted_type}

Generated JSON:
{generated_json}

Judging criteria (apply all):
- Workflow type matching:
  * For Parallel prediction: CORRECT if there is ANY nested "type": "Parallel" structure within the workflow, regardless of top-level type
  * For Loop prediction: CORRECT if there is ANY nested "type": "Loop" structure within the workflow  
  * For LLM prediction: CORRECT if top-level "type": "LLM" with "tools" array
  * For Sequential prediction: CORRECT if simple "type": "Sequential" with NO nested Parallel/Loop structures
  * IMPORTANT: Parallel workflows are wrapped in Sequential containers - this is CORRECT design, not an error
- Agent names: All agents from instruction must be present in the JSON
- Structure correctness: Proper use of tools vs sub_agents, proper nested flows
- Intent matching: Does the workflow capture the instruction's execution pattern (parallel, sequential, etc.)?

Normalization rules:
- Normalize agent names by trimming whitespace and removing any surrounding braces '{{' and '}}' in both the instruction and the JSON before comparison.

CRITICAL VALIDATION RULES:
1. For Parallel prediction: Search the ENTIRE JSON for ANY occurrence of "type": "Parallel" - if found, PASS
2. For Sequential prediction: Only PASS if there are NO nested "type": "Parallel" or "type": "Loop" anywhere
3. For LLM prediction: Only check top-level "type": "LLM"
4. IGNORE top-level "type": "Sequential" when validating Parallel predictions - it's a wrapper

EXAMPLE: This JSON is CORRECT for Parallel prediction:
{{"type": "Sequential", "sub_agents": [{{"flow": {{"type": "Parallel", "sub_agents": [...]}}}}]}}
The nested "type": "Parallel" makes it a valid Parallel workflow.

Output format:
- Return ONLY a single-line minified JSON object with keys "passed" and "reason".
- "passed" must be true or false.
- "reason" must be a concise explanation (<= 120 chars) of the most important factor for the decision.

Example (format only; do NOT include as explanation):
{{"passed": true, "reason": "Contains nested Parallel structure matching prediction, all agents present"}}"""


# JSON to Instruction Conversion Prompt
JSON_TO_INSTRUCTION_PROMPT = """
Convert the given workflow JSON back to a natural Korean instruction that describes the workflow.

You must analyze the JSON structure and generate a clear, natural Korean instruction that would produce this workflow.

Guidelines:
- Use natural Korean language
- Describe the workflow flow logically
- Mention all agent names exactly as they appear in the JSON
- Explain the execution pattern (sequential, parallel, loop, etc.)
- Make it sound like a natural request someone would make

Workflow Types and Descriptions:
1. LLM type: "데이터를 불러와서 응답을 생성하는 시스템" or "문의에 답변하는 시스템"
2. Sequential type: "단계별로 순차적으로 실행되는 플로우"
3. Loop type: "반복적으로 검토하고 수정하는 과정"
4. Parallel type: "동시에 여러 작업을 수행한 후 통합하는 과정"

Examples:

JSON: {{"flow_name": "HRQNAgent", "type": "LLM", "tools": [{{"agent_name": "직원정보Agent"}}, {{"agent_name": "급여정보Agent"}}]}}
Instruction: "인사 부문 문의는 직원정보Agent에서, 급여 관련 요청은 급여정보Agent에서 데이터를 불러와서 응답을 생성하는 시스템을 만들어줘"

JSON: {{"flow_name": "CustomerSupportPipeline", "type": "Sequential", "sub_agents": [{{"agent_name": "고객문의Agent"}}, {{"agent_name": "답변작성Agent"}}, {{"agent_name": "답변검토Agent"}}]}}
Instruction: "고객문의Agent가 고객의 문의를 받고 답변작성Agent가 답변을 작성한 후 답변검토Agent가 검토하는 플로우로 구성해줘"

JSON: {{"flow_name": "IterativeReportPipeline", "type": "Sequential", "sub_agents": [{{"agent_name": "초기작성Agent"}}, {{"flow": {{"flow_name": "RefinementLoop", "type": "Loop", "sub_agents": [{{"agent_name": "비평Agent"}}, {{"agent_name": "수정Agent"}}]}}}}]}}
Instruction: "초기작성Agent를 이용해서 짧은 보고서를 작성하는 에이전트를 만들어줘. 초고를 작성하고 비평Agent와 수정Agent로 몇번의 교정을 거쳐서 완성도를 높혀줄 수 있었으면 좋겠어"

JSON: {{"flow_name": "BusinessDevelopmentPipeline", "type": "Sequential", "sub_agents": [{{"flow": {{"flow_name": "ResearchAndDevelopmentParallel", "type": "Parallel", "sub_agents": [{{"agent_name": "MarketResearchAgent"}}, {{"agent_name": "ProductDevelopmentAgent"}}, {{"agent_name": "MarketingStrategyAgent"}}]}}}}, {{"agent_name": "SynthesisAgent"}}]}}
Instruction: "시장조사와 제품 개발, 마케팅 전략을 한 흐름에서 수행하는 에이전트를 설계해줘. MarketResearchAgent가 조사한 뒤 ProductDevelopmentAgent가 아이디어를 내고 MarketingStrategyAgent가 전략을 짠 다음 SynthesisAgent가 이를 종합하도록 해"

Return ONLY the Korean instruction, no explanations or additional text.

JSON: {workflow_json}"""