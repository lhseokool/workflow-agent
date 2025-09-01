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
- Use exact agent names specified in {{AgentName}} format in the instruction

Examples:

Input: "인사 부문 문의는 {{직원정보Agent}}에서, 급여 관련 요청은 {{급여정보Agent}}에서 데이터를 불러와서 응답을 생성하는 시스템을 만들어줘"
Output: {{"flow_name": "HRQNAagent", "type": "LLM", "tools": [{{"agent_name": "직원정보Agent"}}, {{"agent_name": "급여정보Agent"}}]}}

Input: "{{고객문의Agent}}가 고객의 문의를 받고 {{답변작성Agent}}가 답변을 작성한 후 {{답변검토Agent}}가 검토하는 플로우로 구성해줘"
Output: {{"flow_name": "CustomerSupportPipeline", "type": "Sequential", "sub_agents": [{{"agent_name": "고객문의Agent"}}, {{"agent_name": "답변작성Agent"}}, {{"agent_name": "답변검토Agent"}}]}}

Input: "{{초기작성Agent}}를 이용해서 짧은 보고서를 작성하는 에이전트를 만들어줘. 초고를 작성하고 {{비평Agent}}와 {{수정Agent}}로 몇번의 교정을 거쳐서 완성도를 높혀줄 수 있었으면 좋겠어."
Output: {{"flow_name": "IterativeReportPipeline", "type": "Sequential", "sub_agents": [{{"agent_name": "초기작성Agent"}}, {{"flow": {{"flow_name": "RefinementLoop", "type": "Loop", "sub_agents": [{{"agent_name": "비평Agent"}}, {{"agent_name": "수정Agent"}}]}}}}]}}

Input: "시장조사와 제품 개발, 마케팅 전략을 한 흐름에서 수행하는 에이전트를 설계해줘. {{MarketResearchAgent}}가 조사한 뒤 {{ProductDevelopmentAgent}}가 아이디어를 내고 {{MarketingStrategyAgent}}가 전략을 짠 다음 {{SynthesisAgent}}가 이를 종합하도록 해."
Output: {{"flow_name": "BusinessDevelopmentPipeline", "type": "Sequential", "sub_agents": [{{"flow": {{"flow_name": "ResearchAndDevelopmentParallel", "type": "Parallel", "sub_agents": [{{"agent_name": "MarketResearchAgent"}}, {{"agent_name": "ProductDevelopmentAgent"}}, {{"agent_name": "MarketingStrategyAgent"}}]}}}}, {{"agent_name": "SynthesisAgent"}}]}}

Instruction: {instruction}"""

# LLM-as-Judge Prompt (No Ground Truth needed)
INSTRUCTION_JUDGE_PROMPT = """
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
- Exact usage of agent names specified in {{AgentName}} format

Return ONLY: True or False"""