"""
Workflow Agent Prompts
Template prompts for ADK and JSON generation and evaluation
"""

# ADK Generation Prompt
ADK_PROMPT = """
Convert this instruction to ADK notation. Return ONLY the ADK string.

Patterns:
- Sequential([Agent1, Agent2]) - for step-by-step execution
- Parallel([Agent1, Agent2]) - for simultaneous execution  
- Loop([Agent1, Agent2], max_iterations=N, initial_agent=InitialAgent) - for repetitive execution

Keywords:
- Sequential: "한 후", "다음에", "그 다음", "after", "then", "next"
- Parallel: "동시에", "함께", "병렬로", "simultaneously", "together", "at the same time"
- Loop: "반복", "최대", "회", "repeatedly", "up to", "times"

Examples:
Input: "(CodeWriterAgent)로 코드를 작성한 후, (CodeReviewerAgent)로 점검하세요."
Output: Sequential([CodeWriterAgent, CodeReviewerAgent])

Input: "(AgentA)와 (AgentB)를 동시에 사용하세요."
Output: Parallel([AgentA, AgentB])

Input: "(InitialAgent)로 시작하고, (AgentA)와 (AgentB)를 최대 3회 반복하세요."
Output: Loop([AgentA, AgentB], max_iterations=3, initial_agent=InitialAgent)

Instruction: {instruction}
ADK:"""

# JSON Generation Prompt
JSON_PROMPT = """
Convert instruction to JSON. Return ONLY the JSON object, nothing else.
Do not include any explanations, prefixes, or additional text.

Examples:
Input: "(CodeWriterAgent)로 코드를 작성한 후, (CodeReviewerAgent)로 점검하세요."
Output: {{"type": "Sequential", "sub_agents": [{{"name": "CodeWriterAgent"}}, {{"name": "CodeReviewerAgent"}}]}}

Input: "(AgentA)와 (AgentB)를 동시에 사용하세요."
Output: {{"type": "Parallel", "sub_agents": [{{"name": "AgentA"}}, {{"name": "AgentB"}}]}}

Instruction: {instruction}"""

# LLM-based ADK Evaluation Prompt
ADK_EVALUATION_PROMPT = """
Compare the generated ADK with the expected result for the given instruction.

Original Instruction: {instruction}
Expected ADK: {expected_adk}
Generated ADK: {generated_adk}

Does the generated ADK correctly represent the instruction intent and match the expected result?
Consider: workflow pattern, agent names, execution order, and parameters.

Return ONLY: True or False"""

# LLM-based JSON Evaluation Prompt
JSON_EVALUATION_PROMPT = """
Compare the generated JSON with the expected result for the given instruction.

Original Instruction: {instruction}
Expected JSON: {expected_json}
Generated JSON: {generated_json}

Does the generated JSON correctly represent the instruction intent and match the expected result?
Consider: workflow type, agent structure, completeness, and JSON validity.

Return ONLY: True or False"""
