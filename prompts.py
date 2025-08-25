"""
Workflow Agent Prompts
Template prompts for ADK and JSON generation and evaluation
"""

# ADK Generation Prompt - REMOVED (No longer used)

# JSON Generation Prompt
JSON_PROMPT = """
Convert instruction to workflow JSON. Return ONLY the JSON object, nothing else.
Do not include any explanations, prefixes, or additional text.

You must create one of these 4 types of workflows:

1. LLM type (for Q&A or assistance tasks with main agent + tool agents):
{{"flow_name": "FlowName", "type": "LLM", "sub_agents": [{{"agent_name": "MainAgent"}}], "tools": [{{"agent_name": "ToolAgent1"}}, {{"agent_name": "ToolAgent2"}}]}}

2. Sequential type (for step-by-step execution):
{{"flow_name": "FlowName", "type": "Sequential", "sub_agents": [{{"agent_name": "Agent1"}}, {{"agent_name": "Agent2"}}]}}

3. Sequential with Loop (for iterative refinement):
{{"flow_name": "FlowName", "type": "Sequential", "sub_agents": [{{"agent_name": "InitialAgent"}}, {{"flow": {{"flow_name": "LoopName", "type": "Loop", "sub_agents": [{{"agent_name": "CriticAgent"}}, {{"agent_name": "RefineAgent"}}]}}}}]}}

4. Sequential with Parallel (for complex business processes):
{{"flow_name": "FlowName", "type": "Sequential", "sub_agents": [{{"flow": {{"flow_name": "ParallelName", "type": "Parallel", "sub_agents": [{{"agent_name": "Agent1"}}, {{"agent_name": "Agent2"}}, {{"agent_name": "Agent3"}}]}}}}, {{"agent_name": "SynthesisAgent"}}]}}

Guidelines:
- Use LLM type for Q&A, support, help desk scenarios
- Use Sequential for step-by-step processes
- Use Loop within Sequential for iterative improvement
- Use Parallel within Sequential for simultaneous work followed by synthesis
- Agent names should end with "Agent" 
- Flow names should be descriptive and end with appropriate suffixes

Instruction: {instruction}"""

# ADK Evaluation Prompt - REMOVED (No longer used)

# LLM-based JSON Evaluation Prompt
JSON_EVALUATION_PROMPT = """
Compare the generated JSON with the expected result for the given instruction.

Original Instruction: {instruction}
Expected JSON: {expected_json}
Generated JSON: {generated_json}

Does the generated JSON correctly represent the instruction intent and match the expected result?
Consider: workflow type, agent structure, completeness, and JSON validity.

Return ONLY: True or False"""
