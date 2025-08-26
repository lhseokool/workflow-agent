"""
Workflow Agent Prompts
Simple prompt templates for JSON generation and evaluation
"""

# JSON Generation Prompt
JSON_PROMPT = """
Convert instruction to workflow JSON. Return ONLY the JSON object, nothing else.
Do not include any explanations, prefixes, or additional text.

You must create one of these workflow types:

1. LLM type (for Q&A, assistance, help tasks):
{{"type": "LLM", "sub_agents": [{{"name": "MainAgent"}}], "tools": [{{"name": "ToolAgent1"}}, {{"name": "ToolAgent2"}}]}}

2. Sequential type (for step-by-step processes):
{{"type": "Sequential", "sub_agents": [{{"name": "Agent1"}}, {{"name": "Agent2"}}]}}

3. Parallel type (for simultaneous execution):
{{"type": "Parallel", "sub_agents": [{{"name": "Agent1"}}, {{"name": "Agent2"}}]}}

4. Loop type (for repetitive processes):
{{"type": "Loop", "sub_agents": [{{"name": "Agent1"}}, {{"name": "Agent2"}}], "max_iterations": 3}}

Guidelines:
- Use LLM type for Q&A, support, help scenarios
- Use Sequential for step-by-step processes
- Use Parallel for simultaneous work
- Use Loop for iterative/repetitive tasks
- Agent names should end with "Agent"

Instruction: {instruction}"""

# LLM-as-Judge Prompt (No Ground Truth needed)
INSTRUCTION_JUDGE_PROMPT = """
Evaluate if the generated JSON workflow correctly represents the given instruction.

Original Instruction: {instruction}
Generated JSON: {generated_json}

Does the generated JSON correctly capture the intent and requirements of the instruction?
Consider:
- Workflow type appropriateness (LLM/Sequential/Parallel/Loop)
- Agent relevance and naming
- Overall structure and completeness
- Logical flow matching the instruction

Return ONLY: True or False"""

# LLM-based JSON Evaluation Prompt (With Ground Truth)
JSON_EVALUATION_PROMPT = """
Compare the generated JSON with the expected result for the given instruction.

Original Instruction: {instruction}
Expected JSON: {expected_json}
Generated JSON: {generated_json}

Does the generated JSON correctly represent the instruction intent and match the expected result?
Consider: workflow type, agent structure, completeness, and JSON validity.

Return ONLY: True or False"""