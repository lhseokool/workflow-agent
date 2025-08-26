"""
Workflow Agent Models
Simple and clean workflow agent implementation
"""

import json
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompts import JSON_PROMPT, INSTRUCTION_JUDGE_PROMPT, JSON_EVALUATION_PROMPT
from utils import parse_llm_evaluation

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")


class WorkflowAgent:
    """Clean and simple workflow agent for instruction to JSON conversion"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the workflow agent with OpenAI model"""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=512
        )
        self.parser = StrOutputParser()
        
        # Generation chain
        self.json_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.llm | self.parser
        
        # LLM-as-Judge chain (instruction vs generated result)
        self.judge_chain = ChatPromptTemplate.from_template(INSTRUCTION_JUDGE_PROMPT) | self.llm | self.parser
        
        # LLM evaluation chain (with ground truth)
        self.eval_chain = ChatPromptTemplate.from_template(JSON_EVALUATION_PROMPT) | self.llm | self.parser
    
    def generate_json(self, instruction: str) -> Dict[str, Any]:
        """Generate JSON structure from instruction"""
        try:
            result = self.json_chain.invoke({"instruction": instruction})
            return json.loads(result.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw output: {result}")
            return {"type": "LLM", "sub_agents": []}
        except Exception as e:
            print(f"JSON generation error: {e}")
            return {"type": "LLM", "sub_agents": []}
    
    def generate_workflow(self, instruction: str) -> Dict[str, Any]:
        """Generate complete workflow from instruction"""
        json_result = self.generate_json(instruction)
        return {
            "instruction": instruction,
            "label_json": json_result
        }
    
    def judge_instruction_result(self, instruction: str, generated_json: str) -> bool:
        """
        LLM-as-Judge: Compare instruction with generated result
        No ground truth needed - evaluates if generation matches instruction intent
        """
        try:
            result = self.judge_chain.invoke({
                "instruction": instruction,
                "generated_json": generated_json
            })
            return parse_llm_evaluation(result)
        except Exception as e:
            print(f"LLM judge evaluation error: {e}")
            return False
    
    def evaluate_json_with_llm(self, instruction: str, expected_json: str, generated_json: str) -> bool:
        """
        LLM evaluation with ground truth
        Compares generated result with expected ground truth
        """
        try:
            result = self.eval_chain.invoke({
                "instruction": instruction,
                "expected_json": expected_json,
                "generated_json": generated_json
            })
            return parse_llm_evaluation(result)
        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return False