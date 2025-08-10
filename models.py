"""
Workflow Agent Models
Core workflow agent implementation
"""

import json
from typing import Dict, Any
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompts import ADK_PROMPT, JSON_PROMPT, ADK_EVALUATION_PROMPT, JSON_EVALUATION_PROMPT
from utils import parse_llm_evaluation


class WorkflowAgent:
    """Clean and simple workflow agent for instruction to ADK/JSON conversion"""
    
    def __init__(self, model_name: str = "qwen3:4b"):
        """Initialize the workflow agent with LLM chains"""
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            num_predict=128
        )
        self.parser = StrOutputParser()
        
        # Generation chains
        self.adk_chain = ChatPromptTemplate.from_template(ADK_PROMPT) | self.llm | self.parser
        self.json_chain = ChatPromptTemplate.from_template(JSON_PROMPT) | self.llm | self.parser
        
        # Evaluation chains
        self.adk_eval_chain = ChatPromptTemplate.from_template(ADK_EVALUATION_PROMPT) | self.llm | self.parser
        self.json_eval_chain = ChatPromptTemplate.from_template(JSON_EVALUATION_PROMPT) | self.llm | self.parser
    
    def generate_adk(self, instruction: str) -> str:
        """Generate ADK format from instruction"""
        try:
            result = self.adk_chain.invoke({"instruction": instruction})
            return result.strip()
        except Exception as e:
            print(f"ADK generation error: {e}")
            return ""
    
    def generate_json(self, instruction: str) -> Dict[str, Any]:
        """Generate JSON structure from instruction"""
        try:
            result = self.json_chain.invoke({"instruction": instruction})
            return json.loads(result.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw output: {result}")
            return {"type": "Sequential", "sub_agents": []}
        except Exception as e:
            print(f"JSON generation error: {e}")
            return {"type": "Sequential", "sub_agents": []}
    
    def generate_workflow(self, instruction: str) -> Dict[str, Any]:
        """Generate complete workflow (ADK + JSON) from instruction"""
        return {
            "instruction": instruction,
            "label_adk": self.generate_adk(instruction),
            "label_json": self.generate_json(instruction)
        }
    
    def evaluate_adk_with_llm(self, instruction: str, expected_adk: str, generated_adk: str) -> bool:
        """Evaluate ADK using LLM"""
        try:
            result = self.adk_eval_chain.invoke({
                "instruction": instruction,
                "expected_adk": expected_adk,
                "generated_adk": generated_adk
            })
            return parse_llm_evaluation(result)
        except Exception as e:
            print(f"ADK LLM evaluation error: {e}")
            return False
    
    def evaluate_json_with_llm(self, instruction: str, expected_json: str, generated_json: str) -> bool:
        """Evaluate JSON using LLM"""
        try:
            result = self.json_eval_chain.invoke({
                "instruction": instruction,
                "expected_json": expected_json,
                "generated_json": generated_json
            })
            return parse_llm_evaluation(result)
        except Exception as e:
            print(f"JSON LLM evaluation error: {e}")
            return False
