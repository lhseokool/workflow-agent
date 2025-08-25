"""
Workflow Agent Models - LangGraph Implementation
State Graph-based workflow agent with validation and retry logic
"""

# Import LangGraph implementation
from langgraph_models import LangGraphWorkflowAgent, WorkflowState

# For backward compatibility, alias the new implementation
WorkflowAgent = LangGraphWorkflowAgent

# Export key classes
__all__ = ["WorkflowAgent", "LangGraphWorkflowAgent", "WorkflowState"]
