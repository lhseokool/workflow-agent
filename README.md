# Workflow Agent

Clean and simple system for converting natural language instructions to ADK/JSON workflows with combined evaluation methods.

## ✨ Features

- **ADK Generation**: Convert instructions to `Sequential([Agent1, Agent2])` format
- **JSON Generation**: Convert instructions to `{"type": "Sequential", "sub_agents": [...]}`
- **Dual Evaluation**: Both exact match and LLM-based evaluation simultaneously
- **Excel Export**: Automatic results saving with comprehensive metrics
- **Clean Architecture**: Modular design with separated concerns

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama & download model
ollama pull qwen3:4b

# Run full evaluation
python workflow_agent.py

# Run quick test
python quick_test.py
```

## 🎯 Supported Patterns

1. **Sequential**: Step-by-step execution (`한 후`, `다음에`, `after`, `then`)
2. **Parallel**: Simultaneous execution (`동시에`, `함께`, `simultaneously`, `together`)  
3. **Loop**: Repetitive execution (`반복`, `최대 N회`, `repeatedly`, `up to N times`)

## 📊 Usage Example

```python
from models import WorkflowAgent

agent = WorkflowAgent()
result = agent.generate_workflow(
    "(CodeWriterAgent)로 코드를 작성한 후, (CodeReviewerAgent)로 점검하세요."
)

print(result["label_adk"])    # Sequential([CodeWriterAgent, CodeReviewerAgent])
print(result["label_json"])   # {"type": "Sequential", "sub_agents": [...]}
```

## 📁 Project Structure

```
workflow_agent/
├── models.py           # WorkflowAgent class
├── workflow_agent.py   # Main execution script
├── utils.py            # Utility functions
├── prompts.py          # Prompt templates
├── quick_test.py       # Quick testing script
├── test_data.json      # Test dataset
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎮 How to Run

### Full Evaluation (Recommended)
```bash
python workflow_agent.py
```
- Automatically processes all test cases from `test_data.json`
- Runs both exact match and LLM evaluation simultaneously
- Saves comprehensive results to Excel file
- Displays summary statistics

### Quick Test
```bash
python quick_test.py
```
- Tests single predefined instruction
- Shows both ADK and JSON generation results
- Displays both evaluation methods' results
- Useful for quick verification

## 📊 Excel Output Format

| Test_ID | Instruction | Expected_ADK | Generated_ADK | ADK_Exact_Match | ADK_LLM_Correct | Expected_JSON | Generated_JSON | JSON_Exact_Match | JSON_LLM_Correct | Total_Time | LLM_Eval_Time |
|---------|-------------|--------------|---------------|-----------------|-----------------|---------------|----------------|------------------|------------------|------------|---------------|
| 1 | (Agent1)로... | Sequential([...]) | Sequential([...]) | O | O | {...} | {...} | X | O | 1.2s | 0.8s |

## 🔍 Evaluation Methods

### Exact Match
- **ADK**: Perfect string matching
- **JSON**: Type matching + agent set comparison (order-independent)

### LLM Evaluation
- **Semantic understanding**: Evaluates meaning and intent
- **More flexible**: Handles variations in expression
- **Context-aware**: Considers instruction requirements

## 🛠️ Technical Details

- **Language Model**: Ollama (qwen3:4b recommended)
- **Framework**: LangChain for LLM integration
- **Output**: Excel files with openpyxl
- **Code Style**: Clean, modular, well-documented

## 📈 Performance Metrics

The system provides comprehensive metrics:
- **Accuracy**: Both exact and LLM evaluation percentages
- **Timing**: Generation time vs evaluation time
- **Detailed Results**: Per-test case breakdown
- **Summary Statistics**: Overall performance overview

## 🎯 Use Cases

- **Workflow Design**: Convert natural language to structured workflows
- **Multi-Agent Systems**: Define agent interactions and execution patterns
- **Process Automation**: Specify sequential/parallel task execution
- **AI Research**: Evaluate instruction-to-structure conversion quality

---

Built with ❤️ for clean, efficient workflow generation and evaluation.