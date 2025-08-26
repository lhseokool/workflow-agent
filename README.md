# Workflow Agent

Clean and efficient system for converting natural language instructions to JSON workflows using OpenAI models with dual evaluation methods.

## ✨ Features

- **OpenAI Integration**: Uses GPT-4o and GPT-4o-mini models
- **JSON Generation**: Convert instructions to structured workflow JSON
- **Triple Evaluation**: Exact match + LLM with GT + LLM-as-Judge (no GT)
- **LLM-as-Judge**: Evaluate generation quality without ground truth
- **Excel Export**: Comprehensive results with all evaluation metrics
- **Clean Architecture**: Simple, modular, well-documented

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Method 1: Create .env file (Recommended)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Method 2: Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Method 3: Interactive setup
python setup_api_key.py

# Run full evaluation
python workflow_agent.py

# Quick test
python tests/quick_test.py

# Demo without API key
python demo_without_api.py
```

## 🎯 Supported Workflow Types

1. **LLM**: Q&A and assistance tasks with tools
2. **Sequential**: Step-by-step execution
3. **Parallel**: Simultaneous execution
4. **Loop**: Repetitive execution with iterations

## 📊 Usage Example

```python
from models import WorkflowAgent

# Use GPT-4o-mini (default) or GPT-4o
agent = WorkflowAgent(model_name="gpt-4o-mini")
result = agent.generate_workflow(
    "고객 문의에 답변하는 시스템을 만들어주세요"
)

print(result["label_json"])   # {"type": "LLM", "sub_agents": [...], "tools": [...]}

# LLM-as-Judge evaluation (no GT needed)
is_valid = agent.judge_instruction_result(
    instruction,
    json.dumps(result["label_json"])
)
print(f"Valid workflow: {is_valid}")
```

## 📁 Project Structure

```
workflow_agent/
├── models.py           # WorkflowAgent class (simplified)
├── workflow_agent.py   # Main execution script
├── utils.py            # Utility functions (simplified)
├── prompts.py          # Prompt templates (clean)
├── test_data.json      # Test dataset
├── requirements.txt    # Dependencies (minimal)
├── tests/              # Test files directory
│   ├── quick_test.py   # Quick testing script
│   └── ...             # Other test files
└── README.md          # This file
```

## 🎮 How to Run

### Full Evaluation (Recommended)
```bash
python workflow_agent.py
```
- Processes all test cases from `test_data.json`
- Runs **three evaluation methods**:
  1. **Exact Match** (with ground truth)
  2. **LLM Evaluation** (with ground truth)  
  3. **LLM-as-Judge** (no ground truth needed)
- Saves comprehensive results to Excel

### Quick Test
```bash
python tests/quick_test.py
```
- Tests single instruction
- Shows JSON generation result
- Demonstrates LLM-as-Judge evaluation
- Perfect for quick verification

## 📊 Excel Output Format

| Test_ID | Instruction | Expected_JSON | Generated_JSON | JSON_Exact_Match | JSON_LLM_with_GT | LLM_Judge_no_GT | Total_Time | Judge_Eval_Time |
|---------|-------------|---------------|----------------|------------------|------------------|-----------------|------------|-----------------|
| 1 | 고객 문의... | {...} | {...} | O | O | O | 1.2s | 0.8s |

## 🔍 Evaluation Methods

### 1. Exact Match (with GT)
- **Purpose**: Strict structural comparison
- **Requires**: Ground truth test data
- **Use case**: Precise validation against known correct answers

### 2. LLM Evaluation (with GT) 
- **Purpose**: Semantic comparison with expected results
- **Requires**: Ground truth test data
- **Use case**: Flexible validation that understands meaning

### 3. LLM-as-Judge (no GT needed) ⭐
- **Purpose**: Evaluate if generation matches instruction intent
- **Requires**: Only instruction and generated result
- **Use case**: Real-world deployment where no ground truth exists

## 🛠️ Technical Details

- **Language Models**: OpenAI GPT-4o and GPT-4o-mini
- **Framework**: LangChain for LLM integration
- **Validation**: Simple JSON parsing + LLM semantic evaluation
- **Output**: Excel files with comprehensive metrics
- **Architecture**: Clean, modular, maintainable

## 📈 Performance Metrics

The system provides three evaluation perspectives:
- **Exact Match**: Structural accuracy percentage
- **LLM+GT**: Semantic accuracy with ground truth
- **LLM Judge**: Real-world applicability without ground truth

## 🎯 Key Benefits

1. **Simplified Codebase**: Removed complex Pydantic models
2. **LLM-as-Judge**: No ground truth needed for evaluation
3. **Production Ready**: Evaluate real instructions without test data
4. **Comprehensive Metrics**: Multiple evaluation angles
5. **Easy Maintenance**: Clean, readable code structure

## 💡 Real-World Usage

```python
# For production use - no ground truth needed
agent = WorkflowAgent()

# Generate workflow from user instruction
user_instruction = "사용자 입력을 처리하는 시스템 만들어줘"
result = agent.generate_workflow(user_instruction)

# Validate without ground truth
is_good = agent.judge_instruction_result(
    user_instruction, 
    json.dumps(result["label_json"])
)

if is_good:
    print("✅ Workflow generation successful!")
    # Use the generated workflow
else:
    print("⚠️ May need regeneration or refinement")
```

## 🔧 Troubleshooting

### API Key Issues
```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Method 1: Create .env file (easiest)
python create_env.py

# Method 2: Manual .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Method 3: Environment variable
export OPENAI_API_KEY="your-key-here"

# Test without API key
python demo_without_api.py
```

### Common Errors
- **"api_key client option must be set"**: Run `python setup_api_key.py`
- **"No test data found"**: Ensure `test_data.json` exists in root directory
- **Import errors**: Run `pip install -r requirements.txt`

### Getting API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create account and generate API key
3. Set environment variable or use setup script

---

Built with ❤️ for clean, efficient workflow generation and evaluation.