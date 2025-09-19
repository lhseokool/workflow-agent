# Workflow Designer

AI-powered workflow generation from Korean natural language instructions.

## Features

- **Baseline Model**: Simple JSON generation
- **LangGraph Retry Model**: LangGraph-based model with conditional retry logic
- **3-Stage Model**: Predict → Generate → Validate pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
GPT_OSS_X_API_KEY=your_gpt_oss_api_key
GAUSS_X_API_KEY=your_gauss_api_key

```

## Usage

### Basic Testing

```bash
# Test baseline model
python test.py --model baseline --llm-name gauss2-3-37b

# Test LangGraph model  
python test.py --model langgraph --llm-name gauss2-3-37b

# Test 3-stage model
python test.py --model 3stage --llm-name gauss2-3-37b
```

### JSON Parsing Performance Test

```bash
python json_parsing_test.py --cases 10 --runs 3
```

## Workflow Types

1. **LLM**: Q&A, support systems (tool-based)
2. **Sequential**: Step-by-step sequential execution
3. **Loop**: Iterative refinement processes
4. **Parallel**: Concurrent execution with integration

## Project Structure

```
├── models/           # AI model implementations
├── data/            # Test data
├── results/         # Test results
├── test.py          # Main test script
└── json_parsing_test.py  # JSON parsing performance test
```

## License

MIT License