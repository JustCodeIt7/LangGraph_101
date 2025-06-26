# LangGraph 101: A Comprehensive Tutorial Series

Welcome to LangGraph 101, a hands-on tutorial series designed to help you learn and master [LangGraph](https://github.com/langchain-ai/langgraph), a powerful library for building stateful, multi-actor applications with Large Language Models (LLMs).

---

## 📚 About This Repository

This repository offers a structured learning path from basic concepts to advanced applications, featuring code examples, detailed tutorials, and complete applications. Whether you're new to LangGraph or looking to deepen your expertise, this resource will guide you through effective LangGraph implementation.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **[Ollama](https://ollama.ai/)** (for running local models)
- **Basic understanding of LLMs and Python**
- **OpenAI API key** (optional, for OpenAI models)

### Installation

1.  **Clone this repository**:

    ```bash
    git clone https://github.com/yourusername/LangGraph_101.git
    cd LangGraph_101
    ```

2.  **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables** (if needed):

    ```bash
    cp .env.example .env
    # Edit .env with your API keys and configuration
    ```

---

## 📖 Repository Structure

The repository is organized into several key sections to facilitate a clear learning progression:

### 01-LangGraph_Intro

Get started with the fundamentals of LangGraph:

- **[01-LangGraph_Basics.ipynb](https://www.google.com/search?q=01-LangGraph_Intro/01-LangGraph_Basics.ipynb)**: Interactive Jupyter notebook introduction to core concepts.
- **[01.1-graph_basics.py](https://www.google.com/search?q=01-LangGraph_Intro/01.1-graph_basics.py)**: A basic three-step linear graph example.
- **[02-langgraph_connecting_llms.py](https://www.google.com/search?q=01-LangGraph_Intro/02-langgraph_connecting_llms.py)**: Demonstrates connecting various LLM providers.

### 01-Graphs

Explore core LangGraph concepts and graph-based workflows:

- **[03-Graph_Basics](https://www.google.com/search?q=01-Graphs/03-Graph_Basics/)**: Fundamental graph structures and state management.
- **[04-Multiple_Inputs](https://www.google.com/search?q=01-Graphs/04-Multiple_Inputs/)**: Working with various input types within graphs.
- **[05-Conditional_Routing](https://www.google.com/search?q=01-Graphs/05-Conditional_Routing/)**: Implementing decision logic within graphs.
- **[06-Looping_Logic](https://www.google.com/search?q=01-Graphs/06-Looping_Logic/)**: Creating loops and recursive patterns.
- **[07-Basic_Chat_Graph](https://www.google.com/search?q=01-Graphs/07-Basic_Chat_Graph/)**: Building a complete chat application.

### 02-Agents

Dive into advanced agent implementations and patterns:

- **[01-Running_Agents](https://www.google.com/search?q=02-Agents/01-Running_Agents/)**: Basic agent setup and execution.
- **[02-Agent-State](https://www.google.com/search?q=02-Agents/02-Agent-State/)**: Managing complex agent state.
- **[02-Streaming](https://www.google.com/search?q=02-Agents/02-Streaming/)**: Implementing streaming responses.
- **[03-Models](https://www.google.com/search?q=02-Agents/03-Models/)**: Working with different LLM providers.
- **[04-Tools-v2](https://www.google.com/search?q=02-Agents/04-Tools-v2/)**: Advanced tool usage and integration.
- **[05-MCP](https://www.google.com/search?q=02-Agents/05-MCP/)**: Multi-agent coordination patterns.
- **[06-Context](https://www.google.com/search?q=02-Agents/06-Context/)**: Context management strategies.
- **[07-Memory](https://www.google.com/search?q=02-Agents/07-Memory/)**: Implementing agent memory.
- **[08-Human_in_the_Loop](https://www.google.com/search?q=02-Agents/08-Human_in_the_Loop/)**: Human intervention patterns.
- **[09-Advanced-Agent-Examples-v2](https://www.google.com/search?q=02-Agents/09-Advanced-Agent-Examples-v2/)**: Complex agent implementations.
- **[09-Multi_Agent](https://www.google.com/search?q=02-Agents/09-Multi_Agent/)**: Multi-agent systems.
- **[10-RAG_Agent](https://www.google.com/search?q=02-Agents/10-RAG_Agent/)**: Retrieval-Augmented Generation agents.

### 03-Apps

Explore complete real-world application examples:

- **[01-LangGraph_Stock_Agent](https://www.google.com/search?q=03-Apps/01-LangGraph_Stock_Agent/)**: Financial data analysis.
- **[01-Travel_Planning_Agent](https://www.google.com/search?q=03-Apps/01-Travel_Planning_Agent/)**: Trip planning assistant.
- **[02-Finance_Investment_Agent](https://www.google.com/search?q=03-Apps/02-Finance_Investment_Agent/)**: Investment advisory system.
- **[03-Customer_Support_Agent](https://www.google.com/search?q=03-Apps/03-Customer_Support_Agent/)**: Customer service automation.
- **[04-Medical_Appointment_Scheduler_Agent](https://www.google.com/search?q=03-Apps/04-Medical_Appointment_Scheduler_Agent/)**: Healthcare scheduling.
- **[05-Smart_Home_Automation_Agent](https://www.google.com/search?q=03-Apps/05-Smart_Home_Automation_Agent/)**: IoT control system.
- **[06-Developer_Assistant_Agent](https://www.google.com/search?q=03-Apps/06-Developer_Assistant_Agent/)**: Code assistance and review.
- **[07-Legal_Document_Analyzer_Agent](https://www.google.com/search?q=03-Apps/07-Legal_Document_Analyzer_Agent/)**: Legal document processing.
- **[08-Content_Creation_Pipeline_Agent](https://www.google.com/search?q=03-Apps/08-Content_Creation_Pipeline_Agent/)**: Content generation workflow.
- **[09-Research_Summarization_Agent](https://www.google.com/search?q=03-Apps/09-Research_Summarization_Agent/)**: Academic research assistant.
- **[10-Personalized_Health_Fitness_Planner_Agent](https://www.google.com/search?q=03-Apps/10-Personalized_Health_Fitness_Planner_Agent/)**: Health and fitness planning.
- **[11-E-commerce_Recommendation_Agent](https://www.google.com/search?q=03-Apps/11-E-commerce_Recommendation_Agent/)**: Product recommendation system.
- **[12-Interactive_Storyteller_Agent](https://www.google.com/search?q=03-Apps/12-Interactive_Storyteller_Agent/)**: Creative storytelling assistant.

---

## 💡 Quick Start Examples

### Basic Graph Structure

This example demonstrates the fundamental structure of a LangGraph `StateGraph`.

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List

# Define your state
class WorkflowState(TypedDict):
    user_input: str
    steps: List[str]

# Create a node function
def start_node(state: WorkflowState) -> dict:
    return {"steps": ["started"]}

# Build the graph
builder = StateGraph(WorkflowState)
builder.add_node("start", start_node)
builder.set_entry_point("start")
builder.set_finish_point("start")

app = builder.compile()
```

### Chat Application with LLM

This snippet illustrates how to initialize an LLM for use in a LangGraph application. See the complete example in `01-Graphs/07-Basic_Chat_Graph/` for full details.

```python
from langgraph.graph import StateGraph
from langchain_litellm import ChatLiteLLM

# Initialize LLM
llm = ChatLiteLLM(
    model="ollama/qwen3:0.6b",
    api_base="http://localhost:11434",
    temperature=0.7,
)

# ... (see complete example in 01-Graphs/07-Basic_Chat_Graph/)
```

---

## 🔧 Supported LLM Providers

This repository demonstrates integration with various LLM providers, offering flexibility for your projects:

- **Ollama** (local models): Examples include `llama3.2`, `qwen3`, `deepseek-r1`.
- **OpenAI**: Supports models like `gpt-4o-mini`, `gpt-4`.
- **OpenRouter**: Integrates with various models, including `google/gemma-3-27b-it`.
- **LiteLLM**: Provides a unified interface for multiple providers.

---

## 📋 Dependencies

Key dependencies for this project include:

- `langgraph~=0.4.7`: The core graph framework.
- `langchain-core~=0.3.59`: Foundation for LangChain integration.
- `langchain-openai~=0.3.11`: For seamless OpenAI integration.
- `litellm~=1.67.4`: A versatile multi-provider LLM interface.
- `rich~=14.0.0`: For enhanced terminal output.
- `yfinance~=0.2.61`: Used for financial data in stock agent examples.

For a complete list, please refer to the [`requirements.txt`](https://www.google.com/search?q=requirements.txt) file.

---

## 🎯 Learning Path

We recommend the following learning progression:

1.  **Start with Basics**: Begin with [`01-LangGraph_Intro`](https://www.google.com/search?q=01-LangGraph_Intro/) to grasp core concepts.
2.  **Explore Graphs**: Work through the examples in [`01-Graphs`](https://www.google.com/search?q=01-Graphs/) to master graph structures.
3.  **Build Agents**: Progress to [`02-Agents`](https://www.google.com/search?q=02-Agents/) for advanced agent patterns and implementations.
4.  **Create Applications**: Apply your knowledge with the real-world examples in [`03-Apps`](https://www.google.com/search?q=03-Apps/).

---

## 🛠️ Development Setup

For development and testing, follow these steps:

```bash
# Install in development mode
pip install -e .

# Run with a specific Python version (e.g., Python 3.11)
python3.11 -m pip install -r requirements.txt

# Use with Jupyter notebooks
jupyter notebook 01-LangGraph_Intro/01-LangGraph_Basics.ipynb
```

---

## 📊 Visualization

Many examples within this repository include graph visualization, typically generated as `.png` and `.mermaid` files in their respective directories. This helps in understanding the flow and structure of the LangGraph applications.

---

## 🤝 Contributing

Contributions are highly encouraged and welcome\! Please feel free to submit pull requests or open issues for:

- New example applications.
- Bug fixes and general improvements.
- Documentation enhancements.
- Additional LLM provider integrations.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---
