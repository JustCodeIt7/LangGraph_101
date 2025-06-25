# LangGraph 101: A Comprehensive Tutorial Series

Welcome to LangGraph 101, a hands-on tutorial series designed to help you learn and master [LangGraph](https://github.com/langchain-ai/langgraph), a library for building stateful, multi-actor applications with LLMs.

## 📚 About This Repository

This repository contains code examples, tutorials, and applications that demonstrate how to use LangGraph effectively. Whether you're new to LangGraph or looking to deepen your understanding, this resource provides a structured learning path from basic concepts to advanced applications.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) (for running local models)
- Basic understanding of LLMs and Python

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/LangGraph_101.git
   cd LangGraph_101
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables (if needed):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## 📖 Repository Structure

The repository is organized into several sections:

### 01-Graphs
Introduction to LangGraph's core concepts and graph-based workflows:
- **03-Graph_Basics**: Fundamental graph structures
- **04-Multiple_Inputs**: Working with various input types
- **05-Conditional_Routing**: Implementing decision logic
- **06-Looping_Logic**: Creating loops and recursive patterns
- **07-Basic_Chat_Graph**: Building a simple chat application

### 02-Agents
Advanced agent implementations using LangGraph:
- **01-Running_Agents**: Basic agent setup and execution
- **02-Agent-State**: Managing agent state
- **03-Models**: Working with different LLM models
- **04-Tools**: Implementing tool usage
- **05-MCP**: Multi-agent coordination
- And more advanced topics...

### 03-Apps
Complete application examples:
- Travel planning agents
- Financial investment assistants
- Customer support systems
- Medical appointment schedulers
- And many more real-world applications

## 💡 Usage Examples

### Basic Chat Graph

```python
from langgraph.graph import StateGraph
from langchain_litellm import ChatLiteLLM

# Initialize LLM
llm = ChatLiteLLM(
    model="ollama/qwen3:0.6b",
    api_base="http://localhost:11434",
    temperature=0.7,
)

# Create a graph
# ... (see examples in 01-Graphs/07-Basic_Chat_Graph)
```

Check the individual directories for more detailed examples and step-by-step tutorials.

## 🛠️ Development

This repository is continuously updated with new examples and improvements. Feel free to contribute by submitting pull requests or opening issues for suggestions and bug reports.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) team for creating LangGraph
- Contributors to the open-source LLM ecosystem