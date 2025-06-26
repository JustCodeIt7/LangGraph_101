# LangGraph Agent Examples

This directory contains a comprehensive collection of examples demonstrating how to build various types of agents using LangGraph. Each subdirectory focuses on a specific concept, from basic agent execution to advanced multi-agent systems.

## Table of Contents

- [01-Running_Agents](#01-running_agents)
- [02-Agent-State](#02-agent-state)
- [02-Streaming](#02-streaming)
- [04-Tools-v2](#04-tools-v2)
- [08-Human_in_the_Loop](#08-human_in_the_loop)
- [09-Multi_Agent](#09-multi_agent)
- [09-Advanced-Agent-Examples-v2](#09-advanced-agent-examples-v2)
- [10-RAG_Agent](#10-rag_agent)

---

### `01-Running_Agents`

Learn the fundamentals of creating and running LangGraph agents.
- **Concepts**: Covers synchronous and asynchronous invocation, input/output formats, streaming, and recursion limits.
- **Key Examples**: `01-running_agents-final.py`, `01-running_agents_litellm-final.py`.

### `02-Agent-State`

Explore different patterns for managing state within your agents.
- **Concepts**: Demonstrates tracking multi-step tasks, maintaining conversational context, and coordinating shared state between multiple agents.
- **Key Examples**: `example1_task_tracking_agent_state.py`, `example2_context_aware_agent_state.py`, `example3_collaborative_agent_state.py`.

### `02-Streaming`

Dive into the various streaming capabilities of LangGraph to build responsive applications.
- **Concepts**: Covers agent progress streaming, LLM token streaming, custom tool updates, and asynchronous streaming.
- **Key Examples**: `01-agent_progress_streaming.py`, `02-llm_tokens_streaming.py`, `03-tool_updates_streaming.py`, `04-multiple_streaming_modes.py`, `05-async_streaming.py`.

### `04-Tools-v2`

Master the creation and use of tools within LangGraph agents.
- **Concepts**: Topics include basic function tools, the `@tool` decorator, Pydantic schemas for inputs, advanced features like hiding arguments, disabling parallel calls, and error handling.
- **Key Examples**: `01_basic_tools.py`, `02_advanced_tool_features.py`, `03_error_handling.py`, `04_prebuilt_tools_and_integration.py`.

### `08-Human_in_the_Loop`

Implement workflows that require human approval or intervention.
- **Concepts**: Demonstrates patterns for basic approval, intermediate editing workflows, and complex debugging loops with human input.
- **Key Examples**: `01-basic_human_approval.py`, `02-intermediate_approval_workflow.py`, `LangGraph_HITL_2_Intermediate_Editing_Workflow.ipynb`.

### `09-Multi_Agent`

Build systems with multiple agents collaborating to solve a problem.
- **Concepts**: Patterns include sequential workflows, supervisor-worker delegation, and agent debates.
- **Key Examples**: `example1_basic_sequential_workflow.py`, `example2_supervisor_delegation.py`, `example3_agent_debate.py`.

### `09-Advanced-Agent-Examples-v2`

Explore sophisticated agentic patterns.
- **Concepts**: Architectures include collaborative multi-agent systems, adaptive planning agents, and hierarchical reasoning agents.
- **Key Examples**: `example1_collaborative_agents.py`, `example2_adaptive_planning.py`, `example3_hierarchical_reasoning.py`.

### `10-RAG_Agent`

A practical example of a Retrieval-Augmented Generation (RAG) agent.
- **Concepts**: An interactive CLI application to chat with your documents (PDFs, Markdown) using Ollama and ChromaDB.
- **Key File**: `rag_agent.py`.
