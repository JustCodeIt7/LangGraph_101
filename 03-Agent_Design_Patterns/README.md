# 03 - Agent Design Patterns

This directory contains advanced agent design patterns and implementations using LangGraph. Each subdirectory demonstrates a specific architectural pattern or design approach for building sophisticated AI agents.

## Table of Contents

1. [Multi-Agent Router (`20-Multi_Agent_Router`)](#20-multi_agent_router)
2. [Multi-Agent Parallel (`21-Multi-Agent_Parallel`)](#21-multi-agent_parallel)
3. [Orchestration Agent (`22-Orchestration_Agent`)](#22-orchestration_agent)
4. [Evaluator-Optimizer (`23-Evaluator-Optimizer`)](#23-evaluator-optimizer)
5. [Custom RAG (`24-Custom_RAG`)](#24-custom_rag)
6. [RAG Agent (`25-RAG_Agent`)](#25-rag_agent)
7. [Advanced Agent Examples v2 (`26-Advanced-Agent-Examples-v2`)](#26-advanced-agent-examples-v2)

---

## 20. Multi_Agent_Router

This pattern demonstrates how to route user requests to different specialized agents based on the nature of the query.

### Key Concepts Covered

- **Intent Classification**: Using an LLM to analyze user input and determine which specialized agent should handle it.
- **Dynamic Routing**: Implementing conditional logic that directs traffic between multiple agents.
- **Agent Specialization**: Creating focused agents for specific tasks (e.g., code review, documentation, general assistance).

### How It Works

1. User submits a query
2. A router agent analyzes the input to determine intent
3. Based on the classification, the request is routed to an appropriate specialized agent
4. The selected agent processes the request and returns results

### Example Use Cases

- Customer support systems that route queries to different specialists
- Development tools that direct code questions to a coding assistant and documentation requests to a docs assistant
- Multi-domain assistants that handle different types of tasks (research, writing, analysis)

---

## 21. Multi-Agent Parallel

This pattern demonstrates running multiple agents simultaneously to process the same task or different aspects of a complex problem.

### Key Concepts Covered

- **Parallel Execution**: Running multiple agents concurrently using LangGraph's parallel node execution.
- **Result Aggregation**: Combining outputs from multiple agents into a unified response.
- **Task Decomposition**: Breaking down complex tasks into subtasks that can be handled by different agents.

### How It Works

1. A task is decomposed into independent subtasks
2. Multiple agents are invoked in parallel to handle each subtask
3. Results are collected and aggregated into a final output

### Example Use Cases

- Research tasks where multiple sources need to be analyzed simultaneously
- Content generation that requires different aspects (style, accuracy, creativity)
- Comprehensive analysis requiring multiple expert perspectives

---

## 22. Orchestration Agent

This pattern demonstrates a central coordinator that manages multiple specialized agents in a coordinated workflow.

### Key Concepts Covered

- **Central Coordinator**: A main agent that orchestrates the workflow between sub-agents.
- **State Management**: Maintaining shared state across multiple agent interactions.
- **Workflow Control**: Managing the sequence and dependencies between different processing stages.

### How It Works

1. The orchestrator receives a complex request
2. It breaks down the task into steps and assigns them to appropriate agents
3. Each agent completes its portion and reports back
4. The orchestrator coordinates the next steps based on results
5. Final output is assembled from all agent contributions

### Example Use Cases

- Complex business processes requiring multiple specialized services
- Multi-step research workflows (gather → analyze → synthesize)
- Enterprise applications with multiple microservices

---

## 23. Evaluator-Optimizer

This pattern implements a feedback loop where one agent generates content and another evaluates it, iterating until quality thresholds are met.

### Key Concepts Covered

- **Iterative Refinement**: Creating loops that continuously improve output quality.
- **Quality Gates**: Using evaluation agents to check if outputs meet specific criteria.
- **Self-Correction**: Agents that can identify their own errors and attempt corrections.

### How It Works

1. Generator agent creates initial content
2. Evaluator agent assesses the content against defined criteria
3. If quality is insufficient, feedback is provided back to the generator
4. The process repeats until acceptable quality is achieved or max iterations reached

### Example Use Cases

- Code generation with quality verification
- Document refinement based on style and accuracy checks
- Content creation that must meet multiple validation criteria

---

## 24. Custom RAG

This pattern demonstrates building a custom Retrieval-Augmented Generation system using LangGraph.

### Key Concepts Covered

- **Document Processing**: Loading and chunking documents for retrieval.
- **Vector Storage**: Using embeddings to enable semantic search.
- **Contextual Retrieval**: Augmenting queries with relevant context from the knowledge base.

### How It Works

1. Documents are processed and converted to vector embeddings
2. User queries are matched against the document corpus
3. Retrieved context is combined with the original query
4. An LLM generates responses using both the query and retrieved context

---

## 25. RAG Agent

A practical implementation of a Retrieval-Augmented Generation agent for interactive document-based conversations.

### Key Concepts Covered

- **Conversational Context**: Maintaining chat history across multiple interactions.
- **Dynamic Retrieval**: Determining when to fetch additional context vs. using existing knowledge.
- **Source Attribution**: Tracking which documents contributed to the response.

### How It Works

1. User asks a question about uploaded documents
2. The system retrieves relevant passages based on semantic similarity
3. Retrieved content is combined with conversation history
4. LLM generates an answer citing the source material

---

## 26. Advanced Agent Examples v2

This directory contains sophisticated agentic patterns and advanced implementations.

### Key Concepts Covered

- **Hierarchical Agents**: Multi-level agent structures with supervisor and worker roles.
- **Adaptive Planning**: Agents that dynamically adjust their approach based on task complexity.
- **Collaborative Multi-Agent Systems**: Multiple agents working together with shared goals.

### Example Patterns

- Collaborative agents that share context and coordinate actions
- Adaptive planning systems that modify their strategy mid-execution
- Hierarchical reasoning with multiple levels of abstraction