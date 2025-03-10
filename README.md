# LangGraph_101

LangGraph coding examples

## Outline

## Module 1: Introduction to LangGraph

### Episode 1: What is LangGraph?

LangGraph is a library for building stateful agent workflows as graphs. It extends LangChain by providing a structured
way to orchestrate LLM-powered applications through directed graphs. The framework allows developers to define clear
execution paths with nodes (processing steps) and edges (transitions between steps), with robust state management
throughout the execution flow. Setting up requires Python with `pip install langgraph` to access the core
functionality. [^1]

### Episode 2: Understanding Graph-Based Architectures

LangGraph implements directed computational graphs where each node represents a processing step and edges define the
flow between nodes. This architecture enables complex decision-making patterns including branching, looping, and
conditional execution. The framework maintains a consistent state object that gets passed and transformed through the
graph, allowing for sophisticated applications with memory and persistence. Graph visualization tools help developers
understand execution flow and debug complex workflows. [^2]

### Episode 3: Your First LangGraph Application

Building your first LangGraph application involves defining nodes (functions that process state), creating a StateGraph
object, adding nodes to the graph, connecting them with edges, and compiling the graph. The basic pattern follows:

```python
from langgraph.graph import StateGraph


# Define state schema
class GraphState(TypedDict):
    messages: list[dict]


# Define node functions
def node_function(state: GraphState) -> GraphState:
    # Process state
    return updated_state


# Create graph
graph = StateGraph(GraphState)
graph.add_node("process", node_function)
graph.set_entry_point("process")
graph.set_finish_point("process")
app = graph.compile()

# Run graph
result = app.invoke({"messages": []})
```

This creates a simple sequential workflow that processes messages. [^2]

## Module 2: Graph API Fundamentals

### Episode 4: Building Sequential Workflows

Sequential workflows in LangGraph connect nodes in a linear chain, with each node processing the state and passing it to
the next. State reducers allow you to control how state updates are merged. A text processing pipeline might look like:

```python
graph = StateGraph(GraphState)
graph.add_node("extract_entities", extract_entities)
graph.add_node("analyze_sentiment", analyze_sentiment)
graph.add_node("generate_summary", generate_summary)

graph.set_entry_point("extract_entities")
graph.add_edge("extract_entities", "analyze_sentiment")
graph.add_edge("analyze_sentiment", "generate_summary")
graph.set_finish_point("generate_summary")
```

This creates a three-step pipeline that processes text sequentially. [^3]

### Episode 5: Branching and Conditional Logic

LangGraph supports conditional branching using edge conditions. This allows the graph to take different paths based on
the current state:

```python
def router(state: GraphState) -> str:
    # Logic to determine which branch to take
    return "branch_a" if condition else "branch_b"


graph.add_conditional_edges("decision_node", router, {"branch_a": "node_a", "branch_b": "node_b"})
```

This pattern enables building decision-making agents that can choose different actions based on
context. [Source](https://langchain-ai.github.io/langgraph/concepts/high_level/)

### Episode 6: Creating and Controlling Loops

Loops in LangGraph are implemented through recursive edges that point back to previous nodes. This enables iterative
reasoning and refinement:

```python
def should_continue(state: GraphState) -> str:
    # Check if we need more iterations
    if need_more_iterations:
        return "continue"
    else:
        return "complete"


graph.add_conditional_edges("process", should_continue,
                            {"continue": "process", "complete": "finish"})
```

Recursion limits prevent infinite loops, and conditional edges provide exit conditions. [^4]

### Episode 7: State Management Deep Dive

LangGraph uses Pydantic models or TypedDict to define state schemas, ensuring type safety and validation:

```python
class AgentState(TypedDict):
    messages: list[dict]
    context: dict
    tools: list[dict]
    next_steps: list[str]
```

Private state can be passed between nodes using namespaces, and state reducers control how updates are merged. Best
practices include keeping state immutable and using clear
schemas. [Source](https://langchain-ai.github.io/langgraph/concepts/high_level/)

### Episode 8: Visualizing and Debugging Graphs

LangGraph provides visualization tools to understand graph structure and execution flow. LangGraph Studio, the first
agent IDE, offers a visual interface for building, debugging, and monitoring graphs:

```python
# Generate visualization
graph.show()

# Trace execution for debugging
with tracing.trace() as session:
    result = app.invoke({"messages": []})
    trace = session.get_trace()
```

These tools help identify bottlenecks and debug complex
workflows. [Source](https://blog.langchain.dev/langgraph-studio-the-first-agent-ide/)

## Module 3: Tool Integration and Execution

### Episode 9: Tool Calling Fundamentals

LangGraph integrates with tools through the ToolNode abstraction, which handles tool calling, execution, and state
updates:

```python
from langgraph.prebuilt import ToolNode

tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)
```

This enables building research agents that can access external capabilities like search engines or calculators. [^3]

### Episode 10: Advanced Tool Calling Patterns

Advanced tool calling includes error handling, dynamic tool selection, and state updates from tool results:

```python
def handle_tool_error(state, error):
    # Process error and update state
    return updated_state


tool_node = ToolNode(tools, handle_error=handle_tool_error)
```

This pattern enables robust web browsing agents that can recover from failures and adapt to changing conditions. [^4]

### Episode 11: Managing Large Tool Sets

For large tool sets, LangGraph supports dynamic tool selection and categorization:

```python
def select_tools(state):
    # Dynamically select relevant tools based on state
    return [tool for tool in all_tools if is_relevant(tool, state)]


dynamic_tool_node = ToolNode(select_tools)
```

This enables building multi-capability assistants that can access the right tools for each
task. [Source](https://langchain-ai.github.io/langgraph/concepts/high_level/)

### Episode 12: Structured Output with Tool Calling

LangGraph enforces structured outputs through schemas and validation:

```python
from pydantic import BaseModel, Field


class ExtractedData(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")


def extract_data(state):
    # Use LLM to extract structured data
    return {"extracted": ExtractedData(...)}
```

Error handling and retries ensure robust data extraction even with unreliable LLM outputs. [^3]

## Module 4: Memory and Persistence

### Episode 13: Conversation Memory Fundamentals

LangGraph manages conversation history through state:

```python
class ConversationState(TypedDict):
    messages: list[dict]


def add_message(state, message):
    return {"messages": state["messages"] + [message]}
```

Messages can be added, updated, or filtered as needed, enabling chatbots with sophisticated memory management. [^3]

### Episode 14: Thread-Level Persistence

LangGraph supports persistence through its built-in persistence layer:

```python
from langgraph.persistence import MemorySaver

memory_saver = MemorySaver()
graph = StateGraph(GraphState, checkpointer=memory_saver)
```

This enables checkpointing and state restoration, allowing conversations to be paused and
resumed. [Source](https://blog.langchain.dev/langgraph-studio-the-first-agent-ide/)

### Episode 15: Cross-Thread Persistence

Cross-thread memory allows sharing data between conversations:

```python
from langgraph.persistence import SharedMemorySaver

shared_saver = SharedMemorySaver()
graph = StateGraph(GraphState, checkpointer=shared_saver)
```

This enables assistants with knowledge bases that persist across multiple conversations. [^5]

### Episode 16: Semantic Search and Retrieval

LangGraph integrates with vector databases for semantic search:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())


def retrieve_context(state):
    query = state["messages"][-1]["content"]
    docs = vectorstore.similarity_search(query)
    return {"context": docs}
```

This enables knowledge-augmented agents that can retrieve relevant information. [^3]

## Module 5: Human-in-the-Loop Workflows

### Episode 17: Waiting for User Input

LangGraph supports human-in-the-loop workflows through interrupts:

```python
def needs_human_input(state):
    return "human_needed" if condition else "continue"


graph.add_conditional_edges("process", needs_human_input,
                            {"human_needed": "wait_for_human", "continue": "next_step"})
```

This enables interactive assistants that can pause for user input when needed. [^5]

### Episode 18: Tool Call Review and Approval

Human review for tool calls adds safety and control:

```python
def review_tool_call(state):
    # Present tool call for human review
    return {"approved": True, "modified_call": modified_call}
```

This pattern enables supervised agents where humans can approve or modify actions before
execution. [Source](https://blog.langchain.dev/langgraph-studio-the-first-agent-ide/)

### Episode 19: Time Travel and State Editing

LangGraph Studio supports time travel debugging:

```python
# In LangGraph Studio
trace = studio.get_trace(thread_id)
modified_state = studio.edit_state(trace, step_id, new_state)
result = studio.continue_from_state(thread_id, modified_state)
```

This enables iterative development by allowing developers to modify past states and see how changes affect
outcomes. [Source](https://blog.langchain.dev/langgraph-studio-the-first-agent-ide/)

## Module 6: Streaming and Responsiveness

### Episode 20: Streaming Fundamentals

LangGraph supports streaming for responsive UIs:

```python
# Enable streaming
app = graph.compile(streaming=True)

# Handle stream events
for chunk in app.stream({"messages": [{"role": "user", "content": "Hello"}]}):
    print(chunk)
```

This enables responsive interfaces that show progress in real-time. [^3]

### Episode 21: Advanced Streaming Techniques

Streaming can be configured for specific nodes:

```python
# Configure streaming for specific nodes
app = graph.compile(streaming=["generate_response", "search_results"])
```

This gives fine-grained control over what gets streamed to the
user. [Source](https://langchain-ai.github.io/langgraph/concepts/high_level/)

### Episode 22: Building Responsive UIs with Streaming

Frontend integration with streaming enables responsive UIs:

```javascript
// Frontend code (JavaScript)
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: userInput}),
});

const reader = response.body.getReader();
while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    // Process streaming chunk
    updateUI(new TextDecoder().decode(value));
}
```

This pattern enables web applications with streaming that show progress indicators and partial results. [^5]

## Module 7: Multi-Agent Systems

### Episode 23: Multi-Agent Fundamentals

LangGraph supports multi-agent systems through specialized nodes:

```python
def researcher(state):
    # Agent that researches information
    return {"research": research_results}


def writer(state):
    # Agent that writes content based on research
    return {"content": written_content}


graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge("researcher", "writer")
```

This enables simple two-agent systems with defined roles and responsibilities. [^4]

### Episode 24: Agent Handoffs and Collaboration

Agent handoffs enable collaboration through state sharing:

```python
def determine_next_agent(state):
    # Determine which agent should handle the next step
    return next_agent_id


graph.add_conditional_edges("router", determine_next_agent,
                            {"agent1": "agent1_node", "agent2": "agent2_node"})
```

This enables customer service routing systems that can transfer conversations between specialized agents. [^4]

### Episode 25: Building a Multi-Agent Network

Complex agent networks involve multiple specialized agents:

```python
agents = {
    "researcher": researcher_agent,
    "planner": planner_agent,
    "writer": writer_agent,
    "critic": critic_agent
}

for name, agent in agents.items():
    graph.add_node(name, agent)
```

Coordination and conflict resolution mechanisms ensure effective collaboration. [^4]

### Episode 26: Multi-Turn Conversations in Multi-Agent Systems

Turn-taking mechanisms enable structured conversations:

```python
def next_turn(state):
    # Determine which agent speaks next
    current_speaker = state.get("current_speaker")
    return get_next_speaker(current_speaker)


graph.add_conditional_edges("turn_manager", next_turn,
                            {agent_id: agent_node for agent_id, agent_node in agents.items()})
```

This enables collaborative problem-solving systems where agents take turns contributing to the solution. [^4]

## Module 8: Subgraphs and Modular Design

### Episode 27: Working with Subgraphs

Subgraphs enable modular design by reusing graphs as components:

```python
# Create a subgraph
subgraph = StateGraph(SubgraphState)
# Add nodes and edges to subgraph
subgraph_app = subgraph.compile()


# Use subgraph in main graph
def use_subgraph(state):
    subgraph_result = subgraph_app.invoke(transform_state(state))
    return transform_result(subgraph_result)


main_graph.add_node("subgraph_node", use_subgraph)
```

This enables building modular assistants with reusable
components. [Source](https://langchain-ai.github.io/langgraph/concepts/high_level/)

### Episode 28: Managing State in Subgraphs

State transformation between graphs enables isolation and sharing:

```python
def prepare_subgraph_state(main_state):
    # Extract relevant parts of main state for subgraph
    return subgraph_state


def integrate_subgraph_result(main_state, subgraph_result):
    # Merge subgraph result back into main state
    return updated_main_state
```

This enables hierarchical decision systems with clear separation of
concerns. [Source](https://langchain-ai.github.io/langgraph/concepts/high_level/)

### Episode 29: Subgraph Persistence and Streaming

Subgraphs can have their own persistence and streaming configuration:

```python
subgraph = StateGraph(SubgraphState, checkpointer=subgraph_saver)
subgraph_app = subgraph.compile(streaming=True)
```

This enables complex workflows with independent subcomponents that can be monitored and debugged separately. [^5]

## Module 9: Deployment and Production

### Episode 30: Deployment Options Overview

LangGraph offers both self-hosted and platform deployment options:

- **LangGraph Platform**: Managed service with built-in scaling and monitoring
- **Self-hosted**: Deploy on your own infrastructure for maximum control
- **Serverless**: Deploy as serverless functions for cost optimization

Each option has different infrastructure requirements and cost implications. [^5]

### Episode 31: Setting Up for Deployment

Preparing for deployment involves structuring your application:

```python
# requirements.txt
langgraph == 0.0
.15
langchain == 0.0
.335
pydantic == 2.5
.2

# app.py
from fastapi import FastAPI
from langgraph.graph import StateGraph

app = FastAPI()
graph = StateGraph(GraphState)
# Configure graph
agent = graph.compile()


@app.post("/chat")
async def chat(request: ChatRequest):
    return agent.invoke(request.dict())
```

Dependency management and environment configuration ensure consistent behavior across environments. [^3]

### Episode 32: Deploying with LangGraph Platform

LangGraph Platform simplifies deployment:

```python
from langgraph_sdk import LangGraphClient

# Deploy to LangGraph Platform
client = LangGraphClient(api_key="your_api_key")
deployment = client.deploy(graph, name="my-agent")

# Invoke deployed graph
result = client.invoke(deployment.id, {"messages": [{"role": "user", "content": "Hello"}]})
```

The platform provides monitoring, management, and scaling capabilities. [^5]

### Episode 33: Self-Hosted Deployment

Self-hosted deployment often uses Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Infrastructure setup includes load balancing, monitoring, and scaling considerations. [^3]

## Module 10: Advanced Topics and Case Studies

### Episode 34: Integrating with Other Frameworks

LangGraph can integrate with other agent frameworks:

```python
# Using AutoGen with LangGraph
from autogen import Agent


def autogen_node(state):
    agent = Agent(...)
    result = agent.generate(state["messages"][-1]["content"])
    return {"messages": state["messages"] + [{"role": "assistant", "content": result}]}


graph.add_node("autogen", autogen_node)
```

Hybrid architectures combine the strengths of multiple frameworks. [^6]

### Episode 35: Custom Authentication and Security

Implementing custom authentication adds security:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    user = authenticate_token(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user


@app.post("/chat")
async def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    # Check user permissions
    if not has_permission(user, "chat"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return agent.invoke(request.dict())
```

Access control strategies ensure that only authorized users can access sensitive functionality. [^3]

### Episode 36: Performance Optimization

Identifying and addressing bottlenecks improves performance:

```python
# Caching expensive operations
from functools import lru_cache


@lru_cache(maxsize=100)
def expensive_operation(input_data):
    # Expensive computation
    return result


# Parallel execution
import asyncio


async def parallel_node(state):
    tasks = [asyncio.create_task(process(item)) for item in state["items"]]
    results = await asyncio.gather(*tasks)
    return {"results": results}
```

Caching and parallel execution can significantly improve response
times. [Source](https://langchain-ai.github.io/langgraph/concepts/high_level/)

### Episode 37: Real-World Case Study

A complete implementation might combine multiple techniques:

```python
# Define state
class AgentState(TypedDict):
    messages: list[dict]
    context: dict
    tools: list[dict]


# Create graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("plan", plan_next_steps)
graph.add_node("tools", ToolNode(tools))
graph.add_node("generate_response", generate_response)

# Add edges
graph.set_entry_point("retrieve_context")
graph.add_edge("retrieve_context", "plan")


def route_after_planning(state):
    if state["plan"]["needs_tools"]:
        return "use_tools"
    else:
        return "respond"


graph.add_conditional_edges("plan", route_after_planning,
                            {"use_tools": "tools", "respond": "generate_response"})
graph.add_edge("tools", "generate_response")
graph.set_finish_point("generate_response")

# Compile and deploy
app = graph.compile(streaming=True)
```
