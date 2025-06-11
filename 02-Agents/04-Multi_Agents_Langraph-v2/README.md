# LangGraph Multi-Agent Tutorial - v2

This tutorial series (v2) explores various patterns for building multi-agent systems using LangGraph. It provides three distinct examples, progressing from simple sequential workflows to more complex supervised and collaborative interactions.

## 📚 What You'll Learn

- **Defining Multiple Agents**: How to structure code for different agents with specialized roles and capabilities.
- **State Management in Multi-Agent Systems**: Designing shared state objects that allow agents to collaborate and pass information.
- **Control Flow and Routing**: Implementing logic (often via a supervisor or predefined sequence) to determine which agent acts next.
- **Tool Usage by Specific Agents**: Assigning tools to particular agents within a larger multi-agent graph.
- **Common Multi-Agent Patterns**:
    1.  **Sequential Workflow**: Agents operate one after another in a pipeline.
    2.  **Supervisor-Worker Delegation**: A central agent routes tasks to specialized worker agents.
    3.  **Collaborative Debate/Discussion**: Agents interact in a more dynamic, turn-based manner, potentially moderated.

## ⚙️ Tutorial Examples

This `v2` collection includes the following examples:

### 1. `example1_basic_sequential_workflow.py`
   - **Scenario**: Idea Generation and Refinement.
   - **Agents**:
     - `Idea Generator Agent`: Generates initial ideas based on a theme.
     - `Idea Refiner Agent`: Selects and elaborates on one of the generated ideas.
   - **Pattern**: A simple linear pipeline where the output of Agent A becomes the input for Agent B.
   - **Key Concepts**: Basic state passing, clear task separation, error handling.
   - **Mermaid Diagram**:
     ```mermaid
     graph TD
         A[User Input: Theme] --> B(Idea Generator Agent);
         B -- Generated Ideas --> C(Idea Refiner Agent);
         C -- Refined Idea --> D[END: Final Output];
         B -- Error --> E(Error Handler);
         C -- Error --> E;
         E --> D;
     ```

### 2. `example2_supervisor_delegation.py`
   - **Scenario**: Customer Support System.
   - **Agents**:
     - `Supervisor Agent`: Analyzes user queries and routes them.
     - `Technical Support Agent`: Handles technical issues (can use tools).
     - `Billing Support Agent`: Handles billing questions.
     - `General Inquiry Agent`: Handles other general questions.
   - **Pattern**: A supervisor agent acts as a router, delegating tasks to appropriate worker agents. Workers report back or resolve the query.
   - **Key Concepts**: Centralized routing, specialized workers, conditional logic for delegation, tool use by a specific worker.
   - **Mermaid Diagram**:
     ```mermaid
     graph TD
         A[User Query] --> B{Supervisor Agent};
         B -- Route: Technical --> C(Technical Support Agent);
         B -- Route: Billing --> D(Billing Support Agent);
         B -- Route: General --> E(General Inquiry Agent);
         B -- Route: Resolved --> H[END: Resolution];
         C -- Needs Tool --> CT(Execute Tech Tool);
         CT -- Tool Result --> C;
         C -- Resolution/Info --> B;
         D -- Resolution/Info --> B;
         E -- Resolution/Info --> B;
     ```

### 3. `example3_agent_debate.py`
   - **Scenario**: AI Ethics Debate.
   - **Agents**:
     - `Moderator Agent`: Manages debate flow, introduces topic, enforces turns, summarizes.
     - `Pro-AI Agent`: Argues in favor of the AI topic.
     - `Con-AI Agent`: Argues against or raises concerns about the AI topic.
   - **Pattern**: Turn-based interaction between multiple agents with distinct roles, managed by a moderator.
   - **Key Concepts**: Cyclic graph flow, managing conversational history, role-playing by LLMs, dynamic turn-taking.
   - **Mermaid Diagram**:
     ```mermaid
     graph TD
         A[User Input: Debate Topic] --> B(Moderator Agent);
         B -- Next: Pro-AI --> C(Pro-AI Agent);
         C -- Argument --> B;
         B -- Next: Con-AI --> D(Con-AI Agent);
         D -- Argument --> B;
         B -- Debate End --> E[END: Final Summary];
     ```
     *(Note: The loop B->C->B->D->B represents multiple turns)*

## 🚀 Running the Tutorials

### Prerequisites

Ensure you have the necessary Python packages:
```bash
pip install langgraph langchain_openai langchain_core typing_extensions
```
And set your OpenAI API key (or configure for your chosen LLM provider):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Execution

Navigate to this directory (`02-Agents/04-Multi_Agents_Langraph-v2/`) and run each example script individually:
```bash
python example1_basic_sequential_workflow.py
python example2_supervisor_delegation.py
python example3_agent_debate.py
```
Each script is interactive and will prompt you for input relevant to its scenario. Follow the on-screen instructions and observe the agent interactions printed to the console.

## 💡 Core LangGraph Concepts for Multi-Agent Systems

- **StateGraph**: The foundation for defining the structure and flow of interactions.
- **TypedDict for State**: Crucial for defining the shared information space between agents. Fields in the state can hold data, messages, control flags (like `next_agent` or `current_speaker`), etc.
- **Nodes as Agents (or Agent Steps)**: Each node in the graph typically represents an agent performing an action or a specific step within an agent's logic.
- **Conditional Edges**: Essential for routing control to different agents based on the current state (e.g., a supervisor's decision, the outcome of a task, or whose turn it is).
- **`add_messages`**: Useful for accumulating a shared history of interactions (like a chat transcript) in the state.
- **Checkpointer (`SqliteSaver`)**: Enables persistence of state, allowing multi-agent interactions to be paused and resumed, or to run over extended periods.

These examples provide a starting point for building sophisticated multi-agent applications with LangGraph, showcasing its flexibility in orchestrating complex workflows.
