# LangGraph Advanced Agent Examples Tutorial

This tutorial demonstrates more sophisticated agentic patterns using LangGraph, focusing on a research assistant agent that can plan, execute, reflect, and refine its work.

## 📚 What You'll Learn

- **Multi-step Planning**: How an agent can create a sequence of actions to achieve a goal.
- **Plan Execution with Tools**: Integrating tool usage within the execution of a plan.
- **Reflection and Self-Correction**: Enabling the agent to critique its own progress and findings, and decide to re-plan or refine information.
- **Looping for Refinement**: Implementing loops where the agent can iterate on its plan or information gathering based on reflection.
- **Complex State Management**: Handling a more detailed agent state to track the plan, executed steps, research summaries, reflections, and loop controls.
- **Role-Specific LLMs**: Using different LLM configurations for different tasks within the agent (planning, execution, reflection, generation).

## 🔧 Scenario: AI Research Assistant

The agent is tasked with researching a user-provided topic. Its workflow involves:

1.  **Planning**: Generates an initial multi-step research plan.
2.  **Execution**: Iterates through the plan steps. For each step:
    *   Decides if a tool (e.g., `simulated_web_search`) is needed or if it can generate information directly.
    *   Executes the tool if required and processes its output.
    *   Accumulates information into a `research_summary`.
3.  **Reflection**: After completing all plan steps (or a loop iteration):
    *   Critiques its progress, the quality of information, and the effectiveness of the plan.
    *   Decides whether to `PROCEED` to generating a report, `REPLAN` if the plan was flawed, or `REFINE_INFO` if more/better information is needed.
4.  **Looping**: If reflection suggests `REPLAN` or `REFINE_INFO`, the agent can loop back to the planning or execution phase, up to a maximum number of iterations.
5.  **Report Generation**: Once the agent (or loop limit) decides to proceed, it generates a final research report based on the accumulated summary and final reflections.

**Tool Used:**
- `simulated_web_search(query)`: A mock tool that returns pre-defined search results for specific queries to simulate web research.

## 🏗️ Graph Architecture

The agent's workflow is structured as a stateful graph with several key nodes and conditional edges:

- **`create_initial_plan_node`**: Generates the initial research plan.
- **`execute_plan_step_node`**: Determines action for the current plan step (use tool or generate text).
- **`execute_tool_node`**: Executes a tool if called by `execute_plan_step_node`.
- **`reflection_node`**: Critiques the current state of research and suggests next actions (proceed, replan, refine).
- **`generate_final_report_node`**: Synthesizes all information into a final report.

**Conditional Routing Logic:**
- **`route_after_plan_execution`**:
    - If `execute_plan_step_node` proposes a tool, routes to `execute_tool_node`.
    - If a step is completed (either by direct text generation or after `execute_tool_node`):
        - If more plan steps remain, routes back to `execute_plan_step_node` for the next step.
        - If all plan steps are done, routes to `reflection_node`.
- **`route_after_reflection`**:
    - If reflection suggests `REPLAN`, routes back to `create_initial_plan_node`.
    - If reflection suggests `REFINE_INFO`, routes back to `execute_plan_step_node` (to re-iterate through the plan with new context/reflection).
    - If reflection suggests `PROCEED` (or max loops reached), routes to `generate_final_report_node`.

```mermaid
graph TD
    A[User Input: Research Topic] --> B(create_initial_plan_node);
    B --> C(execute_plan_step_node);
    C -- Tool Call Proposed? --> D{route_after_plan_execution};
    D -- Yes --> E(execute_tool_node);
    D -- No, More Steps --> C;
    D -- No, All Steps Done --> F(reflection_node);
    E --> D; % After tool, re-evaluate plan execution state
    F -- Reflection Outcome? --> G{route_after_reflection};
    G -- REPLAN --> B;
    G -- REFINE_INFO --> C;
    G -- PROCEED / Max Loops --> H(generate_final_report_node);
    H --> I[END - Final Report];
```

## 🚀 Running the Tutorial

### Prerequisites

Ensure you have the necessary packages:
```bash
pip install langgraph langchain_openai langchain_core typing_extensions
```
And set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Execution

Run the script:
```bash
python 05-Advanced-Agent-Examples.py
```
You will be prompted to enter a research topic. Try:
- "History of AI"
- "Benefits of LangGraph"
- Or any other topic (the mock search tool will provide generic results for unknown topics).

The script will output the agent's internal steps, including planning, execution, tool calls, reflections, and the final report.

## 💡 Key Advanced Concepts Demonstrated

- **Autonomous Looping and Refinement**: The agent can decide to loop back and refine its work based on self-critique, a hallmark of more advanced autonomous systems.
- **Separation of Concerns with Nodes**: Different nodes handle distinct responsibilities (planning, acting, reflecting, generating), making the agent's logic modular.
- **State as a "Scratchpad"**: The `ResearchAgentState` acts as a comprehensive memory or scratchpad for the agent, holding its plan, findings, reflections, and control variables.
- **Checkpointing for Long-Running Tasks**: `SqliteSaver` is used, implying that such complex, multi-step, and potentially long-running tasks can be made resilient and resumable.
- **Controlled Iteration**: The use of `max_loops` and `loop_count` in the state prevents infinite loops and provides a control mechanism for the reflection cycle.

This tutorial showcases how LangGraph can be used to build agents that exhibit more complex reasoning patterns, moving beyond simple request-response or single tool use towards more autonomous, iterative problem-solving.
