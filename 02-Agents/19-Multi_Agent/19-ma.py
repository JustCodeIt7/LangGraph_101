# LangGraph Multi-Agent Workflow Examples

## Example 1: Simple Supervisor Architecture
# A supervisor agent manages two specialized agents

from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings



def create_supervisor_workflow():
    # model = ChatOpenAI(model="gpt-4")
    model = ChatOllama(model='llama3.2')
    def supervisor(state: MessagesState) -> Command[Literal["research_agent", "writer_agent", END]]:
        """Supervisor decides which agent to call next based on the task"""
        messages = state["messages"]
        last_message = messages[-1].content
        
        # Simple routing logic
        if "research" in last_message.lower() or "find" in last_message.lower():
            return Command(goto="research_agent")
        elif "write" in last_message.lower() or "draft" in last_message.lower():
            return Command(goto="writer_agent")
        else:
            return Command(goto=END)

    def research_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
        """Specialized agent for research tasks"""
        messages = state["messages"]
        
        # Simulate research work
        research_response = model.invoke([
            {"role": "system", "content": "You are a research specialist. Provide factual information."},
            {"role": "user", "content": messages[-1].content}
        ])
        
        return Command(
            goto="supervisor",
            update={"messages": [research_response]},
        )

    def writer_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
        """Specialized agent for writing tasks"""
        messages = state["messages"]
        
        # Simulate writing work
        writing_response = model.invoke([
            {"role": "system", "content": "You are a professional writer. Create engaging content."},
            {"role": "user", "content": messages[-1].content}
        ])
        
        return Command(
            goto="supervisor",
            update={"messages": [writing_response]},
        )

    # Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("research_agent", research_agent)
    builder.add_node("writer_agent", writer_agent)
    
    builder.add_edge(START, "supervisor")
    
    return builder.compile()

## Example 2: Sequential Custom Workflow
# Agents work in a predefined sequence

def create_sequential_workflow():
    model = ChatOpenAI(model="gpt-4")

    def planner_agent(state: MessagesState):
        """Creates a plan for the task"""
        messages = state["messages"]
        
        plan_response = model.invoke([
            {"role": "system", "content": "You are a planning specialist. Break down tasks into steps."},
            {"role": "user", "content": f"Create a plan for: {messages[-1].content}"}
        ])
        
        return {"messages": [plan_response]}

    def executor_agent(state: MessagesState):
        """Executes the plan"""
        messages = state["messages"]
        
        execution_response = model.invoke([
            {"role": "system", "content": "You are an execution specialist. Implement the given plan."},
            {"role": "user", "content": f"Execute this plan: {messages[-1].content}"}
        ])
        
        return {"messages": [execution_response]}

    def reviewer_agent(state: MessagesState):
        """Reviews and refines the output"""
        messages = state["messages"]
        
        review_response = model.invoke([
            {"role": "system", "content": "You are a quality reviewer. Improve and finalize the work."},
            {"role": "user", "content": f"Review and improve: {messages[-1].content}"}
        ])
        
        return {"messages": [review_response]}

    # Build the graph with explicit flow
    builder = StateGraph(MessagesState)
    builder.add_node("planner", planner_agent)
    builder.add_node("executor", executor_agent)
    builder.add_node("reviewer", reviewer_agent)
    
    # Define the sequential flow
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reviewer")
    
    return builder.compile()

## Example 3: Tool-Based Supervisor with Agent Tools
# Agents are exposed as tools to a supervisor

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from typing import Annotated

def create_tool_based_workflow():
    model = ChatOpenAI(model="gpt-4")

    @tool
    def math_agent(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Solve mathematical problems and calculations"""
        math_response = model.invoke([
            {"role": "system", "content": "You are a math expert. Solve mathematical problems step by step."},
            {"role": "user", "content": query}
        ])
        return math_response.content

    @tool
    def code_agent(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Write and explain code solutions"""
        code_response = model.invoke([
            {"role": "system", "content": "You are a coding expert. Write clean, documented code."},
            {"role": "user", "content": query}
        ])
        return code_response.content

    @tool
    def general_agent(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Handle general questions and conversations"""
        general_response = model.invoke([
            {"role": "system", "content": "You are a helpful general assistant."},
            {"role": "user", "content": query}
        ])
        return general_response.content

    # Create the supervisor using the prebuilt ReAct agent
    tools = [math_agent, code_agent, general_agent]
    supervisor = create_react_agent(model, tools)
    
    return supervisor

# Usage Examples
if __name__ == "__main__":
    # Example 1: Supervisor Architecture
    print("=== Supervisor Architecture Example ===")
    supervisor_workflow = create_supervisor_workflow()
    
    result = supervisor_workflow.invoke({
        "messages": [{"role": "user", "content": "Please research the benefits of renewable energy"}]
    })
    print("Supervisor Result:", result["messages"][-1].content)
    
    # Example 2: Sequential Workflow
    print("\n=== Sequential Workflow Example ===")
    sequential_workflow = create_sequential_workflow()
    
    result = sequential_workflow.invoke({
        "messages": [{"role": "user", "content": "Organize a team building event"}]
    })
    print("Sequential Result:", result["messages"][-1].content)
    
    # Example 3: Tool-Based Workflow
    print("\n=== Tool-Based Workflow Example ===")
    tool_workflow = create_tool_based_workflow()
    
    result = tool_workflow.invoke({
        "messages": [{"role": "user", "content": "Calculate the area of a circle with radius 5"}]
    })
    print("Tool-Based Result:", result["messages"][-1].content)