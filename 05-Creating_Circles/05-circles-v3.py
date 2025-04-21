from typing import Annotated, Dict, List, TypedDict, Tuple, Optional
import operator
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END


# Define our state schema using TypedDict
class ReActAgentState(TypedDict):
    """State for a ReAct agent that performs reasoning and action in cycles."""
    messages: Annotated[List, operator.add]  # Accumulate messages
    actions_taken: Annotated[List[str], operator.add]  # Track actions
    iteration_count: int  # Count iterations to prevent infinite loops
    final_answer: Optional[str]  # Final answer when complete


# Define some simple tools for our agent to use
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information on a specific topic."""
    # In a real implementation, this would use an actual Wikipedia API
    # This is a stub for demonstration purposes
    if "python" in query.lower():
        return "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected."
    elif "langchain" in query.lower():
        return "LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware, reason, and learn from feedback."
    elif "langgraph" in query.lower():
        return "LangGraph is a library for building stateful, multi-actor applications with LLMs. It allows for the creation of directed graphs where LLMs and tools can be composed as nodes."
    else:
        return f"Information about '{query}' not found in the mock Wikipedia database."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safety: restrict to basic arithmetic
        allowed = set("0123456789+-*/() .")
        if not all(c in allowed for c in expression):
            return "Error: Only basic arithmetic operations are supported."
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def current_date() -> str:
    """Return the current date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


# Create a list of available tools
tools = [search_wikipedia, calculate, current_date]


# Node 1: Agent reasoning - decide what to do next
def agent_reasoning(state: ReActAgentState) -> Dict:
    """Agent analyzes the problem and decides what to do next."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Extract the conversation history
    messages = state["messages"]
    
    # Include the number of iterations so far for context
    iteration_info = f"Iteration {state['iteration_count'] + 1}"
    
    # Include available tools information
    tools_info = "Available tools:\n"
    for tool in tools:
        tools_info += f"- {tool.name}: {tool.description}\n"
    
    # Create system message with instructions for the ReAct pattern
    system_message = SystemMessage(content=f"""
    You are a problem-solving agent that follows the ReAct (Reasoning + Action) approach.
    
    {iteration_info}
    
    {tools_info}
    
    For each step:
    1. THINK: Analyze the problem and what information you have so far
    2. PLAN: Decide what tool to use next or if you have the final answer
    
    If you need to use a tool, format your response as:
    THOUGHT: <your reasoning>
    ACTION: <tool_name>
    ACTION_INPUT: <input for the tool>
    
    If you have the final answer, format your response as:
    THOUGHT: <your reasoning>
    FINAL_ANSWER: <your final answer to the original question>
    
    Always use these exact formats so I can parse your response correctly.
    """)
    
    # Create the full prompt with history
    full_messages = [system_message] + messages
    
    # Get the agent's response
    response = llm.invoke(full_messages)
    
    # Add the AI's response to messages
    return {
        "messages": [AIMessage(content=response.content)],
        "iteration_count": state["iteration_count"] + 1
    }


# Node 2: Parse agent output and decide whether to continue or finish
def process_agent_response(state: ReActAgentState) -> Dict:
    """Parse the agent's response and determine next steps."""
    # Get the latest agent message
    latest_message = state["messages"][-1].content
    
    # Check for FINAL_ANSWER
    if "FINAL_ANSWER:" in latest_message:
        # Extract the final answer
        final_answer_parts = latest_message.split("FINAL_ANSWER:")
        if len(final_answer_parts) > 1:
            final_answer = final_answer_parts[1].strip()
            return {"final_answer": final_answer}
    
    # If not final answer, extract ACTION and ACTION_INPUT
    if "ACTION:" in latest_message and "ACTION_INPUT:" in latest_message:
        # Get the action name
        action_parts = latest_message.split("ACTION:")
        action_and_rest = action_parts[1].strip().split("\n")[0].strip()
        
        # Get the action input
        input_parts = latest_message.split("ACTION_INPUT:")
        action_input = input_parts[1].strip().split("\n")[0].strip()
        
        return {
            "actions_taken": [f"{action_and_rest}({action_input})"]
        }
    
    # If neither ACTION nor FINAL_ANSWER is found, default to ending the cycle
    return {"final_answer": "I couldn't determine what to do next. Please rephrase your question."}


# Node 3: Execute the tool selected by the agent
def execute_tool(state: ReActAgentState) -> Dict:
    """Execute the tool specified by the agent."""
    # Get the latest agent message
    latest_message = state["messages"][-1].content
    
    # Extract ACTION and ACTION_INPUT
    action_name = None
    action_input = None
    
    if "ACTION:" in latest_message and "ACTION_INPUT:" in latest_message:
        # Get the action name
        action_parts = latest_message.split("ACTION:")
        action_name = action_parts[1].strip().split("\n")[0].strip()
        
        # Get the action input
        input_parts = latest_message.split("ACTION_INPUT:")
        action_input = input_parts[1].strip().split("\n")[0].strip()
    
    # If we couldn't parse the action, return an error
    if not action_name or not action_input:
        return {
            "messages": [FunctionMessage(
                content="Error: Could not parse action and input from your response.",
                name="system"
            )]
        }
    
    # Find the matching tool
    tool_to_use = None
    for tool in tools:
        if tool.name == action_name:
            tool_to_use = tool
            break
    
    # If we couldn't find the tool, return an error
    if not tool_to_use:
        return {
            "messages": [FunctionMessage(
                content=f"Error: Tool '{action_name}' not found. Available tools: {', '.join(t.name for t in tools)}",
                name="system"
            )]
        }
    
    # Execute the tool
    try:
        result = tool_to_use.invoke(action_input)
        return {
            "messages": [FunctionMessage(
                content=result,
                name=action_name
            )]
        }
    except Exception as e:
        return {
            "messages": [FunctionMessage(
                content=f"Error executing {action_name}({action_input}): {str(e)}",
                name="system"
            )]
        }


# Create a ReAct agent graph with cycles
def create_react_agent():
    """Create a ReAct agent graph with cycles."""
    # Initialize the graph
    graph = StateGraph(ReActAgentState)
    
    # Add nodes
    graph.add_node("reasoning", agent_reasoning)
    graph.add_node("process_response", process_agent_response)
    graph.add_node("execute_tool", execute_tool)
    
    # Set the entry point
    graph.set_entry_point("reasoning")
    
    # Add regular edge from reasoning to process_response
    graph.add_edge("reasoning", "process_response")
    
    # Define the condition function for routing
    def should_continue(state: ReActAgentState) -> str:
        """Determine if we should continue the cycle or end."""
        # Check if we have a final answer
        if state.get("final_answer"):
            return "end"
        
        # Check if we've reached the maximum number of iterations
        # This is a safety measure to prevent infinite loops
        if state.get("iteration_count", 0) >= 10:
            return "end"
        
        # If we don't have a final answer and haven't reached the limit, continue
        return "continue"
    
    # Add conditional edges from process_response
    # FIXED: Changed from source_node to just using the node directly as first argument
    graph.add_conditional_edges(
        "process_response",  # Changed from source_node="process_response"
        should_continue,     # Changed from condition_function=should_continue
        {
            "continue": "execute_tool",
            "end": END
        }
    )
    
    # Create the cycle by connecting execute_tool back to reasoning
    graph.add_edge("execute_tool", "reasoning")
    
    # Compile the graph with a recursion limit for safety
    return graph.compile()


# Example usage
if __name__ == "__main__":
    # Create the ReAct agent
    react_agent = create_react_agent()
    
    # Sample questions that require multi-step reasoning
    questions = [
        "What is Python programming language? Summarize in 3 bullet points.",
        "If I have 5 apples and give away 2, then buy 3 more, how many do I have? Show your calculations.",
        "What is LangGraph and how does it relate to LangChain?"
    ]
    
    # Run each question and demonstrate the cycle
    for i, question in enumerate(questions):
        print(f"\n\n{'='*50}")
        print(f"QUESTION {i+1}: {question}")
        print(f"{'='*50}")
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "actions_taken": [],
            "iteration_count": 0,
            "final_answer": None
        }
        
        # Set recursion limit and thread ID for this run
        config = {"recursion_limit": 10}
        
        # Use stream to see intermediate steps
        print("SOLVING (watch the reasoning-action cycle):")
        print("-" * 40)
        
        for chunk in react_agent.stream(initial_state, config=config):
            # If this is a node transition, print it
            if chunk.get("steps"):
                for step in chunk["steps"]:
                    if step[0] == "start":
                        print(f"\n>> ENTERING NODE: {step[1]}")
                    elif step[0] == "end":
                        print(f">> EXITING NODE: {step[1]}")
            
            # If this is a state update with new messages, print them
            if chunk.get("messages") and chunk["messages"] != initial_state.get("messages", []):
                latest_message = chunk["messages"][-1]
                print(f"\n{latest_message.type.upper()}: {latest_message.content}")
        
        # Get the final state
        final_state = react_agent.invoke(initial_state, config=config)
        
        # Print the final answer and stats
        print("\n" + "=" * 50)
        print(f"ITERATIONS: {final_state['iteration_count']}")
        print(f"ACTIONS TAKEN: {', '.join(final_state['actions_taken'])}")
        print(f"FINAL ANSWER: {final_state.get('final_answer', 'No final answer provided')}")
        print("=" * 50)