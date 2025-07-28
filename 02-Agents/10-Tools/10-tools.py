"""
LangGraph Tools Tutorial: Three Examples of Agent Tool Usage

This tutorial demonstrates how to create LangGraph agents that can use tools to extend their capabilities.
We'll cover three progressively complex examples:
1. Agent with a single calculator tool
2. Agent with a single search tool  
3. Agent with multiple tools that can intelligently choose which tool to use

Requirements:
- langgraph
- langchain-core
- litellm

To run: python 10-tools.py
"""

import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
# from litellm import ChatLiteLLM
from langchain_litellm import ChatLiteLLM

# =============================================================================
## Tool Definitions
# =============================================================================

@tool
def add_calculator(a: int, b: int) -> int:
    """Add two integers together.
    
    Args:
        a: First integer to add
        b: Second integer to add
        
    Returns:
        The sum of a and b
    """
    return a + b


@tool
def search_tool(query: str) -> str:
    """Search for information about a given query.
    
    This is a mock search tool that returns predefined information.
    In a real implementation, this would connect to a search API.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as a string
    """
    # Mock search results for common queries
    search_results = {
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows.",
        "python": "Python is a high-level, general-purpose programming language known for its simplicity and readability.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn.",
        "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."
    }
    
    # Simple keyword matching for demonstration
    query_lower = query.lower()
    for key, value in search_results.items():
        if key in query_lower:
            return f"Search results for '{query}': {value}"
    
    return f"Search results for '{query}': No specific information found, but this is a general search result."


# =============================================================================
## State Definition
# =============================================================================

class AgentState(TypedDict):
    """State definition for our LangGraph agents.
    
    This defines the structure of data that flows through the graph.
    """
    # The list of messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]


# =============================================================================
## Helper Functions
# =============================================================================

def create_agent_node(llm_with_tools, system_message: str):
    """Create an agent node that can call tools.
    
    Args:
        llm_with_tools: LLM instance with tools bound to it
        system_message: System message to guide the agent's behavior
        
    Returns:
        A function that processes the agent state
    """
    def agent(state: AgentState):
        # Add system message if this is the first message
        messages = state["messages"]
        if not messages or not any(isinstance(msg, HumanMessage) and msg.content.startswith("System:") for msg in messages):
            system_msg = HumanMessage(content=f"System: {system_message}")
            messages = [system_msg] + list(messages)
        
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    return agent


def should_continue(state: AgentState) -> str:
    """Determine whether to continue to tools or end the conversation.
    
    Args:
        state: Current agent state
        
    Returns:
        "tools" if the last message has tool calls, "end" otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"


def create_graph(llm_with_tools, system_message: str):
    """Create a LangGraph workflow for an agent with tools.
    
    Args:
        llm_with_tools: LLM instance with tools bound to it
        system_message: System message to guide the agent's behavior
        
    Returns:
        Compiled graph ready for execution
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", create_agent_node(llm_with_tools, system_message))
    workflow.add_node("tools", ToolNode(llm_with_tools.bound_tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()


def run_example(graph, user_input: str, example_name: str):
    """Run an example with the given graph and user input.
    
    Args:
        graph: Compiled LangGraph workflow
        user_input: User's question or request
        example_name: Name of the example for display purposes
    """
    print(f"\n{'-'*60}")
    print(f"Running {example_name}")
    print(f"User Input: {user_input}")
    print(f"{'-'*60}")
    
    # Run the graph
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    
    # Print the final response
    final_message = result["messages"][-1]
    print(f"Agent Response: {final_message.content}")


# =============================================================================
## Example 1: Agent with Calculator Tool
# =============================================================================

def example_1_calculator_agent():
    """
    Example 1: Agent with Calculator Tool
    
    This example demonstrates an agent that can perform arithmetic calculations
    using a simple add_calculator tool. The agent can understand mathematical
    questions and use the tool to provide accurate answers.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: AGENT WITH CALCULATOR TOOL")
    print("="*80)
    print("This agent can perform addition calculations using the add_calculator tool.")
    
    # Initialize the LLM
    llm = ChatLiteLLM(model_name='ollama/llama3.2', temperature=0)
    
    # Bind the calculator tool to the LLM
    llm_with_tools = llm.bind_tools([add_calculator])
    
    # System message to guide the agent's behavior
    system_message = (
        "You are a helpful assistant that can perform mathematical calculations. "
        "When users ask mathematical questions involving addition, use the add_calculator tool. "
        "Be precise and explain your reasoning."
    )
    
    # Create the graph
    graph = create_graph(llm_with_tools, system_message)
    
    # Test cases for the calculator agent
    test_cases = [
        "What is 15 + 27?",
        "Can you add 100 and 250?",
        "I need to calculate 42 plus 58",
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        run_example(graph, test_case, f"Calculator Test {i}")


# =============================================================================
## Example 2: Agent with Search Tool
# =============================================================================

def example_2_search_agent():
    """
    Example 2: Agent with Search Tool
    
    This example demonstrates an agent that can search for information
    using a mock search tool. The agent can understand information requests
    and use the search tool to provide relevant answers.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: AGENT WITH SEARCH TOOL")
    print("="*80)
    print("This agent can search for information using the search_tool.")
    
    # Initialize the LLM
    llm = ChatLiteLLM(model_name='ollama/llama3.2', temperature=0)
    
    # Bind the search tool to the LLM
    llm_with_tools = llm.bind_tools([search_tool])
    
    # System message to guide the agent's behavior
    system_message = (
        "You are a knowledgeable assistant that can search for information. "
        "When users ask questions about topics, use the search_tool to find relevant information. "
        "Provide comprehensive answers based on the search results."
    )
    
    # Create the graph
    graph = create_graph(llm_with_tools, system_message)
    
    # Test cases for the search agent
    test_cases = [
        "What is LangGraph?",
        "Tell me about Python programming",
        "What is artificial intelligence?",
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        run_example(graph, test_case, f"Search Test {i}")


# =============================================================================
## Example 3: Agent with Multiple Tools
# =============================================================================

def example_3_multi_tool_agent():
    """
    Example 3: Agent with Multiple Tools
    
    This example demonstrates an agent that can intelligently choose between
    multiple tools based on the user's request. The agent has access to both
    the calculator tool and the search tool, and will decide which one to use
    (or whether to use any tool at all) based on the context of the question.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: AGENT WITH MULTIPLE TOOLS")
    print("="*80)
    print("This agent can intelligently choose between calculator and search tools.")
    
    # Initialize the LLM
    llm = ChatLiteLLM(model_name='ollama/llama3.2', temperature=0)
    
    # Bind both tools to the LLM
    llm_with_tools = llm.bind_tools([add_calculator, search_tool])
    
    # System message to guide the agent's behavior
    system_message = (
        "You are a versatile assistant with access to multiple tools. "
        "You can perform calculations using the add_calculator tool and search for information using the search_tool. "
        "Analyze each user request carefully and choose the appropriate tool(s):\n"
        "- Use add_calculator for arithmetic addition problems\n"
        "- Use search_tool for information queries about topics, concepts, or general knowledge\n"
        "- For simple questions that don't require tools, answer directly\n"
        "Always explain your reasoning and provide helpful, accurate responses."
    )
    
    # Create the graph
    graph = create_graph(llm_with_tools, system_message)
    
    # Test cases that demonstrate tool selection
    test_cases = [
        # Mathematical questions (should use calculator)
        "What is 25 + 37?",
        "Calculate 150 plus 275",
        
        # Information questions (should use search)
        "What is machine learning?",
        "Tell me about LangGraph",
        
        # Mixed/ambiguous questions (agent should decide)
        "Hello, how are you?",  # Should answer directly
        "What's 5 + 5 and also tell me about AI?",  # Might use both tools
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        run_example(graph, test_case, f"Multi-Tool Test {i}")


# =============================================================================
## Main Execution
# =============================================================================

def main():
    """
    Main function to run all three examples.
    
    This will demonstrate the progression from simple single-tool agents
    to more sophisticated multi-tool agents that can make intelligent
    decisions about which tools to use.
    """
    print("LangGraph Tools Tutorial")
    print("========================")
    print("This tutorial demonstrates three examples of LangGraph agents using tools:")
    print("1. Agent with calculator tool only")
    print("2. Agent with search tool only") 
    print("3. Agent with multiple tools that can choose intelligently")
    
    try:
        # Run all examples
        example_1_calculator_agent()
        example_2_search_agent()
        example_3_multi_tool_agent()
        
        print("\n" + "="*80)
        print("TUTORIAL COMPLETE")
        print("="*80)
        print("You've seen how to:")
        print("• Create simple tools using the @tool decorator")
        print("• Bind tools to LLM instances")
        print("• Build LangGraph workflows that incorporate tools")
        print("• Create agents that can intelligently select between multiple tools")
        print("• Handle tool execution and responses in a conversational flow")
        
    except Exception as e:
        print(f"\nError running tutorial: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install langgraph langchain-core litellm")


if __name__ == "__main__":
    main()