import os
from typing import Annotated, TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # Replace with your actual key

# Define the state
class State(TypedDict):
    messages: List[BaseMessage]
    ask_human: bool  # Flag to indicate if human input is needed

# Define the LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tool for demonstration (optional, can be expanded)
def dummy_tool(query: str) -> str:
    """A dummy tool that echoes the query."""
    return f"Echo: {query}"

tools = [dummy_tool]
model_with_tools = model.bind_tools(tools)

# System prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a helpful assistant. Respond to the user. "
         "If the user's message contains the word 'approve', set ask_human to False and continue. "
         "If you need human approval for something, set ask_human to True."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the agent node
def agent(state: State):
    chain = prompt | model_with_tools
    response = chain.invoke(state["messages"])
    ask_human = False
    if "need approval" in response.content.lower():  # Simple condition for demo
        ask_human = True
    return {
        "messages": [response],
        "ask_human": ask_human,
    }

# Define the human node (this will interrupt for human input)
def human_review(state: State):
    # In a real app, this would wait for human input via some interface.
    # For this basic example, we'll simulate it by printing and input.
    print("Human review needed!")
    human_input = input("Please provide your input or approval: ")
    return {
        "messages": [HumanMessage(content=human_input)],
        "ask_human": False,  # Reset after human input
    }

# Tool node
tool_node = ToolNode(tools)

# Define the graph
def should_ask_human(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)  # If tools are called, go to tools

workflow = StateGraph(state_schema=State)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.add_node("human", human_review)

# Edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_ask_human, {"human": "human", "tools": "tools", END: END})
workflow.add_edge("tools", "agent")
workflow.add_edge("human", "agent")  # After human, go back to agent

# Set up memory
memory = MemorySaver()

# Compile the graph
graph = workflow.compile(checkpointer=memory, interrupt_before=["human"])

# Function to run the graph
def run_conversation(user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initial input
    events = graph.stream({"messages": [HumanMessage(content=user_input)], "ask_human": False}, config, stream_mode="values")
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
        
        # Check for interruption
        if graph.get_state(config).next:
            # Interrupted before human node
            human_input = input("Human input required: ")
            # Continue with human input
            continue_events = graph.stream({
                "messages": [HumanMessage(content=human_input)],
                "ask_human": False
            }, config, stream_mode="values")
            for cont_event in continue_events:
                if "messages" in cont_event:
                    cont_event["messages"][-1].pretty_print()

# Example usage
if __name__ == "__main__":
    print("Starting Human-in-the-Loop LangGraph App")
    user_query = input("Enter your message: ")
    run_conversation(user_query)

# Note: This is a basic console-based example. In a production app, you'd integrate with a UI for human input.
# Total lines: Approximately 80 (excluding comments and blanks).