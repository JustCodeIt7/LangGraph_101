# main.py
import os
import uuid
from typing import List, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


# --- 1. Define a simple tool the agent can use ---
# This tool simulates searching for information. In a real application,
# this could be a database lookup, an API call, or a web search.
@tool
def search(query: str):
    """
    Simulates searching for information about a query.
    For this example, it returns a fixed string, but in a real-world
    scenario, this would perform an actual search.
    """
    print(f"\n-- Executing Search Tool with query: '{query}' --")
    return f"Information about '{query}' was found."

tools = [search]
tool_node = ToolNode(tools)

# --- 2. Set up the language model ---
# We'll use OpenAI's model for this example. Make sure to set your
# OPENAI_API_KEY environment variable.
# You can replace this with any other LangChain-compatible model.
try:
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    model = model.bind_tools(tools)
except Exception as e:
    print(f"Error initializing OpenAI model: {e}")
    print("Please make sure your OPENAI_API_KEY is set correctly.")
    # Fallback to a dummy model if API key is not set
    class DummyModel:
        def invoke(self, messages):
            print("\n-- Using Dummy Model --")
            ai_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": "search",
                    "args": {"query": "latest AI trends"},
                    "id": "tool_call_123"
                }]
            )
            return ai_message
    model = DummyModel()


# --- 3. Define the Agent State ---
# This is the state that will be passed between nodes in our graph.
# It's a dictionary that must contain a "messages" key, which is a list.
class AgentState(TypedDict):
    messages: List[BaseMessage]


# --- 4. Define the Agent Node ---
# This function is the core of our agent. It takes the current state
# (the conversation history) and calls the model to decide the next step.
def call_model(state: AgentState):
    """
    The primary node for the agent. It invokes the language model
    with the current conversation history and returns an AIMessage.
    """
    print("\n-- Calling Model --")
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a dictionary with the model's response appended to the messages list
    return {"messages": [response]}


# --- 5. Define the Conditional Logic ---
# This function determines the next step after the model has been called.
# It checks if the model's last message contains tool calls.
def should_continue(state: AgentState):
    """
    This is our conditional edge. It decides whether to:
    a) Continue to the tool node if the model produced a tool call.
    b) End the graph if the model did not produce a tool call.
    """
    print("\n-- Checking for Tool Calls --")
    last_message = state["messages"][-1]
    # If there are no tool calls, we end the conversation.
    if not last_message.tool_calls:
        print("  - Decision: End of conversation.")
        return "end"
    # Otherwise, we continue to the tool node.
    else:
        print("  - Decision: Call tools.")
        return "continue"


# --- 6. Construct the Graph ---
# This is where we wire all our components together.
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entry point of the graph
workflow.set_entry_point("agent")

# Add the conditional edge
# This edge routes the flow from the 'agent' node to either the 'action' node or the 'end'
# based on the output of the 'should_continue' function.
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# Add a regular edge
# This edge connects the 'action' node back to the 'agent' node, creating a loop.
workflow.add_edge("action", "agent")

# --- 7. Compile the graph and implement the Human-in-the-Loop ---
# We compile the graph into a runnable object. We also specify where to
# interrupt the execution. In this case, we want to pause *before* the
# 'action' node is executed, allowing a human to approve the tool call.
app = workflow.compile(interrupt_before=["action"])


# --- 8. Run the application ---
def run_human_in_the_loop():
    """
    Executes the main application loop, handling user input,
    graph execution, and the human approval step.
    """
    print("Human-in-the-Loop LangGraph Agent")
    print("Type 'quit' to exit.")
    print("-" * 50)

    # A unique identifier for the conversation thread
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Exiting.")
            break

        # Initial state for the graph
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # Invoke the graph. It will run until it hits an interruption point
        # or finishes.
        state = app.invoke(inputs, config)

        # The graph has paused because it's about to execute the 'action' node.
        # We can inspect the state to see what the agent plans to do.
        if state and state.get("messages") and state["messages"][-1].tool_calls:
            tool_call = state["messages"][-1].tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print("\n-- HUMAN-IN-THE-LOOP --")
            print(f"Agent wants to call the '{tool_name}' tool with arguments: {tool_args}")

            # Ask for human approval
            approval = input("Do you approve? (yes/no): ").lower()

            if approval == "yes":
                print("  - You approved. Resuming execution.")
                # If approved, we continue the graph's execution.
                # The ToolNode will now run.
                state = app.invoke(None, config)
            else:
                print("  - You denied. Modifying the state.")
                # If denied, we don't proceed with the tool call.
                # Instead, we inject a new message into the state to inform
                # the agent that the request was denied.
                tool_message = ToolMessage(
                    content="The user denied this tool call. Please ask for clarification or try something else.",
                    tool_call_id=tool_call["id"],
                )
                # We resume execution with this modified state.
                state = app.invoke({"messages": [tool_message]}, config)

        # Print the final response from the agent
        if state and state.get("messages"):
            final_response = state["messages"][-1]
            if final_response.content:
                print(f"\nAgent: {final_response.content}")
        print("-" * 50)

# --- Entry point of the script ---
if __name__ == "__main__":
    # To run this, you need to have langchain, langgraph, and langchain-openai installed:
    # pip install langchain langgraph langchain-openai
    #
    # You also need to set your OpenAI API key as an environment variable:
    # export OPENAI_API_KEY='your-api-key-here'

    # Set a dummy key if it's not present, to avoid errors on import
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "key-not-set"

    run_human_in_the_loop()
