import os
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph, END

# --- LLM Setup ---
# This assumes Ollama is running locally with the llama3.2 model.
# Make sure to run `ollama pull llama3.2` in your terminal first.
llm = ChatLiteLLM(model='ollama/llama3.2', temperature=0)
class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- 3. Graph Node Definitions ---
# These are the functions that will be called as nodes in our graph.


def call_model(state: AgentState):
    """Invokes the LLM to get a response or decide on a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # We return a dictionary with the new message to add to the state
    return {'messages': [response]}


def call_tool(state: AgentState):
    """Executes the tool chosen by the LLM."""
    last_message = state['messages'][-1]  # This will be the AIMessage with tool_calls
    tool_calls = last_message.tool_calls

    # The ToolExecutor runs the tools and returns the output
    tool_outputs = llm_with_tools.tool_executor.invoke(tool_calls)

    # We create ToolMessage instances to represent the tool's output
    tool_messages = [
        ToolMessage(content=str(output), tool_call_id=call['id']) for call, output in zip(tool_calls, tool_outputs)
    ]
    return {'messages': tool_messages}
# --- 4. Conditional Edge Logic ---
def should_continue(state: AgentState):
    """Determines the next step: call a tool or finish."""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        # If the LLM made a tool call, we route to the 'call_tool' node
        return 'call_tool'
    else:
        # Otherwise, we're done
        return END
# --- 1. Tool Definitions ---
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers 'a' and 'b'."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers 'a' and 'b'."""
    return a * b


# Create a list of our tools
tools = [add, multiply]

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# --- 2. Agent State & Nodes (Re-used) ---
# The AgentState, call_model, call_tool, and should_continue functions
# from Example 1 are identical and can be reused here without changes.

# --- 3. Graph Construction (Identical to Example 1) ---
workflow = StateGraph(AgentState)
workflow.add_node('agent', call_model)
workflow.add_node('call_tool', call_tool)
workflow.set_entry_point('agent')
workflow.add_conditional_edges('agent', should_continue, {'call_tool': 'call_tool', END: END})
workflow.add_edge('call_tool', 'agent')

app = workflow.compile()

# --- 4. Run the Graph ---
print('\n--- Example 2: Multiple Tools ---')
inputs = {'messages': [HumanMessage(content='What is 6 multiplied by 9?')]}
for event in app.stream(inputs):
    for k, v in event.items():
        if k != '__end__':
            print(v)