from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from rich import print

# --- Example 1: Basic Message State ---
# This state just tracks messages. The graph has two nodes: one to "think" and one to "respond".


class BasicAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def think_node(state: BasicAgentState) -> BasicAgentState:
    # Simulate thinking by adding an AI message
    new_message = AIMessage(content="I'm thinking about your query...")
    return {'messages': [new_message]}


def respond_node(state: BasicAgentState) -> BasicAgentState:
    # Respond based on the last message
    last_message = state['messages'][-1].content if state['messages'] else ''
    response = AIMessage(content=f'Response to: {last_message}')
    return {'messages': [response]}


# Build the graph
basic_workflow = StateGraph(state_schema=BasicAgentState)
basic_workflow.add_node('think', think_node)
basic_workflow.add_node('respond', respond_node)
basic_workflow.add_edge('think', 'respond')
basic_workflow.add_edge('respond', END)
basic_workflow.set_entry_point('think')

# Compile and run example
basic_graph = basic_workflow.compile()
initial_state = {'messages': [HumanMessage(content='Hello!')]}
print('Example 1 Output - Basic Message State:')
print(basic_graph.invoke(initial_state))