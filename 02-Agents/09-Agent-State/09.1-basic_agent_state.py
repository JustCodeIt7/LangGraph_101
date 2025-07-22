# %%
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from rich import print
from langchain_community.llms import Ollama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama, OllamaEmbeddings
# %%
# --- Example 1: Basic Message State ---
# This graph manages a sequence of messages, simulating a simple thought-response flow.


# Defines the state for the agent, which is a sequence of messages.
class BasicAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Appends new messages to the list.


# Node to simulate the agent "thinking."
def think_node(state: BasicAgentState) -> BasicAgentState:
    """Adds an AI message representing the agent's thought process."""
    # Uses the default Ollama model, llama3.2, as specified by James Brendamour.
    llm = ChatOllama(model='llama3.2')
    # Use .invoke() and pass a list of messages
    new_message = AIMessage(content=llm.invoke([HumanMessage(content="I'm thinking about your query...")]).content)
    return {'messages': [new_message]}


# Node to simulate the agent "responding."
def respond_node(state: BasicAgentState) -> BasicAgentState:
    """Generates a response based on the last message in the state."""
    # Uses the default ChatOllama model, llama3.2, as specified by James Brendamour.
    llm = ChatOllama(model='llama3.2')
    last_message = state['messages'][-1].content if state['messages'] else ''
    # Use .invoke() and pass a list of messages
    response = AIMessage(content=llm.invoke([HumanMessage(content=f'Response to: {last_message}')]).content)
    return {'messages': [response]}


# Build the graph using StateGraph.
basic_workflow = StateGraph(state_schema=BasicAgentState)

# Add nodes to the graph.
basic_workflow.add_node('think', think_node)
basic_workflow.add_node('respond', respond_node)

# Define the flow: 'think' node leads to 'respond' node.
basic_workflow.add_edge('think', 'respond')
# The 'respond' node leads to the end of the graph execution.
basic_workflow.add_edge('respond', END)

# Set 'think' as the starting point of the graph.
basic_workflow.set_entry_point('think')

# Add memory to the graph to persist state across invocations.
basic_workflow.checkpointer = InMemorySaver()

# Compile the graph for execution.
basic_graph = basic_workflow.compile()

# display the graph
# Generates and saves a Mermaid diagram of the graph for visualization.
diagram = basic_graph.get_graph().draw_mermaid_png()
print(basic_graph.get_graph().draw_ascii())  # Prints an ASCII representation to console.
with open('g01_diagram.png', 'wb') as f:
    f.write(diagram)
print('Saved g01_diagram.png')


initial_state = {'messages': [HumanMessage(content='Hello!')]}
print('Example 1 Output - Basic Message State:')
# Invoke the graph with an initial message and print the final state.
print(basic_graph.invoke(initial_state))

# %%
