# This file demonstrates three examples of defining and using AgentState in LangGraph.

from typing import TypedDict, Annotated, Sequence, Dict, Any
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

# --- Example 2: State with Task Tracking ---
# Adds task_id, retries, and is_complete. Graph includes a retry mechanism.


class TaskAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_id: str
    retries: Annotated[int, operator.add]
    is_complete: bool


def init_task_node(state: TaskAgentState) -> TaskAgentState:
    return {
        'task_id': 'task_123',
        'retries': 0,
        'is_complete': False,
        'messages': [AIMessage(content='Task initialized.')],
    }


def process_node(state: TaskAgentState) -> TaskAgentState:
    if state['retries'] < 2:
        # Simulate failure and retry
        return {'retries': 1, 'messages': [AIMessage(content=f'Processing... Retry {state["retries"] + 1}')]}
    else:
        return {'is_complete': True, 'messages': [AIMessage(content='Task completed!')]}


def check_complete(state: TaskAgentState) -> str:
    return END if state['is_complete'] else 'process'


# Build the graph
task_workflow = StateGraph(state_schema=TaskAgentState)
task_workflow.add_node('init', init_task_node)
task_workflow.add_node('process', process_node)
task_workflow.add_edge('init', 'process')
task_workflow.add_conditional_edges('process', check_complete, {'process': 'process', END: END})
task_workflow.set_entry_point('init')

# Compile and run example
task_graph = task_workflow.compile()
initial_state = {'messages': [HumanMessage(content='Start task')]}
print('\nExample 2 Output - Task Status:')
print(task_graph.invoke(initial_state))

# --- Example 3: Complex State with Nested Data ---
# Includes nested SubTaskState for subtasks. Graph processes subtasks and summarizes.


class SubTaskState(TypedDict):
    subtask_name: str
    result: Dict[str, Any]  # Stores arbitrary results from subtasks


class ComplexAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    subtasks: Annotated[Sequence[SubTaskState], operator.add]
    overall_summary: str


def subtask_node(state: ComplexAgentState) -> ComplexAgentState:
    # Add a subtask
    new_subtask = SubTaskState(subtask_name='subtask1', result={'data': 'Processed data'})
    return {'subtasks': [new_subtask], 'messages': [AIMessage(content='Subtask added.')]}


def summarize_node(state: ComplexAgentState) -> ComplexAgentState:
    # Summarize subtasks
    summary = 'Summary: ' + ', '.join([st['subtask_name'] for st in state['subtasks']])
    return {'overall_summary': summary, 'messages': [AIMessage(content='Summarized.')]}


# Build the graph
complex_workflow = StateGraph(state_schema=ComplexAgentState)
complex_workflow.add_node('subtask', subtask_node)
complex_workflow.add_node('summarize', summarize_node)
complex_workflow.add_edge('subtask', 'summarize')
complex_workflow.add_edge('summarize', END)
complex_workflow.set_entry_point('subtask')

# Compile and run example
complex_graph = complex_workflow.compile()
initial_state = {'messages': [HumanMessage(content='Run complex task')]}
print('\nExample 3 Output - Complex State with Nested Data:')
print(complex_graph.invoke(initial_state))
