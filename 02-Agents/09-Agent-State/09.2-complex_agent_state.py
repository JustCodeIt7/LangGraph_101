from typing import TypedDict, Annotated, Sequence, Dict, Any
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from rich import print

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