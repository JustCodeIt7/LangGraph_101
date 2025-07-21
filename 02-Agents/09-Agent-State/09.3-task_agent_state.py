from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from rich import print
from typing import TypedDict, Annotated, Sequence

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