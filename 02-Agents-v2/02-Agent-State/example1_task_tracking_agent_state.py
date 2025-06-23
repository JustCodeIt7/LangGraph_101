"""
Example 1: Task Tracking Agent State
====================================

This example demonstrates how to use LangGraph agent state to track and manage
multi-step tasks with progress tracking, subtask management, and completion status.

Key Features:
- Task queue management
- Progress tracking with percentages
- Subtask creation and completion
- Priority-based task handling
- Task status tracking (pending, in_progress, completed, failed)
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

class Task(TypedDict):
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: int  # 1-5, where 5 is highest
    created_at: str
    completed_at: Optional[str]
    subtasks: List[str]  # List of subtask IDs
    progress: float  # 0.0 to 1.0

class TaskTrackingAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    current_task_id: Optional[str]
    tasks: Dict[str, Task]  # Task ID -> Task
    task_queue: List[str]  # List of task IDs ordered by priority
    overall_progress: float  # Overall completion percentage
    active_goals: List[str]  # High-level goals being worked on
    session_stats: Dict[str, int]  # Statistics for current session

def task_analyzer_node(state: TaskTrackingAgentState):
    """
    Analyzes user input to identify tasks, update progress, or manage task queue.
    """
    print("---ANALYZING TASK REQUEST---")

    messages = state['messages']
    if not messages:
        return state

    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return state

    user_input = latest_message.content.lower()

    # Simple task detection logic (in real implementation, use NLP)
    if "new task" in user_input or "add task" in user_input:
        return _extracted_from_task_analyzer_node_20(state, user_input, latest_message)
    elif 'complete' in user_input and state.get('current_task_id'):
        # Mark current task as completed
        current_id = state['current_task_id']
        if current_id in state['tasks']:
            return _extracted_from_task_analyzer_node_53(state, current_id)
    return state


# TODO Rename this here and in `task_analyzer_node`
def _extracted_from_task_analyzer_node_53(state, current_id):
    updated_tasks = state['tasks'].copy()
    updated_tasks[current_id] = {
        **updated_tasks[current_id],
        'status': TaskStatus.COMPLETED,
        'progress': 1.0,
        'completed_at': datetime.now().isoformat(),
    }

    # Update session stats
    updated_stats = state['session_stats'].copy()
    updated_stats['completed'] = updated_stats.get('completed', 0) + 1

    print(f'Completed task: {current_id}')
    return {'tasks': updated_tasks, 'session_stats': updated_stats}


# TODO Rename this here and in `task_analyzer_node`
def _extracted_from_task_analyzer_node_20(state, user_input, latest_message):
    # Extract task details (simplified)
    task_id = f'task_{len(state["tasks"]) + 1}'
    new_task: Task = {
        'id': task_id,
        'title': f'Task from: {user_input[:50]}...',
        'description': latest_message.content,
        'status': TaskStatus.PENDING,
        'priority': 3,  # Default priority
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'subtasks': [],
        'progress': 0.0,
    }

    # Update state
    updated_tasks = state['tasks'].copy()
    updated_tasks[task_id] = new_task

    updated_queue = state['task_queue'].copy()
    updated_queue.append(task_id)
    # Sort by priority (higher priority first)
    updated_queue.sort(key=lambda tid: updated_tasks[tid]['priority'], reverse=True)

    print(f'Added new task: {task_id}')
    return {'tasks': updated_tasks, 'task_queue': updated_queue, 'current_task_id': task_id}


def progress_tracker_node(state: TaskTrackingAgentState):
    """
    Calculates and updates overall progress and task priorities.
    """
    print("---TRACKING PROGRESS---")
    
    total_tasks = len(state["tasks"])
    if total_tasks == 0:
        return {"overall_progress": 0.0}
    
    completed_tasks = sum(1 for task in state["tasks"].values() 
                         if task["status"] == TaskStatus.COMPLETED)
    
    overall_progress = completed_tasks / total_tasks
    
    # Update current task status if there's an active task
    current_id = state.get("current_task_id")
    if current_id and current_id in state["tasks"]:
        current_task = state["tasks"][current_id]
        if current_task["status"] == TaskStatus.PENDING:
            updated_tasks = state["tasks"].copy()
            updated_tasks[current_id] = {
                **current_task,
                "status": TaskStatus.IN_PROGRESS
            }
            
            print(f"Started working on task: {current_id}")
            return {
                "tasks": updated_tasks,
                "overall_progress": overall_progress
            }
    
    print(f"Overall progress: {overall_progress:.1%}")
    return {"overall_progress": overall_progress}

def task_assistant_node(state: TaskTrackingAgentState):
    """
    Provides AI assistance based on current task context and progress.
    """
    print("---TASK ASSISTANT RESPONDING---")

    messages = state['messages']
    current_task_id = state.get('current_task_id')
    overall_progress = state.get('overall_progress', 0.0)

    # Build context about current state
    context_info = [
        f'Overall Progress: {overall_progress:.1%}',
        f'Total Tasks: {len(state["tasks"])}',
    ]
    if current_task_id and current_task_id in state["tasks"]:
        current_task = state["tasks"][current_task_id]
        context_info.extend(
            (
                f'Current Task: {current_task["title"]}',
                f'Task Status: {current_task["status"].value}',
                f'Task Progress: {current_task["progress"]:.1%}',
            )
        )
    if pending_tasks := [task for task in state['tasks'].values() if task['status'] == TaskStatus.PENDING]:
        context_info.append(f"Pending Tasks: {len(pending_tasks)}")

    system_prompt = f"""
    You are a task management assistant. Here's the current context:
    
    {chr(10).join(context_info)}
    
    Help the user manage their tasks effectively. You can:
    - Provide guidance on current tasks
    - Suggest next steps
    - Help prioritize work
    - Break down complex tasks into subtasks
    - Offer progress updates and motivation
    
    Be concise and actionable in your responses.
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Create messages with system context
    context_messages = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(context_messages)
    print(f"Assistant Response: {response.content}")

    return {"messages": [AIMessage(content=response.content)]}

# Build the graph
workflow = StateGraph(TaskTrackingAgentState)

workflow.add_node("analyzer", task_analyzer_node)
workflow.add_node("tracker", progress_tracker_node)
workflow.add_node("assistant", task_assistant_node)

# Set up the flow
workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "tracker")
workflow.add_edge("tracker", "assistant")
workflow.add_edge("assistant", END)

# Compile the graph
task_tracking_agent = workflow.compile()

if __name__ == "__main__":
    print("Starting Task Tracking Agent...")
    print("Commands: 'new task [description]', 'complete', 'status', 'exit'")
    
    # Initialize state
    initial_state: TaskTrackingAgentState = {
        "messages": [],
        "current_task_id": None,
        "tasks": {},
        "task_queue": [],
        "overall_progress": 0.0,
        "active_goals": [],
        "session_stats": {"created": 0, "completed": 0}
    }
    
    current_state = initial_state
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            break
        
        if user_input.lower() == "status":
            print(f"\n--- TASK STATUS ---")
            print(f"Overall Progress: {current_state.get('overall_progress', 0):.1%}")
            print(f"Total Tasks: {len(current_state['tasks'])}")
            for task_id, task in current_state["tasks"].items():
                status_symbol = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌"}
                print(f"{status_symbol.get(task['status'].value, '?')} {task['title']} - {task['status'].value}")
            continue
        
        # Add user message to state
        current_state["messages"] = [HumanMessage(content=user_input)]
        
        # Run the graph
        result = task_tracking_agent.invoke(current_state)
        
        # Update current state with results
        current_state.update(result)
        
        # Display AI response
        if "messages" in result and result["messages"]:
            ai_message = result["messages"][-1]
            if isinstance(ai_message, AIMessage):
                print(f"Assistant: {ai_message.content}")
    
    print("\n--- SESSION SUMMARY ---")
    print(f"Tasks Created: {len(current_state['tasks'])}")
    print(f"Tasks Completed: {current_state['session_stats'].get('completed', 0)}")
    print(f"Final Progress: {current_state.get('overall_progress', 0):.1%}")