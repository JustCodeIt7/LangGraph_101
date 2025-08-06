#!/usr/bin/env python3
"""
Basic Human-in-the-Loop Approval Workflow

This example demonstrates:
1. A simple agent that needs human approval before executing sensitive actions
2. Basic interrupt mechanism for human review
3. Simple approve/deny workflow

Perfect for understanding core human-in-the-loop concepts.
"""

import json
from typing import TypedDict, Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    approved: bool


# Define a sensitive tool that requires human approval
@tool
def transfer_money(amount: float, recipient: str) -> str:
    """Transfer money to a recipient. Requires human approval."""
    return f'Transferred ${amount} to {recipient}'


# Define a safe tool that doesn't require approval
@tool
def check_balance() -> str:
    """Check account balance."""
    return 'Current balance: $10,000'


# Create tool list
tools = [transfer_money, check_balance]


# Node functions
def call_model(state: State):
    """Call the AI model."""
    system_message = SystemMessage(
        content='You are a banking assistant. You can check balances and transfer money. '
        'For transfers, you must get human approval. Use tools appropriately.'
    )

    # Add system message if not present
    messages = state['messages']
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [system_message] + messages

    model = ChatOpenAI(model='gpt-4o')
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {'messages': [response]}


def call_tool(state: State):
    """Call tools based on model response."""
    messages = state['messages']
    last_message = messages[-1]

    # Check if this is an approval request
    if 'transfer_money' in last_message.content and 'approval' in last_message.content.lower():
        return {'messages': [AIMessage(content='Waiting for human approval...')], 'approved': False}

    # Execute tool if approved or if it's a safe tool
    for tool_call in last_message.tool_calls:
        if tool_call['name'] == 'check_balance':
            result = check_balance.invoke(tool_call['args'])
            tool_message = ToolMessage(content=result, name=tool_call['name'], tool_call_id=tool_call['id'])
            return {'messages': [tool_message]}
        elif tool_call['name'] == 'transfer_money' and state.get('approved', False):
            result = transfer_money.invoke(tool_call['args'])
            tool_message = ToolMessage(content=result, name=tool_call['name'], tool_call_id=tool_call['id'])
            return {'messages': [tool_message]}

    # If we get here, we need approval
    return {'messages': [AIMessage(content='This action requires human approval.')]}


# Human approval function
def human_approval(state: State):
    """Handle human approval."""
    print('\n--- HUMAN APPROVAL REQUIRED ---')
    print('AI wants to perform an action that requires your approval.')
    print('Last AI message:', state['messages'][-1].content)

    # In a real app, this would be a UI prompt
    # For this example, we'll simulate approval
    approval = input('Do you approve? (yes/no): ').strip().lower()

    if approval in ['yes', 'y']:
        print('Action approved!')
        return {'approved': True}
    else:
        print('Action denied!')
        return {'approved': False, 'messages': [HumanMessage(content='User denied the action.')]}


# Router functions
def route_model_output(state: State):
    """Route based on model output."""
    messages = state['messages']
    last_message = messages[-1]

    # If the model is asking for approval
    if 'approval' in last_message.content.lower() or 'requires approval' in last_message.content.lower():
        return 'human_approval'

    # If the model called tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tools'

    # If we're waiting for approval
    if not state.get('approved', True) and 'transfer_money' in str(last_message.content):
        return 'human_approval'

    # Otherwise end
    return 'end'


def route_after_approval(state: State):
    """Route after human approval."""
    if state.get('approved', False):
        return 'tools'
    else:
        return 'model'


# Create the graph
graph = StateGraph(State)

# Add nodes
graph.add_node('model', call_model)
graph.add_node('tools', call_tool)
graph.add_node('human_approval', human_approval)

# Add edges
graph.add_edge('tools', 'model')
graph.add_edge('human_approval', 'model')

# Add conditional edges
graph.add_conditional_edges(
    'model', route_model_output, {'tools': 'tools', 'human_approval': 'human_approval', 'end': END}
)

# Set entry point
graph.set_entry_point('model')

# Compile graph with memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# Example usage
if __name__ == '__main__':
    print('=== Basic Human Approval Example ===')
    print('Ask me to check your balance or transfer money!')

    # Start conversation
    config = {'configurable': {'thread_id': '1'}}

    while True:
        user_input = input('\nYou: ').strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # Process user input
        result = app.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)

        # Print last message
        last_message = result['messages'][-1]
        print(f'Assistant: {last_message.content}')
