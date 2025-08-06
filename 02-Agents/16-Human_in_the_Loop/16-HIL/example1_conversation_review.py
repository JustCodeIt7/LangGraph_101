#!/usr/bin/env python3
"""
Human-in-the-Loop: Conversation Review and Correction

This example demonstrates:
1. AI agent that generates responses but requires human review
2. Human can edit, approve, or reject AI responses
3. Conversation continues based on human feedback
4. Perfect for customer service, content moderation, or quality control

Pattern: Review → Edit/Approve → Continue
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
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_review: bool
    review_status: str  # "pending", "approved", "edited", "rejected"
    human_feedback: str
    conversation_history: List[dict]


# Define tools for the agent
@tool
def search_knowledge_base(query: str) -> str:
    """Search knowledge base for information."""
    knowledge_base = {
        'refund policy': 'Our refund policy allows returns within 30 days of purchase.',
        'shipping': 'Standard shipping takes 3-5 business days. Express shipping is available.',
        'hours': 'We are open Monday-Friday 9AM-6PM EST.',
        'contact': 'You can reach us at support@example.com or call 1-800-123-4567.',
    }

    for key, value in knowledge_base.items():
        if key in query.lower():
            return value

    return "I don't have specific information about that in my knowledge base."


@tool
def create_ticket(subject: str, description: str) -> str:
    """Create a support ticket."""
    return f'Support ticket created: #{hash(subject + description) % 10000}. Subject: {subject}'


# Create tool list
tools = [search_knowledge_base, create_ticket]


# Node functions
def generate_response(state: ConversationState):
    """Generate AI response to customer query."""
    system_message = SystemMessage(
        content='You are a customer service agent. Help customers with their questions. '
        'Use the knowledge base search tool when you need information. '
        'Create tickets for complex issues that need follow-up. '
        'Be helpful, professional, and concise.'
    )

    messages = state['messages']
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [system_message] + messages

    model = ChatOpenAI(model='gpt-4o')
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    return {'messages': [response], 'pending_review': True, 'review_status': 'pending'}


def review_response(state: ConversationState):
    """Handle human review of AI response."""
    print('\n' + '=' * 50)
    print('HUMAN REVIEW REQUIRED')
    print('=' * 50)

    # Show conversation context
    print('\nConversation Context:')
    for i, msg in enumerate(state['messages'][-3:]):  # Show last 3 messages
        if isinstance(msg, HumanMessage):
            print(f'Customer: {msg.content}')
        elif isinstance(msg, AIMessage):
            print(f'AI Draft: {msg.content}')

    print('\nReview Options:')
    print('1. Approve - Send response as-is')
    print('2. Edit - Modify the response before sending')
    print("3. Reject - Don't send this response")

    choice = input('\nChoose option (1/2/3): ').strip()

    if choice == '1':
        print('✓ Response approved!')
        return {'review_status': 'approved', 'pending_review': False, 'human_feedback': 'Response approved by human'}
    elif choice == '2':
        print('\nCurrent AI response:')
        ai_response = state['messages'][-1].content
        print(f'"{ai_response}"')

        edited_response = input('\nEnter your edited response: ').strip()
        print('✓ Response edited!')

        return {
            'review_status': 'edited',
            'pending_review': False,
            'human_feedback': f'Human edited response: {edited_response}',
            'messages': [AIMessage(content=edited_response)],
        }
    elif choice == '3':
        reason = input('Reason for rejection: ').strip()
        print('✓ Response rejected!')

        return {
            'review_status': 'rejected',
            'pending_review': False,
            'human_feedback': f'Response rejected: {reason}',
            'messages': [HumanMessage(content=f'Please provide a different response. Reason: {reason}')],
        }
    else:
        print('Invalid choice. Defaulting to approval.')
        return {
            'review_status': 'approved',
            'pending_review': False,
            'human_feedback': 'Response approved by human (default)',
        }


def execute_tools(state: ConversationState):
    """Execute any tools called by the AI."""
    messages = state['messages']
    last_message = messages[-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {'messages': []}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        if tool_name == 'search_knowledge_base':
            result = search_knowledge_base.invoke(tool_args)
        elif tool_name == 'create_ticket':
            result = create_ticket.invoke(tool_args)
        else:
            result = f'Unknown tool: {tool_name}'

        tool_message = ToolMessage(content=result, name=tool_name, tool_call_id=tool_call['id'])
        tool_messages.append(tool_message)

    return {'messages': tool_messages}


def log_conversation(state: ConversationState):
    """Log the conversation for quality assurance."""
    last_message = state['messages'][-1]

    log_entry = {
        'timestamp': '2025-08-06T12:00:00Z',  # In real app, use actual timestamp
        'review_status': state['review_status'],
        'human_feedback': state['human_feedback'],
        'final_response': last_message.content if isinstance(last_message, AIMessage) else None,
    }

    # In a real app, save to database
    print(f'\n📝 Conversation logged: {json.dumps(log_entry, indent=2)}')

    return {'conversation_history': [log_entry]}


# Router functions
def route_after_generation(state: ConversationState):
    """Route after AI generates response."""
    if state.get('pending_review', False):
        return 'review'
    else:
        return 'end'


def route_after_review(state: ConversationState):
    """Route after human review."""
    status = state.get('review_status', '')

    if status == 'rejected':
        return 'generate'  # Generate new response
    elif status in ['approved', 'edited']:
        return 'log'  # Log and end
    else:
        return 'end'


def route_after_tools(state: ConversationState):
    """Route after tool execution."""
    # After tools, generate final response
    return 'generate'


# Create the graph
graph = StateGraph(ConversationState)

# Add nodes
graph.add_node('generate', generate_response)
graph.add_node('review', review_response)
graph.add_node('tools', execute_tools)
graph.add_node('log', log_conversation)

# Add edges
graph.add_edge('tools', 'generate')
graph.add_edge('log', END)

# Add conditional edges
graph.add_conditional_edges('generate', route_after_generation, {'review': 'review', 'end': END})

graph.add_conditional_edges('review', route_after_review, {'generate': 'generate', 'log': 'log', 'end': END})


# Check if tools were called and route accordingly
def route_from_start(state: ConversationState):
    """Route from start based on whether tools are needed."""
    messages = state['messages']
    last_message = messages[-1]

    if isinstance(last_message, HumanMessage):
        # Check if we need to search knowledge base first
        query_lower = last_message.content.lower()
        if any(keyword in query_lower for keyword in ['refund', 'shipping', 'hours', 'contact', 'policy']):
            return 'tools'

    return 'generate'


# Set entry point with conditional routing
graph.add_conditional_edges('__start__', route_from_start, {'tools': 'tools', 'generate': 'generate'})

# Compile graph with memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# Example usage
if __name__ == '__main__':
    print('=== Conversation Review System ===')
    print('Customer Service Agent with Human Review')
    print('Try asking about refunds, shipping, hours, or contact info.')

    config = {'configurable': {'thread_id': 'conversation_review_1'}}

    while True:
        user_input = input('\nCustomer: ').strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # Initialize state
        initial_state = {
            'messages': [HumanMessage(content=user_input)],
            'pending_review': False,
            'review_status': 'pending',
            'human_feedback': '',
            'conversation_history': [],
        }

        # Run the conversation
        result = app.invoke(initial_state, config=config)

        # Show final response
        final_message = result['messages'][-1]
        if isinstance(final_message, AIMessage):
            print(f'\nAgent: {final_message.content}')

        print(f'\nStatus: {result["review_status"]}')
        if result['human_feedback']:
            print(f'Feedback: {result["human_feedback"]}')
