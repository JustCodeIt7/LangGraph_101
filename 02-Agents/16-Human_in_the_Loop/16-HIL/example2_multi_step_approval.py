#!/usr/bin/env python3
"""
Human-in-the-Loop: Multi-Step Approval Workflow

This example demonstrates:
1. Complex workflows requiring approval at multiple stages
2. Different approval levels for different types of actions
3. Escalation paths for high-risk decisions
4. Perfect for financial transactions, content publishing, or compliance workflows

Pattern: Stage 1 → Approve → Stage 2 → Approve → Stage 3 → Complete
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
class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    current_stage: str  # "planning", "review", "execution", "verification"
    approval_level: str  # "low", "medium", "high"
    approvals: dict  # Track approvals for each stage
    workflow_data: dict  # Store workflow-specific data
    risk_level: str  # "low", "medium", "high"


# Define tools for different workflow stages
@tool
def assess_risk(action_type: str, amount: float, details: str) -> dict:
    """Assess risk level of a proposed action."""
    risk_matrix = {
        'transfer': {'low': 100, 'medium': 1000, 'high': 10000},
        'access': {'low': 1, 'medium': 2, 'high': 3},
        'change': {'low': 1, 'medium': 2, 'high': 3},
    }

    if action_type in risk_matrix:
        thresholds = risk_matrix[action_type]
        if action_type == 'transfer':
            if amount <= thresholds['low']:
                return {'level': 'low', 'reason': 'Amount below low threshold'}
            elif amount <= thresholds['medium']:
                return {'level': 'medium', 'reason': 'Amount below medium threshold'}
            else:
                return {'level': 'high', 'reason': 'Amount exceeds medium threshold'}
        else:
            if details.count('change') <= thresholds['low']:
                return {'level': 'low', 'reason': 'Minimal changes requested'}
            elif details.count('change') <= thresholds['medium']:
                return {'level': 'medium', 'reason': 'Moderate changes requested'}
            else:
                return {'level': 'high', 'reason': 'Extensive changes requested'}

    return {'level': 'medium', 'reason': 'Default risk assessment'}


@tool
def execute_transfer(amount: float, recipient: str, reference: str) -> str:
    """Execute a financial transfer."""
    return f'Transfer executed: ${amount} to {recipient} (Ref: {reference})'


@tool
def update_permissions(user: str, permissions: List[str]) -> str:
    """Update user permissions."""
    return f'Permissions updated for {user}: {", ".join(permissions)}'


@tool
def send_notification(recipient: str, message: str, priority: str) -> str:
    """Send notification about workflow completion."""
    return f'Notification sent to {recipient} with priority {priority}'


# Create tool list
tools = [assess_risk, execute_transfer, update_permissions, send_notification]


# Node functions
def plan_workflow(state: WorkflowState):
    """Plan the workflow based on user request."""
    system_message = SystemMessage(
        content="You are a workflow planner. Analyze the user's request and determine "
        'what type of workflow is needed. Use the assess_risk tool to evaluate risk level. '
        'Provide a clear plan of action.'
    )

    messages = state['messages']
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [system_message] + messages

    model = ChatOpenAI(model='gpt-4o')
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    return {'messages': [response], 'current_stage': 'planning', 'approvals': {}, 'workflow_data': {}}


def review_stage(state: WorkflowState):
    """Handle human review at current stage."""
    stage = state['current_stage']
    risk_level = state.get('risk_level', 'medium')

    print('\n' + '=' * 60)
    print(f'HUMAN APPROVAL REQUIRED - {stage.upper()} STAGE')
    print('=' * 60)
    print(f'Risk Level: {risk_level.upper()}')

    # Show context based on stage
    if stage == 'planning':
        print('\nProposed Workflow Plan:')
        last_message = state['messages'][-1]
        print(last_message.content)

        # Extract risk assessment if available
        for msg in reversed(state['messages']):
            if hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    if tool_call['name'] == 'assess_risk':
                        print(f'\nRisk Assessment: {tool_call["args"]}')
                        break

    elif stage == 'execution':
        print('\nReady to execute the approved plan.')
        print('Workflow data:', json.dumps(state.get('workflow_data', {}), indent=2))

    elif stage == 'verification':
        print('\nWorkflow execution completed. Verification needed.')
        print('Results:', json.dumps(state.get('workflow_data', {}), indent=2))

    # Determine approval requirements
    approvers = {'low': 'Any team member', 'medium': 'Team lead or manager', 'high': 'Director or above'}

    print(f'\nRequired approver: {approvers[risk_level]}')

    # Get approval
    approval = input('\nApprove this stage? (yes/no/escalate): ').strip().lower()

    if approval == 'yes':
        print('✓ Stage approved!')
        state['approvals'][stage] = {
            'status': 'approved',
            'approver': 'human_user',
            'timestamp': '2025-08-06T12:00:00Z',
        }
        return {'approvals': state['approvals']}

    elif approval == 'escalate':
        print('⚠️ Escalating to higher authority...')
        state['approvals'][stage] = {
            'status': 'escalated',
            'approver': 'pending_escalation',
            'timestamp': '2025-08-06T12:00:00Z',
        }
        return {
            'approvals': state['approvals'],
            'messages': [HumanMessage(content='Stage escalated for higher-level approval.')],
        }

    else:
        print('✗ Stage rejected!')
        state['approvals'][stage] = {
            'status': 'rejected',
            'approver': 'human_user',
            'timestamp': '2025-08-06T12:00:00Z',
        }
        return {
            'approvals': state['approvals'],
            'messages': [HumanMessage(content='Stage rejected by human reviewer.')],
        }


def execute_workflow(state: WorkflowState):
    """Execute the workflow based on approved plan."""
    print('\n🚀 Executing workflow...')

    # Extract workflow data from previous messages
    workflow_data = state.get('workflow_data', {})

    # Execute based on workflow type
    if 'transfer' in str(state['messages']).lower():
        # Simulate transfer execution
        amount = workflow_data.get('amount', 100)
        recipient = workflow_data.get('recipient', 'unknown')
        reference = f'TXN{hash(str(amount) + recipient) % 100000}'

        result = execute_transfer.invoke({'amount': amount, 'recipient': recipient, 'reference': reference})

        workflow_data.update({'transaction_id': reference, 'execution_result': result, 'status': 'completed'})

    elif 'permission' in str(state['messages']).lower():
        # Simulate permission update
        user = workflow_data.get('user', 'unknown')
        permissions = workflow_data.get('permissions', ['read'])

        result = update_permissions.invoke({'user': user, 'permissions': permissions})

        workflow_data.update({'execution_result': result, 'status': 'completed'})

    else:
        workflow_data.update({'execution_result': 'Generic workflow executed', 'status': 'completed'})

    return {
        'workflow_data': workflow_data,
        'current_stage': 'verification',
        'messages': [AIMessage(content=f'Workflow executed: {workflow_data.get("execution_result", "Unknown")}')],
    }


def verify_completion(state: WorkflowState):
    """Verify workflow completion and send notifications."""
    workflow_data = state.get('workflow_data', {})

    print('\n✅ Verifying workflow completion...')

    # Send notification
    notification_result = send_notification.invoke(
        {
            'recipient': 'admin@company.com',
            'message': f'Workflow completed: {workflow_data.get("execution_result", "Unknown")}',
            'priority': state.get('risk_level', 'medium'),
        }
    )

    # Generate completion report
    report = {
        'workflow_id': f'WF{hash(str(workflow_data)) % 100000}',
        'stages_completed': list(state['approvals'].keys()),
        'approvals': state['approvals'],
        'final_status': workflow_data.get('status', 'unknown'),
        'notification_sent': notification_result,
        'risk_level': state.get('risk_level', 'medium'),
    }

    print(f'\n📊 Completion Report:')
    print(json.dumps(report, indent=2))

    return {
        'workflow_data': {**workflow_data, 'completion_report': report},
        'messages': [AIMessage(content=f'Workflow completed successfully! {notification_result}')],
    }


def process_tools(state: WorkflowState):
    """Process any tool calls."""
    messages = state['messages']
    last_message = messages[-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {'messages': []}

    tool_messages = []
    workflow_data = state.get('workflow_data', {})

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        if tool_name == 'assess_risk':
            result = assess_risk.invoke(tool_args)
            risk_level = result.get('level', 'medium')

            # Update state with risk assessment
            workflow_data.update({'risk_assessment': result, 'risk_level': risk_level})

            # Store workflow-specific data
            if 'amount' in tool_args:
                workflow_data['amount'] = tool_args['amount']
            if 'recipient' in tool_args:
                workflow_data['recipient'] = tool_args['recipient']
            if 'user' in tool_args:
                workflow_data['user'] = tool_args['user']
            if 'permissions' in tool_args:
                workflow_data['permissions'] = tool_args['permissions']

        tool_message = ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call['id'])
        tool_messages.append(tool_message)

    return {
        'messages': tool_messages,
        'workflow_data': workflow_data,
        'risk_level': workflow_data.get('risk_level', 'medium'),
    }


# Router functions
def route_after_planning(state: WorkflowState):
    """Route after planning stage."""
    if state['current_stage'] == 'planning':
        return 'review'
    return 'end'


def route_after_review(state: WorkflowState):
    """Route after review stage."""
    stage = state['current_stage']
    approvals = state.get('approvals', {})

    if stage in approvals:
        status = approvals[stage].get('status', '')

        if status == 'approved':
            if stage == 'planning':
                return 'execute'
            elif stage == 'execution':
                return 'verify'
            elif stage == 'verification':
                return 'end'
        elif status == 'escalated':
            return 'review'  # Wait for higher-level approval
        elif status == 'rejected':
            return 'end'

    return 'end'


def route_after_tools(state: WorkflowState):
    """Route after tool processing."""
    # After tools, go to review for planning stage
    if state['current_stage'] == 'planning':
        return 'review'
    return 'end'


# Create the graph
graph = StateGraph(WorkflowState)

# Add nodes
graph.add_node('plan', plan_workflow)
graph.add_node('review', review_stage)
graph.add_node('execute', execute_workflow)
graph.add_node('verify', verify_completion)
graph.add_node('tools', process_tools)

# Add edges
graph.add_edge('verify', END)

# Add conditional edges
graph.add_conditional_edges('plan', route_after_planning, {'review': 'review', 'end': END})

graph.add_conditional_edges(
    'review', route_after_review, {'execute': 'execute', 'verify': 'verify', 'review': 'review', 'end': END}
)

graph.add_conditional_edges('tools', route_after_tools, {'review': 'review', 'end': END})

# Set entry point
graph.set_entry_point('plan')

# Compile graph with memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# Example usage
if __name__ == '__main__':
    print('=== Multi-Step Approval Workflow ===')
    print('Try requests like:')
    print("- 'Transfer $500 to John Doe for services'")
    print("- 'Update permissions for user Alice to include admin access'")
    print("- 'Change system configuration settings'")

    config = {'configurable': {'thread_id': 'multi_step_approval_1'}}

    while True:
        user_input = input('\nRequest: ').strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # Initialize state
        initial_state = {
            'messages': [HumanMessage(content=user_input)],
            'current_stage': 'planning',
            'approval_level': 'medium',
            'approvals': {},
            'workflow_data': {},
            'risk_level': 'medium',
        }

        # Run the workflow
        result = app.invoke(initial_state, config=config)

        # Show final status
        print(f'\n🎯 Workflow Status: {result.get("current_stage", "completed")}')
        print(f'Risk Level: {result.get("risk_level", "unknown")}')

        if result.get('workflow_data', {}).get('execution_result'):
            print(f'Result: {result["workflow_data"]["execution_result"]}')
