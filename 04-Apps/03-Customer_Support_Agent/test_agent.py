#!/usr/bin/env python3
"""
Test script for the Customer Support Agent
"""

import asyncio
import os
from langchain_core.messages import HumanMessage
from customer_support_agent import SupportWorkflow, SupportState

# Set up environment
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'  # Replace with your key


async def test_customer_support_agent():
    """Test the customer support agent with various scenarios"""

    workflow = SupportWorkflow()

    # Test scenarios
    test_cases = [
        {
            'name': 'Greeting',
            'message': 'Hello, I need help',
            'customer_data': {'id': 'CUST001', 'name': 'John Doe', 'is_vip': False},
        },
        {
            'name': 'Question about billing',
            'message': 'How do I update my payment method?',
            'customer_data': {'id': 'CUST002', 'name': 'Jane Smith', 'is_vip': False},
        },
        {
            'name': 'Report issue',
            'message': "My account is locked and I can't log in",
            'customer_data': {'id': 'CUST003', 'name': 'Bob Johnson', 'is_vip': False},
        },
        {
            'name': 'Check ticket status',
            'message': "What's the status of ticket TK123?",
            'customer_data': {'id': 'CUST004', 'name': 'Alice Brown', 'is_vip': False},
        },
        {
            'name': 'VIP escalation',
            'message': 'This is urgent, I need to speak to a manager',
            'customer_data': {'id': 'CUST005', 'name': 'VIP Customer', 'is_vip': True},
        },
    ]

    for test_case in test_cases:
        print(f'\n=== Testing: {test_case["name"]} ===')

        initial_state = SupportState(
            messages=[HumanMessage(content=test_case['message'])],
            customer_data=test_case['customer_data'],
            intent='',
            entities={},
            pending_action=None,
            ticket_id=None,
            response='',
            language='en',
            timestamp='2024-01-01T00:00:00',
        )

        try:
            result = await workflow.graph.ainvoke(initial_state)
            print(f'User: {test_case["message"]}')
            print(f'Agent: {result["response"]}')
            print(f'Intent: {result["intent"]}')
            if result.get('ticket_id'):
                print(f'Ticket ID: {result["ticket_id"]}')
            if result.get('pending_action'):
                print(f'Pending Action: {result["pending_action"]}')
        except Exception as e:
            print(f'Error: {e}')


if __name__ == '__main__':
    print('Customer Support Agent Test Suite')
    print('===================================')
    asyncio.run(test_customer_support_agent())
