# LangGraph Multi-Agent Implementation Examples
# Install required packages: pip install langgraph langgraph-supervisor langgraph-swarm langchain-openai langchain-anthropic
#
# SETUP: You need to set your API keys as environment variables:
# export OPENAI_API_KEY="your-openai-key"
# export ANTHROPIC_API_KEY="your-anthropic-key"
#
# Alternative: Use the fake LLM version at the bottom for testing without API keys

import os
from dotenv import load_dotenv

load_dotenv()

# Check if API keys are available
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_KEY = os.getenv('ANTHROPIC_API_KEY')

if not OPENAI_KEY:
    print('Warning: OPENAI_API_KEY not found. Please set it as an environment variable.')
if not ANTHROPIC_KEY:
    print('Warning: ANTHROPIC_API_KEY not found. Please set it as an environment variable.')

# Only proceed with real examples if keys are available
if OPENAI_KEY and ANTHROPIC_KEY:
    # ================================
    # 1. SUPERVISOR PATTERN EXAMPLE
    # ================================
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    # Define simple tools
    def calculate_sum(a: float, b: float) -> str:
        """Calculate the sum of two numbers"""
        return f'The sum of {a} and {b} is {a + b}'

    def calculate_product(a: float, b: float) -> str:
        """Calculate the product of two numbers"""
        return f'The product of {a} and {b} is {a * b}'

    # Create specialized agents
    math_agent = create_react_agent(
        model=ChatOpenAI(model='gpt-4o-mini'),
        tools=[calculate_sum, calculate_product],
        prompt='You are a math assistant. Perform calculations when requested.',
        name='math_agent',
    )

    # Create supervisor
    supervisor = create_supervisor(
        agents=[math_agent],
        model=ChatOpenAI(model='gpt-4o-mini'),
        prompt='You manage a math assistant. Delegate math tasks to the math agent.',
    ).compile()

    # Example usage
    print('=== SUPERVISOR EXAMPLE ===')
    for chunk in supervisor.stream({'messages': [{'role': 'user', 'content': "What's 15 + 27?"}]}):
        if 'messages' in chunk:
            print(f'Supervisor: {chunk["messages"][-1]["content"]}')

    # ================================
    # 2. SWARM PATTERN EXAMPLE
    # ================================
    from langgraph_swarm import create_swarm, create_handoff_tool

    # Define tools for different agents
    def weather_check(city: str) -> str:
        """Check weather for a city"""
        return f'The weather in {city} is sunny and 75°F'

    def restaurant_recommendation(city: str, cuisine: str) -> str:
        """Recommend restaurants"""
        return f'Great {cuisine} restaurant in {city}: Local Favorite Bistro'

    # Create handoff tools
    transfer_to_restaurant = create_handoff_tool(
        agent_name='restaurant_agent', description='Transfer to restaurant recommendation agent'
    )

    transfer_to_weather = create_handoff_tool(
        agent_name='weather_agent', description='Transfer to weather checking agent'
    )

    # Create agents with handoff capabilities
    weather_agent = create_react_agent(
        model=ChatAnthropic(model='claude-3-5-sonnet-20241022'),
        tools=[weather_check, transfer_to_restaurant],
        prompt='You help with weather. Transfer to restaurant agent for dining recommendations.',
        name='weather_agent',
    )

    restaurant_agent = create_react_agent(
        model=ChatAnthropic(model='claude-3-5-sonnet-20241022'),
        tools=[restaurant_recommendation, transfer_to_weather],
        prompt='You recommend restaurants. Transfer to weather agent for weather info.',
        name='restaurant_agent',
    )

    # Create swarm
    swarm = create_swarm(agents=[weather_agent, restaurant_agent], default_active_agent='weather_agent').compile()

    print('\n=== SWARM EXAMPLE ===')
    for chunk in swarm.stream({'messages': [{'role': 'user', 'content': "What's the weather in Paris?"}]}):
        if 'messages' in chunk:
            print(f'Swarm: {chunk["messages"][-1]["content"]}')

    # ================================
    # 3. CUSTOM HANDOFFS EXAMPLE
    # ================================
    from typing import Annotated
    from langchain_core.tools import tool, InjectedToolCallId
    from langgraph.prebuilt import InjectedState
    from langgraph.graph import StateGraph, START, MessagesState
    from langgraph.types import Command

    # Custom handoff tool creator
    def create_custom_handoff(agent_name: str, description: str):
        @tool(f'handoff_to_{agent_name}', description=description)
        def handoff_tool(
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            return Command(
                goto=agent_name,
                update={
                    'messages': state['messages']
                    + [
                        {
                            'role': 'tool',
                            'content': f'Transferred to {agent_name}',
                            'name': f'handoff_to_{agent_name}',
                            'tool_call_id': tool_call_id,
                        }
                    ]
                },
                graph=Command.PARENT,
            )

        return handoff_tool

    # Define specialized tools
    def create_task(task_name: str) -> str:
        """Create a new task"""
        return f'Created task: {task_name}'

    def send_email(recipient: str, subject: str) -> str:
        """Send an email"""
        return f'Email sent to {recipient} with subject: {subject}'

    # Create handoff tools
    to_email_agent = create_custom_handoff('email_agent', 'Transfer to email handling agent')
    to_task_agent = create_custom_handoff('task_agent', 'Transfer to task management agent')

    # Create specialized agents
    task_agent = create_react_agent(
        model=ChatAnthropic(model='claude-3-5-sonnet-20241022'),
        tools=[create_task, to_email_agent],
        prompt='You manage tasks. Use handoff for email-related requests.',
        name='task_agent',
    )

    email_agent = create_react_agent(
        model=ChatAnthropic(model='claude-3-5-sonnet-20241022'),
        tools=[send_email, to_task_agent],
        prompt='You handle emails. Use handoff for task-related requests.',
        name='email_agent',
    )

    # Build custom multi-agent graph
    custom_graph = (
        StateGraph(MessagesState)
        .add_node('task_agent', task_agent)
        .add_node('email_agent', email_agent)
        .add_edge(START, 'task_agent')
    ).compile()

    print('\n=== CUSTOM HANDOFFS EXAMPLE ===')
    for chunk in custom_graph.stream({'messages': [{'role': 'user', 'content': "Create a task 'Review proposal'"}]}):
        if 'messages' in chunk:
            print(f'Custom: {chunk["messages"][-1]["content"]}')

    print('\n=== All examples completed! ===')

else:
    print('\n' + '=' * 50)
    print('API KEYS NOT FOUND - SHOWING FAKE LLM VERSION')
    print('=' * 50)

    # ================================
    # FAKE LLM VERSION (NO API KEYS NEEDED)
    # ================================
    from langchain_core.language_models.fake import FakeListChatModel
    from langgraph.prebuilt import create_react_agent
    from langgraph.graph import StateGraph, START, MessagesState

    # Create a fake LLM for testing
    fake_llm = FakeListChatModel(
        responses=[
            "I'll help you with that calculation. Let me use the calculator tool.",
            'The calculation is complete. The result is 42.',
            'Task completed successfully!',
        ]
    )

    # Simple tool
    def add_numbers(a: float, b: float) -> str:
        """Add two numbers together"""
        return f'Result: {a + b}'

    # Create a simple agent with fake LLM
    simple_agent = create_react_agent(
        model=fake_llm, tools=[add_numbers], prompt='You are a helpful calculator assistant.'
    )

    # Create simple graph
    simple_graph = (StateGraph(MessagesState).add_node('agent', simple_agent).add_edge(START, 'agent')).compile()

    print('=== FAKE LLM EXAMPLE (No API keys needed) ===')
    try:
        for chunk in simple_graph.stream({'messages': [{'role': 'user', 'content': 'Add 20 and 22'}]}):
            if 'messages' in chunk and chunk['messages']:
                print(f'Agent: {chunk["messages"][-1]["content"]}')
    except Exception as e:
        print(f'Demo completed with fake responses: {e}')

    print('\nTo run the full examples with real LLMs, set these environment variables:')
    print("export OPENAI_API_KEY='your-openai-api-key'")
    print("export ANTHROPIC_API_KEY='your-anthropic-api-key'")
