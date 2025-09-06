# =============================================================================
# SETUP: Make sure to set your API keys as environment variables
# export OPENAI_API_KEY="your-openai-api-key-here"
# =============================================================================

import os
from dotenv import load_dotenv
from rich import print

load_dotenv()


# =============================================================================
# EXAMPLE 1: SUPERVISOR MULTI-AGENT SYSTEM
# =============================================================================
def run_supervisor_example():
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    # Define simple booking tools
    def book_hotel(hotel_name: str):
        """Book a hotel"""
        return f'Successfully booked a stay at {hotel_name}.'

    def book_flight(from_airport: str, to_airport: str):
        """Book a flight"""
        return f'Successfully booked a flight from {from_airport} to {to_airport}.'

    # Create specialized agents
    flight_assistant = create_react_agent(
        model=ChatOpenAI(model='gpt-4o-mini'),
        tools=[book_flight],
        prompt='You are a flight booking assistant. Help users book flights.',
        name='flight_assistant',
    )

    hotel_assistant = create_react_agent(
        model=ChatOpenAI(model='gpt-4o-mini'),
        tools=[book_hotel],
        prompt='You are a hotel booking assistant. Help users book hotels.',
        name='hotel_assistant',
    )

    # Create supervisor
    supervisor = create_supervisor(
        agents=[flight_assistant, hotel_assistant],
        model=ChatOpenAI(model='gpt-4o-mini'),
        prompt='You manage hotel and flight booking assistants. Assign work to them.',
    ).compile()

    print('=== SUPERVISOR EXAMPLE ===')
    for chunk in supervisor.stream({'messages': [{'role': 'user', 'content': 'Book a flight from LAX to NYC'}]}):
        if chunk:
            print(f'Chunk: {chunk}')


# =============================================================================
# EXAMPLE 2: SWARM MULTI-AGENT SYSTEM
# =============================================================================
def run_swarm_example():
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_swarm import create_swarm, create_handoff_tool

    # Define booking tools
    def book_hotel_swarm(hotel_name: str):
        """Book a hotel"""
        return f'Hotel booked: {hotel_name}'

    def book_flight_swarm(from_airport: str, to_airport: str):
        """Book a flight"""
        return f'Flight booked: {from_airport} to {to_airport}'

    # Create handoff tools
    transfer_to_hotel = create_handoff_tool(
        agent_name='hotel_assistant',
        description='Transfer to hotel booking assistant.',
    )

    transfer_to_flight = create_handoff_tool(
        agent_name='flight_assistant',
        description='Transfer to flight booking assistant.',
    )

    # Create agents
    flight_assistant = create_react_agent(
        model=ChatOpenAI(model='gpt-4o-mini'),
        tools=[book_flight_swarm, transfer_to_hotel],
        prompt='You are a flight booking assistant.',
        name='flight_assistant',
    )

    hotel_assistant = create_react_agent(
        model=ChatOpenAI(model='gpt-4o-mini'),
        tools=[book_hotel_swarm, transfer_to_flight],
        prompt='You are a hotel booking assistant.',
        name='hotel_assistant',
    )

    # Create swarm
    swarm = create_swarm(agents=[flight_assistant, hotel_assistant], default_active_agent='flight_assistant').compile()

    print('\n=== SWARM EXAMPLE ===')
    for chunk in swarm.stream({'messages': [{'role': 'user', 'content': 'Book a hotel at Marriott'}]}):
        if chunk:
            print(f'Chunk: {chunk}')


# =============================================================================
# EXAMPLE 3: CUSTOM HANDOFFS IMPLEMENTATION
# =============================================================================
def run_handoffs_example():
    from typing import Annotated
    from langchain_core.tools import tool, InjectedToolCallId
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent, InjectedState
    from langgraph.graph import StateGraph, START, MessagesState
    from langgraph.types import Command

    # Define booking tools
    def book_hotel_custom(hotel_name: str):
        """Book a hotel"""
        return f'Custom hotel booking: {hotel_name}'

    def book_flight_custom(from_airport: str, to_airport: str):
        """Book a flight"""
        return f'Custom flight booking: {from_airport} to {to_airport}'

    # Custom handoff tool factory
    def create_handoff_tool(*, agent_name: str, description: str):
        name = f'transfer_to_{agent_name}'

        @tool(name, description=description)
        def handoff_tool(
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            tool_message = {
                'role': 'tool',
                'content': f'Transferred to {agent_name}',
                'name': name,
                'tool_call_id': tool_call_id,
            }
            return Command(
                goto=agent_name,
                update={'messages': state['messages'] + [tool_message]},
                graph=Command.PARENT,
            )

        return handoff_tool

    # Create handoff tools
    transfer_to_hotel_custom = create_handoff_tool(
        agent_name='hotel_assistant',
        description='Transfer to hotel assistant.',
    )

    transfer_to_flight_custom = create_handoff_tool(
        agent_name='flight_assistant',
        description='Transfer to flight assistant.',
    )

    # Define agents
    flight_assistant = create_react_agent(
        model=ChatOpenAI(model='gpt-4o-mini'),
        tools=[book_flight_custom, transfer_to_hotel_custom],
        prompt='You are a flight booking assistant.',
        name='flight_assistant',
    )

    hotel_assistant = create_react_agent(
        model=ChatOpenAI(model='gpt-4o-mini'),
        tools=[book_hotel_custom, transfer_to_flight_custom],
        prompt='You are a hotel booking assistant.',
        name='hotel_assistant',
    )

    # Build graph
    multi_agent_graph = (
        StateGraph(MessagesState)
        .add_node(flight_assistant)
        .add_node(hotel_assistant)
        .add_edge(START, 'flight_assistant')
        .compile()
    )

    print('\n=== CUSTOM HANDOFFS EXAMPLE ===')
    for chunk in multi_agent_graph.stream({'messages': [{'role': 'user', 'content': 'Book a flight from SFO to LAX'}]}):
        if chunk:
            print(f'Chunk: {chunk}')


# =============================================================================
# RUN ALL EXAMPLES
# =============================================================================
if __name__ == '__main__':
    try:
        run_supervisor_example()
    except Exception as e:
        print(f'Supervisor example error: {e}')

    try:
        run_swarm_example()
    except Exception as e:
        print(f'Swarm example error: {e}')

    try:
        run_handoffs_example()
    except Exception as e:
        print(f'Handoffs example error: {e}')

    print('\n=== EXAMPLES COMPLETED ===')
