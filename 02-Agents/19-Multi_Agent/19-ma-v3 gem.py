# %%
import os
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessageGraph, MessagesState, START, StateGraph
from langgraph.types import Command

# ===============================
# 1. SUPERVISOR ARCHITECTURE
# ===============================
# A central supervisor agent routes tasks to specialized worker agents.
# This example is simple, using keyword matching for routing.

print('=' * 30)
print('RUNNING SUPERVISOR EXAMPLE')
print('=' * 30)


# Define placeholder tools and agents
def research_tool(query: str):
    """A placeholder for a real research tool."""
    print(f"Executing research_tool for query: '{query}'")
    return f"Found valuable information on: '{query}'"


def research_agent(messages: list[AnyMessage]):
    """A worker agent that performs research."""
    print('---CALLING RESEARCH AGENT---')
    # In a real app, an LLM would intelligently call the tool.
    # Here, we call it directly for simplicity.
    query = messages[-1].content
    result = research_tool(query)
    return HumanMessage(content=result)


def math_agent(messages: list[AnyMessage]):
    """A worker agent that performs math."""
    print('---CALLING MATH AGENT---')
    # This agent would typically use a calculator tool or an LLM.
    return HumanMessage(content='The final answer is 42.')


# The supervisor function decides which agent (or the user) should act next.
def supervisor_router(messages: list[AnyMessage]) -> Literal['research', 'math', END]:
    """The central supervisor that routes messages to the correct agent."""
    last_message = messages[-1].content.lower()
    if 'math' in last_message or 'calculate' in last_message:
        return 'math'
    if 'research' in last_message or 'look up' in last_message:
        return 'research'
    # End the conversation if the task is not recognized.
    return END


# Build the graph
supervisor_builder = MessageGraph()
supervisor_builder.add_node('research', research_agent)
supervisor_builder.add_node('math', math_agent)

# The supervisor is the entry point and decides the first step.
supervisor_builder.add_conditional_edges(START, supervisor_router, {'research': 'research', 'math': 'math', END: END})
# After each worker agent finishes, the conversation ends.
supervisor_builder.add_edge('research', END)
supervisor_builder.add_edge('math', END)

supervisor_graph = supervisor_builder.compile(checkpointer=MemorySaver())

# Run and demonstrate the supervisor graph
config = {'configurable': {'thread_id': 'supervisor-thread'}}
# Example 1: Routing to the research agent
print('\n--- Task: Research LangGraph ---')
for chunk in supervisor_graph.stream([HumanMessage(content='Could you research LangGraph?')], config=config):
    print(chunk)

# Example 2: Routing to the math agent
print('\n--- Task: Calculate 1+1 ---')
for chunk in supervisor_graph.stream(
    [HumanMessage(content='What is 1+1 in math?')], config=config, stream_mode='values'
):
    chunk[-1].pretty_print()


# ===============================
# 2. NETWORK ARCHITECTURE
# ===============================
# A collaborative network where agents can pass control to each other
# without a single, fixed supervisor.

print('\n\n' + '=' * 30)
print('RUNNING NETWORK EXAMPLE')
print('=' * 30)


# Define the state for the network
class NetworkState(MessagesState):
    """State for a network of collaborating agents, including an iteration counter."""

    iteration_count: int = 0


# Define the network agents
def analyst_agent(state: NetworkState) -> dict:
    """Analyzes data and routes to other agents or ends the process."""
    last_message = state['messages'][-1].content
    response = f"Analyst: I've reviewed '{last_message}'. The data suggests a need for deeper investigation."
    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state['iteration_count'] + 1}

    # Use an iteration limit to prevent infinite loops
    if state['iteration_count'] >= 3:
        return {'messages': [AIMessage(content='Analyst: Reached iteration limit. Ending analysis.')]}

    # Route based on content
    if 'strategy' in last_message.lower():
        return Command(goto='strategist_agent', update=new_state)
    else:
        return Command(goto='researcher_agent', update=new_state)


def researcher_agent(state: NetworkState) -> dict:
    """Gathers information and routes to other agents."""
    last_message = state['messages'][-1].content
    response = f"Researcher: Based on my investigation of '{last_message}', I found relevant background information."
    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state['iteration_count'] + 1}

    if state['iteration_count'] >= 3:
        return {'messages': [AIMessage(content='Researcher: Reached iteration limit. Ending research.')]}
    elif 'analysis' in last_message.lower():
        return Command(goto='analyst_agent', update=new_state)
    else:
        return Command(goto='strategist_agent', update=new_state)


def strategist_agent(state: NetworkState) -> dict:
    """Develops plans and provides the final recommendation."""
    last_message = state['messages'][-1].content
    response = f"Strategist: For '{last_message}', I recommend a comprehensive approach."
    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state['iteration_count'] + 1}

    # The strategist provides the final word
    if state['iteration_count'] >= 2:
        final_response = 'Strategist: Final recommendation compiled. All agents have contributed.'
        return {'messages': [AIMessage(content=final_response)]}
    else:
        return Command(goto='analyst_agent', update=new_state)


# Build the network graph
network_builder = StateGraph(NetworkState)
network_builder.add_node('analyst_agent', analyst_agent)
network_builder.add_node('researcher_agent', researcher_agent)
network_builder.add_node('strategist_agent', strategist_agent)
# Any agent can be the entry point
network_builder.set_entry_point('analyst_agent')
network_graph = network_builder.compile()

# Run and demonstrate the network graph
print('\n--- Task: Develop a market expansion strategy ---')
result = network_graph.invoke(
    {'messages': [HumanMessage(content='We need to develop a strategy for market expansion')], 'iteration_count': 0}
)
print('\nNetwork Collaboration Result:')
for i, message in enumerate(result['messages']):
    print(f'Step {i + 1}: {message.pretty_repr()}')


# ===============================
# 3. HIERARCHICAL ARCHITECTURE
# ===============================
# A multi-level structure where a top-level supervisor routes tasks
# to specialized teams, which are themselves subgraphs.

print('\n\n' + '=' * 30)
print('RUNNING HIERARCHICAL EXAMPLE')
print('=' * 30)


# Define the state for the hierarchy
class HierarchicalState(MessagesState):
    """State for a hierarchical agent system."""

    team_assigned: str = ''


# --- Research Team ---
def data_collector(state: HierarchicalState) -> dict:
    """A worker agent within the research team."""
    last_message = state['messages'][-1].content
    response = f"Data Collector: Gathered comprehensive data on '{last_message}'."
    return {'messages': [AIMessage(content=response)]}


def research_team_supervisor(state: HierarchicalState) -> Command:
    """The supervisor for the research team."""
    response = 'Research Supervisor: Coordinating data collection and analysis.'
    return Command(goto='data_collector', update={'messages': [AIMessage(content=response)]})


# --- Analysis Team ---
def data_analyst(state: HierarchicalState) -> dict:
    """A worker agent within the analysis team."""
    last_message = state['messages'][-1].content
    response = f"Data Analyst: Completed statistical analysis of '{last_message}'."
    return {'messages': [AIMessage(content=response)]}


def analysis_team_supervisor(state: HierarchicalState) -> Command:
    """The supervisor for the analysis team."""
    response = 'Analysis Supervisor: Initiating data analysis workflow.'
    return Command(goto='data_analyst', update={'messages': [AIMessage(content=response)]})


# --- Create Team Subgraphs ---
def create_team_subgraph(supervisor_node, worker_node, team_name):
    """Factory function to create a team subgraph."""
    builder = StateGraph(HierarchicalState)
    builder.add_node(f'{team_name}_supervisor', supervisor_node)
    builder.add_node(f'{team_name}_worker', worker_node)
    builder.add_edge(f'{team_name}_worker', END)
    builder.set_entry_point(f'{team_name}_supervisor')
    return builder.compile()


research_team_graph = create_team_subgraph(research_team_supervisor, data_collector, 'research')
analysis_team_graph = create_team_subgraph(analysis_team_supervisor, data_analyst, 'analysis')


# --- Top-Level Supervisor ---
def top_supervisor_router(state: HierarchicalState) -> Command:
    """Top-level supervisor that routes tasks to the appropriate team."""
    last_message = state['messages'][-1].content.lower()
    if any(word in last_message for word in ['collect', 'gather', 'research', 'find']):
        return Command(goto='research_team', update={'team_assigned': 'research'})
    elif any(word in last_message for word in ['analyze', 'calculate', 'process', 'examine']):
        return Command(goto='analysis_team', update={'team_assigned': 'analysis'})
    else:
        return Command(goto='research_team', update={'team_assigned': 'research'})  # Default


# Build the main hierarchical graph
hierarchical_builder = StateGraph(HierarchicalState)
hierarchical_builder.add_node('top_supervisor', top_supervisor_router)
hierarchical_builder.add_node('research_team', research_team_graph)
hierarchical_builder.add_node('analysis_team', analysis_team_graph)
hierarchical_builder.add_edge('research_team', END)
hierarchical_builder.add_edge('analysis_team', END)
hierarchical_builder.set_entry_point('top_supervisor')
hierarchical_graph = hierarchical_builder.compile()

# Run and demonstrate the hierarchical graph
# Example 1: Routing to the research team
print('\n--- Task: Research customer preferences ---')
result1 = hierarchical_graph.invoke(
    {'messages': [HumanMessage(content='We need to research information about customer preferences')]}
)
print('\nResearch Task Result:')
for message in result1['messages']:
    print(f'- {message.pretty_repr()}')

# Example 2: Routing to the analysis team
print('\n--- Task: Analyze sales data ---')
result2 = hierarchical_graph.invoke({'messages': [HumanMessage(content='Analyze the sales data trends')]})
print('\nAnalysis Task Result:')
for message in result2['messages']:
    print(f'- {message.pretty_repr()}')


# %%
