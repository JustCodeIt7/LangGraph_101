# ================================
# MULTI-AGENT LANGGRAPH EXAMPLES FOR TEACHING
# ================================

import os
from typing import Annotated, Any, Literal, TypedDict
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessageGraph, MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

llm = ChatOllama(model='llama3.2')
embedding = OllamaEmbeddings(model='nomic-embed-text')

# ================================
# 1. SUPERVISOR ARCHITECTURE
# ================================
# Central supervisor routes tasks to specialized worker agents


class SupervisorState(MessagesState):
    """State that includes messages and any additional context"""

    pass


def research_agent(state: SupervisorState) -> SupervisorState:
    """Handles research-related queries"""
    last_message = state['messages'][-1].content
    response = f"Research Agent: Based on my analysis of '{last_message}', here are the key findings: This topic involves complex data gathering and analysis."
    return {'messages': [AIMessage(content=response)]}


def math_agent(state: SupervisorState) -> SupervisorState:
    """Handles mathematical calculations"""
    last_message = state['messages'][-1].content
    response = f"Math Agent: Analyzing the mathematical aspects of '{last_message}': This involves numerical computation and statistical analysis."
    return {'messages': [AIMessage(content=response)]}


def supervisor_agent(state: SupervisorState) -> Command[Literal['research_agent', 'math_agent', '__end__']]:
    """Central supervisor that routes to appropriate worker agents"""
    last_message = state['messages'][-1].content.lower()

    # Simple routing logic based on keywords
    if any(word in last_message for word in ['research', 'study', 'analyze', 'investigate']):
        return Command(goto='research_agent')
    elif any(word in last_message for word in ['calculate', 'math', 'number', 'formula']):
        return Command(goto='math_agent')
    else:
        # Default to research for ambiguous queries
        return Command(goto='research_agent')


# Build the supervisor graph
def create_supervisor_graph():
    workflow = StateGraph(SupervisorState)

    # Add nodes
    workflow.add_node('supervisor', supervisor_agent)
    workflow.add_node('research_agent', research_agent)
    workflow.add_node('math_agent', math_agent)

    # Define edges
    workflow.add_edge('research_agent', '__end__')
    workflow.add_edge('math_agent', '__end__')

    # Set entry point
    workflow.set_entry_point('supervisor')

    return workflow.compile()


# ================================
# 2. NETWORK ARCHITECTURE
# ================================
# Agents can communicate with any other agent in a collaborative network


class NetworkState(MessagesState):
    """State for network of collaborating agents"""

    iteration_count: int = 0


def analyst_agent(state: NetworkState) -> Command[Literal['researcher_agent', 'strategist_agent', '__end__']]:
    """Analyst agent that processes data and routes accordingly"""
    last_message = state['messages'][-1].content
    response = f"Analyst: I've reviewed '{last_message}'. The data suggests we need deeper investigation."

    # Add response to state
    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state.get('iteration_count', 0) + 1}

    # Routing logic with iteration limit to prevent infinite loops
    if state.get('iteration_count', 0) >= 3:
        return Command(goto='__end__', update=new_state)
    elif 'strategy' in last_message.lower():
        return Command(goto='strategist_agent', update=new_state)
    else:
        return Command(goto='researcher_agent', update=new_state)


def researcher_agent(state: NetworkState) -> Command[Literal['analyst_agent', 'strategist_agent', '__end__']]:
    """Researcher agent that gathers information"""
    last_message = state['messages'][-1].content
    response = f"Researcher: Based on my investigation of '{last_message}', I found relevant background information."

    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state.get('iteration_count', 0) + 1}

    # Route based on content and iteration count
    if state.get('iteration_count', 0) >= 3:
        return Command(goto='__end__', update=new_state)
    elif 'analysis' in last_message.lower():
        return Command(goto='analyst_agent', update=new_state)
    else:
        return Command(goto='strategist_agent', update=new_state)


def strategist_agent(state: NetworkState) -> Command[Literal['analyst_agent', 'researcher_agent', '__end__']]:
    """Strategist agent that develops plans"""
    last_message = state['messages'][-1].content
    response = (
        f"Strategist: For '{last_message}', I recommend a comprehensive approach combining multiple perspectives."
    )

    new_state = {'messages': [AIMessage(content=response)], 'iteration_count': state.get('iteration_count', 0) + 1}

    # End the conversation after strategist provides final recommendation
    if state.get('iteration_count', 0) >= 2:
        final_response = 'Strategist: Final recommendation compiled. All agents have contributed to the solution.'
        return Command(goto='__end__', update={'messages': [AIMessage(content=final_response)]})
    else:
        return Command(goto='analyst_agent', update=new_state)


# Build the network graph
def create_network_graph():
    workflow = StateGraph(NetworkState)

    # Add all agent nodes
    workflow.add_node('analyst_agent', analyst_agent)
    workflow.add_node('researcher_agent', researcher_agent)
    workflow.add_node('strategist_agent', strategist_agent)

    # Set entry point (could be any agent)
    workflow.set_entry_point('analyst_agent')

    return workflow.compile()


# ================================
# 3. HIERARCHICAL ARCHITECTURE
# ================================
# Multi-level organization with teams and supervisors


class HierarchicalState(MessagesState):
    """State for hierarchical agent system"""

    team_assigned: str = ''


# Research Team Agents
def data_collector(state: HierarchicalState) -> HierarchicalState:
    """Collects data for research team"""
    last_message = state['messages'][-1].content
    response = f"Data Collector: Gathered comprehensive data on '{last_message}'"
    return {'messages': [AIMessage(content=response)]}


def research_supervisor(state: HierarchicalState) -> Command[Literal['data_collector', '__end__']]:
    """Supervises research team"""
    response = 'Research Supervisor: Coordinating data collection and analysis.'
    return Command(goto='data_collector', update={'messages': [AIMessage(content=response)]})


# Analysis Team Agents
def data_analyst(state: HierarchicalState) -> HierarchicalState:
    """Analyzes data for analysis team"""
    last_message = state['messages'][-1].content
    response = f"Data Analyst: Completed statistical analysis of '{last_message}'"
    return {'messages': [AIMessage(content=response)]}


def analysis_supervisor(state: HierarchicalState) -> Command[Literal['data_analyst', '__end__']]:
    """Supervises analysis team"""
    response = 'Analysis Supervisor: Initiating data analysis workflow.'
    return Command(goto='data_analyst', update={'messages': [AIMessage(content=response)]})


# Top-level supervisor
def top_supervisor(state: HierarchicalState) -> Command[Literal['research_team', 'analysis_team', '__end__']]:
    """Top-level supervisor that routes to appropriate teams"""
    last_message = state['messages'][-1].content.lower()

    if any(word in last_message for word in ['collect', 'gather', 'research', 'find']):
        return Command(goto='research_team', update={'team_assigned': 'research'})
    elif any(word in last_message for word in ['analyze', 'calculate', 'process', 'examine']):
        return Command(goto='analysis_team', update={'team_assigned': 'analysis'})
    else:
        # Default to research team
        return Command(goto='research_team', update={'team_assigned': 'research'})


# Create subgraphs for teams
def create_research_team():
    """Create research team subgraph"""
    team_workflow = StateGraph(HierarchicalState)
    team_workflow.add_node('research_supervisor', research_supervisor)
    team_workflow.add_node('data_collector', data_collector)
    team_workflow.add_edge('data_collector', '__end__')
    team_workflow.set_entry_point('research_supervisor')
    return team_workflow.compile()


def create_analysis_team():
    """Create analysis team subgraph"""
    team_workflow = StateGraph(HierarchicalState)
    team_workflow.add_node('analysis_supervisor', analysis_supervisor)
    team_workflow.add_node('data_analyst', data_analyst)
    team_workflow.add_edge('data_analyst', '__end__')
    team_workflow.set_entry_point('analysis_supervisor')
    return team_workflow.compile()


# Build the main hierarchical graph
def create_hierarchical_graph():
    workflow = StateGraph(HierarchicalState)

    # Add top supervisor
    workflow.add_node('top_supervisor', top_supervisor)

    # Add team subgraphs
    workflow.add_node('research_team', create_research_team())
    workflow.add_node('analysis_team', create_analysis_team())

    # Connect teams back to end
    workflow.add_edge('research_team', '__end__')
    workflow.add_edge('analysis_team', '__end__')

    # Set entry point
    workflow.set_entry_point('top_supervisor')

    return workflow.compile()


# ================================
# USAGE EXAMPLES
# ================================

if __name__ == '__main__':
    # Test Supervisor Architecture
    print('=== SUPERVISOR ARCHITECTURE ===')
    supervisor_graph = create_supervisor_graph()
    result1 = supervisor_graph.invoke({'messages': [HumanMessage(content='I need to research climate change impacts')]})
    print('Research Query Result:', result1['messages'][-1].content)

    result2 = supervisor_graph.invoke({'messages': [HumanMessage(content='Calculate the probability distribution')]})
    print('Math Query Result:', result2['messages'][-1].content)

    # Test Network Architecture
    print('\n=== NETWORK ARCHITECTURE ===')
    network_graph = create_network_graph()
    result3 = network_graph.invoke(
        {'messages': [HumanMessage(content='We need to develop a strategy for market expansion')], 'iteration_count': 0}
    )
    print('Network Collaboration Result:')
    for i, message in enumerate(result3['messages']):
        print(f'{i + 1}. {message.content}')

    # Test Hierarchical Architecture
    print('\n=== HIERARCHICAL ARCHITECTURE ===')
    hierarchical_graph = create_hierarchical_graph()
    result4 = hierarchical_graph.invoke(
        {'messages': [HumanMessage(content='We need to gather information about customer preferences')]}
    )
    print('Hierarchical Research Task Result:')
    for message in result4['messages']:
        print(f'- {message.content}')

    result5 = hierarchical_graph.invoke({'messages': [HumanMessage(content='Analyze the sales data trends')]})
    print('Hierarchical Analysis Task Result:')
    for message in result5['messages']:
        print(f'- {message.content}')
