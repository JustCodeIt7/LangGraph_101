# %% [markdown]
# # Multi-Agent System
# ## 1. Supervisor Architecture

# %%


# %%
import os
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt import ToolNode


# This is a placeholder for a real tool; in a real app, you would use something like Tavily
def research_tool(query: str):
    print(f'Researching: {query}')
    return f'Found information on: {query}'


# Define the agents
def research_agent(messages: list[AnyMessage]):
    print('---CALLING RESEARCH AGENT---')
    # In a real app, you'd use an LLM to call the tool
    # For this example, we'll just call it directly
    result = research_tool(messages[-1].content)
    return HumanMessage(content=result)


def math_agent(messages: list[AnyMessage]):
    print('---CALLING MATH AGENT---')
    return HumanMessage(content='The answer is 2.')


# The supervisor decides which agent to call next
def supervisor(messages: list[AnyMessage]) -> Literal['research', 'math', END]:
    last_message = messages[-1].content.lower()
    if 'math' in last_message:
        return 'math'
    if 'research' in last_message:
        return 'research'
    return END


# Build the graph
graph_builder = MessageGraph()
graph_builder.add_node('research', research_agent)
graph_builder.add_node('math', math_agent)
graph_builder.add_conditional_edges('__start__', supervisor, {'research': 'research', 'math': 'math', END: END})
graph_builder.add_edge('research', END)
graph_builder.add_edge('math', END)
graph = graph_builder.compile(checkpointer=MemorySaver())

# Run the graph
config = {'configurable': {'thread_id': '1'}}
for chunk in graph.stream([HumanMessage(content='Could you research langgraph?')], config=config):
    print(chunk)

for chunk in graph.stream([HumanMessage(content='What is 1 + 1?')], config=config, stream_mode='values'):
    chunk[-1].pretty_print()


# %% [markdown]
# #### Example 1: Supervisor Architecture (with ChatOllama)

# %%
# pip install langgraph langchain langchain-ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

llm = ChatOllama(model='llama3.2')
embedding = OllamaEmbeddings(model='nomic-embed-text')


class State(MessagesState):
    result: str


def last_user(state: State) -> str:
    for msg in reversed(state['messages']):
        if getattr(msg, 'type', getattr(msg, 'role', '')) in ('human', 'user'):
            return msg.content if hasattr(msg, 'content') else msg.get('content', '')
    return ''


def supervisor(state: State):
    if state.get('result'):
        return Command(goto='finish', update={})
    q = last_user(state).lower()
    if any(op in q for op in ['+', '-', '*', '/', 'solve', 'calc']):
        return Command(goto='math', update={})
    return Command(goto='research', update={})


def research_agent(state: State):
    q = last_user(state)
    resp = llm.invoke([HumanMessage(content=f'Answer briefly and factually: {q}')])
    return Command(goto='supervisor', update={'result': resp.content})


def math_agent(state: State):
    q = last_user(state)
    prompt = f'Solve the math expression or question. Reply with the final answer only:\n{q}'
    resp = llm.invoke([HumanMessage(content=prompt)])
    return Command(goto='supervisor', update={'result': resp.content.strip()})


def finish(state: State):
    return Command(goto=END, update={'messages': [AIMessage(content=state.get('result', 'No result'))]})


builder = StateGraph(State)
builder.add_node('supervisor', supervisor)
builder.add_node('research', research_agent)
builder.add_node('math', math_agent)
builder.add_node('finish', finish)
builder.add_edge(START, 'supervisor')
graph = builder.compile()

out = graph.invoke({'messages': [HumanMessage("What's the capital of Japan?")]})
for m in out['messages']:
    m.pretty_print()
out = graph.invoke({'messages': [HumanMessage('Compute 12*(3+4)')]})
for m in out['messages']:
    m.pretty_print()

# %% [markdown]
# ## 2. Network Architecture
#

# %%
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command


# Define our state schema
class NetworkState(MessagesState):
    """State for network of collaborating agents"""

    iteration_count: int = 0


# Network Agents - each can route to any other
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


# Usage example
if __name__ == '__main__':
    graph = create_network_graph()

    # Test network collaboration
    result = graph.invoke(
        {'messages': [HumanMessage(content='We need to develop a strategy for market expansion')], 'iteration_count': 0}
    )

    print('Network Collaboration Result:')
    for i, message in enumerate(result['messages']):
        print(f'{i + 1}. {message.content}')
        print()

# %%


# %%
# pip install langgraph langchain langchain-ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

llm = ChatOllama(model='llama3.2')
embedding = OllamaEmbeddings(model='nomic-embed-text')


class NetState(MessagesState):
    turns: int


MAX_TURNS = 4


def last_user(state: NetState) -> str:
    for msg in reversed(state['messages']):
        if getattr(msg, 'type', getattr(msg, 'role', '')) in ('human', 'user'):
            return msg.content if hasattr(msg, 'content') else msg.get('content', '')
    return ''


def stop(state: NetState) -> bool:
    if state.get('turns', 0) >= MAX_TURNS:
        return True
    last = state['messages'][-1]
    txt = getattr(last, 'content', '')
    return 'stop' in str(txt).lower()


def alpha(state: NetState):
    t = state.get('turns', 0) + 1
    topic = last_user(state)
    resp = llm.invoke([HumanMessage(content=f'As agent alpha, add one sentence about: {topic}')])
    upd = {'messages': [AIMessage(content=resp.content)], 'turns': t}
    return Command(goto='end' if stop({**state, **upd}) else 'beta', update=upd)


def beta(state: NetState):
    t = state.get('turns', 0) + 1
    topic = last_user(state)
    resp = llm.invoke([HumanMessage(content=f'As agent beta, refine in one sentence: {topic}')])
    upd = {'messages': [AIMessage(content=resp.content)], 'turns': t}
    return Command(goto='end' if stop({**state, **upd}) else 'gamma', update=upd)


def gamma(state: NetState):
    t = state.get('turns', 0) + 1
    topic = last_user(state)
    resp = llm.invoke([HumanMessage(content=f'As agent gamma, propose a conclusion in one sentence about: {topic}')])
    upd = {'messages': [AIMessage(content=resp.content)], 'turns': t}
    return Command(goto='end' if stop({**state, **upd}) else 'alpha', update=upd)


def end_node(state: NetState):
    return Command(goto=END, update={'messages': [AIMessage(content='Consensus reached. Done.')]})


builder = StateGraph(NetState)
builder.add_node('alpha', alpha)
builder.add_node('beta', beta)
builder.add_node('gamma', gamma)
builder.add_node('end', end_node)
builder.add_edge(START, 'alpha')
graph = builder.compile()

if __name__ == '__main__':
    init = {'messages': [HumanMessage('Discuss pros/cons of remote work.')], 'turns': 0}
    out = graph.invoke(init)
    for m in out['messages']:
        m.pretty_print()

# %% [markdown]
# ## 3. Hierarchical Architecture
#

# %%
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessageGraph
from typing_extensions import TypedDict


# Define teams as subgraphs
def research_node(messages: list[AnyMessage]):
    print('---CALLING RESEARCH TEAM---')
    # In a real app, this would be a more complex graph
    return HumanMessage(content='LangGraph is a framework for building stateful, multi-agent applications.')


def analysis_node(messages: list[AnyMessage]):
    print('---CALLING ANALYSIS TEAM---')
    # This team "analyzes" the research output
    return HumanMessage(content=f'Analysis complete. Summary: {messages[-1].content}')


research_team = MessageGraph()
research_team.add_node('researcher', research_node)
research_team.add_edge('researcher', END)
research_graph = research_team.compile()

analysis_team = MessageGraph()
analysis_team.add_node('analyzer', analysis_node)
analysis_team.add_edge('analyzer', END)
analysis_graph = analysis_team.compile()


# Top-level supervisor routes to the appropriate team
def top_supervisor(messages: list[AnyMessage]) -> Literal['research_team', 'analysis_team', END]:
    last_message = messages[-1]
    if 'research' in last_message.content.lower():
        return 'research_team'
    # If the last message came from the research team, pass to analysis
    if isinstance(last_message, HumanMessage) and 'LangGraph is a framework' in last_message.content:
        return 'analysis_team'
    return END


# Build the final graph by composing the subgraphs
graph_builder = MessageGraph()
graph_builder.add_node('research_team', research_graph)
graph_builder.add_node('analysis_team', analysis_graph)
graph_builder.add_conditional_edges('__start__', top_supervisor)
graph_builder.add_conditional_edges('research_team', top_supervisor)
graph_builder.add_edge('analysis_team', END)
graph = graph_builder.compile(checkpointer=MemorySaver())

# Run the full hierarchy
config = {'configurable': {'thread_id': '3'}}
initial_message = [HumanMessage(content='Can you research LangGraph and then analyze the result?')]
for chunk in graph.stream(initial_message, config=config):
    print(chunk)


# %%
# pip install langgraph langchain langchain-ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

llm = ChatOllama(model='llama3.2')
embedding = OllamaEmbeddings(model='nomic-embed-text')


class TopState(MessagesState):
    data: str
    notes: str


def last_user(s: TopState) -> str:
    for m in reversed(s['messages']):
        if getattr(m, 'type', getattr(m, 'role', '')) in ('human', 'user'):
            return m.content if hasattr(m, 'content') else m.get('content', '')
    return ''


DOCS = [
    'Python was created by Guido van Rossum.',
    'The capital of Japan is Tokyo.',
    'LangGraph helps build multi-agent workflows.',
]
V_DOCS = [embedding.embed_query(d) for d in DOCS]


def gather(state: TopState):
    q = last_user(state)
    qv = embedding.embed_query(q)
    idx = max(range(len(DOCS)), key=lambda i: sum(a * b for a, b in zip(V_DOCS[i], qv)))
    return {'data': DOCS[idx]}


def synthesize(state: TopState):
    prompt = f'Using this note, answer the user briefly:\nNote: {state.get("data", "")}\nUser: {last_user(state)}'
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {'notes': resp.content}


def compute(state: TopState):
    q = last_user(state)
    resp = llm.invoke([HumanMessage(content=f'Analyze and give a score 0-1 with a reason for: {q}')])
    return {'data': resp.content}


def summarize(state: TopState):
    resp = llm.invoke([HumanMessage(content=f'Summarize to one sentence: {state.get("data", "")}')])
    return {'notes': resp.content}


# Research subgraph
r = StateGraph(TopState)
r.add_node('gather', gather)
r.add_node('synth', synthesize)
r.add_edge(START, 'gather')
r.add_edge('gather', 'synth')
r.add_edge('synth', END)
research_team = r.compile()

# Analysis subgraph
a = StateGraph(TopState)
a.add_node('compute', compute)
a.add_node('summary', summarize)
a.add_edge(START, 'compute')
a.add_edge('compute', 'summary')
a.add_edge('summary', END)
analysis_team = a.compile()


def supervisor(state: TopState):
    q = last_user(state).lower()
    if state.get('notes'):
        return Command(goto='final', update={})
    if any(op in q for op in ['+', '-', '*', '/', 'calc', 'solve']):
        return Command(goto='analysis_team', update={})
    return Command(goto='research_team', update={})


def final_node(state: TopState):
    return Command(goto=END, update={'messages': [AIMessage(content=state.get('notes', 'No notes'))]})


b = StateGraph(TopState)
b.add_node('supervisor', supervisor)
b.add_node('research_team', research_team)
b.add_node('analysis_team', analysis_team)
b.add_node('final', final_node)
b.add_edge(START, 'supervisor')
b.add_edge('research_team', 'final')
b.add_edge('analysis_team', 'final')
graph = b.compile()

if __name__ == '__main__':
    out = graph.invoke({'messages': [HumanMessage('Who created Python?')]})
    for m in out['messages']:
        m.pretty_print()
    out = graph.invoke({'messages': [HumanMessage('Quickly estimate 20*(5-2).')]})
    for m in out['messages']:
        m.pretty_print()

# %%
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command


# Define our state schema
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


# Usage example
if __name__ == '__main__':
    graph = create_hierarchical_graph()

    # Test with research task
    result1 = graph.invoke(
        {'messages': [HumanMessage(content='We need to gather information about customer preferences')]}
    )
    print('Research Task Result:')
    for message in result1['messages']:
        print(f'- {message.content}')
    print()

    # Test with analysis task
    result2 = graph.invoke({'messages': [HumanMessage(content='Analyze the sales data trends')]})
    print('Analysis Task Result:')
    for message in result2['messages']:
        print(f'- {message.content}')

# %% [markdown]
# ## CC

# %%
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command


# Define our state schema
class SupervisorState(MessagesState):
    """State that includes messages and any additional context"""

    pass


# Worker Agents
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


# Build the graph
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


# Usage example
if __name__ == '__main__':
    graph = create_supervisor_graph()

    # Test with research query
    result1 = graph.invoke({'messages': [HumanMessage(content='I need to research climate change impacts')]})
    print('Research Query Result:')
    print(result1['messages'][-1].content)
    print()

    # Test with math query
    result2 = graph.invoke({'messages': [HumanMessage(content='Calculate the probability distribution')]})
    print('Math Query Result:')
    print(result2['messages'][-1].content)

# %%
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command


# Define our state schema
class NetworkState(MessagesState):
    """State for network of collaborating agents"""

    iteration_count: int = 0


# Network Agents - each can route to any other
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


# Usage example
if __name__ == '__main__':
    graph = create_network_graph()

    # Test network collaboration
    result = graph.invoke(
        {'messages': [HumanMessage(content='We need to develop a strategy for market expansion')], 'iteration_count': 0}
    )

    print('Network Collaboration Result:')
    for i, message in enumerate(result['messages']):
        print(f'{i + 1}. {message.content}')
        print()

# %%
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command


# Define our state schema
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


# Usage example
if __name__ == '__main__':
    graph = create_hierarchical_graph()

    # Test with research task
    result1 = graph.invoke(
        {'messages': [HumanMessage(content='We need to gather information about customer preferences')]}
    )
    print('Research Task Result:')
    for message in result1['messages']:
        print(f'- {message.content}')
    print()

    # Test with analysis task
    result2 = graph.invoke({'messages': [HumanMessage(content='Analyze the sales data trends')]})
    print('Analysis Task Result:')
    for message in result2['messages']:
        print(f'- {message.content}')

v
