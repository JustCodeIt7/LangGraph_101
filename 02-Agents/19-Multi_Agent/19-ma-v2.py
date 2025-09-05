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
