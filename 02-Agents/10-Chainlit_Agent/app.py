from typing import Literal, List, Dict
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState


# ----------------------------------
# Tool Definition
# ----------------------------------
@tool
def get_weather(city: Literal['nyc', 'sf']) -> str:
    """Fetch basic weather info for supported cities."""
    weather_map = {
        'nyc': 'It might be cloudy in NYC',
        'sf': "It's always sunny in SF",
    }
    try:
        return weather_map[city]
    except KeyError:
        raise ValueError(f'Unknown city: {city}')


# ----------------------------------
# LLM Setup
# ----------------------------------
def create_llm(model_name: str, temperature: float = 0.0, tags: List[str] = None) -> ChatOpenAI:
    """Initialize and configure a ChatOpenAI instance."""
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    if tags:
        llm = llm.with_config(tags=tags)
    return llm.bind_tools([get_weather])


# Instantiate base and final LLMs
base_llm = create_llm('gpt-3.5-turbo')
final_llm = create_llm('gpt-3.5-turbo', tags=['final_node'])

tool_node = ToolNode(tools=[get_weather])


# ----------------------------------
# Graph Node Functions
# ----------------------------------
def should_route(state: MessagesState) -> Literal['tools', 'final']:
    """Decide whether to invoke tools or finish the conversation."""
    last = state['messages'][-1]
    return 'tools' if last.tool_calls else 'final'


def call_base_llm(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Invoke the base LLM and append its response."""
    response = base_llm.invoke(state['messages'])
    return {'messages': [response]}


def call_final_llm(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """Invoke the final LLM to rewrite the last message."""
    last = state['messages'][-1]
    rewritten = final_llm.invoke([
        SystemMessage(content='Rewrite this in the voice of Al Roker.'),
        HumanMessage(content=last.content),
    ])
    rewritten.id = last.id  # preserve original ID
    return {'messages': [rewritten]}


# ----------------------------------
# Build State Graph
# ----------------------------------
def build_state_graph() -> StateGraph:
    builder = StateGraph(MessagesState)
    builder.add_node('agent', call_base_llm)
    builder.add_node('tools', tool_node)
    builder.add_node('final', call_final_llm)

    builder.add_edge(START, 'agent')
    builder.add_conditional_edges('agent', should_route)
    builder.add_edge('tools', 'agent')
    builder.add_edge('final', END)
    return builder.compile()


agent_graph = build_state_graph()


# ----------------------------------
# Chainlit Event Handler
# ----------------------------------
@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming messages and stream the final response."""
    thread_id = cl.context.session.id
    callbacks = [cl.LangchainCallbackHandler()]
    config = RunnableConfig(callbacks=callbacks, configurable={'thread_id': thread_id})

    history = [HumanMessage(content=msg.content)]
    reply = cl.Message(content='')

    for response, meta in agent_graph.stream(
        {'messages': history},
        stream_mode='messages',
        config=config,
    ):
        if response.content and meta.get('langgraph_node') == 'final':
            await reply.stream_token(response.content)

    await reply.send()
