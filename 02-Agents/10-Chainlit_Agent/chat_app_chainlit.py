from typing import Literal, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
import chainlit as cl

# --- Constants ---
NYC = 'nyc'
SF = 'sf'
FINAL_NODE_TAG = 'final_node'
GPT_MODEL_NAME = 'gpt-3.5-turbo'
AL_ROKER_VOICE_PROMPT = 'Rewrite this in the voice of Al Roker'


# --- Tools ---
@tool
def get_weather(city: Literal[NYC, SF]) -> str:
    """Use this to get weather information for a specified city."""
    if city == NYC:
        return 'It might be cloudy in nyc'
    elif city == SF:
        return "It's always sunny in sf"
    else:
        # This case should ideally not be reached due to Literal type hint
        raise ValueError(f'Unknown city: {city}. Only "{NYC}" and "{SF}" are supported.')


# --- Language Model Setup ---
def setup_llms():
    """Initializes and configures the language models."""
    # Main model for agentic reasoning
    agent_model = ChatOpenAI(model_name=GPT_MODEL_NAME, temperature=0).bind_tools([get_weather])

    # Final model for rephrasing, tagged for specific event filtering
    final_rephrasing_model = ChatOpenAI(model_name=GPT_MODEL_NAME, temperature=0).with_config(tags=[FINAL_NODE_TAG])
    return agent_model, final_rephrasing_model


# Initialize models globally or pass them through a class/state if preferred
AGENT_MODEL, FINAL_REPHRASING_MODEL = setup_llms()


# --- Graph Node Definitions ---
def call_agent_model(state: MessagesState) -> MessagesState:
    """Invokes the agent model with the current messages and returns its response."""
    messages = state['messages']
    response = AGENT_MODEL.invoke(messages)
    # Append the AI's response to the message history
    return {'messages': [response]}


def call_final_rephrasing_model(state: MessagesState) -> MessagesState:
    """
    Rewrites the last AI message using the final rephrasing model in Al Roker's voice.
    The ID of the original AI message is preserved for continuity in UI.
    """
    messages = state['messages']
    last_ai_message: AIMessage = messages[-1]  # Assuming the last message is from the AI

    # Construct the prompt for the final rephrasing model
    rephrase_prompt = [
        SystemMessage(content=AL_ROKER_VOICE_PROMPT),
        HumanMessage(content=last_ai_message.content),
    ]
    response = FINAL_REPHRASING_MODEL.invoke(rephrase_prompt)

    # Preserve the original message ID for UI updates in Chainlit
    response.id = last_ai_message.id

    # Overwrite the last AI message with the rephrased version
    return {'messages': [response]}


# --- Graph Edge Logic ---
def should_continue(state: MessagesState) -> Literal['tools', 'final']:
    """
    Determines the next step in the graph based on the last message from the agent.
    If the agent made a tool call, route to 'tools'; otherwise, route to 'final'.
    """
    last_message: BaseMessage = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return 'final'


# --- Graph Construction ---
def build_graph():
    """Constructs and compiles the LangGraph state machine."""
    workflow = StateGraph(MessagesState)

    # Add nodes to the graph
    workflow.add_node('agent', call_agent_model)
    workflow.add_node('tools', ToolNode(tools=[get_weather]))
    workflow.add_node('final', call_final_rephrasing_model)  # Separate node for final processing

    # Define the entry point
    workflow.add_edge(START, 'agent')

    # Define conditional routing from the agent
    workflow.add_conditional_edges(
        'agent',
        should_continue,
        {'tools': 'tools', 'final': 'final'},  # Map return values of should_continue to node names
    )

    # Define paths from tool execution
    workflow.add_edge('tools', 'agent')  # After tool execution, go back to agent for next steps

    # Define the exit point for the final processed message
    workflow.add_edge('final', END)

    return workflow.compile()


# Compile the graph
graph = build_graph()


# --- Chainlit Integration ---
@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages from the Chainlit UI, runs them through the LangGraph,
    and streams the final AI response back to the user.
    """
    # Configure LangGraph for Chainlit session
    # thread_id ensures message history is tied to the current chat session
    config = {'configurable': {'thread_id': cl.context.session.id}}

    # Chainlit callback handler for real-time streaming and visibility
    chainlit_callback = cl.LangchainCallbackHandler()

    # Prepare a Chainlit message to stream the final answer into
    final_answer_message = cl.Message(content='')

    # Stream graph execution, focusing on the 'final' node's output
    # `stream_mode='messages'` provides a stream of message dictionaries
    async for msg_dict, metadata in graph.stream(
        {'messages': [HumanMessage(content=message.content)]},
        stream_mode='messages',
        config=RunnableConfig(callbacks=[chainlit_callback], **config),
    ):
        # Only process and stream content from the 'final' node's AI message
        # Ensure it's not a HumanMessage (which indicates input or internal messages)
        is_ai_message_from_final_node = (
            msg_dict.get('messages')
            and isinstance(msg_dict['messages'][-1], AIMessage)
            and metadata.get('langgraph_node') == 'final'
        )

        if is_ai_message_from_final_node:
            content_to_stream = msg_dict['messages'][-1].content
            if content_to_stream:
                await final_answer_message.stream_token(content_to_stream)

    # Send the completed streamed message to the user
    await final_answer_message.send()
