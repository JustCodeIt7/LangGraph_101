from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
import chainlit as cl

# Initialize a single LLM
# model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

model = ChatOllama(model_name='qwen3:4b', base_url='http://eos-james.local:11434', temperature=0)  # Uncomment to use Ollama

# Define the node that just calls the LLM on the full message history
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages': [response]}


# Build a simple graph: START → chat → END
builder = StateGraph(MessagesState)
builder.add_node('chat', call_model)
builder.add_edge(START, 'chat')
builder.add_edge('chat', END)
graph = builder.compile()


# Chainlit handler: feed user input into the graph and stream back the LLM’s reply
@cl.on_message
async def on_message(msg: cl.Message):
    config = {'configurable': {'thread_id': cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    reply = cl.Message(content='')

    # Stream through the graph
    for out_msg, meta in graph.stream(
        {'messages': [HumanMessage(content=msg.content)]},
        stream_mode='messages',
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        # Only stream tokens from our "chat" node
        if out_msg.content and meta['langgraph_node'] == 'chat':
            await reply.stream_token(out_msg.content)

    await reply.send()
