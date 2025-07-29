from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import chainlit as cl

from dotenv import load_dotenv

load_dotenv()


workflow = StateGraph(state_schema=MessagesState)
model = ChatOpenAI(model='gpt-4o-mini', temperature=0)


def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {'messages': response}


workflow.add_edge(START, 'model')
workflow.add_node('model', call_model)

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass


@cl.on_message
async def main(message: cl.Message):
    answer = cl.Message(content='')
    await answer.send()

    config: RunnableConfig = {'configurable': {'thread_id': cl.context.session.thread_id}}

    for msg, _ in app.stream(
        {'messages': [HumanMessage(content=message.content)]},
        config,
        stream_mode='messages',
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content  # type: ignore
            await answer.update()
