from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import chainlit as cl
import os
from dotenv import load_dotenv

################################## Setup ##################################
load_dotenv()  # Load environment variables from a .env file

openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

############################# Graph Definition ############################
# Initialize the state graph with a schema for storing messages
workflow = StateGraph(state_schema=MessagesState)

model = ChatOpenAI(
    model='google/gemini-2.5-flash-lite',
    base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
    api_key=openrouter_api_key,
    temperature=0,  # Set for deterministic, less creative responses
)


def call_model(state: MessagesState):
    """Invokes the model with the current state and returns the new message."""
    response = model.invoke(state['messages'])
    return {'messages': response}  # Update the state with the model's response


workflow.add_node('model', call_model)  # Add the model-calling function as a node
workflow.add_edge(START, 'model')  # Set the 'model' node as the entry point

############################ Graph Compilation ############################
memory = MemorySaver()  # Initialize in-memory storage for conversation history

# Compile the graph into a runnable, adding the memory checkpointer
app = workflow.compile(checkpointer=memory)


############################### Chainlit UI ###############################
@cl.on_message
async def main(message: cl.Message):
    """Process incoming user messages and stream back the AI's response."""
    answer = cl.Message(content='')  # Create an empty message to stream the response into
    await answer.send()

    # Configure the runnable to associate the conversation with the user's session
    config: RunnableConfig = {'configurable': {'thread_id': cl.context.session.thread_id}}

    # Stream the graph's output
    for msg, _ in app.stream(
        {'messages': [HumanMessage(content=message.content)]},  # Pass the user's message
        config,
        stream_mode='messages',  # Stream individual message chunks
    ):
        # Check if the current streamed item is an AI message chunk
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content  # type: ignore # Append the content chunk
            await answer.update()  # Update the UI with the appended content
