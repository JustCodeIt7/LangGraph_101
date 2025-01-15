from typing import TypedDict, Annotated, Sequence
from uuid import UUID
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
import operator
from langchain_ollama import ChatOllama, OllamaEmbeddings

model = ChatOllama(model='llama3.2', temperature=0.3)
embeddings = OllamaEmbeddings(model='snowflake-arctic-embed:33m')

# Define our state structure
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next_step: str

# Define the function that processes messages
def chat_agent(state: AgentState) -> AgentState:
    # Get messages from state
    messages = state["messages"]
    
    # Generate response using the chat model
    response = model.invoke(messages)
    
    # Add AI's response to messages
    new_messages = list(messages) + [response]
    
    return {
        "messages": new_messages,
        "next_step": "end"
    }

# Define condition for continuing the conversation
def should_continue(state: AgentState) -> str:
    return state["next_step"]

# Create the workflow graph
workflow = Graph()

# Add nodes and edges
workflow.add_node("chat", chat_agent)

# Add conditional edges
workflow.add_conditional_edges(
    "chat",
    should_continue,
    {
        "end": None,
    }
)

# Compile the graph
app = workflow.compile()

# Function to run a chat interaction
def chat(message: str, chat_history: list[BaseMessage] = None) -> list[BaseMessage]:
    if chat_history is None:
        chat_history = []
    
    # Create initial state
    state = {
        "messages": chat_history + [HumanMessage(content=message)],
        "next_step": "chat"
    }
    
    # Run the workflow
    result = app.invoke(state)
    
    return result["messages"]

# Example usage
if __name__ == "__main__":
    # Start a conversation
    messages = chat("Hello! How are you today?")
    print("AI:", messages[-1].content)
    
    # Continue the conversation
    messages = chat("Tell me about yourself.", messages)
    print("AI:", messages[-1].content)
