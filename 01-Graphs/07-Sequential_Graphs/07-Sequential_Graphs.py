# %% [markdown]
# # LangGraph Tutorial: Interactive Chatbot with Advanced Features
# 
# ## Introduction
# 
# This notebook provides a comprehensive tutorial on building interactive chatbot workflows using LangGraph. It covers fundamental concepts such as:
# 
# - **Nodes:** Representing individual processing steps.
# - **Edges:** Defining the flow of execution between nodes.
# - **Conditional Routing:** Making decisions based on the state of the graph.
# - **Looping:** Repeating certain steps based on specific conditions.
# 
# ## Environment Setup
# 

# %%
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich import print
from IPython.display import Image, display
import os

# Load environment variables from .env file
load_dotenv("../.env")

# Ensure the OpenAI API key is set
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# %% [markdown]
# ## 1. Define the State
# 
# The state represents the information that is passed between nodes in the graph.

# %%
class AgentState(TypedDict):
    user_name: str                  # User's name
    user_question: str              # User's current question
    chat_history: list[BaseMessage] # List of messages in the conversation
    knowledge_cutoff: str           # Knowledge cutoff date for the model
    conversation_finished: bool     # Flag to indicate if the conversation is finished

# %% [markdown]
# ## 2. Define the Nodes
# 
# Nodes are the individual steps in the graph. Each node is a function that takes the current state as input and returns a dictionary that updates the state.

# %%
def greet_user(state: AgentState) -> Dict[str, Any]:
    """Greets the user and initializes the conversation."""
    user_name = input("Hello! What is your name? ")
    return {
        "user_name": user_name,
        "chat_history": [AIMessage(content=f"Hi {user_name}! How can I help you today?")],
        "conversation_finished": False
    }


def get_knowledge_cutoff(state: AgentState) -> Dict[str, Any]:
    """Gets the knowledge cutoff date for the model."""
    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = model.invoke("What is your knowledge cutoff date?")
    return {"knowledge_cutoff": response.content}


def ask_question(state: AgentState) -> Dict[str, Any]:
    """
    Asks the user for their question. 
    If the user types '/exit', sets conversation_finished = True.
    """
    user_question = input("You (type /exit to quit): ")
    if user_question.strip() == "/exit":
        # Directly signal that we should end the conversation
        return {"user_question": user_question, "conversation_finished": True}
    return {"user_question": user_question}


def generate_response(state: AgentState) -> Dict[str, Any]:
    """Generates a response to the user's question using a ChatOpenAI model."""
    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    # Append the user's question to history and get a response
    messages = state["chat_history"] + [HumanMessage(content=state["user_question"])]
    response = model.invoke(messages)
    new_history = state["chat_history"] + [HumanMessage(content=state["user_question"]), response]
    return {"chat_history": new_history}


def evaluate_response(state: AgentState) -> Dict[str, bool]:
    """
    Evaluates if the conversation should continue or end.
    If conversation_finished is already True, we stay True.
    Otherwise, ask the user if they want to continue.
    """
    if state.get("conversation_finished", False):
        return {"conversation_finished": True}

    user_input = input("Do you have any more questions? (yes/no): ").lower()
    if user_input in ("no", "n"):
        return {"conversation_finished": True}
    return {"conversation_finished": False}


def goodbye_message(state: AgentState) -> Dict[str, Any]:
    """Sends a goodbye message to the user."""
    print("Goodbye!")
    return {}

# %% [markdown]
# ## 3. Define the Graph
# 
# The graph defines the flow of execution between the nodes. Note how “ask” now conditionally jumps to “goodbye” if the user typed `/exit`.

# %%
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("greet", greet_user)
graph.add_node("get_cutoff", get_knowledge_cutoff)
graph.add_node("ask", ask_question)
graph.add_node("respond", generate_response)
graph.add_node("evaluate", evaluate_response)
graph.add_node("goodbye", goodbye_message)

# Define edges
graph.add_edge(START, "greet")
graph.add_edge("greet", "get_cutoff")

# Instead of a direct edge from "ask" → "respond", we add a conditional edge:
#   - If the user typed "/exit", state["conversation_finished"] is True, so send to "goodbye"
#   - Otherwise, go to "respond"
def ask_outcome(state: AgentState) -> str:
    return "goodbye" if state.get("conversation_finished", False) else "respond"

graph.add_edge("get_cutoff", "ask")
graph.add_conditional_edges("ask", ask_outcome, {"respond": "respond", "goodbye": "goodbye"})

# After "respond", always check if we should loop or finish
graph.add_edge("respond", "evaluate")

# From "evaluate", either loop back to "ask" or go to "goodbye"
def should_continue(state: AgentState) -> str:
    return "ask" if not state.get("conversation_finished", False) else "goodbye"

graph.add_conditional_edges("evaluate", should_continue, {"ask": "ask", "goodbye": "goodbye"})
graph.add_edge("goodbye", END)

# Compile the graph
app = graph.compile()

# %% [markdown]
# ## 4. Visualize the Graph
# 
# Visualizing the graph can help you understand the flow of execution.

# %%
# Visualize the Graph (requires graphviz and pygraphviz or pydot)
display(Image(app.get_graph().draw_mermaid_png()))

# Save the graph to a file
with open("../output/07-Interactive_Chatbot.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

# %% [markdown]
# ## 5. Run the Graph
# 
# Now, let's run the graph with an initial state. Typing `/exit` at any “You:” prompt will immediately jump to the goodbye node.

# %%
# Initial state
initial_state = {
    "user_name": "",
    "user_question": "",
    "chat_history": [],
    "knowledge_cutoff": "",
    "conversation_finished": False,
}

# Run the graph
result = app.invoke(initial_state)

# Print the chat history (if any)
print("\n--- Chat History ---")
for message in result.get("chat_history", []):
    message.pretty_print()

