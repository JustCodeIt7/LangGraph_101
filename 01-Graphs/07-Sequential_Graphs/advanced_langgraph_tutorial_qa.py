# %% [markdown]
# # Building Dynamic Chatbot Workflows with LangGraph
#
# ## Introduction
#
# This tutorial will guide you through building a dynamic chatbot using LangGraph, a library for orchestrating LLM-powered applications. We'll explore core LangGraph concepts like **nodes**, **edges**, **conditional routing**, and **looping** to create intelligent, multi-step workflows.
#
# ## Environment Setup
#
# First, let's set up our environment and import the necessary libraries. We'll use `langgraph` for graph orchestration, `langchain` for interacting with LLMs, and `rich` for pretty printing.
#
# Make sure you have your `.env` file configured with your API keys (e.g., for OpenAI).

# %%
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from dotenv import load_dotenv
from rich import print
from IPython.display import Image, display

# Load environment variables from .env file
load_dotenv("../.env")

# Initialize the LLM (using OpenAI for this tutorial, but you can swap it)
# Make sure OPENAI_API_KEY is set in your .env file
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# %% [markdown]
# ## Core Concepts in LangGraph
#
# Before diving into the code, let's understand the fundamental building blocks of LangGraph:
#
# * **Nodes**: These are the individual steps or functions in your workflow. Each node takes the current state as input and returns an updated state.
# * **Edges**: These define the transitions between nodes. An edge specifies which node to execute after a particular node finishes.
# * **State**: The central data structure that is passed between nodes. Each node can read from and write to this shared state.
# * **Conditional Routing**: This allows the graph to make decisions about which path to take next, based on the current state.
# * **Looping**: Enables the graph to revisit previous nodes, creating iterative processes (e.g., for clarification or multi-turn conversations).
#
# ---
#
# ## Example 1: Simple Sequential Chatbot (Nodes and Edges)
#
# Let's start with a basic chatbot to illustrate nodes and edges. This bot will greet the user, ask a question, and then answer it.
#
# ### 1. Define the Shared State Schema
#
# The `AgentState` will hold all the information relevant to our conversation. We'll use `TypedDict` for clear type hinting.

# %%
class AgentState(TypedDict):
    """
    Represents the state of our chatbot's conversation.

    Attributes:
        name: The user's name.
        user_question: The specific question the user asks.
        chat_history: A list of all messages in the conversation (Human and AI).
        clarification_needed: A flag to indicate if the chatbot needs more info.
    """
    name: str
    user_question: str
    chat_history: list[AnyMessage]
    clarification_needed: bool

# %% [markdown]
# ### 2. Define Node Functions
#
# Each node will be a Python function that takes the `AgentState` as input and returns a dictionary to update the state.

# %%
def greet_user(state: AgentState) -> dict:
    """
    Node 1: Greets the user and captures their name.
    This simulates an initial interaction where we learn about the user.
    """
    print("\n--- Executing Node: greet_user ---")
    user_name = input("[Chatbot] Hello! What's your name? ")
    # For a real app, you might get this from an initial state or user profile.
    # For this tutorial, we'll hardcode it for demonstration.
    # state["name"] = user_name # If we wanted to take live input.
    name = "Alice" # Hardcoding for consistent tutorial output
    greeting_message = AIMessage(f"Hi {name}! I'm here to answer your questions.")
    print(f"[Chatbot] {greeting_message.content}")
    return {"name": name, "chat_history": [greeting_message]}

def get_user_question(state: AgentState) -> dict:
    """
    Node 2: Prompts the user for their question and stores it in the state.
    """
    print("\n--- Executing Node: get_user_question ---")
    question = input(f"[Chatbot] What would you like to know, {state['name']}? ")
    user_message = HumanMessage(question)
    print(f"[You] {question}")
    return {"user_question": question, "chat_history": state["chat_history"] + [user_message]}

def answer_question(state: AgentState) -> dict:
    """
    Node 3: Uses the LLM to answer the user's question based on the chat history.
    """
    print("\n--- Executing Node: answer_question ---")
    # Build the full message history to send to the LLM
    # We include a system message to set the context for the LLM
    messages_for_llm = [
        AIMessage("You are a helpful and knowledgeable AI assistant. Answer the user's questions concisely."),
    ] + state["chat_history"]

    print(f"[Chatbot] Thinking about '{state['user_question']}'...")
    response_message: AIMessage = llm.invoke(messages_for_llm)
    print(f"[Chatbot] {response_message.content}")
    return {"chat_history": state["chat_history"] + [response_message]}

# %% [markdown]
# ### 3. Assemble the Sequential Graph
#
# We'll use `StateGraph` to define our workflow. We add nodes and then connect them using `add_edge`.
#
# * `START`: The entry point of the graph.
# * `END`: The exit point of the graph.

# %%
# Build the graph
simple_chatbot_graph = StateGraph(AgentState)

# Add nodes
simple_chatbot_graph.add_node("greet", greet_user)
simple_chatbot_graph.add_node("get_question", get_user_question)
simple_chatbot_graph.add_node("answer", answer_question)

# Define edges: control flow from one node to the next
simple_chatbot_graph.add_edge(START, "greet")
simple_chatbot_graph.add_edge("greet", "get_question")
simple_chatbot_graph.add_edge("get_question", "answer")
simple_chatbot_graph.add_edge("answer", END) # The conversation ends after answering

# Compile the graph into an executable application
simple_chatbot_app = simple_chatbot_graph.compile()

# %% [markdown]
# ### 4. Visualize the Graph
#
# It's always helpful to visualize your graph to understand its flow. This requires `graphviz` to be installed (e.g., `brew install graphviz` on macOS, `sudo apt-get install graphviz` on Debian/Ubuntu, or install the `graphviz` Python package).

# %%
# Visualize the Simple Chatbot Graph
print("\n--- Visualizing Simple Chatbot Graph ---")
display(Image(simple_chatbot_app.get_graph().draw_mermaid_png()))

# Save the graph to a file for easy reference
with open("../output/simple_chatbot_graph.png", "wb") as f:
    f.write(simple_chatbot_app.get_graph().draw_mermaid_png())

# %% [markdown]
# ### 5. Invoke the Simple Graph
#
# Now, let's run our simple chatbot. We provide an initial state (though for this sequential graph, much of it gets populated by nodes).

# %%
print("\n--- Invoking Simple Chatbot ---")
# The initial state can be empty or partially filled, as nodes will populate it.
initial_state_simple = {"name": "", "user_question": "", "chat_history": [], "clarification_needed": False}
final_state_simple = simple_chatbot_app.invoke(initial_state_simple)

print("\n--- Simple Chatbot Conversation Log ---")
for msg in final_state_simple["chat_history"]:
    if isinstance(msg, HumanMessage):
        print(f"[You] {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"[Chatbot] {msg.content}")

# %% [markdown]
# ---
#
# ## Example 2: Chatbot with Conditional Routing and Looping
#
# Now, let's make our chatbot smarter by introducing conditional logic and the ability to loop for clarification.
#
# **Scenario**: If the user's question is unclear or the LLM indicates a need for clarification, the bot should ask for more details and then try to answer again.
#
# ### 1. Refined Node Functions (for conditional logic)
#
# We'll add a new node for clarification and modify `answer_question` to determine if clarification is needed.

# %%
def check_clarification(state: AgentState) -> Literal["clarify", "answer"]:
    """
    Conditional Edge: Determines if the chatbot needs to ask for clarification.
    This function analyzes the last AI message for keywords or patterns that suggest
    the LLM couldn't fully answer and needs more information.
    """
    print("\n--- Executing Conditional Edge: check_clarification ---")
    last_ai_message = state["chat_history"][-1].content.lower()

    # Simple heuristic: if the AI response contains "unclear" or "more information",
    # we assume clarification is needed. In a real app, this would be more robust,
    # perhaps using another LLM call to assess clarity.
    if "unclear" in last_ai_message or "more information" in last_ai_message or "could you please elaborate" in last_ai_message:
        print("[Decision] Clarification needed.")
        return "clarify"
    else:
        print("[Decision] No clarification needed, proceeding to final answer.")
        return "answer"

def clarify_question(state: AgentState) -> dict:
    """
    Node 4: Asks the user for clarification.
    Sets the 'clarification_needed' flag to True.
    """
    print("\n--- Executing Node: clarify_question ---")
    clarification_prompt = AIMessage(f"I apologize, {state['name']}. Your previous question was a bit unclear. Could you please provide more details or rephrase it?")
    print(f"[Chatbot] {clarification_prompt.content}")
    return {
        "chat_history": state["chat_history"] + [clarification_prompt],
        "clarification_needed": True
    }

def get_clarified_question(state: AgentState) -> dict:
    """
    Node 5: Gets the user's clarified question.
    This effectively acts as a loop back to getting input.
    """
    print("\n--- Executing Node: get_clarified_question ---")
    new_question = input("[You - Clarification] ")
    clarified_user_message = HumanMessage(new_question)
    return {
        "user_question": new_question,
        "chat_history": state["chat_history"] + [clarified_user_message],
        "clarification_needed": False # Reset flag after getting clarification
    }


# The `answer_question` node from the previous example will be reused.
# It will now be called multiple times if clarification is needed.

# %% [markdown]
# ### 2. Assemble the Graph with Conditional Routing and Looping
#
# Here, we'll use `add_conditional_edges` to introduce decision points.

# %%
# Build the graph
conditional_chatbot_graph = StateGraph(AgentState)

# Add nodes (reusing some from the simple example)
conditional_chatbot_graph.add_node("greet", greet_user)
conditional_chatbot_graph.add_node("get_question", get_user_question)
conditional_chatbot_graph.add_node("answer", answer_question)
conditional_chatbot_graph.add_node("clarify", clarify_question)
conditional_chatbot_graph.add_node("get_clarified_question", get_clarified_question)


# Define starting point
conditional_chatbot_graph.add_edge(START, "greet")
conditional_chatbot_graph.add_edge("greet", "get_question")

# After getting a question, we attempt to answer.
conditional_chatbot_graph.add_edge("get_question", "answer")

# --- Conditional Routing after "answer" node ---
# Based on the output of `check_clarification`, the graph will either:
# - go to "clarify" (if clarification is needed)
# - or go to END (if the answer is satisfactory)
conditional_chatbot_graph.add_conditional_edges(
    "answer",                  # Source node
    check_clarification,       # Function to determine the next path
    {
        "clarify": "clarify",  # If check_clarification returns "clarify", go to "clarify" node
        "answer": END          # If check_clarification returns "answer", the graph ends
    }
)

# If clarification is needed, we ask for it and then get the clarified question.
conditional_chatbot_graph.add_edge("clarify", "get_clarified_question")

# --- Looping back ---
# After getting the clarified question, we loop back to the "answer" node
# to try answering again with the new information. This creates a loop.
conditional_chatbot_graph.add_edge("get_clarified_question", "answer")

# Compile the graph
conditional_chatbot_app = conditional_chatbot_graph.compile()

# %% [markdown]
# ### 3. Visualize the Graph with Conditional Routing and Looping
#
# Notice how the conditional edge and the loop back to `answer` are represented.

# %%
# Visualize the Conditional Chatbot Graph
print("\n--- Visualizing Conditional Chatbot Graph ---")
display(Image(conditional_chatbot_app.get_graph().draw_mermaid_png()))

# Save the graph
with open("../output/conditional_chatbot_graph.png", "wb") as f:
    f.write(conditional_chatbot_app.get_graph().draw_mermaid_png())

# %% [markdown]
# ### 4. Invoke the Conditional Graph
#
# Let's test the conditional logic. Try a vague question first, then a clear one.

# %%
print("\n--- Invoking Conditional Chatbot (Scenario 1: Needs Clarification) ---")
initial_state_conditional = {"name": "", "user_question": "", "chat_history": [], "clarification_needed": False}
# In a real scenario, you'd feed inputs interactively.
# For this tutorial, we'll simulate inputs to demonstrate the flow.

# Simulating interaction for demonstration
print("Simulating interaction...")

# Simulate the first run, expecting clarification
# The `stream` method allows us to see state updates in real-time
inputs_for_run1 = {
    "name": "Bob",
    "user_question": "Tell me about stuff." # Vague question to trigger clarification
}
print("\n--- First Pass (Vague Question) ---")
# Manually pass inputs to simulate the `input()` calls in nodes
# This is a bit manual for a tutorial, but shows the state changes
# For a fully interactive demo, you'd run this in a Jupyter notebook
# and literally type into the `input()` prompts.

# Simulate the `greet_user` node's output
simulated_state_after_greet = greet_user(initial_state_conditional)
simulated_state_after_greet["name"] = "Bob" # Override for consistent demo
simulated_state_after_greet["chat_history"].append(AIMessage(f"Hi Bob! I'm here to answer your questions."))


# Simulate the `get_user_question` node's output
simulated_state_after_question = get_user_question(simulated_state_after_greet)
simulated_state_after_question["user_question"] = "Tell me about stuff."
simulated_state_after_question["chat_history"].append(HumanMessage("Tell me about stuff."))


print("\n--- Running conditional_chatbot_app with simulated vague input ---")
# The `invoke` call here will run the graph end-to-end,
# including the internal `input()` calls within nodes.
# For a true hands-on tutorial, encourage the user to type in the prompts.

# We'll make a more direct call to `invoke` but prepare the `AgentState` carefully
# for the first vague input to trigger the clarification path.
# IMPORTANT: The `input()` calls within nodes will still require user interaction
# if you run this in a live Python environment (e.g., Jupyter).
# For a non-interactive demonstration, you'd remove `input()` and pre-fill state.

# Let's manually trigger the state changes to show the flow without live input prompts for the tutorial's clarity.
# In a real interactive demo, the `input()` calls would drive this.

# Simplified for tutorial: directly set initial state to trigger the flow.
# This bypasses the interactive `input()` for "name" and initial "question" for a cleaner demo run.

print("\n--- Running Conditional Chatbot (Scenario 1: Needs Clarification) ---")
# Define an initial state that will lead to a vague answer from the LLM
# (This assumes the LLM would deem "Tell me about stuff." as vague)
# For a live demo, let the user type "Tell me about stuff." when prompted.
first_run_state = {
    "name": "Bob",
    "user_question": "Tell me about stuff.",
    "chat_history": [
        AIMessage("Hi Bob! I'm here to answer your questions."),
        HumanMessage("Tell me about stuff.")
    ],
    "clarification_needed": False
}

# Manually invoking the graph to demonstrate the flow.
# In a true interactive session, the `input()` functions would drive this.
# To make this a clearer tutorial without halting for `input()`,
# we'll simulate the state changes for the "clarification" path.

print("\n[Chatbot] Running scenario: Asking a vague question 'Tell me about stuff.'")
print("[Chatbot] (Simulating LLM response that requires clarification...)")
# Simulate the LLM's response to be vague, leading to clarification
simulated_vague_llm_response = AIMessage("I can tell you about many things! Could you please provide more information or be more specific about what 'stuff' you're interested in?")

# Now let's manually step through the graph with this simulated response
current_state = {
    "name": "Bob",
    "user_question": "Tell me about stuff.",
    "chat_history": [
        AIMessage("Hi Bob! I'm here to answer your questions."),
        HumanMessage("Tell me about stuff."),
        simulated_vague_llm_response # This is the simulated AI response
    ],
    "clarification_needed": False
}

print("\n--- State after initial vague answer from LLM ---")
print(f"Chat History: {[m.content for m in current_state['chat_history']]}")

# Now, `check_clarification` would be called. Let's simulate its output.
next_step = check_clarification(current_state)
print(f"Check clarification result: {next_step}")

if next_step == "clarify":
    print("\n--- Chatbot is now asking for clarification ---")
    current_state = clarify_question(current_state)
    print(f"Chat History: {[m.content for m in current_state['chat_history']]}")

    # Simulate getting clarified question
    print("\n[You] (Simulating user providing clarification: 'I want to know about LangGraph.')")
    clarified_q_state = get_clarified_question(current_state)
    clarified_q_state["user_question"] = "I want to know about LangGraph."
    clarified_q_state["chat_history"].append(HumanMessage("I want to know about LangGraph."))
    current_state = clarified_q_state
    print(f"Chat History: {[m.content for m in current_state['chat_history']]}")

    # Now, the graph would loop back to `answer_question` with the new input
    print("\n[Chatbot] (Simulating LLM re-attempting to answer with clarification...)")
    # For this example, we'll assume the LLM now gives a good answer.
    simulated_clear_llm_response = AIMessage("LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain by allowing cycles in your LLM application graphs, enabling more complex conversational patterns like agents that loop or perform tool use.")

    current_state["chat_history"].append(simulated_clear_llm_response)
    print(f"Chat History: {[m.content for m in current_state['chat_history']]}")

    # Finally, check_clarification would be called again
    final_step = check_clarification(current_state)
    print(f"Final clarification check result: {final_step}")
    if final_step == "answer":
        print("\n--- Conversation concluded after clarification ---")

# ---
print("\n--- Running Conditional Chatbot (Scenario 2: No Clarification Needed) ---")
# Define an initial state that will lead to a direct answer from the LLM
print("[Chatbot] Running scenario: Asking a clear question 'What is the capital of France?'")

clear_question_state = {
    "name": "Alice",
    "user_question": "What is the capital of France?",
    "chat_history": [
        AIMessage("Hi Alice! I'm here to answer your questions."),
        HumanMessage("What is the capital of France?")
    ],
    "clarification_needed": False
}

# Simulate the LLM's response to be direct
simulated_clear_llm_response_direct = AIMessage("The capital of France is Paris.")

current_state_direct = {
    "name": "Alice",
    "user_question": "What is the capital of France?",
    "chat_history": [
        AIMessage("Hi Alice! I'm here to answer your questions."),
        HumanMessage("What is the capital of France?"),
        simulated_clear_llm_response_direct # This is the simulated AI response
    ],
    "clarification_needed": False
}

print("\n--- State after clear answer from LLM ---")
print(f"Chat History: {[m.content for m in current_state_direct['chat_history']]}")

# Now, `check_clarification` would be called.
final_step_direct = check_clarification(current_state_direct)
print(f"Check clarification result: {final_step_direct}")

if final_step_direct == "answer":
    print("\n--- Conversation concluded directly ---")


print("\n--- Conditional Chatbot Demo Concluded ---")
# %%
