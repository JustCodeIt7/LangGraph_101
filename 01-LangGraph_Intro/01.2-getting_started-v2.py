from IPython.core.display import Image
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_openai import OpenAI
from langchain_ollama import ChatOllama
from rich import print

# Create a ChatOpenAI instance with desired parameters.
# Talking Point: LangChain’s ChatOpenAI handles the OpenAI API call.
llm = ChatOllama(temperature=0, model="llama3.2", base_url="http://localhost:11434")


# Define the state schema for the graph.
# Talking Point: The state now includes an optional 'question' alongside the 'messages' list.
class GraphState(TypedDict, total=False):
    messages: list[dict]
    question: str


# Define the first node function that processes the state.
def process_node(state: GraphState) -> GraphState:
    # Append a processed message.
    new_message = {"content": "Processed message", "status": "processed"}
    state["messages"].append(new_message)
    # Talking Point: This node simulates the initial processing step.
    return state


# Define an additional node function that finalizes the state.
def finalize_node(state: GraphState) -> GraphState:
    # Append a finalized message.
    final_message = {"content": "Finalized message", "status": "finalized"}
    state["messages"].append(final_message)
    # Talking Point: This node shows how to extend the workflow before invoking the LLM.
    return state


# Define the final node function that uses LangChain's OpenAI for generating a response.
def llm_node(state: GraphState) -> GraphState:
    # Retrieve the question from the state.
    question = state.get("question", "No question provided.")
    # Use LangChain's ChatOpenAI to generate an answer.
    answer = llm.invoke(question)
    llm_message = {"content": answer, "status": "llm_response"}
    state["messages"].append(llm_message)
    # Talking Point: This node integrates a real LLM response as the final output.
    return state


# Create the StateGraph instance with the defined state schema.
graph = StateGraph(GraphState)

# Add nodes to the graph.
graph.add_node("process", process_node)
graph.add_node("llm", llm_node)

# Define edges to set the flow: process -> finalize -> llm.
graph.add_edge("process", "llm")

# Set the entry point to "process" and the finish point to "llm".
graph.set_entry_point("process")
graph.set_finish_point("llm")

# Compile the graph into an executable application.
app = graph.compile()

# Visualize the graph structure as ASCII art.
print(app.get_graph().draw_ascii())

# Generate and save the graph structure as a PNG image.
img = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(img)

# Optionally, print the Mermaid diagram as text.
print(app.get_graph().draw_mermaid())

# Run the graph with an initial state that includes a question.
initial_state = {"messages": [], "question": "What is LangGraph?"}
result = app.invoke(initial_state)

# Output the final state to observe the accumulated changes, including the LLM response.
print("Final state after graph execution:", result)
