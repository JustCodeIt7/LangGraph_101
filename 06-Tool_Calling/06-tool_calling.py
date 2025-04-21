# --- Imports ---
import operator
import json
from typing import TypedDict, Annotated, List, Literal
from uuid import uuid4 # To generate unique tool call IDs

# LangChain core imports for messages and tools
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode # Import the prebuilt ToolNode

# --- Tool Definition ---
# Define a simple tool using the @tool decorator.
# This tool simulates fetching weather information.
@tool
def get_weather(city: str) -> str:
    """
    Gets the current weather for a specified city.

    Args:
        city: The name of the city.

    Returns:
        A string describing the weather (mocked response).
    """
    print(f"--- Calling get_weather tool for city: {city} ---")
    # In a real scenario, this would call an API.
    # Here, we just return a mock response.
    if city.lower() == "indian hill":
        return f"The weather in {city} is sunny and 75°F."
    elif city.lower() == "cincinnati":
        return f"The weather in {city} is cloudy with a chance of rain."
    else:
        return f"Sorry, I don't have weather information for {city}."

# List of tools to be used by the ToolNode
tools = [get_weather]

# --- State Definition ---
# Define the state for our agent.
# 'messages' will accumulate the conversation history, including tool calls and results.
# We use Annotated and operator.add to ensure messages are appended, not overwritten.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- Node Functions ---

# Mock LLM Node (Agent)
def agent_node(state: AgentState) -> dict:
    """
    Simulates an LLM call. It checks the latest message and decides
    whether to respond directly, call a tool, or respond based on tool results.
    """
    print("--- Agent Node ---")
    last_message = state['messages'][-1]
    print(f"Agent received message of type: {type(last_message)}")
    print(f"Content: {last_message.content}")

    # If the last message is a ToolMessage, process the result
    if isinstance(last_message, ToolMessage):
        print("Agent processes ToolMessage result.")
        # Generate a final response based on the tool's output
        final_response_content = f"Based on the tool result: {last_message.content}"
        return {"messages": [AIMessage(content=final_response_content)]}

    # If the last message is a HumanMessage, decide if a tool is needed
    elif isinstance(last_message, HumanMessage):
        print("Agent processes HumanMessage.")
        # Simple logic: If the last message asks for weather, simulate a tool call.
        if "weather" in last_message.content.lower():
            tool_call_id = str(uuid4()) # Generate a unique ID for the tool call
            # Extract city or use a default - more robust parsing needed for real use
            city = "Indian Hill" # Hardcoded for simplicity, LLM would extract this
            if "cincinnati" in last_message.content.lower():
                city = "Cincinnati"

            tool_call_content = json.dumps({"city": city})
            print(f"Agent decides to call tool: get_weather with args {tool_call_content}")
            # Create an AIMessage containing the tool call information
            ai_response = AIMessage(
                content="", # Can be empty if only calling tools
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "get_weather",
                        "args": {"city": city}, # Pass arguments as a dictionary
                    }
                ]
            )
            return {"messages": [ai_response]}
        else:
            # If no tool call needed, simulate a final response.
            print("Agent decides to respond directly to HumanMessage.")
            final_response = AIMessage(content="I can help with weather information. What city are you interested in?")
            return {"messages": [final_response]}

    # Handle other message types if necessary, or default behavior
    else:
        print("Agent received unexpected message type, ending.")
        # Or return a default response
        return {"messages": [AIMessage(content="I'm not sure how to proceed.")]}

# --- Conditional Edge Logic ---
def should_call_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Checks the latest AI message in the state to see if it contains tool calls.
    Routes to ToolNode if tools need to be called, otherwise ends the graph.
    """
    print("--- Condition Check: Should Call Tools? ---")
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(f"Decision: Route to ToolNode (Tool calls found: {last_message.tool_calls})")
        return "tools" # Route to the ToolNode
    else:
        print("Decision: End Graph (No tool calls)")
        return "__end__" # End the graph execution

# --- Graph Definition ---
# Instantiate the graph and define its structure.

# Instantiate ToolNode with the defined tools
tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node) # Add the ToolNode

# Set the entry point
workflow.set_entry_point("agent")

# Add conditional edges
# After the 'agent' node, check if tools should be called.
workflow.add_conditional_edges(
    "agent",
    should_call_tools,
    {
        "tools": "tools",   # If tool calls exist, go to the 'tools' (ToolNode)
        "__end__": END      # Otherwise, end the graph
    }
)

# Add edge from ToolNode back to the agent
# After tools are executed, the results (as ToolMessages) are added to the state.
# We route back to the agent node so it can process the tool results.
workflow.add_edge("tools", "agent")

# --- Compile the Graph ---
app = workflow.compile()

# --- Run the Graph ---

print("\n--- Running Graph: Asking about weather ---")
# Start with a human message asking for weather
initial_state_weather = {"messages": [HumanMessage(content="What's the weather like today?")]}
# Invoke the graph
final_state_weather = app.invoke(initial_state_weather, {"recursion_limit": 5})

print("\n--- Final State (Weather Query) ---")
# The final state should contain the HumanMessage, the AIMessage with the tool call,
# the ToolMessage with the result, and the final AIMessage from the agent.
for message in final_state_weather['messages']:
    print(f"- {message.type}: {message.content}")
    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"  Tool Calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"  Tool Call ID: {message.tool_call_id}")


print("\n" + "="*30 + "\n")

print("--- Running Graph: General question ---")
# Start with a human message not related to weather
initial_state_general = {"messages": [HumanMessage(content="Hello, how are you?")]}
final_state_general = app.invoke(initial_state_general, {"recursion_limit": 5})

print("\n--- Final State (General Query) ---")
for message in final_state_general['messages']:
    print(f"- {message.type}: {message.content}")
# Expected: HumanMessage, AIMessage (direct response), graph ends.

# --- Visualization (Optional) ---
try:
    print("\n--- Graph Structure (ASCII) ---")
    print(app.get_graph().print_ascii())

    # # Generate PNG (requires graphviz and pillow)
    # from PIL import Image
    # img_bytes = app.get_graph().draw_png()
    # with open("tool_node_graph.png", "wb") as f:
    #     f.write(img_bytes)
    # print("\nGraph image saved to tool_node_graph.png")

except ImportError:
    print("\nInstall pygraphviz and pillow to generate graph images: pip install pygraphviz pillow")
except Exception as e:
    print(f"\nCould not generate graph visualization: {e}")

