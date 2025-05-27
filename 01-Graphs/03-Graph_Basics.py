#%%
"""
LangGraph Tutorial: Hello World and Multiple Inputs
YouTube Video Code Examples
"""

# =============================================================================
# PART 1: HELLO WORLD GRAPH
# =============================================================================

# Step 1: Import required modules
from typing import TypedDict, List
from langgraph.graph import StateGraph
#%%
# Step 2: Define the Agent State Schema
class AgentState(TypedDict):
    message: str

print("✅ AgentState defined with 'message' field")

# Step 3: Define the greeting node
def greeting_node(state: AgentState) -> AgentState:
    """
    Simple node that adds a greeting message to the state.
    Takes the current message (which should be a name) and creates a greeting.
    """
    print(f"📥 Input to greeting_node: {state}")

    # Modify the message to create a greeting
    state["message"] = f"Hey {state['message']}, how's your day going?"

    print(f"📤 Output from greeting_node: {state}")
    return state

# Step 4: Build and compile the graph
def build_hello_world_graph():
    """Build and compile the Hello World graph"""
    print("\n🔧 Building Hello World Graph...")

    # Create the graph with our state schema
    graph = StateGraph(state_schema=AgentState)

    # Add our greeting node
    graph.add_node("greeter", greeting_node)

    # Set entry and finish points
    graph.set_entry_point("greeter")
    graph.set_finish_point("greeter")

    # Compile the graph
    app = graph.compile()
    print("✅ Graph compiled successfully!")

    return app

# Step 5: Test the Hello World graph
def test_hello_world():
    print("\n🚀 Testing Hello World Graph...")

    app = build_hello_world_graph()

    # Test with different names
    test_names = ["Bob", "Alice", "Charlie"]

    for name in test_names:
        print(f"\n--- Testing with name: {name} ---")
        result = app.invoke({"message": name})
        print(f"🎯 Final result: {result['message']}")

# Run Hello World example
test_hello_world()

print("\n" + "=" * 60)
print("EXERCISE 1: PERSONALIZED COMPLIMENT AGENT")
print("=" * 60)

# Exercise 1 Solution
def compliment_node(state: AgentState) -> AgentState:
    """
    Node that creates a personalized compliment message.
    Instead of replacing, we'll concatenate to the existing message.
    """
    print(f"📥 Input to compliment_node: {state}")

    # Get the name from the current message
    name = state["message"]

    # Create a compliment message
    state["message"] = f"{name}, you're doing an amazing job learning LangGraph!"

    print(f"📤 Output from compliment_node: {state}")
    return state

def build_compliment_graph():
    """Build the compliment graph for Exercise 1"""
    print("\n🔧 Building Compliment Graph...")

    graph = StateGraph(state_schema=AgentState)
    graph.add_node("complimenter", compliment_node)
    graph.set_entry_point("complimenter")
    graph.set_finish_point("complimenter")

    app = graph.compile()
    print("✅ Compliment Graph compiled!")

    return app

def test_compliment_agent():
    print("\n🚀 Testing Compliment Agent...")

    app = build_compliment_graph()

    test_names = ["Alice", "Bob", "Sarah"]

    for name in test_names:
        print(f"\n--- Complimenting: {name} ---")
        result = app.invoke({"message": name})
        print(f"🎯 Compliment: {result['message']}")

# Run Exercise 1
test_compliment_agent()

# =============================================================================
# PART 2: MULTIPLE INPUTS GRAPH
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: MULTIPLE INPUTS GRAPH")
print("=" * 60)

# Step 1: Define extended Agent State Schema
class MultiInputAgentState(TypedDict):
    values: List[int]
    name: str
    result: str

print("✅ MultiInputAgentState defined with 'values', 'name', and 'result' fields")

# Step 2: Define the processing node
def process_values(state: MultiInputAgentState) -> MultiInputAgentState:
    """
    Handles multiple different inputs.
    Takes a list of values, sums them, and creates a greeting with the result.
    """
    print(f"📥 Input to process_values: {state}")

    # Calculate the sum of values
    total = sum(state["values"])

    # Create result message
    state["result"] = f"Hi there {state['name']}, your sum is {total}"

    print(f"📤 Output from process_values: {state}")
    return state

# Step 3: Build and compile the multiple inputs graph
def build_multi_input_graph():
    """Build and compile the Multiple Inputs graph"""
    print("\n🔧 Building Multiple Inputs Graph...")

    graph = StateGraph(state_schema=MultiInputAgentState)
    graph.add_node("processor", process_values)
    graph.set_entry_point("processor")
    graph.set_finish_point("processor")

    app = graph.compile()
    print("✅ Multiple Inputs Graph compiled!")

    return app

# Step 4: Test the multiple inputs graph
def test_multi_input():
    print("\n🚀 Testing Multiple Inputs Graph...")

    app = build_multi_input_graph()

    # Test cases
    test_cases = [
        {"values": [1, 2, 3, 4], "name": "Steve", "result": ""},
        {"values": [10, 20, 30], "name": "Maria", "result": ""},
        {"values": [5, 15, 25, 35], "name": "David", "result": ""},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_case}")
        result = app.invoke(test_case)
        print(f"🎯 Result: {result['result']}")

# Run Multiple Inputs example
test_multi_input()

print("\n" + "=" * 60)
print("EXERCISE 2: CONDITIONAL OPERATION AGENT")
print("=" * 60)

# Exercise 2 Solution
class ConditionalAgentState(TypedDict):
    values: List[int]
    name: str
    operation: str  # "+" or "×"
    result: str

def conditional_operation_node(state: ConditionalAgentState) -> ConditionalAgentState:
    """
    Performs either addition or multiplication based on the operation parameter.
    """
    print(f"📥 Input to conditional_operation_node: {state}")

    values = state["values"]
    operation = state["operation"]
    name = state["name"]

    if operation == "+":
        # Sum the values
        computed = sum(values)
        op_name = "sum"
    elif operation == "×" or operation == "*":
        # Multiply the values
        computed = 1
        for value in values:
            computed *= value
        op_name = "product"
    else:
        # Handle invalid operation
        computed = 0
        op_name = "unknown operation"

    # Create result message
    state["result"] = f"Hi {name}, your {op_name} is {computed}"

    print(f"📤 Output from conditional_operation_node: {state}")
    return state

def build_conditional_graph():
    """Build the conditional operation graph for Exercise 2"""
    print("\n🔧 Building Conditional Operation Graph...")

    graph = StateGraph(state_schema=ConditionalAgentState)
    graph.add_node("conditional_processor", conditional_operation_node)
    graph.set_entry_point("conditional_processor")
    graph.set_finish_point("conditional_processor")

    app = graph.compile()
    print("✅ Conditional Operation Graph compiled!")

    return app

def test_conditional_agent():
    print("\n🚀 Testing Conditional Operation Agent...")

    app = build_conditional_graph()

    # Test cases for both addition and multiplication
    test_cases = [
        {"values": [1, 2, 3, 4], "name": "Alice", "operation": "+", "result": ""},
        {"values": [2, 3, 4], "name": "Bob", "operation": "×", "result": ""},
        {"values": [10, 20, 30], "name": "Charlie", "operation": "+", "result": ""},
        {"values": [2, 5], "name": "Diana", "operation": "×", "result": ""},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_case}")
        result = app.invoke(test_case)
        print(f"🎯 Result: {result['result']}")

# Run Exercise 2
test_conditional_agent()

