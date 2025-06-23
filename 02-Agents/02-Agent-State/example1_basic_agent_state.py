"""
Example 1: Basic Agent State
===========================

This is the simplest example of LangGraph agent state management.
It demonstrates the fundamental concepts of state definition, initialization,
and basic state updates with minimal complexity.

Key Concepts:
- Basic TypedDict state definition
- Simple state initialization
- Turn counting and user identification
- Message history management
"""

import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage

# 1. Define Basic Agent State (minimal example)
class BasicAgentState(TypedDict):
    messages: Annotated[List, add_messages]  # Required for conversation
    turn_count: int                          # Simple counter

# 2. Define Simple Nodes
def increment_turn_node(state: BasicAgentState):
    """
    Simple node that increments the turn counter.
    Shows basic state reading and updating.
    """
    print("---INCREMENTING TURN---")
    current_count = state.get("turn_count", 0)
    new_count = current_count + 1
    print(f"Turn: {current_count} -> {new_count}")
    
    # Return only the fields that need updating
    return {"turn_count": new_count}

def simple_llm_node(state: BasicAgentState):
    """
    Basic LLM node that responds to messages.
    Demonstrates accessing messages from state.
    """
    print("---SIMPLE LLM RESPONSE---")
    messages = state["messages"]
    turn_count = state["turn_count"]
    
    print(f"Processing turn {turn_count} with {len(messages)} messages")
    
    # Simple LLM call
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(messages)
    
    print(f"LLM Response: {response.content}")
    
    # Return the new AI message (add_messages will append it)
    return {"messages": [AIMessage(content=response.content)]}

# 3. Build Simple Graph
workflow = StateGraph(BasicAgentState)

# Add nodes
workflow.add_node("turn_counter", increment_turn_node)
workflow.add_node("llm", simple_llm_node)

# Define simple flow
workflow.set_entry_point("turn_counter")
workflow.add_edge("turn_counter", "llm")
workflow.add_edge("llm", END)

# 4. Compile Graph
basic_agent = workflow.compile()

# 5. Simple Interactive Loop
if __name__ == "__main__":
    print("Basic Agent State Example")
    print("=" * 30)
    print("This example shows the simplest form of agent state management.")
    print("The agent only tracks turn count and conversation messages.\n")
    
    # Initialize with minimal state
    current_state = {
        "messages": [],
        "turn_count": 0
    }
    
    print("Type 'exit' to quit, or start chatting!")
    
    while True:
        user_input = input(f"\nTurn {current_state['turn_count'] + 1} - You: ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
            
        if not user_input.strip():
            print("Please enter some text.")
            continue
        
        # Add user message to state
        current_state["messages"] = [HumanMessage(content=user_input)]
        
        print("\n--- Processing ---")
        
        # Run the graph and get final state
        result = basic_agent.invoke(current_state)
        
        # Update our current state with the results
        current_state.update(result)
        
        # Display the AI response
        if result["messages"]:
            ai_message = result["messages"][-1]
            if isinstance(ai_message, AIMessage):
                print(f"AI: {ai_message.content}")
    
    print(f"\nFinal State Summary:")
    print(f"Total turns: {current_state['turn_count']}")
    print(f"Total messages: {len(current_state['messages'])}")