#!/usr/bin/env python3
"""
LangGraph Multi-Agent Tutorial - v2
Example 1: Basic Sequential Multi-Agent Workflow

This script demonstrates a simple multi-agent setup where agents perform tasks
in a predefined sequence. Agent A processes input, and its output becomes
the input for Agent B.

Scenario: Idea Generation and Refinement
- Agent A (Idea Generator): Generates a list of initial ideas based on a user-provided theme.
- Agent B (Idea Refiner): Takes the generated ideas, selects one, and refines it.

Concepts Demonstrated:
- Defining distinct agent responsibilities.
- Passing state (data) sequentially between agents.
- A simple, linear flow of control in a multi-agent system.
- Basic state management for multi-agent collaboration.
"""

import uuid
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages # Though not strictly needed for this simple state
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 70)
print("Multi-Agent Example 1: Basic Sequential Workflow")
print("(Idea Generation & Refinement)")
print("=" * 70)

# ============================================================================
# PART 1: DEFINE AGENT STATE
# ============================================================================
print("\n📊 PART 1: Defining Shared Agent State")
print("-" * 50)

class SequentialWorkflowState(TypedDict):
    # messages: Annotated[List[BaseMessage], add_messages] # For full chat history if needed
    user_theme: str
    generated_ideas: Optional[List[str]]
    selected_idea: Optional[str]
    refined_idea: Optional[str]
    error_message: Optional[str]

print("✅ Defined SequentialWorkflowState with fields for theme, ideas, and refinement.")

# ============================================================================
# PART 2: DEFINE INDIVIDUAL AGENT LOGIC
# ============================================================================
print("\n🤖 PART 2: Defining Agent Logic (Nodes)")
print("-" * 50)

# LLM Configurations
idea_generator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
idea_refiner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def agent_a_idea_generator_node(state: SequentialWorkflowState):
    """Agent A: Generates initial ideas based on the user's theme."""
    print("   [Agent A Running] Idea Generator")
    theme = state["user_theme"]
    
    if not theme:
        print("   Agent A: Error - No theme provided.")
        return {"error_message": "User theme is missing for Agent A."}

    prompt = f"""You are an Idea Generation Bot.
Given the theme: "{theme}"
Generate a list of 3-5 creative and distinct ideas related to this theme.
Each idea should be a short phrase or sentence.
Output the ideas as a numbered list. For example:
1. First idea.
2. Second idea.
...
"""
    try:
        response = idea_generator_llm.invoke([HumanMessage(content=prompt)])
        ideas_str = response.content
        # Parse ideas into a list
        ideas_list = [idea.strip() for idea in ideas_str.split('\n') if idea.strip() and idea[0].isdigit()]
        
        if not ideas_list:
            ideas_list = [ideas_str.strip()] # Fallback if parsing fails but content exists

        print(f"   Agent A generated ideas:\n{ideas_str}")
        return {"generated_ideas": ideas_list}
    except Exception as e:
        print(f"   Agent A: Error during idea generation - {e}")
        return {"error_message": f"Agent A failed: {str(e)}"}

def agent_b_idea_refiner_node(state: SequentialWorkflowState):
    """Agent B: Selects one idea and refines it."""
    print("   [Agent B Running] Idea Refiner")
    generated_ideas = state.get("generated_ideas")
    theme = state["user_theme"]

    if state.get("error_message"): # If Agent A failed
        print(f"   Agent B: Skipping due to previous error: {state['error_message']}")
        return {} 

    if not generated_ideas:
        print("   Agent B: Error - No ideas provided by Agent A.")
        return {"error_message": "No ideas received from Agent A for refinement."}

    # For simplicity, Agent B selects the first idea.
    # A more complex agent might have logic to choose or ask the user.
    selected_idea_for_refinement = generated_ideas[0]
    print(f"   Agent B selected idea for refinement: '{selected_idea_for_refinement}'")

    prompt = f"""You are an Idea Refinement Bot.
The overall theme is: "{theme}"
The idea to refine is: "{selected_idea_for_refinement}"
Please elaborate on this idea, adding more detail, potential benefits, or a unique angle.
Provide the refined idea as a short paragraph.
"""
    try:
        response = idea_refiner_llm.invoke([HumanMessage(content=prompt)])
        refined_idea_text = response.content
        print(f"   Agent B refined idea:\n{refined_idea_text}")
        return {
            "selected_idea": selected_idea_for_refinement,
            "refined_idea": refined_idea_text
        }
    except Exception as e:
        print(f"   Agent B: Error during idea refinement - {e}")
        return {"error_message": f"Agent B failed: {str(e)}"}

def error_handler_node(state: SequentialWorkflowState):
    """Handles and reports errors."""
    print("   [Error Handler Running]")
    if state.get("error_message"):
        print(f"   ERROR DETECTED: {state['error_message']}")
        # Potentially add error message to a main message list if using one
    return {}


print("✅ Defined logic for Idea Generator (Agent A) and Idea Refiner (Agent B).")

# ============================================================================
# PART 3: BUILD THE SEQUENTIAL MULTI-AGENT GRAPH
# ============================================================================
print("\n🏗️  PART 3: Building the Graph")
print("-" * 50)

# Using an in-memory checkpointer for simplicity in this example
memory = SqliteSaver.from_conn_string(":memory:")

workflow = StateGraph(SequentialWorkflowState)

# Add nodes for each agent
workflow.add_node("agent_a_generator", agent_a_idea_generator_node)
workflow.add_node("agent_b_refiner", agent_b_idea_refiner_node)
workflow.add_node("error_handler", error_handler_node)

# Define the sequential flow
workflow.set_entry_point("agent_a_generator")

# Conditional edge after Agent A
def route_after_agent_a(state: SequentialWorkflowState):
    if state.get("error_message"):
        return "error_handler"
    return "agent_b_refiner"

workflow.add_conditional_edges(
    "agent_a_generator",
    route_after_agent_a,
    {
        "agent_b_refiner": "agent_b_refiner",
        "error_handler": "error_handler"
    }
)

# Conditional edge after Agent B
def route_after_agent_b(state: SequentialWorkflowState):
    if state.get("error_message") and not state.get("refined_idea"): # Error specifically from B
        return "error_handler"
    return END # If successful or error was from A and B skipped

workflow.add_conditional_edges(
    "agent_b_refiner",
    route_after_agent_b,
    {
        "error_handler": "error_handler",
        END: END
    }
)

workflow.add_edge("error_handler", END)


# Compile the graph
sequential_multi_agent_graph = workflow.compile(checkpointer=memory)
print("✅ Sequential multi-agent graph compiled successfully!")
print("   Flow: Agent A (Generate) -> Agent B (Refine) -> END (or Error Handler -> END)")

# ============================================================================
# PART 4: INTERACTIVE SESSION
# ============================================================================
print("\n🎮 PART 4: Interactive Session")
print("-" * 50)

def run_sequential_workflow():
    session_id = f"smag-{str(uuid.uuid4())}"
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting new session: {session_id}")
    user_theme = input("Enter a theme for idea generation (e.g., 'sustainable living', 'future of education'): ").strip()
    
    if not user_theme:
        print("No theme provided. Exiting.")
        return

    initial_input = {"user_theme": user_theme}

    print("\n--- Running Sequential Multi-Agent Workflow ---")
    
    final_state = None
    try:
        for event_part in sequential_multi_agent_graph.stream(initial_input, config, stream_mode="values"):
            print(f"\n--- Graph State Update ---")
            if event_part.get("agent_a_generator"):
                print("   Agent A (Generator) has run.")
            if event_part.get("agent_b_refiner"):
                print("   Agent B (Refiner) has run.")
            if event_part.get("error_handler"):
                print("   Error Handler has run.")
            
            # Log key state changes
            if "generated_ideas" in event_part and event_part["generated_ideas"]:
                print(f"   Generated Ideas: {event_part['generated_ideas']}")
            if "selected_idea" in event_part and event_part["selected_idea"]:
                print(f"   Selected Idea for Refinement: {event_part['selected_idea']}")
            if "refined_idea" in event_part and event_part["refined_idea"]:
                print(f"   Refined Idea: {event_part['refined_idea'][:150]}...")
            if "error_message" in event_part and event_part["error_message"]:
                print(f"   ERROR: {event_part['error_message']}")

            final_state = event_part
            # time.sleep(0.1) # Optional: slow down for readability

        print("\n--- Workflow Complete ---")
        if final_state:
            if final_state.get("error_message"):
                print(f"\n❌ Workflow ended with an error: {final_state['error_message']}")
            elif final_state.get("refined_idea"):
                print("\n✅ Successfully generated and refined an idea!")
                print(f"Theme: {final_state.get('user_theme')}")
                print(f"Initial Ideas (sample): {final_state.get('generated_ideas', [])[:1]}...")
                print(f"Selected Idea: {final_state.get('selected_idea')}")
                print(f"Refined Idea:\n{final_state.get('refined_idea')}")
            else:
                print("\n⚠️ Workflow completed, but no refined idea was produced (check logs).")
        else:
            print("Workflow did not produce a final state.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during workflow execution: {e}")
        if final_state:
            print("\nLast known state:")
            for key, value in final_state.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    run_sequential_workflow()
    print("\n" + "="*70)
    print("Basic Sequential Multi-Agent Workflow Example Complete!")
    print("Key takeaways:")
    print("  - Agents can be defined as nodes performing distinct tasks.")
    print("  - State is passed between agents to enable sequential processing.")
    print("  - Conditional edges can handle simple error routing.")
    print("  - This forms a foundational pattern for more complex multi-agent systems.")
    print("="*70)
