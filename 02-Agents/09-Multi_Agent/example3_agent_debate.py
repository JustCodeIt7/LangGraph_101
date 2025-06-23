#!/usr/bin/env python3
"""
LangGraph Multi-Agent Tutorial - v2
Example 3: Multi-Agent Debate with Reflection

This script demonstrates a more advanced multi-agent setup where agents
take on different perspectives to debate a topic, moderated by another agent.
Agents may also reflect on the debate to refine their arguments.

Scenario: AI Ethics Debate
- User provides a debate topic related to AI ethics.
- Moderator Agent: Sets the stage, manages turns, and summarizes the debate.
- Pro-AI Agent: Argues for the benefits and positive aspects of the AI topic.
- Con-AI Agent: Argues for the concerns and potential negative aspects.
- (Optional) Reflection step where agents refine arguments.

Concepts Demonstrated:
- Cyclic interactions between multiple agents.
- Managing different perspectives and roles.
- A moderator agent to control the flow of discussion.
- Shared conversational state for the debate.
- Potential for agents to reflect and adapt arguments (simplified here).
"""

import uuid
import time
from typing import TypedDict, Annotated, List, Optional, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 70)
print("Multi-Agent Example 3: Multi-Agent Debate with Reflection")
print("(AI Ethics Debate)")
print("=" * 70)

# ============================================================================
# PART 1: DEFINE AGENT STATE
# ============================================================================
print("\n📊 PART 1: Defining Shared Agent State for the Debate")
print("-" * 50)

class DebateState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] # Full debate transcript
    debate_topic: str
    current_speaker: str # 'moderator', 'pro_ai', 'con_ai', or 'reflection'
    turn_count: int
    max_turns: int # Max turns per debater before summarization
    pro_ai_arguments: List[str]
    con_ai_arguments: List[str]
    final_summary: Optional[str]
    error_message: Optional[str]

print("✅ Defined DebateState for topic, speakers, arguments, and transcript.")

# ============================================================================
# PART 2: DEFINE INDIVIDUAL AGENT LOGIC (NODES)
# ============================================================================
print("\n🤖 PART 2: Defining Agent Logic (Moderator, Debaters)")
print("-" * 50)

# LLM Configurations
moderator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
pro_ai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, model_kwargs={"seed": 100}) # Seed for some consistency
con_ai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, model_kwargs={"seed": 200})

# --- Moderator Agent ---
def moderator_agent_node(state: DebateState):
    """Moderator: Manages the debate flow, introduces topic, and summarizes."""
    print("   [Moderator Agent Running]")
    turn = state["turn_count"]
    max_turns = state["max_turns"]
    topic = state["debate_topic"]
    
    # Get last few messages for context
    history = state.get("messages", [])
    context_messages = history[-3:] # Last 3 messages as context

    if turn == 0: # Initial turn
        intro_message = f"Welcome to our debate on the topic: '{topic}'. We will have {max_turns} rounds of arguments. First, let's hear from the Pro-AI perspective."
        print(f"   Moderator: {intro_message}")
        return {
            "messages": [AIMessage(content=intro_message, name="Moderator")],
            "current_speaker": "pro_ai",
            "turn_count": 1
        }
    elif turn > (max_turns * 2) : # Both debaters have spoken max_turns times
        summary_prompt = f"""You are the Moderator. The debate on '{topic}' has concluded its argument phase.
Debate Transcript:
{format_transcript(history)}

Please provide a concise summary of the main arguments presented by both sides and conclude the debate.
"""
        summary_response = moderator_llm.invoke([SystemMessage(content=summary_prompt)])
        final_summary_text = summary_response.content
        print(f"   Moderator (Final Summary): {final_summary_text[:150]}...")
        return {
            "messages": [AIMessage(content=final_summary_text, name="Moderator")],
            "final_summary": final_summary_text,
            "current_speaker": "finish"
        }
    else: # Mid-debate, manage turns
        next_speaker = "pro_ai" if state["current_speaker"] == "con_ai" else "con_ai"
        round_num = (turn // 2) + 1
        instruction = f"Round {round_num}. Now, let's hear from the {next_speaker.replace('_', '-')} perspective."
        if turn % 2 == 0 and next_speaker == "pro_ai": # Pro AI starts a new round
             instruction = f"Round {round_num}. {next_speaker.replace('_', '-')} agent, please present your argument."
        elif turn % 2 != 0 and next_speaker == "con_ai": # Con AI starts its turn in a round
             instruction = f"Round {round_num}. {next_speaker.replace('_', '-')} agent, please present your counter-argument or new point."


        print(f"   Moderator: {instruction}")
        return {
            "messages": [AIMessage(content=instruction, name="Moderator")],
            "current_speaker": next_speaker
        }

# --- Pro-AI Agent ---
def pro_ai_agent_node(state: DebateState):
    """Pro-AI Agent: Argues for the positive aspects."""
    print("   [Pro-AI Agent Running]")
    topic = state["debate_topic"]
    history = state.get("messages", [])
    # Get arguments from opponent and moderator's last instruction
    context_messages = history[-3:] # Moderator instruction + opponent's last arg (if any)

    prompt = f"""You are the Pro-AI Debater. Your stance is to argue FOR the benefits and positive aspects of the AI topic.
Debate Topic: "{topic}"
Current Debate Context:
{format_transcript(context_messages)}

Present your argument clearly and concisely. Respond to the last point if relevant, or introduce a new argument.
Keep your argument to 1-2 paragraphs.
"""
    response = pro_ai_llm.invoke([SystemMessage(content=prompt)])
    argument = response.content
    print(f"   Pro-AI Argument: {argument[:100]}...")
    current_args = state.get("pro_ai_arguments", [])
    return {
        "messages": [AIMessage(content=argument, name="Pro-AI_Agent")],
        "pro_ai_arguments": current_args + [argument],
        "current_speaker": "moderator", # Hand back to moderator
        "turn_count": state["turn_count"] + 1
    }

# --- Con-AI Agent ---
def con_ai_agent_node(state: DebateState):
    """Con-AI Agent: Argues for the concerns and negative aspects."""
    print("   [Con-AI Agent Running]")
    topic = state["debate_topic"]
    history = state.get("messages", [])
    context_messages = history[-3:]

    prompt = f"""You are the Con-AI Debater. Your stance is to argue AGAINST, highlighting concerns and potential negative aspects of the AI topic.
Debate Topic: "{topic}"
Current Debate Context:
{format_transcript(context_messages)}

Present your counter-argument or concerns clearly and concisely. Respond to the last point if relevant, or introduce a new argument.
Keep your argument to 1-2 paragraphs.
"""
    response = con_ai_llm.invoke([SystemMessage(content=prompt)])
    argument = response.content
    print(f"   Con-AI Argument: {argument[:100]}...")
    current_args = state.get("con_ai_arguments", [])
    return {
        "messages": [AIMessage(content=argument, name="Con-AI_Agent")],
        "con_ai_arguments": current_args + [argument],
        "current_speaker": "moderator", # Hand back to moderator
        "turn_count": state["turn_count"] + 1
    }

def format_transcript(messages: Sequence[BaseMessage]) -> str:
    """Helper to format messages for context."""
    formatted = []
    for msg in messages:
        sender_name = getattr(msg, 'name', type(msg).__name__) # Get name if AIMessage, else class name
        if isinstance(msg, HumanMessage): sender_name = "User" # Or some other consistent name
        formatted.append(f"{sender_name}: {msg.content}")
    return "\n".join(formatted)

print("✅ Defined logic for Moderator, Pro-AI, and Con-AI agents.")

# ============================================================================
# PART 3: BUILD THE DEBATE GRAPH
# ============================================================================
print("\n🏗️  PART 3: Building the Debate Graph")
print("-" * 50)

memory = SqliteSaver.from_conn_string(":memory:")
workflow = StateGraph(DebateState)

# Add nodes
workflow.add_node("moderator", moderator_agent_node)
workflow.add_node("pro_ai_agent", pro_ai_agent_node)
workflow.add_node("con_ai_agent", con_ai_agent_node)

# Define entry point (Moderator starts)
workflow.set_entry_point("moderator")

# Routing logic
def route_debate(state: DebateState):
    speaker = state["current_speaker"]
    print(f"   Router (Debate): Next speaker is '{speaker}'")
    if speaker == "pro_ai":
        return "pro_ai_agent"
    elif speaker == "con_ai":
        return "con_ai_agent"
    elif speaker == "moderator":
        return "moderator"
    elif speaker == "finish":
        return END
    return END # Should not happen

workflow.add_conditional_edges("moderator", route_debate)
workflow.add_conditional_edges("pro_ai_agent", route_debate) # Pro-AI always hands back to moderator
workflow.add_conditional_edges("con_ai_agent", route_debate) # Con-AI always hands back to moderator

# Compile the graph
debate_graph = workflow.compile(checkpointer=memory)
print("✅ Multi-agent debate graph compiled successfully!")

# ============================================================================
# PART 4: INTERACTIVE DEBATE SESSION
# ============================================================================
print("\n🎮 PART 4: Interactive AI Ethics Debate Session")
print("-" * 50)

def run_debate_session():
    session_id = f"dbmg-{str(uuid.uuid4())}"
    config = {"configurable": {"thread_id": session_id}}
    
    debate_topic = input("Enter an AI ethics debate topic (e.g., 'Autonomous Weapons', 'AI in Hiring', 'Universal Basic Income due to AI'): ").strip()
    if not debate_topic:
        print("No debate topic provided. Exiting.")
        return

    max_turns_per_side = 2 # Each debater gets 2 turns
    initial_input = {
        "debate_topic": debate_topic,
        "messages": [HumanMessage(content=f"Let's start a debate on: {debate_topic}")],
        "turn_count": 0,
        "max_turns": max_turns_per_side,
        "current_speaker": "moderator", # Moderator starts
        "pro_ai_arguments": [],
        "con_ai_arguments": []
    }

    print(f"\n--- Starting Debate on '{debate_topic}' (Session: {session_id}) ---")
    
    final_state = None
    try:
        for event_part in debate_graph.stream(initial_input, config, stream_mode="values"):
            print(f"\n   --- Debate Turn ---")
            active_node = [k for k, v in event_part.items() if isinstance(v, dict) and "messages" not in v and v is not None] # Heuristic to find active node
            if active_node: print(f"    Active Node: {active_node[0]}")
            
            if event_part.get("messages"):
                last_msg = event_part["messages"][-1]
                sender_name = getattr(last_msg, 'name', type(last_msg).__name__)
                print(f"    Speaker [{sender_name}]: {last_msg.content[:200]}...") # Print message snippet
            
            final_state = event_part
            time.sleep(1) # Slow down for readability

        print("\n--- Debate Concluded ---")
        if final_state and final_state.get("final_summary"):
            print("\n✅ Final Debate Summary by Moderator:")
            print(final_state["final_summary"])
        elif final_state and final_state.get("error_message"):
            print(f"\n❌ Debate ended with an error: {final_state['error_message']}")
        else:
            print("\n⚠️ Debate finished, but no final summary was generated (check logs).")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the debate: {e}")
        if final_state:
            print("\nLast known debate state:")
            for key, value in final_state.items():
                if key == "messages": print(f"  {key}: [{len(value)} messages]")
                else: print(f"  {key}: {str(value)[:100]}...")


if __name__ == "__main__":
    run_debate_session()
    print("\n" + "="*70)
    print("Multi-Agent Debate Example Complete!")
    print("Key takeaways:")
    print("  - Multiple agents can engage in cyclic interactions like a debate.")
    print("  - A moderator agent can manage turns and summarize.")
    print("  - State needs to track whose turn it is, arguments, and the overall transcript.")
    print("  - This pattern can be extended for more complex discussions or negotiations.")
    print("="*70)
