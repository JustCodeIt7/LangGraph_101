#!/usr/bin/env python3
"""
LangGraph Multi-Agent Example 1: Basic Multi-Agent System (EASY)

This is the simplest multi-agent example showing:
- Two specialized agents working in sequence
- Basic message passing between agents
- Simple routing logic
- Clear separation of responsibilities

Scenario: Question & Answer System
- Agent 1 (Researcher): Finds information about a topic
- Agent 2 (Summarizer): Creates a concise summary of the research
"""

import uuid
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 70)
print("🤖 LangGraph Multi-Agent Example 1: Basic Multi-Agent System")
print("=" * 70)

# ============================================================================
# PART 1: DEFINE SIMPLE MULTI-AGENT STATE
# ============================================================================
print("\n📊 PART 1: Simple Multi-Agent State")
print("-" * 50)

class BasicMultiAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    topic: str
    research_result: str
    summary_result: str
    next_agent: Literal["researcher", "summarizer", "END"]

print("✅ Defined BasicMultiAgentState with simple agent coordination")

# ============================================================================
# PART 2: DEFINE AGENT NODES
# ============================================================================
print("\n🤖 PART 2: Agent Node Functions")
print("-" * 50)

# Simple LLMs for each agent
researcher_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
summarizer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

def researcher_agent(state: BasicMultiAgentState):
    """Agent 1: Researches information about the given topic."""
    print("   [Agent 1] Researcher: Gathering information...")
    
    topic = state["topic"]
    
    prompt = f"""You are a research agent. Research and provide detailed information about: {topic}

Please provide:
1. Key facts and definitions
2. Current trends or developments
3. Important statistics or data
4. Notable examples or case studies

Be comprehensive but focus on the most important and relevant information."""
    
    response = researcher_llm.invoke([HumanMessage(content=prompt)])
    research_content = response.content
    
    print(f"   Researcher found information about: {topic}")
    
    return {
        "research_result": research_content,
        "next_agent": "summarizer",
        "messages": [AIMessage(content=f"Research completed on {topic}")]
    }

def summarizer_agent(state: BasicMultiAgentState):
    """Agent 2: Creates a concise summary of the research."""
    print("   [Agent 2] Summarizer: Creating summary...")
    
    topic = state["topic"]
    research_result = state["research_result"]
    
    prompt = f"""You are a summarization agent. Create a concise, well-structured summary of this research about {topic}:

RESEARCH CONTENT:
{research_result}

Please provide:
1. A brief overview (2-3 sentences)
2. Key points (3-5 bullet points)
3. Main takeaway or conclusion

Keep it clear, concise, and easy to understand."""
    
    response = summarizer_llm.invoke([HumanMessage(content=prompt)])
    summary_content = response.content
    
    print(f"   Summarizer created summary for: {topic}")
    
    return {
        "summary_result": summary_content,
        "next_agent": "END",
        "messages": [AIMessage(content=f"Summary completed:\n\n{summary_content}")]
    }

# ============================================================================
# PART 3: ROUTING LOGIC
# ============================================================================
print("\n🔀 PART 3: Simple Routing Logic")
print("-" * 50)

def route_agents(state: BasicMultiAgentState):
    """Simple router that directs to the next agent."""
    next_agent = state.get("next_agent", "researcher")
    
    print(f"   Router: Directing to {next_agent}")
    
    if next_agent == "researcher":
        return "researcher_agent"
    elif next_agent == "summarizer":
        return "summarizer_agent"
    else:
        return "END"

print("✅ Simple routing function defined")

# ============================================================================
# PART 4: BUILD THE MULTI-AGENT GRAPH
# ============================================================================
print("\n🏗️  PART 4: Building Basic Multi-Agent Graph")
print("-" * 50)

# Create the graph
workflow = StateGraph(BasicMultiAgentState)

# Add agent nodes
workflow.add_node("researcher_agent", researcher_agent)
workflow.add_node("summarizer_agent", summarizer_agent)

# Set entry point
workflow.set_entry_point("researcher_agent")

# Add simple routing
workflow.add_conditional_edges(
    "researcher_agent",
    route_agents,
    {
        "summarizer_agent": "summarizer_agent",
        "END": END
    }
)

workflow.add_conditional_edges(
    "summarizer_agent", 
    route_agents,
    {
        "END": END
    }
)

# Compile the graph
memory = SqliteSaver.from_conn_string(":memory:")
multi_agent_graph = workflow.compile(checkpointer=memory)

print("✅ Basic multi-agent graph compiled successfully!")

# ============================================================================
# PART 5: INTERACTIVE SESSION
# ============================================================================
print("\n🎮 PART 5: Interactive Multi-Agent Session")
print("-" * 50)

def run_basic_multi_agent():
    """Run the basic multi-agent system."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting basic multi-agent session: {session_id}")
    
    # Get topic from user
    topic = input("Enter a topic to research (e.g., 'artificial intelligence', 'climate change'): ").strip()
    if not topic:
        topic = "artificial intelligence"
        print(f"No topic provided. Using default: {topic}")
    
    # Initial input
    initial_input = {
        "topic": topic,
        "research_result": "",
        "summary_result": "",
        "next_agent": "researcher",
        "messages": [HumanMessage(content=f"Please research and summarize: {topic}")]
    }
    
    print(f"\n--- Starting Multi-Agent Research on: {topic} ---")
    print("-" * 60)
    
    try:
        # Run the multi-agent workflow
        for event in multi_agent_graph.stream(initial_input, config, stream_mode="values"):
            # Show progress
            if event.get("research_result") and not event.get("summary_result"):
                print("\n✅ Research Phase Complete")
                print("📄 Research findings gathered")
                
            elif event.get("summary_result"):
                print("\n✅ Summary Phase Complete")
                print("📋 Final summary created")
        
        # Get final state and display results
        final_state = multi_agent_graph.get_state(config)
        if final_state:
            values = final_state.values
            
            print("\n" + "="*60)
            print("🎯 MULTI-AGENT RESEARCH RESULTS")
            print("="*60)
            
            print(f"\n📝 Topic: {values.get('topic', 'N/A')}")
            
            print(f"\n🔍 Research Results:")
            print("-" * 30)
            research = values.get('research_result', 'No research conducted')
            print(research[:300] + "..." if len(research) > 300 else research)
            
            print(f"\n📋 Final Summary:")
            print("-" * 30)
            summary = values.get('summary_result', 'No summary created')
            print(summary)
            
            print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Error during multi-agent session: {e}")

if __name__ == "__main__":
    run_basic_multi_agent()
    print("\n" + "="*70)
    print("Basic Multi-Agent System Complete!")
    print("Key features demonstrated:")
    print("  ✓ Two specialized agents working in sequence")
    print("  ✓ Simple message passing between agents")
    print("  ✓ Basic routing logic")
    print("  ✓ Clear separation of responsibilities")
    print("="*70)