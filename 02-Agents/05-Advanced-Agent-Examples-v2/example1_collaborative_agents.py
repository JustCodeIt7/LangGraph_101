#!/usr/bin/env python3
"""
LangGraph Advanced Agent Example 1: Collaborative Multi-Agent System

This script demonstrates a collaborative multi-agent system where specialized agents
work together to solve complex problems. The system includes:

1. Coordinator Agent: Orchestrates the workflow and makes high-level decisions
2. Analyst Agent: Analyzes data and identifies patterns
3. Researcher Agent: Gathers additional information when needed
4. Synthesizer Agent: Combines insights from other agents into final output

Scenario: Market Analysis System
- Takes a market analysis request (e.g., "Analyze the AI startup market")
- Coordinator breaks down the task and delegates to specialized agents
- Agents collaborate and share findings
- Synthesizer creates comprehensive market analysis report
"""

import uuid
import time
from typing import TypedDict, Annotated, List, Union, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 80)
print("LangGraph Advanced Example 1: Collaborative Multi-Agent System")
print("=" * 80)

# ============================================================================
# PART 1: DEFINE TOOLS FOR THE COLLABORATIVE AGENTS
# ============================================================================
print("\n🔧 PART 1: Defining Collaborative Agent Tools")
print("-" * 60)

@tool
def market_data_search(query: str, data_type: str = "general") -> str:
    """
    Simulates market data search with different data types.
    
    Args:
        query: The search query
        data_type: Type of data to search for (general, financial, trends, competitors)
    
    Returns:
        Mock market data based on query and data type
    """
    print(f"   [Tool] market_data_search: {data_type} data for '{query}'")
    time.sleep(0.5)
    
    if "AI startup" in query and data_type == "financial":
        return """Financial Data - AI Startup Market:
• Total funding in AI startups: $50B+ in 2024
• Average Series A: $15M (up 25% from 2023)
• Top funded sectors: Healthcare AI (30%), Enterprise AI (25%), AutoML (20%)
• Valuation multiples: 8-12x revenue for established AI companies
• IPO activity: 12 AI companies went public in 2024"""
    elif "AI startup" in query and data_type == "trends":
        return """Trend Analysis - AI Startup Market:
• Generative AI dominates (40% of all AI funding)
• Shift toward specialized AI solutions vs general platforms
• Growing focus on AI safety and responsible AI
• Enterprise adoption accelerating (70% of Fortune 500 piloting AI)
• Consolidation trend: Larger players acquiring specialized startups"""
    elif data_type == "competitors":
        return f"""Competitive Landscape for {query}:
• Major players consolidating market share
• High barrier to entry due to compute costs
• Differentiation through specialized domain expertise
• Partnership strategies with cloud providers common
• Open source vs proprietary model tensions"""
    else:
        return f"General market data for {query}: Market showing growth with increasing competition."

@tool
def trend_analyzer(data: str, timeframe: str = "1year") -> str:
    """
    Analyzes trends in provided data over specified timeframe.
    
    Args:
        data: Data to analyze
        timeframe: Analysis timeframe (1year, 2year, 5year)
    
    Returns:
        Trend analysis summary
    """
    print(f"   [Tool] trend_analyzer: Analyzing {timeframe} trends")
    time.sleep(0.3)
    
    return f"""Trend Analysis ({timeframe}):
GROWTH PATTERNS:
• Exponential growth phase identified
• Market maturity indicators emerging
• Seasonal fluctuations in Q4 typically stronger

KEY INDICATORS:
• Investment velocity increasing 40% YoY
• Market penetration reaching 15% of TAM
• Customer acquisition costs stabilizing

FUTURE PROJECTIONS:
• Continued growth expected at 25-30% CAGR
• Market consolidation likely in next 18-24 months
• Technology commoditization risk in 3-5 years"""

@tool
def synthesis_formatter(inputs: List[str], format_type: str = "executive_summary") -> str:
    """
    Formats multiple inputs into a structured output.
    
    Args:
        inputs: List of input texts to synthesize
        format_type: Output format (executive_summary, detailed_report, presentation)
    
    Returns:
        Formatted synthesis of inputs
    """
    print(f"   [Tool] synthesis_formatter: Creating {format_type}")
    time.sleep(0.4)
    
    if format_type == "executive_summary":
        return """EXECUTIVE SUMMARY FORMAT:
• Key findings organized by priority
• Financial metrics highlighted
• Strategic recommendations provided
• Risk factors identified
• Timeline for implementation suggested"""
    else:
        return "Formatted output combining all agent insights into structured report."

tools = [market_data_search, trend_analyzer, synthesis_formatter]
tool_executor = ToolExecutor(tools)

print(f"✅ Defined {len(tools)} collaborative tools")

# ============================================================================
# PART 2: DEFINE COLLABORATIVE AGENT STATE
# ============================================================================
print("\n📊 PART 2: Collaborative Agent State Definition")
print("-" * 60)

class CollaborativeAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    market_request: str
    current_agent: Literal["coordinator", "analyst", "researcher", "synthesizer"]
    task_assignments: Optional[List[str]]  # Tasks assigned by coordinator
    analyst_findings: Optional[str]        # Findings from analyst
    researcher_data: Optional[str]         # Data gathered by researcher
    coordinator_decisions: Optional[str]   # High-level decisions/direction
    final_synthesis: Optional[str]         # Final combined output
    collaboration_round: int               # Current round of collaboration
    max_collaboration_rounds: int         # Maximum rounds before conclusion

print("✅ Defined CollaborativeAgentState with agent-specific fields and collaboration tracking")

# ============================================================================
# PART 3: DEFINE SPECIALIZED AGENT NODES
# ============================================================================
print("\n🤖 PART 3: Specialized Agent Node Functions")
print("-" * 60)

# Specialized LLMs for different agent roles
coordinator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
analyst_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).bind_tools(tools)
researcher_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1).bind_tools(tools)
synthesizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(tools)

def coordinator_agent_node(state: CollaborativeAgentState):
    """Coordinator agent that orchestrates the collaboration."""
    print("   [Agent] Coordinator: Orchestrating collaboration...")
    
    request = state["market_request"]
    round_num = state.get("collaboration_round", 0)
    
    if round_num == 0:
        # Initial coordination - create task assignments
        prompt = f"""You are a market analysis coordinator.
Task: {request}

Create a collaboration plan by assigning specific tasks to specialized agents:
1. Analyst Agent - data analysis and pattern identification
2. Researcher Agent - information gathering and market research
3. Synthesizer Agent - combining insights into final report

For the request "{request}", create 2-3 specific tasks for each agent.
Output format:
ANALYST TASKS:
- Task 1
- Task 2

RESEARCHER TASKS:
- Task 1
- Task 2

SYNTHESIZER TASKS:
- Task 1
- Task 2
"""
    else:
        # Review progress and provide guidance
        analyst_findings = state.get("analyst_findings", "None")
        researcher_data = state.get("researcher_data", "None")
        
        prompt = f"""You are reviewing the collaboration progress for: {request}

Current findings:
ANALYST: {analyst_findings}
RESEARCHER: {researcher_data}

Provide coordination guidance:
1. Are we on track to answer the original request?
2. What additional work is needed?
3. Should we proceed to synthesis or gather more information?

Give clear direction for the next steps."""

    response = coordinator_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "coordinator",
        "coordinator_decisions": response.content,
        "messages": [AIMessage(content=f"Coordinator: {response.content}")],
        "collaboration_round": round_num + 1
    }

def analyst_agent_node(state: CollaborativeAgentState):
    """Analyst agent that analyzes data and identifies patterns."""
    print("   [Agent] Analyst: Analyzing market data...")
    
    request = state["market_request"]
    coordinator_guidance = state.get("coordinator_decisions", "")
    
    prompt = f"""You are a market analysis specialist.
Original request: {request}
Coordinator guidance: {coordinator_guidance}

Your role is to:
1. Analyze available market data using tools if needed
2. Identify key patterns and trends
3. Provide data-driven insights

Use the available tools to gather and analyze market data. Focus on quantitative analysis and trend identification.
"""
    
    response = analyst_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "analyst",
        "messages": [response],
        "analyst_findings": "Analyst working on market data analysis..."
    }

def researcher_agent_node(state: CollaborativeAgentState):
    """Researcher agent that gathers additional market information."""
    print("   [Agent] Researcher: Gathering market intelligence...")
    
    request = state["market_request"]
    coordinator_guidance = state.get("coordinator_decisions", "")
    
    prompt = f"""You are a market research specialist.
Original request: {request}
Coordinator guidance: {coordinator_guidance}

Your role is to:
1. Gather comprehensive market information using available tools
2. Research competitive landscape
3. Identify market opportunities and threats

Use the market research tools to gather relevant data. Focus on qualitative insights and market intelligence.
"""
    
    response = researcher_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "researcher", 
        "messages": [response],
        "researcher_data": "Researcher gathering market intelligence..."
    }

def synthesizer_agent_node(state: CollaborativeAgentState):
    """Synthesizer agent that combines insights from all agents."""
    print("   [Agent] Synthesizer: Combining insights...")
    
    request = state["market_request"]
    analyst_findings = state.get("analyst_findings", "")
    researcher_data = state.get("researcher_data", "")
    coordinator_decisions = state.get("coordinator_decisions", "")
    
    prompt = f"""You are a synthesis specialist combining insights from multiple agents.
Original request: {request}

Available inputs:
COORDINATOR GUIDANCE: {coordinator_decisions}
ANALYST FINDINGS: {analyst_findings}  
RESEARCHER DATA: {researcher_data}

Create a comprehensive market analysis report that synthesizes all available information.
Use the synthesis_formatter tool to create a well-structured final report.
The report should be actionable and directly address the original request.
"""
    
    response = synthesizer_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "synthesizer",
        "messages": [response],
        "final_synthesis": "Synthesizer creating comprehensive report..."
    }

def execute_tool_node(state: CollaborativeAgentState):
    """Executes tools called by any agent."""
    print("   [Tool Execution] Processing agent tool calls...")
    
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {}
    
    tool_results = []
    current_agent = state["current_agent"]
    
    for tool_call in last_message.tool_calls:
        print(f"   Executing {tool_call['name']} for {current_agent} agent")
        try:
            tool_invocation = ToolInvocation(tool=tool_call["name"], tool_input=tool_call["args"])
            tool_output = tool_executor.invoke(tool_invocation)
            tool_results.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
            # Update agent-specific findings based on tool results
            if current_agent == "analyst":
                return {
                    "messages": tool_results,
                    "analyst_findings": f"Analyst findings updated with {tool_call['name']} results"
                }
            elif current_agent == "researcher":
                return {
                    "messages": tool_results,
                    "researcher_data": f"Research data updated with {tool_call['name']} results"
                }
            elif current_agent == "synthesizer":
                return {
                    "messages": tool_results,
                    "final_synthesis": f"Synthesis updated with {tool_call['name']} results"
                }
                
        except Exception as e:
            error_msg = f"Error executing {tool_call['name']}: {e}"
            tool_results.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
    
    return {"messages": tool_results}

# ============================================================================
# PART 4: COLLABORATION ROUTING LOGIC
# ============================================================================
print("\n🔀 PART 4: Collaboration Routing Logic")
print("-" * 60)

def route_collaboration_flow(state: CollaborativeAgentState):
    """Routes between different agents based on collaboration state."""
    current_round = state.get("collaboration_round", 0)
    max_rounds = state.get("max_collaboration_rounds", 4)
    current_agent = state.get("current_agent", "coordinator")
    
    # Check if any agent has pending tool calls
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and last_message.tool_calls:
        print(f"   Router: Tool call pending from {current_agent}. Routing to execute_tool_node")
        return "execute_tool_node"
    
    # Collaboration flow logic
    if current_round >= max_rounds:
        print("   Router: Max collaboration rounds reached. Finalizing with synthesizer")
        return "synthesizer_agent_node"
    
    if current_round == 1:  # After initial coordination
        print("   Router: Starting with analyst agent")
        return "analyst_agent_node"
    elif current_agent == "analyst":
        print("   Router: Moving from analyst to researcher")
        return "researcher_agent_node"
    elif current_agent == "researcher":
        print("   Router: Moving from researcher to synthesizer")
        return "synthesizer_agent_node"
    elif current_agent == "synthesizer":
        print("   Router: Synthesis complete, back to coordinator for review")
        return "coordinator_agent_node"
    else:  # coordinator
        if current_round < max_rounds - 1:
            print("   Router: Coordinator review complete, continuing collaboration")
            return "analyst_agent_node"
        else:
            print("   Router: Final coordination, proceeding to synthesis")
            return "synthesizer_agent_node"

def route_from_tools(state: CollaborativeAgentState):
    """Routes back to appropriate agent after tool execution."""
    current_agent = state.get("current_agent", "coordinator")
    print(f"   Router: Tool execution complete, returning to {current_agent}_agent_node")
    return f"{current_agent}_agent_node"

def check_completion(state: CollaborativeAgentState):
    """Checks if collaboration should end."""
    current_agent = state.get("current_agent")
    current_round = state.get("collaboration_round", 0)
    max_rounds = state.get("max_collaboration_rounds", 4)
    
    if current_agent == "synthesizer" and current_round >= max_rounds:
        print("   Router: Collaboration complete")
        return "END"
    else:
        print("   Router: Continuing collaboration")
        return "continue"

print("✅ Collaboration routing functions defined")

# ============================================================================
# PART 5: BUILD THE COLLABORATIVE AGENT GRAPH
# ============================================================================
print("\n🏗️  PART 5: Building Collaborative Agent Graph")
print("-" * 60)

memory = SqliteSaver.from_conn_string(":memory:")
workflow = StateGraph(CollaborativeAgentState)

# Add agent nodes
workflow.add_node("coordinator_agent_node", coordinator_agent_node)
workflow.add_node("analyst_agent_node", analyst_agent_node)
workflow.add_node("researcher_agent_node", researcher_agent_node)
workflow.add_node("synthesizer_agent_node", synthesizer_agent_node)
workflow.add_node("execute_tool_node", execute_tool_node)

# Set entry point
workflow.set_entry_point("coordinator_agent_node")

# Add conditional edges for collaboration flow
workflow.add_conditional_edges(
    "coordinator_agent_node",
    route_collaboration_flow,
    {
        "analyst_agent_node": "analyst_agent_node",
        "synthesizer_agent_node": "synthesizer_agent_node",
        "execute_tool_node": "execute_tool_node"
    }
)

workflow.add_conditional_edges(
    "analyst_agent_node", 
    route_collaboration_flow,
    {
        "researcher_agent_node": "researcher_agent_node",
        "execute_tool_node": "execute_tool_node"
    }
)

workflow.add_conditional_edges(
    "researcher_agent_node",
    route_collaboration_flow,
    {
        "synthesizer_agent_node": "synthesizer_agent_node", 
        "execute_tool_node": "execute_tool_node"
    }
)

workflow.add_conditional_edges(
    "synthesizer_agent_node",
    check_completion,
    {
        "END": END,
        "continue": "coordinator_agent_node"
    }
)

workflow.add_conditional_edges(
    "execute_tool_node",
    route_from_tools,
    {
        "coordinator_agent_node": "coordinator_agent_node",
        "analyst_agent_node": "analyst_agent_node", 
        "researcher_agent_node": "researcher_agent_node",
        "synthesizer_agent_node": "synthesizer_agent_node"
    }
)

# Compile the graph
collaborative_graph = workflow.compile(checkpointer=memory)
print("✅ Collaborative agent graph compiled successfully!")

# ============================================================================
# PART 6: INTERACTIVE SESSION FOR COLLABORATIVE AGENTS
# ============================================================================
print("\n🎮 PART 6: Interactive Collaborative Agent Session")
print("-" * 60)

def run_collaborative_session():
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting collaborative agent session: {session_id}")
    market_request = input("Enter market analysis request (e.g., 'Analyze the AI startup market'): ").strip()
    
    if not market_request:
        print("No request provided. Using default: 'Analyze the AI startup market'")
        market_request = "Analyze the AI startup market"

    initial_input = {
        "market_request": market_request,
        "messages": [HumanMessage(content=f"Market analysis request: {market_request}")],
        "collaboration_round": 0,
        "max_collaboration_rounds": 4,
        "current_agent": "coordinator"
    }

    print(f"\n--- Starting Collaborative Agent System ---")
    print(f"Request: {market_request}")
    print("-" * 60)
    
    try:
        for event in collaborative_graph.stream(initial_input, config, stream_mode="values"):
            current_agent = event.get("current_agent", "unknown")
            round_num = event.get("collaboration_round", 0)
            
            print(f"\n[Round {round_num}] Current Agent: {current_agent.title()}")
            
            if event.get("coordinator_decisions"):
                print(f"📋 Coordinator Decisions:")
                print(f"   {event['coordinator_decisions'][:200]}...")
                
            if event.get("analyst_findings"):
                print(f"📊 Analyst Progress:")
                print(f"   {event['analyst_findings'][:150]}...")
                
            if event.get("researcher_data"): 
                print(f"🔍 Researcher Progress:")
                print(f"   {event['researcher_data'][:150]}...")
                
            if event.get("final_synthesis"):
                print(f"📝 Synthesis Progress:")
                print(f"   {event['final_synthesis'][:150]}...")
                
            # Show agent messages
            if event.get("messages"):
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    if not last_msg.tool_calls:
                        print(f"💬 Agent Output:")
                        print(f"   {last_msg.content[:200]}...")
                elif isinstance(last_msg, ToolMessage):
                    print(f"🔧 Tool Result: {last_msg.content[:100]}...")
            
            time.sleep(0.5)  # Pause for readability

        print("\n" + "="*30 + " COLLABORATION COMPLETE " + "="*30)
        
        # Get final state
        final_state = collaborative_graph.get_state(config)
        if final_state and final_state.values.get("final_synthesis"):
            print("\n📋 FINAL COLLABORATIVE ANALYSIS:")
            print(final_state.values["final_synthesis"])
        else:
            print("Collaborative analysis completed. Check agent messages above for results.")
            
    except Exception as e:
        print(f"\n❌ Error during collaborative session: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_collaborative_session()
    print("\n" + "="*80)
    print("Collaborative Multi-Agent System Complete!")
    print("Key features demonstrated:")
    print("  ✓ Multiple specialized agents working together")
    print("  ✓ Coordinator orchestrating workflow") 
    print("  ✓ Agent-specific tool usage and findings")
    print("  ✓ Collaborative rounds with state sharing")
    print("  ✓ Synthesis of multiple agent perspectives")
    print("="*80)