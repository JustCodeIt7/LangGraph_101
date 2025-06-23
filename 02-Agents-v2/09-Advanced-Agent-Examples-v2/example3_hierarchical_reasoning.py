#!/usr/bin/env python3
"""
LangGraph Advanced Agent Example 3: Hierarchical Reasoning Agent

This script demonstrates a hierarchical reasoning agent that can:
1. Break down complex problems into smaller sub-problems
2. Solve problems at different levels of abstraction
3. Combine solutions from lower levels to solve higher-level problems
4. Make strategic decisions at multiple hierarchical levels

Scenario: Strategic Business Decision Agent
- Takes a complex business challenge (e.g., "Enter the European market")
- Breaks it down into strategic, tactical, and operational levels
- Reasons through each level with appropriate depth and timeframe
- Combines insights across levels for comprehensive recommendations
"""

import uuid
import time
from typing import TypedDict, Annotated, List, Union, Optional, Dict, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 80)
print("LangGraph Advanced Example 3: Hierarchical Reasoning Agent")
print("=" * 80)

# ============================================================================
# PART 1: DEFINE TOOLS FOR HIERARCHICAL REASONING
# ============================================================================
print("\n🔧 PART 1: Defining Hierarchical Reasoning Tools")
print("-" * 60)

@tool
def strategic_analysis(business_challenge: str, timeframe: str = "3-5 years") -> Dict[str, Union[str, List[str]]]:
    """
    Performs high-level strategic analysis for business challenges.
    
    Args:
        business_challenge: The business challenge to analyze
        timeframe: Strategic timeframe (1-2 years, 3-5 years, 5+ years)
    
    Returns:
        Strategic analysis including market positioning, competitive advantage, and long-term vision
    """
    print(f"   [Tool] strategic_analysis: Analyzing '{business_challenge}' over {timeframe}")
    time.sleep(0.6)
    
    if "european market" in business_challenge.lower():
        return {
            "market_opportunity": "Europe represents $2.8T market with strong regulatory framework",
            "competitive_landscape": "Fragmented market with local leaders in each region",
            "strategic_objectives": [
                "Establish brand presence in 3 key markets within 18 months",
                "Build strategic partnerships with regional players",
                "Adapt product portfolio to European preferences and regulations"
            ],
            "success_metrics": [
                "Market share in target segments",
                "Brand recognition scores", 
                "Revenue growth from European operations"
            ],
            "risk_factors": [
                "Regulatory compliance complexity",
                "Currency fluctuation exposure",
                "Cultural adaptation challenges"
            ],
            "timeframe": timeframe,
            "analysis_level": "strategic"
        }
    elif "digital transformation" in business_challenge.lower():
        return {
            "market_opportunity": "Digital transformation market growing at 22% CAGR",
            "competitive_landscape": "Mix of traditional IT vendors and cloud-native disruptors",
            "strategic_objectives": [
                "Modernize core business processes",
                "Build digital-first customer experience",
                "Create data-driven decision making culture"
            ],
            "success_metrics": [
                "Digital revenue percentage",
                "Customer digital adoption rate",
                "Operational efficiency gains"
            ],
            "risk_factors": [
                "Technology integration complexity",
                "Change management resistance",
                "Cybersecurity vulnerabilities"
            ],
            "timeframe": timeframe,
            "analysis_level": "strategic"
        }
    else:
        return {
            "market_opportunity": f"Strategic opportunity analysis for {business_challenge}",
            "competitive_landscape": "Competitive dynamics require detailed analysis",
            "strategic_objectives": ["Define clear strategic direction", "Build competitive advantages"],
            "success_metrics": ["Market position", "Financial performance"],
            "risk_factors": ["Market uncertainty", "Execution risks"],
            "timeframe": timeframe,
            "analysis_level": "strategic"
        }

@tool
def tactical_planning(strategic_objectives: List[str], resources: str = "standard") -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """
    Creates tactical plans to achieve strategic objectives.
    
    Args:
        strategic_objectives: List of strategic objectives to plan for
        resources: Resource availability (limited, standard, abundant)
    
    Returns:
        Tactical plans with specific initiatives, timelines, and resource requirements
    """
    print(f"   [Tool] tactical_planning: Planning for {len(strategic_objectives)} objectives with {resources} resources")
    time.sleep(0.5)
    
    tactical_initiatives = []
    
    for i, objective in enumerate(strategic_objectives[:3]):  # Limit to first 3 objectives
        if "brand presence" in objective.lower():
            tactical_initiatives.append({
                "initiative": "Multi-channel Brand Launch Campaign",
                "objective_addressed": objective,
                "timeline": "12-18 months",
                "key_activities": "Market research, brand localization, channel partnerships, launch events",
                "resource_requirements": "Marketing team, local agencies, campaign budget",
                "success_criteria": "Brand awareness > 25% in target markets",
                "dependencies": "Regulatory approvals, local partnerships"
            })
        elif "partnership" in objective.lower():
            tactical_initiatives.append({
                "initiative": "Strategic Partnership Development",
                "objective_addressed": objective,
                "timeline": "6-12 months",
                "key_activities": "Partner identification, due diligence, negotiation, integration",
                "resource_requirements": "Business development team, legal support, integration resources",
                "success_criteria": "3-5 strategic partnerships established",
                "dependencies": "Partner availability, alignment of objectives"
            })
        elif "product portfolio" in objective.lower():
            tactical_initiatives.append({
                "initiative": "Product Localization Program",
                "objective_addressed": objective,
                "timeline": "9-15 months",
                "key_activities": "Requirements analysis, product adaptation, compliance testing, rollout",
                "resource_requirements": "Product team, compliance experts, testing resources",
                "success_criteria": "Product-market fit scores > 70%",
                "dependencies": "Market research completion, regulatory clarity"
            })
        else:
            tactical_initiatives.append({
                "initiative": f"Initiative for: {objective[:50]}...",
                "objective_addressed": objective,
                "timeline": "6-12 months",
                "key_activities": "Planning, execution, monitoring, optimization",
                "resource_requirements": "Cross-functional team, budget allocation",
                "success_criteria": "Objective-specific KPIs achieved",
                "dependencies": "Resource availability, stakeholder alignment"
            })
    
    return {
        "tactical_initiatives": tactical_initiatives,
        "resource_optimization": f"Plans optimized for {resources} resource environment",
        "coordination_needs": "Cross-initiative coordination required for maximum impact",
        "analysis_level": "tactical",
        "planning_horizon": "6-18 months"
    }

@tool
def operational_breakdown(tactical_initiative: str, urgency: str = "normal") -> Dict[str, Union[str, List[str]]]:
    """
    Breaks down tactical initiatives into operational tasks and workflows.
    
    Args:
        tactical_initiative: Tactical initiative to break down
        urgency: Urgency level (low, normal, high, critical)
    
    Returns:
        Operational breakdown with specific tasks, timelines, and resource assignments
    """
    print(f"   [Tool] operational_breakdown: Breaking down '{tactical_initiative}' with {urgency} urgency")
    time.sleep(0.4)
    
    if "brand launch" in tactical_initiative.lower():
        return {
            "immediate_tasks": [
                "Conduct market research in target countries",
                "Analyze competitor positioning and messaging",
                "Define brand positioning for European market"
            ],
            "short_term_tasks": [
                "Develop localized brand assets and messaging",
                "Identify and evaluate potential agency partners",
                "Create brand guidelines for European markets",
                "Establish legal framework for brand protection"
            ],
            "medium_term_tasks": [
                "Execute pilot campaign in primary market",
                "Build relationships with key media outlets",
                "Develop influencer partnership program",
                "Create measurement and optimization framework"
            ],
            "resource_assignments": {
                "Market Research": "External research firm + internal analyst",
                "Brand Development": "Creative agency + brand manager",
                "Legal/Compliance": "Legal team + local counsel",
                "Campaign Execution": "Marketing team + media agency"
            },
            "critical_path": "Market research → Brand positioning → Asset development → Pilot launch",
            "risk_mitigation": [
                "Regular stakeholder reviews and feedback loops",
                "Phased rollout to minimize risk exposure",
                "Contingency plans for regulatory changes"
            ],
            "analysis_level": "operational",
            "execution_timeframe": "3-6 months for initial phases"
        }
    elif "partnership" in tactical_initiative.lower():
        return {
            "immediate_tasks": [
                "Create partner evaluation criteria and scoring framework",
                "Identify potential strategic partners in target markets",
                "Develop partnership value proposition and term sheets"
            ],
            "short_term_tasks": [
                "Conduct initial outreach and screening calls",
                "Perform due diligence on priority partners",
                "Draft partnership agreements and frameworks",
                "Create integration and onboarding processes"
            ],
            "medium_term_tasks": [
                "Negotiate and execute partnership agreements",
                "Implement joint go-to-market strategies",
                "Establish performance monitoring and governance",
                "Scale successful partnership models"
            ],
            "resource_assignments": {
                "Partner Identification": "Business development team",
                "Due Diligence": "Strategy team + external consultants",
                "Legal Negotiations": "Legal team + external counsel",
                "Integration": "Operations team + partner management"
            },
            "critical_path": "Criteria definition → Partner identification → Due diligence → Negotiation",
            "risk_mitigation": [
                "Multiple partnership options to reduce dependency",
                "Pilot programs before full commitment",
                "Clear performance metrics and exit clauses"
            ],
            "analysis_level": "operational",
            "execution_timeframe": "2-4 months for partner selection and negotiation"
        }
    else:
        return {
            "immediate_tasks": [
                "Define detailed requirements and success criteria",
                "Identify and allocate necessary resources",
                "Create project timeline and milestone framework"
            ],
            "short_term_tasks": [
                "Execute core initiative activities",
                "Monitor progress against defined metrics",
                "Address blockers and adjust approach as needed"
            ],
            "medium_term_tasks": [
                "Scale successful approaches",
                "Optimize based on learnings and feedback",
                "Prepare for next phase or handoff"
            ],
            "resource_assignments": {
                "Project Management": "Dedicated project manager",
                "Execution": "Cross-functional team",
                "Quality Assurance": "Internal stakeholders"
            },
            "critical_path": "Requirements → Resource allocation → Execution → Optimization",
            "risk_mitigation": [
                "Regular progress reviews and course correction",
                "Stakeholder communication and alignment",
                "Contingency planning for key risks"
            ],
            "analysis_level": "operational",
            "execution_timeframe": "1-3 months per phase"
        }

tools = [strategic_analysis, tactical_planning, operational_breakdown]
tool_executor = ToolExecutor(tools)

print(f"✅ Defined {len(tools)} hierarchical reasoning tools")

# ============================================================================
# PART 2: DEFINE HIERARCHICAL REASONING STATE
# ============================================================================
print("\n📊 PART 2: Hierarchical Reasoning State Definition")
print("-" * 60)

class HierarchicalReasoningState(TypedDict):
    messages: Annotated[List, add_messages]
    business_challenge: str
    current_level: Literal["strategic", "tactical", "operational", "synthesis"]
    strategic_analysis: Optional[Dict[str, Union[str, List[str]]]]
    tactical_plans: Optional[Dict[str, Union[str, List[Dict[str, str]]]]]
    operational_breakdown: Optional[Dict[str, Union[str, List[str]]]]
    synthesis_report: Optional[str]
    reasoning_depth: int  # Current depth of hierarchical reasoning (1=strategic, 2=tactical, 3=operational)
    max_depth: int        # Maximum depth to explore
    cross_level_insights: List[str]  # Insights that span multiple levels

print("✅ Defined HierarchicalReasoningState with multi-level reasoning tracking")

# ============================================================================
# PART 3: DEFINE HIERARCHICAL REASONING NODES
# ============================================================================
print("\n🤖 PART 3: Hierarchical Reasoning Node Functions")
print("-" * 60)

# Specialized LLMs for different reasoning levels
strategic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2).bind_tools(tools)
tactical_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(tools) 
operational_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1).bind_tools(tools)
synthesis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

def strategic_reasoning_node(state: HierarchicalReasoningState):
    """Performs high-level strategic reasoning and analysis."""
    print("   [Node] Strategic Reasoning: Analyzing at strategic level...")
    
    challenge = state["business_challenge"]
    depth = state.get("reasoning_depth", 1)
    
    prompt = f"""You are a strategic business analyst conducting high-level analysis.
Business Challenge: {challenge}

Perform strategic-level analysis considering:
1. Market opportunities and competitive landscape
2. Long-term strategic objectives and positioning
3. Key success metrics and risk factors
4. Resource requirements and capabilities needed

Use the strategic_analysis tool to conduct this analysis. Focus on the 3-5 year strategic horizon.
Consider how this challenge fits into broader business strategy and market dynamics.
"""
    
    response = strategic_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_level": "strategic",
        "reasoning_depth": depth,
        "messages": [response]
    }

def tactical_planning_node(state: HierarchicalReasoningState):
    """Translates strategic objectives into tactical plans and initiatives."""
    print("   [Node] Tactical Planning: Developing tactical implementation plans...")
    
    challenge = state["business_challenge"]
    strategic_analysis = state.get("strategic_analysis", {})
    depth = state.get("reasoning_depth", 2)
    
    strategic_objectives = strategic_analysis.get("strategic_objectives", [])
    if not strategic_objectives:
        strategic_objectives = [f"Address key aspects of {challenge}"]
    
    prompt = f"""You are a tactical planning specialist translating strategic objectives into actionable plans.
Business Challenge: {challenge}
Strategic Objectives: {strategic_objectives}

Based on the strategic analysis, create tactical plans that:
1. Break down strategic objectives into specific initiatives
2. Define timelines and resource requirements
3. Identify dependencies and coordination needs
4. Establish success criteria and milestones

Use the tactical_planning tool to develop these plans. Focus on the 6-18 month tactical horizon.
Consider how different initiatives can be coordinated for maximum impact.
"""
    
    response = tactical_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_level": "tactical", 
        "reasoning_depth": depth,
        "messages": [response]
    }

def operational_execution_node(state: HierarchicalReasoningState):
    """Breaks down tactical plans into operational tasks and workflows."""
    print("   [Node] Operational Execution: Creating detailed operational plans...")
    
    challenge = state["business_challenge"]
    tactical_plans = state.get("tactical_plans", {})
    depth = state.get("reasoning_depth", 3)
    
    tactical_initiatives = tactical_plans.get("tactical_initiatives", [])
    if not tactical_initiatives:
        tactical_initiatives = [{"initiative": f"Core initiative for {challenge}"}]
    
    # Focus on the first tactical initiative for detailed breakdown
    primary_initiative = tactical_initiatives[0]["initiative"] if tactical_initiatives else f"Primary initiative for {challenge}"
    
    prompt = f"""You are an operational planning specialist creating detailed execution plans.
Business Challenge: {challenge}
Primary Tactical Initiative: {primary_initiative}

Break down this tactical initiative into operational components:
1. Immediate tasks (next 2-4 weeks)
2. Short-term tasks (1-3 months)  
3. Medium-term tasks (3-6 months)
4. Resource assignments and responsibilities
5. Critical path and dependencies
6. Risk mitigation strategies

Use the operational_breakdown tool to create this detailed plan. Focus on actionable tasks with clear timelines and ownership.
"""
    
    response = operational_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_level": "operational",
        "reasoning_depth": depth,
        "messages": [response]
    }

def synthesis_node(state: HierarchicalReasoningState):
    """Synthesizes insights across all hierarchical levels into comprehensive recommendations."""
    print("   [Node] Synthesis: Combining insights across all hierarchical levels...")
    
    challenge = state["business_challenge"]
    strategic_analysis = state.get("strategic_analysis", {})
    tactical_plans = state.get("tactical_plans", {})
    operational_breakdown = state.get("operational_breakdown", {})
    cross_level_insights = state.get("cross_level_insights", [])
    
    prompt = f"""You are a senior business strategist synthesizing multi-level analysis.
Business Challenge: {challenge}

Available Analysis:
STRATEGIC LEVEL: {strategic_analysis}
TACTICAL LEVEL: {tactical_plans}  
OPERATIONAL LEVEL: {operational_breakdown}
CROSS-LEVEL INSIGHTS: {cross_level_insights}

Create a comprehensive recommendation that:
1. Integrates insights from all hierarchical levels
2. Identifies connections and dependencies between levels
3. Provides clear action priorities and sequencing
4. Addresses potential conflicts or gaps between levels
5. Offers specific next steps and success measures

Focus on creating actionable recommendations that bridge strategic vision with operational reality.
The synthesis should provide a clear roadmap from current state to desired outcomes.
"""
    
    response = synthesis_llm.invoke([HumanMessage(content=prompt)])
    
    synthesis_report = response.content
    
    return {
        "current_level": "synthesis",
        "synthesis_report": synthesis_report,
        "messages": [AIMessage(content=f"Hierarchical Analysis Complete:\n{synthesis_report}")]
    }

def execute_tool_node(state: HierarchicalReasoningState):
    """Executes hierarchical reasoning tools and updates state with results."""
    print("   [Tool Execution] Processing hierarchical reasoning tool calls...")
    
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {}
    
    tool_results = []
    current_level = state.get("current_level", "strategic")
    cross_level_insights = state.get("cross_level_insights", []).copy()
    
    for tool_call in last_message.tool_calls:
        print(f"   Executing {tool_call['name']} at {current_level} level")
        
        try:
            tool_invocation = ToolInvocation(tool=tool_call["name"], tool_input=tool_call["args"])
            tool_output = tool_executor.invoke(tool_invocation)
            tool_results.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
            # Store results based on tool type and add cross-level insights
            if tool_call["name"] == "strategic_analysis":
                cross_level_insights.append(f"Strategic insight: {tool_output.get('market_opportunity', 'Key strategic opportunity identified')}")
                return {
                    "messages": tool_results,
                    "strategic_analysis": tool_output,
                    "cross_level_insights": cross_level_insights
                }
            elif tool_call["name"] == "tactical_planning":
                cross_level_insights.append(f"Tactical insight: {len(tool_output.get('tactical_initiatives', []))} initiatives planned")
                return {
                    "messages": tool_results,
                    "tactical_plans": tool_output,
                    "cross_level_insights": cross_level_insights
                }
            elif tool_call["name"] == "operational_breakdown":
                cross_level_insights.append(f"Operational insight: {len(tool_output.get('immediate_tasks', []))} immediate tasks identified")
                return {
                    "messages": tool_results,
                    "operational_breakdown": tool_output,
                    "cross_level_insights": cross_level_insights
                }
                
        except Exception as e:
            error_msg = f"Error executing {tool_call['name']}: {e}"
            tool_results.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
    
    return {
        "messages": tool_results,
        "cross_level_insights": cross_level_insights
    }

# ============================================================================
# PART 4: HIERARCHICAL ROUTING LOGIC
# ============================================================================
print("\n🔀 PART 4: Hierarchical Routing Logic")
print("-" * 60)

def route_hierarchical_flow(state: HierarchicalReasoningState):
    """Routes between different hierarchical levels based on reasoning progress."""
    current_level = state.get("current_level", "strategic")
    reasoning_depth = state.get("reasoning_depth", 1)
    max_depth = state.get("max_depth", 3)
    
    # Check for pending tool calls
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and last_message.tool_calls:
        print(f"   Router: Tool call pending at {current_level} level. Routing to execute_tool_node")
        return "execute_tool_node"
    
    # Progress through hierarchical levels
    if current_level == "strategic" and reasoning_depth < max_depth:
        if state.get("strategic_analysis"):
            print("   Router: Strategic analysis complete. Moving to tactical level")
            return "tactical_planning_node"
        else:
            print("   Router: Continuing strategic analysis")
            return "strategic_reasoning_node"
            
    elif current_level == "tactical" and reasoning_depth < max_depth:
        if state.get("tactical_plans"):
            print("   Router: Tactical planning complete. Moving to operational level")
            return "operational_execution_node"
        else:
            print("   Router: Continuing tactical planning")
            return "tactical_planning_node"
            
    elif current_level == "operational" or reasoning_depth >= max_depth:
        if state.get("operational_breakdown") or reasoning_depth >= max_depth:
            print("   Router: All levels analyzed. Moving to synthesis")
            return "synthesis_node"
        else:
            print("   Router: Continuing operational planning")
            return "operational_execution_node"
            
    else:  # synthesis or completed
        print("   Router: Hierarchical reasoning complete")
        return "END"

def route_from_tools(state: HierarchicalReasoningState):
    """Routes back to appropriate reasoning level after tool execution."""
    current_level = state.get("current_level", "strategic")
    print(f"   Router: Tool execution complete at {current_level} level. Returning to {current_level}_node")
    
    if current_level == "strategic":
        return "strategic_reasoning_node"
    elif current_level == "tactical":
        return "tactical_planning_node"
    elif current_level == "operational":
        return "operational_execution_node"
    else:
        return "synthesis_node"

print("✅ Hierarchical routing functions defined")

# ============================================================================
# PART 5: BUILD THE HIERARCHICAL REASONING GRAPH
# ============================================================================
print("\n🏗️  PART 5: Building Hierarchical Reasoning Graph")
print("-" * 60)

memory = SqliteSaver.from_conn_string(":memory:")
workflow = StateGraph(HierarchicalReasoningState)

# Add reasoning level nodes
workflow.add_node("strategic_reasoning_node", strategic_reasoning_node)
workflow.add_node("tactical_planning_node", tactical_planning_node)
workflow.add_node("operational_execution_node", operational_execution_node)
workflow.add_node("synthesis_node", synthesis_node)
workflow.add_node("execute_tool_node", execute_tool_node)

# Set entry point
workflow.set_entry_point("strategic_reasoning_node")

# Add conditional edges for hierarchical flow
workflow.add_conditional_edges(
    "strategic_reasoning_node",
    route_hierarchical_flow,
    {
        "strategic_reasoning_node": "strategic_reasoning_node",
        "tactical_planning_node": "tactical_planning_node",
        "execute_tool_node": "execute_tool_node"
    }
)

workflow.add_conditional_edges(
    "tactical_planning_node",
    route_hierarchical_flow,
    {
        "tactical_planning_node": "tactical_planning_node",
        "operational_execution_node": "operational_execution_node",
        "execute_tool_node": "execute_tool_node"
    }
)

workflow.add_conditional_edges(
    "operational_execution_node",
    route_hierarchical_flow,
    {
        "operational_execution_node": "operational_execution_node",
        "synthesis_node": "synthesis_node",
        "execute_tool_node": "execute_tool_node"
    }
)

workflow.add_conditional_edges(
    "execute_tool_node",
    route_from_tools,
    {
        "strategic_reasoning_node": "strategic_reasoning_node",
        "tactical_planning_node": "tactical_planning_node", 
        "operational_execution_node": "operational_execution_node",
        "synthesis_node": "synthesis_node"
    }
)

workflow.add_edge("synthesis_node", END)

# Compile the graph
hierarchical_graph = workflow.compile(checkpointer=memory)
print("✅ Hierarchical reasoning graph compiled successfully!")

# ============================================================================
# PART 6: INTERACTIVE SESSION FOR HIERARCHICAL REASONING
# ============================================================================
print("\n🎮 PART 6: Interactive Hierarchical Reasoning Session")
print("-" * 60)

def run_hierarchical_reasoning_session():
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting hierarchical reasoning session: {session_id}")
    business_challenge = input("Enter business challenge (e.g., 'Enter the European market', 'Digital transformation strategy'): ").strip()
    
    if not business_challenge:
        print("No challenge provided. Using default: 'Enter the European market'")
        business_challenge = "Enter the European market"

    initial_input = {
        "business_challenge": business_challenge,
        "messages": [HumanMessage(content=f"Business challenge: {business_challenge}")],
        "current_level": "strategic",
        "reasoning_depth": 1,
        "max_depth": 3,
        "cross_level_insights": []
    }

    print(f"\n--- Starting Hierarchical Reasoning Analysis ---")
    print(f"Challenge: {business_challenge}")
    print("-" * 60)
    
    try:
        for event in hierarchical_graph.stream(initial_input, config, stream_mode="values"):
            current_level = event.get("current_level", "unknown")
            reasoning_depth = event.get("reasoning_depth", 0)
            
            print(f"\n[Level {reasoning_depth}] Current Analysis: {current_level.title()}")
            
            # Show level-specific progress
            if event.get("strategic_analysis"):
                analysis = event["strategic_analysis"]
                print(f"📈 Strategic Analysis:")
                print(f"   Market Opportunity: {analysis.get('market_opportunity', 'N/A')[:100]}...")
                if analysis.get("strategic_objectives"):
                    print(f"   Objectives: {len(analysis['strategic_objectives'])} defined")
                    
            if event.get("tactical_plans"):
                plans = event["tactical_plans"]
                initiatives = plans.get("tactical_initiatives", [])
                print(f"🎯 Tactical Planning:")
                print(f"   Initiatives: {len(initiatives)} tactical initiatives planned")
                if initiatives:
                    print(f"   Primary Initiative: {initiatives[0].get('initiative', 'N/A')[:80]}...")
                    
            if event.get("operational_breakdown"):
                breakdown = event["operational_breakdown"]
                immediate_tasks = breakdown.get("immediate_tasks", [])
                print(f"⚙️  Operational Breakdown:")
                print(f"   Immediate Tasks: {len(immediate_tasks)} tasks identified")
                if immediate_tasks:
                    print(f"   First Task: {immediate_tasks[0][:80]}...")
                    
            # Show cross-level insights
            if event.get("cross_level_insights"):
                insights = event["cross_level_insights"]
                if insights and len(insights) > len(initial_input.get("cross_level_insights", [])):
                    print(f"🔗 Cross-Level Insight: {insights[-1]}")
            
            # Show agent messages
            if event.get("messages"):
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content and not last_msg.tool_calls:
                    if current_level == "synthesis":
                        print(f"📋 Synthesis Complete")
                    else:
                        print(f"🤖 {current_level.title()} Agent: Working...")
                elif isinstance(last_msg, ToolMessage):
                    print(f"🔧 Tool Result: Analysis completed at {current_level} level")
            
            time.sleep(0.4)  # Pause for readability

        print("\n" + "="*30 + " HIERARCHICAL ANALYSIS COMPLETE " + "="*30)
        
        # Get final state
        final_state = hierarchical_graph.get_state(config)
        if final_state and final_state.values.get("synthesis_report"):
            print("\n📊 COMPREHENSIVE HIERARCHICAL ANALYSIS:")
            print("="*80)
            print(final_state.values["synthesis_report"])
            print("="*80)
            
            # Show summary of insights across levels
            insights = final_state.values.get("cross_level_insights", [])
            if insights:
                print(f"\n🔗 Cross-Level Insights Summary ({len(insights)} insights):")
                for insight in insights[-5:]:  # Show last 5 insights
                    print(f"   • {insight}")
        else:
            print("Hierarchical analysis completed. Check above for level-specific results.")
            
    except Exception as e:
        print(f"\n❌ Error during hierarchical reasoning session: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_hierarchical_reasoning_session()
    print("\n" + "="*80)
    print("Hierarchical Reasoning Agent Complete!")
    print("Key features demonstrated:")
    print("  ✓ Multi-level problem decomposition (Strategic → Tactical → Operational)")
    print("  ✓ Level-appropriate reasoning depth and timeframes")
    print("  ✓ Cross-level insight integration and dependency tracking")
    print("  ✓ Hierarchical synthesis combining all analysis levels")
    print("  ✓ Tool specialization for different reasoning levels")
    print("  ✓ Comprehensive business challenge analysis framework")
    print("="*80)