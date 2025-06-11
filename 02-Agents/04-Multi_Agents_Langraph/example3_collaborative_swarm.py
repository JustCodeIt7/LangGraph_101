#!/usr/bin/env python3
"""
LangGraph Multi-Agent Example 3: Collaborative Agent Swarm (ADVANCED)

This example demonstrates the most sophisticated multi-agent architecture with:
- Multiple agents collaborating dynamically on complex tasks
- Peer-to-peer communication between agents
- Adaptive role assignment and task decomposition
- Consensus building and conflict resolution
- Real-time coordination and knowledge sharing

Scenario: Strategic Business Planning Swarm
- Strategy Agent: Develops high-level strategic approaches
- Market Agent: Analyzes market conditions and opportunities
- Finance Agent: Evaluates financial implications and projections
- Risk Agent: Identifies and assesses potential risks
- Innovation Agent: Proposes creative solutions and innovations
- Coordinator Agent: Facilitates collaboration and builds consensus
"""

import uuid
import time
import random
from typing import TypedDict, Annotated, List, Optional, Dict, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 70)
print("🤖 LangGraph Multi-Agent Example 3: Collaborative Agent Swarm")
print("=" * 70)

# ============================================================================
# PART 1: DEFINE COLLABORATIVE TOOLS
# ============================================================================
print("\n🔧 PART 1: Collaborative Agent Tools")
print("-" * 50)

@tool
def market_research(query: str, market_type: str = "global") -> str:
    """
    Conducts market research and analysis.
    
    Args:
        query: Research query or market segment
        market_type: Type of market (global, regional, niche)
    
    Returns:
        Market research findings
    """
    print(f"   [Market Tool] Researching {market_type} market for: {query}")
    time.sleep(0.4)
    
    return f"""Market Research Results for {query} ({market_type} market):

Market Size: $45-80B projected by 2027
Growth Rate: 15-25% CAGR over next 5 years
Key Drivers: Digital transformation, automation demand, regulatory changes
Market Leaders: 3-4 dominant players control 60% market share
Emerging Trends: AI integration, sustainability focus, customer experience priority
Competitive Landscape: Fragmented with high innovation potential
Entry Barriers: Moderate - technology expertise and capital requirements
Opportunity Score: 8.2/10 - Strong growth potential with manageable risks"""

@tool
def financial_modeling(scenario: str, timeframe: str = "5-year") -> str:
    """
    Creates financial models and projections.
    
    Args:
        scenario: Business scenario to model
        timeframe: Projection timeframe (1-year, 3-year, 5-year, 10-year)
    
    Returns:
        Financial analysis and projections
    """
    print(f"   [Finance Tool] Creating {timeframe} financial model for: {scenario}")
    time.sleep(0.5)
    
    # Simulate financial projections
    base_revenue = random.randint(50, 200)
    growth_rate = random.randint(15, 35)
    
    return f"""Financial Model - {scenario} ({timeframe} projection):

Revenue Projections:
Year 1: ${base_revenue}M (baseline)
Year 3: ${int(base_revenue * 1.5)}M ({growth_rate}% CAGR)
Year 5: ${int(base_revenue * 2.2)}M (sustained growth)

Cost Structure:
COGS: 45-55% of revenue
OpEx: 25-35% of revenue
R&D Investment: 8-12% of revenue

Profitability Timeline:
Break-even: Month 18-24
Positive Cash Flow: Month 30-36
Target Margins: 20-25% by Year 3

ROI Analysis: 280-350% over {timeframe}
Payback Period: 2.5-3.2 years
NPV (10% discount): ${int(base_revenue * 1.8)}M"""

@tool
def risk_assessment(domain: str, risk_type: str = "comprehensive") -> str:
    """
    Assesses risks across different business domains.
    
    Args:
        domain: Business domain to assess
        risk_type: Type of risk analysis (operational, strategic, financial, comprehensive)
    
    Returns:
        Risk assessment results
    """
    print(f"   [Risk Tool] Assessing {risk_type} risks in: {domain}")
    time.sleep(0.3)
    
    return f"""Risk Assessment - {domain} ({risk_type}):

HIGH RISKS (Probability: 20-30%, Impact: Severe):
• Market saturation in core segments
• Regulatory changes affecting business model
• Key talent acquisition and retention challenges

MEDIUM RISKS (Probability: 40-50%, Impact: Moderate):
• Technology disruption by competitors
• Economic downturn affecting customer spending
• Supply chain disruptions and cost increases

LOW RISKS (Probability: 10-15%, Impact: Minor):
• Minor operational inefficiencies
• Short-term customer preference shifts
• Temporary market volatility

MITIGATION STRATEGIES:
• Diversify revenue streams and market presence
• Build regulatory compliance and government relations
• Invest in talent development and retention programs
• Develop technology partnerships and innovation capabilities

Overall Risk Score: MODERATE (6.2/10)
Recommended Actions: Implement comprehensive risk monitoring and mitigation framework"""

@tool
def innovation_brainstorm(challenge: str, innovation_type: str = "disruptive") -> str:
    """
    Generates innovative solutions and creative approaches.
    
    Args:
        challenge: Challenge or problem to solve
        innovation_type: Type of innovation (incremental, disruptive, radical)
    
    Returns:
        Innovative solutions and approaches
    """
    print(f"   [Innovation Tool] Brainstorming {innovation_type} solutions for: {challenge}")
    time.sleep(0.6)
    
    return f"""Innovation Solutions - {challenge} ({innovation_type} approach):

BREAKTHROUGH IDEAS:
1. AI-Powered Predictive Platform
   - Machine learning algorithms for trend forecasting
   - Real-time market adaptation capabilities
   - Personalized customer experience optimization

2. Ecosystem Partnership Model
   - Multi-stakeholder collaboration platform
   - Shared value creation framework
   - Network effect amplification strategy

3. Sustainable Innovation Loop
   - Circular economy integration
   - Environmental impact optimization
   - Social responsibility embedded in core operations

IMPLEMENTATION ROADMAP:
Phase 1 (0-6 months): Proof of concept development
Phase 2 (6-18 months): Pilot testing and iteration
Phase 3 (18-36 months): Scale and market deployment

Innovation Score: 9.1/10 - High potential for market differentiation
Technology Readiness: Level 6-7 (demonstrated in relevant environment)
Market Readiness: 75% - Strong customer demand signals"""

@tool
def consensus_builder(proposals: str, criteria: str = "strategic_value") -> str:
    """
    Builds consensus among different proposals and viewpoints.
    
    Args:
        proposals: Different proposals or viewpoints to evaluate
        criteria: Evaluation criteria (strategic_value, feasibility, impact, risk_reward)
    
    Returns:
        Consensus analysis and recommendations
    """
    print(f"   [Consensus Tool] Building consensus on proposals using {criteria} criteria")
    time.sleep(0.4)
    
    return f"""Consensus Analysis ({criteria} evaluation):

PROPOSAL EVALUATION MATRIX:
Proposal A: Strategic Value (8.5/10), Feasibility (7.2/10), Risk (Moderate)
Proposal B: Strategic Value (7.8/10), Feasibility (8.7/10), Risk (Low)
Proposal C: Strategic Value (9.2/10), Feasibility (6.1/10), Risk (High)

CONSENSUS RECOMMENDATIONS:
Primary Choice: Hybrid approach combining Proposals A & B
- Leverage high strategic value of A with feasibility of B
- Phased implementation to manage risk exposure
- 85% stakeholder alignment achieved

Alternative Path: Modified Proposal C with risk mitigation
- Address feasibility gaps through partnerships
- Staged rollout to minimize high-risk exposure
- 70% stakeholder alignment with conditions

DECISION FRAMEWORK:
• Prioritize strategic value while ensuring feasibility
• Balance innovation potential with risk tolerance
• Maintain stakeholder alignment and buy-in
• Create measurable milestones and success metrics

Consensus Score: 8.1/10 - Strong alignment on recommended approach"""

# Create tool collections for different agent types
market_tools = [market_research]
finance_tools = [financial_modeling]
risk_tools = [risk_assessment]
innovation_tools = [innovation_brainstorm]
coordination_tools = [consensus_builder]

# Tool executors
market_executor = ToolExecutor(market_tools)
finance_executor = ToolExecutor(finance_tools)
risk_executor = ToolExecutor(risk_tools)
innovation_executor = ToolExecutor(innovation_tools)
coordination_executor = ToolExecutor(coordination_tools)

print(f"✅ Defined {len(market_tools + finance_tools + risk_tools + innovation_tools + coordination_tools)} collaborative tools")

# ============================================================================
# PART 2: DEFINE COLLABORATIVE SWARM STATE
# ============================================================================
print("\n📊 PART 2: Collaborative Swarm State")
print("-" * 50)

class CollaborativeSwarmState(TypedDict):
    messages: Annotated[List, add_messages]
    business_challenge: str
    active_agents: List[str]
    current_agent: Optional[str]
    collaboration_round: int
    max_rounds: int
    agent_contributions: Dict[str, str]
    cross_agent_insights: List[str]
    consensus_status: Optional[str]
    final_strategy: Optional[str]
    coordination_notes: List[str]

print("✅ Defined CollaborativeSwarmState with dynamic multi-agent coordination")

# ============================================================================
# PART 3: DEFINE COLLABORATIVE AGENT NODES
# ============================================================================
print("\n🤖 PART 3: Collaborative Agent Nodes")
print("-" * 50)

# Specialized LLMs for each agent type
strategy_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
market_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2).bind_tools(market_tools)
finance_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1).bind_tools(finance_tools)
risk_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2).bind_tools(risk_tools)
innovation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).bind_tools(innovation_tools)
coordinator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).bind_tools(coordination_tools)

def strategy_agent_node(state: CollaborativeSwarmState):
    """Strategy agent that develops high-level strategic approaches."""
    print("   [Strategy Agent] Developing strategic approaches...")
    
    challenge = state["business_challenge"]
    round_num = state["collaboration_round"]
    other_contributions = state.get("agent_contributions", {})
    
    prompt = f"""You are a strategic planning specialist in a collaborative team.

Business Challenge: {challenge}
Collaboration Round: {round_num}
Other Team Contributions: {other_contributions}

Develop strategic approaches that:
1. Address the core business challenge
2. Consider inputs from other team members
3. Propose high-level strategic directions
4. Identify key success factors and objectives

Build upon insights from other agents while providing your unique strategic perspective."""
    
    response = strategy_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "strategy",
        "messages": [response]
    }

def market_agent_node(state: CollaborativeSwarmState):
    """Market agent that analyzes market conditions and opportunities."""
    print("   [Market Agent] Analyzing market landscape...")
    
    challenge = state["business_challenge"]
    other_contributions = state.get("agent_contributions", {})
    
    prompt = f"""You are a market analysis specialist in a collaborative team.

Business Challenge: {challenge}
Team Insights So Far: {other_contributions}

Use the market_research tool to analyze:
1. Market opportunities and threats
2. Competitive landscape and positioning
3. Customer needs and market segments
4. Market trends and growth potential

Integrate your market insights with perspectives from other team members."""
    
    response = market_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "market",
        "messages": [response]
    }

def finance_agent_node(state: CollaborativeSwarmState):
    """Finance agent that evaluates financial implications."""
    print("   [Finance Agent] Evaluating financial implications...")
    
    challenge = state["business_challenge"]
    other_contributions = state.get("agent_contributions", {})
    
    prompt = f"""You are a financial analysis specialist in a collaborative team.

Business Challenge: {challenge}
Team Insights So Far: {other_contributions}

Use the financial_modeling tool to analyze:
1. Financial projections and business case
2. Investment requirements and ROI potential
3. Revenue models and cost structures
4. Financial risks and mitigation strategies

Align your financial analysis with strategic and market insights from the team."""
    
    response = finance_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "finance",
        "messages": [response]
    }

def risk_agent_node(state: CollaborativeSwarmState):
    """Risk agent that identifies and assesses potential risks."""
    print("   [Risk Agent] Assessing risks and mitigation strategies...")
    
    challenge = state["business_challenge"]
    other_contributions = state.get("agent_contributions", {})
    
    prompt = f"""You are a risk management specialist in a collaborative team.

Business Challenge: {challenge}
Team Insights So Far: {other_contributions}

Use the risk_assessment tool to analyze:
1. Strategic, operational, and financial risks
2. Market and competitive risks
3. Risk mitigation and contingency planning
4. Risk-adjusted recommendations

Consider risks identified by other team members and provide comprehensive risk perspective."""
    
    response = risk_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "risk",
        "messages": [response]
    }

def innovation_agent_node(state: CollaborativeSwarmState):
    """Innovation agent that proposes creative solutions."""
    print("   [Innovation Agent] Generating innovative solutions...")
    
    challenge = state["business_challenge"]
    other_contributions = state.get("agent_contributions", {})
    
    prompt = f"""You are an innovation specialist in a collaborative team.

Business Challenge: {challenge}
Team Insights So Far: {other_contributions}

Use the innovation_brainstorm tool to develop:
1. Creative and disruptive solutions
2. Novel approaches to address the challenge
3. Technology and business model innovations
4. Future-oriented strategic options

Build on insights from strategy, market, finance, and risk perspectives to create breakthrough solutions."""
    
    response = innovation_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "innovation",
        "messages": [response]
    }

def coordinator_agent_node(state: CollaborativeSwarmState):
    """Coordinator agent that facilitates collaboration and builds consensus."""
    print("   [Coordinator Agent] Facilitating collaboration and building consensus...")
    
    challenge = state["business_challenge"]
    agent_contributions = state.get("agent_contributions", {})
    round_num = state["collaboration_round"]
    max_rounds = state["max_rounds"]
    
    prompt = f"""You are a collaboration coordinator managing a strategic planning team.

Business Challenge: {challenge}
Round: {round_num}/{max_rounds}

Team Contributions:
{chr(10).join([f"{agent}: {contribution[:200]}..." for agent, contribution in agent_contributions.items()])}

Use the consensus_builder tool to:
1. Synthesize different perspectives and proposals
2. Identify areas of alignment and conflict
3. Build consensus on recommended approach
4. Create integrated strategic recommendations

Facilitate effective collaboration and decision-making among team members."""
    
    response = coordinator_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_agent": "coordinator",
        "messages": [response]
    }

def execute_collaborative_tools_node(state: CollaborativeSwarmState):
    """Executes tools called by collaborative agents."""
    print("   [Tool Execution] Processing collaborative tool calls...")
    
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {}
    
    current_agent = state.get("current_agent")
    tool_results = []
    
    # Route to appropriate tool executor
    executor_map = {
        "strategy": None,  # No tools for strategy agent
        "market": market_executor,
        "finance": finance_executor,
        "risk": risk_executor,
        "innovation": innovation_executor,
        "coordinator": coordination_executor
    }
    
    tool_executor = executor_map.get(current_agent)
    if not tool_executor:
        return {}
    
    for tool_call in last_message.tool_calls:
        try:
            tool_invocation = ToolInvocation(tool=tool_call["name"], tool_input=tool_call["args"])
            tool_output = tool_executor.invoke(tool_invocation)
            tool_results.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
            # Store agent contribution
            agent_contributions = state.get("agent_contributions", {}).copy()
            agent_contributions[current_agent] = str(tool_output)
            
            # Add cross-agent insight
            cross_insights = state.get("cross_agent_insights", []).copy()
            cross_insights.append(f"Round {state['collaboration_round']}: {current_agent} agent contributed {tool_call['name']} analysis")
            
            return {
                "messages": tool_results,
                "agent_contributions": agent_contributions,
                "cross_agent_insights": cross_insights
            }
                
        except Exception as e:
            error_msg = f"Error executing {tool_call['name']}: {e}"
            tool_results.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
    
    return {"messages": tool_results}

# ============================================================================
# PART 4: COLLABORATIVE ROUTING LOGIC
# ============================================================================
print("\n🔀 PART 4: Collaborative Swarm Routing")
print("-" * 50)

def route_collaborative_flow(state: CollaborativeSwarmState):
    """Routes between agents in the collaborative swarm."""
    current_round = state.get("collaboration_round", 1)
    max_rounds = state.get("max_rounds", 3)
    current_agent = state.get("current_agent")
    agent_contributions = state.get("agent_contributions", {})
    
    # Check for pending tool calls
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and last_message.tool_calls:
        print(f"   Router: Tool call pending from {current_agent}. Routing to execute_collaborative_tools_node")
        return "execute_collaborative_tools_node"
    
    # Determine active agents for this round
    if current_round == 1:
        # First round: Core analysis agents
        active_agents = ["strategy", "market", "finance"]
    elif current_round == 2:
        # Second round: Risk and innovation perspectives
        active_agents = ["risk", "innovation"]
    else:
        # Final round: Coordination and consensus
        active_agents = ["coordinator"]
    
    # Check if current round is complete
    round_complete = all(agent in agent_contributions for agent in active_agents)
    
    if round_complete and current_round < max_rounds:
        # Move to next round
        next_round = current_round + 1
        print(f"   Router: Round {current_round} complete. Moving to round {next_round}")
        
        # Determine first agent for next round
        if next_round == 2:
            next_agent = "risk"
        elif next_round == 3:
            next_agent = "coordinator"
        else:
            next_agent = "coordinator"
        
        # Update state for next round
        return {
            "collaboration_round": next_round,
            "current_agent": next_agent
        }
    
    elif round_complete and current_round >= max_rounds:
        # All rounds complete
        print("   Router: All collaboration rounds complete. Ending.")
        return "END"
    
    else:
        # Continue current round with next agent
        remaining_agents = [agent for agent in active_agents if agent not in agent_contributions]
        if remaining_agents:
            next_agent = remaining_agents[0]
            print(f"   Router: Continuing round {current_round}. Routing to {next_agent}_agent_node")
            return f"{next_agent}_agent_node"
        else:
            print("   Router: Round complete but max rounds not reached. Moving to coordinator.")
            return "coordinator_agent_node"

def route_from_collaborative_tools(state: CollaborativeSwarmState):
    """Routes back to appropriate agent after tool execution."""
    current_agent = state.get("current_agent")
    print(f"   Router: Tool execution complete. Returning to {current_agent}_agent_node")
    return f"{current_agent}_agent_node"

print("✅ Collaborative swarm routing functions defined")

# ============================================================================
# PART 5: BUILD THE COLLABORATIVE SWARM GRAPH
# ============================================================================
print("\n🏗️  PART 5: Building Collaborative Swarm Graph")
print("-" * 50)

workflow = StateGraph(CollaborativeSwarmState)

# Add agent nodes
workflow.add_node("strategy_agent_node", strategy_agent_node)
workflow.add_node("market_agent_node", market_agent_node)
workflow.add_node("finance_agent_node", finance_agent_node)
workflow.add_node("risk_agent_node", risk_agent_node)
workflow.add_node("innovation_agent_node", innovation_agent_node)
workflow.add_node("coordinator_agent_node", coordinator_agent_node)
workflow.add_node("execute_collaborative_tools_node", execute_collaborative_tools_node)

# Set entry point
workflow.set_entry_point("strategy_agent_node")

# Add comprehensive conditional routing for all agents
for agent in ["strategy", "market", "finance", "risk", "innovation", "coordinator"]:
    workflow.add_conditional_edges(
        f"{agent}_agent_node",
        route_collaborative_flow,
        {
            "strategy_agent_node": "strategy_agent_node",
            "market_agent_node": "market_agent_node",
            "finance_agent_node": "finance_agent_node",
            "risk_agent_node": "risk_agent_node",
            "innovation_agent_node": "innovation_agent_node",
            "coordinator_agent_node": "coordinator_agent_node",
            "execute_collaborative_tools_node": "execute_collaborative_tools_node",
            "END": END
        }
    )

workflow.add_conditional_edges(
    "execute_collaborative_tools_node",
    route_from_collaborative_tools,
    {
        "strategy_agent_node": "strategy_agent_node",
        "market_agent_node": "market_agent_node",
        "finance_agent_node": "finance_agent_node",
        "risk_agent_node": "risk_agent_node",
        "innovation_agent_node": "innovation_agent_node",
        "coordinator_agent_node": "coordinator_agent_node"
    }
)

# Compile the graph
memory = SqliteSaver.from_conn_string(":memory:")
collaborative_swarm_graph = workflow.compile(checkpointer=memory)

print("✅ Collaborative swarm graph compiled successfully!")

# ============================================================================
# PART 6: INTERACTIVE SESSION
# ============================================================================
print("\n🎮 PART 6: Interactive Collaborative Swarm Session")
print("-" * 50)

def run_collaborative_swarm():
    """Run the collaborative agent swarm system."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting collaborative swarm session: {session_id}")
    
    # Get business challenge from user
    challenge = input("Enter a complex business challenge (e.g., 'Enter the European market', 'Launch AI product line'): ").strip()
    if not challenge:
        challenge = "Develop a sustainable competitive advantage in the rapidly evolving AI market while maintaining ethical standards and regulatory compliance"
        print(f"No input provided. Using sample challenge.")
    
    # Initial input
    initial_input = {
        "business_challenge": challenge,
        "active_agents": ["strategy", "market", "finance"],
        "current_agent": "strategy",
        "collaboration_round": 1,
        "max_rounds": 3,
        "agent_contributions": {},
        "cross_agent_insights": [],
        "consensus_status": None,
        "final_strategy": None,
        "coordination_notes": [],
        "messages": [HumanMessage(content=f"Collaborative strategic planning for: {challenge}")]
    }
    
    print(f"\n--- Starting Collaborative Swarm Analysis ---")
    print(f"Challenge: {challenge}")
    print("-" * 60)
    
    try:
        step_count = 0
        # Run the collaborative swarm workflow
        for event in collaborative_swarm_graph.stream(initial_input, config, stream_mode="values"):
            step_count += 1
            current_round = event.get("collaboration_round", 1)
            current_agent = event.get("current_agent", "unknown")
            contributions = event.get("agent_contributions", {})
            
            print(f"\n[Round {current_round}] Agent: {current_agent.title()}")
            
            # Show progress indicators
            if len(contributions) > len(initial_input.get("agent_contributions", {})):
                latest_agent = list(contributions.keys())[-1]
                print(f"✅ {latest_agent.title()} agent completed analysis")
            
            # Show round transitions
            if current_round > initial_input.get("collaboration_round", 1):
                print(f"🔄 Advanced to Round {current_round}")
            
            # Show insights
            insights = event.get("cross_agent_insights", [])
            if insights and len(insights) > len(initial_input.get("cross_agent_insights", [])):
                print(f"💡 New insight: {insights[-1]}")
            
            time.sleep(0.2)  # Pace for readability
        
        # Display comprehensive results
        final_state = collaborative_swarm_graph.get_state(config)
        if final_state:
            values = final_state.values
            
            print("\n" + "="*60)
            print("🌐 COLLABORATIVE SWARM ANALYSIS RESULTS")
            print("="*60)
            
            print(f"\n🎯 Business Challenge:")
            print(values.get('business_challenge', 'N/A'))
            
            print(f"\n👥 Agent Contributions ({len(values.get('agent_contributions', {}))} agents):")
            contributions = values.get('agent_contributions', {})
            for agent, contribution in contributions.items():
                print(f"\n{agent.upper()} AGENT:")
                print("-" * 20)
                print(contribution[:300] + "..." if len(contribution) > 300 else contribution)
            
            print(f"\n🔗 Cross-Agent Insights:")
            insights = values.get('cross_agent_insights', [])
            for insight in insights[-5:]:  # Show last 5 insights
                print(f"   • {insight}")
            
            if values.get('consensus_status'):
                print(f"\n🤝 Consensus Status:")
                print(values['consensus_status'])
            
            print(f"\n📊 Collaboration Summary:")
            print(f"   • Rounds Completed: {values.get('collaboration_round', 0)}")
            print(f"   • Agents Participated: {len(contributions)}")
            print(f"   • Cross-Agent Insights: {len(insights)}")
            
            print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Error during collaborative swarm session: {e}")

if __name__ == "__main__":
    run_collaborative_swarm()
    print("\n" + "="*70)
    print("Collaborative Agent Swarm Complete!")
    print("Key features demonstrated:")
    print("  ✓ Multiple agents collaborating dynamically on complex tasks")
    print("  ✓ Peer-to-peer communication and knowledge sharing")
    print("  ✓ Adaptive role assignment and task decomposition")
    print("  ✓ Multi-round collaboration with consensus building")
    print("  ✓ Cross-agent insights and integrated decision making")
    print("  ✓ Advanced coordination and conflict resolution")
    print("="*70)