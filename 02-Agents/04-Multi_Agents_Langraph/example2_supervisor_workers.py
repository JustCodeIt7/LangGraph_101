#!/usr/bin/env python3
"""
LangGraph Multi-Agent Example 2: Supervisor-Worker Pattern (INTERMEDIATE)

This example demonstrates a more sophisticated multi-agent architecture with:
- A supervisor agent that coordinates multiple worker agents
- Dynamic task assignment based on content type
- Tool usage by specialized worker agents
- Conditional routing and decision making

Scenario: Content Analysis Team
- Supervisor: Analyzes requests and assigns tasks to appropriate workers
- Text Analyst: Analyzes text content and extracts insights
- Data Analyst: Processes numerical data and statistics
- Sentiment Analyst: Evaluates emotional tone and sentiment
"""

import uuid
import time
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 70)
print("🤖 LangGraph Multi-Agent Example 2: Supervisor-Worker Pattern")
print("=" * 70)

# ============================================================================
# PART 1: DEFINE WORKER AGENT TOOLS
# ============================================================================
print("\n🔧 PART 1: Worker Agent Tools")
print("-" * 50)

@tool
def extract_key_topics(text: str) -> str:
    """
    Extracts key topics and themes from text content.
    
    Args:
        text: Text content to analyze
    
    Returns:
        Key topics and themes found
    """
    print(f"   [Text Tool] Extracting topics from text ({len(text)} chars)")
    time.sleep(0.3)
    
    # Simulate topic extraction
    if "climate" in text.lower():
        return "Key topics: Climate change, environmental impact, sustainability, global warming, renewable energy"
    elif "technology" in text.lower() or "ai" in text.lower():
        return "Key topics: Artificial intelligence, technology trends, innovation, digital transformation, automation"
    elif "business" in text.lower():
        return "Key topics: Business strategy, market analysis, competitive landscape, growth opportunities"
    else:
        return "Key topics: General content analysis, main themes, central concepts, important subjects"

@tool
def analyze_statistics(data: str) -> str:
    """
    Analyzes numerical data and provides statistical insights.
    
    Args:
        data: Data content to analyze
    
    Returns:
        Statistical analysis and insights
    """
    print(f"   [Data Tool] Analyzing statistical data")
    time.sleep(0.4)
    
    # Simulate statistical analysis
    return """Statistical Analysis Results:
- Data points identified: 15-25 key metrics
- Trend direction: Positive growth trajectory observed
- Variance analysis: Moderate fluctuation within normal range
- Correlation strength: Strong positive correlation (r=0.75-0.85)
- Confidence interval: 95% confidence in primary findings
- Outliers detected: 2-3 anomalous data points requiring investigation"""

@tool
def sentiment_analysis(content: str) -> str:
    """
    Analyzes sentiment and emotional tone of content.
    
    Args:
        content: Content to analyze for sentiment
    
    Returns:
        Sentiment analysis results
    """
    print(f"   [Sentiment Tool] Analyzing emotional tone")
    time.sleep(0.2)
    
    # Simulate sentiment analysis
    if "positive" in content.lower() or "good" in content.lower() or "success" in content.lower():
        return "Sentiment Analysis: POSITIVE (Score: 0.75) - Optimistic tone with confident language and constructive outlook"
    elif "negative" in content.lower() or "problem" in content.lower() or "issue" in content.lower():
        return "Sentiment Analysis: NEGATIVE (Score: -0.45) - Concerned tone with critical language and cautious outlook"
    else:
        return "Sentiment Analysis: NEUTRAL (Score: 0.15) - Balanced tone with objective language and informative approach"

# Create tool executors
text_tools = [extract_key_topics]
data_tools = [analyze_statistics]
sentiment_tools = [sentiment_analysis]

text_tool_executor = ToolExecutor(text_tools)
data_tool_executor = ToolExecutor(data_tools)
sentiment_tool_executor = ToolExecutor(sentiment_tools)

print(f"✅ Defined {len(text_tools + data_tools + sentiment_tools)} specialized worker tools")

# ============================================================================
# PART 2: DEFINE SUPERVISOR-WORKER STATE
# ============================================================================
print("\n📊 PART 2: Supervisor-Worker State")
print("-" * 50)

class SupervisorWorkerState(TypedDict):
    messages: Annotated[List, add_messages]
    user_request: str
    assigned_workers: List[str]
    current_worker: Optional[str]
    text_analysis: Optional[str]
    data_analysis: Optional[str]
    sentiment_analysis: Optional[str]
    supervisor_decision: Optional[str]
    final_report: Optional[str]
    workflow_complete: bool

print("✅ Defined SupervisorWorkerState with task coordination")

# ============================================================================
# PART 3: DEFINE AGENT NODES
# ============================================================================
print("\n🤖 PART 3: Agent Node Functions")
print("-" * 50)

# Specialized LLMs
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
text_analyst_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3).bind_tools(text_tools)
data_analyst_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1).bind_tools(data_tools)
sentiment_analyst_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2).bind_tools(sentiment_tools)

def supervisor_node(state: SupervisorWorkerState):
    """Supervisor that analyzes requests and assigns tasks to workers."""
    print("   [Supervisor] Analyzing request and assigning tasks...")
    
    user_request = state["user_request"]
    text_analysis = state.get("text_analysis")
    data_analysis = state.get("data_analysis")
    sentiment_analysis = state.get("sentiment_analysis")
    
    # If no work has been done yet, assign tasks
    if not any([text_analysis, data_analysis, sentiment_analysis]):
        prompt = f"""You are a supervisor managing a content analysis team.

User Request: {user_request}

Analyze this request and determine which workers should be assigned:
1. Text Analyst - for extracting topics, themes, and content structure
2. Data Analyst - for numerical data, statistics, and quantitative analysis  
3. Sentiment Analyst - for emotional tone, sentiment, and subjective analysis

Based on the request, which workers should be assigned? Consider:
- Does the request involve text content that needs topic extraction?
- Does it involve numerical data or statistics?
- Does it require sentiment or emotional analysis?

Provide your decision and reasoning."""
        
        response = supervisor_llm.invoke([HumanMessage(content=prompt)])
        decision = response.content
        
        # Determine assigned workers based on content
        assigned_workers = []
        if "text" in decision.lower() or "topic" in decision.lower():
            assigned_workers.append("text_analyst")
        if "data" in decision.lower() or "statistic" in decision.lower():
            assigned_workers.append("data_analyst")
        if "sentiment" in decision.lower() or "emotion" in decision.lower():
            assigned_workers.append("sentiment_analyst")
        
        # Default assignment if nothing specific detected
        if not assigned_workers:
            assigned_workers = ["text_analyst", "sentiment_analyst"]
        
        print(f"   Supervisor assigned workers: {assigned_workers}")
        
        return {
            "assigned_workers": assigned_workers,
            "current_worker": assigned_workers[0] if assigned_workers else None,
            "supervisor_decision": decision,
            "messages": [AIMessage(content=f"Supervisor Decision:\n{decision}")]
        }
    
    else:
        # Create final report combining all analyses
        prompt = f"""You are a supervisor creating a final report.

Original Request: {user_request}
Text Analysis: {text_analysis or 'Not performed'}
Data Analysis: {data_analysis or 'Not performed'}  
Sentiment Analysis: {sentiment_analysis or 'Not performed'}

Create a comprehensive final report that:
1. Summarizes the key findings from each analysis
2. Identifies patterns and connections between analyses
3. Provides actionable insights and recommendations
4. Addresses the original user request directly

Make it clear, professional, and well-structured."""
        
        response = supervisor_llm.invoke([HumanMessage(content=prompt)])
        final_report = response.content
        
        print("   Supervisor created final report")
        
        return {
            "final_report": final_report,
            "workflow_complete": True,
            "messages": [AIMessage(content=f"Final Report:\n{final_report}")]
        }

def text_analyst_node(state: SupervisorWorkerState):
    """Text analyst that extracts topics and analyzes content structure."""
    print("   [Text Analyst] Analyzing text content...")
    
    user_request = state["user_request"]
    
    prompt = f"""You are a text analysis specialist.

Content to analyze: {user_request}

Use the extract_key_topics tool to identify the main topics and themes.
Focus on:
- Key subjects and themes
- Content structure and organization
- Important concepts and ideas
- Recurring patterns or motifs

Provide comprehensive text analysis insights."""
    
    response = text_analyst_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_worker": "text_analyst",
        "messages": [response]
    }

def data_analyst_node(state: SupervisorWorkerState):
    """Data analyst that processes numerical information and statistics."""
    print("   [Data Analyst] Analyzing data and statistics...")
    
    user_request = state["user_request"]
    
    prompt = f"""You are a data analysis specialist.

Content to analyze: {user_request}

Use the analyze_statistics tool to examine any numerical data or quantitative aspects.
Focus on:
- Statistical patterns and trends
- Numerical relationships and correlations
- Data quality and reliability
- Quantitative insights and metrics

Provide detailed statistical analysis."""
    
    response = data_analyst_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_worker": "data_analyst", 
        "messages": [response]
    }

def sentiment_analyst_node(state: SupervisorWorkerState):
    """Sentiment analyst that evaluates emotional tone and sentiment."""
    print("   [Sentiment Analyst] Analyzing emotional tone...")
    
    user_request = state["user_request"]
    
    prompt = f"""You are a sentiment analysis specialist.

Content to analyze: {user_request}

Use the sentiment_analysis tool to evaluate the emotional tone and sentiment.
Focus on:
- Overall emotional tone and mood
- Positive, negative, or neutral sentiment
- Subjective vs objective language
- Emotional intensity and nuance

Provide detailed sentiment analysis insights."""
    
    response = sentiment_analyst_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_worker": "sentiment_analyst",
        "messages": [response]
    }

def execute_tools_node(state: SupervisorWorkerState):
    """Executes tools called by worker agents."""
    print("   [Tool Execution] Processing worker tool calls...")
    
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {}
    
    current_worker = state.get("current_worker")
    tool_results = []
    
    # Route to appropriate tool executor
    if current_worker == "text_analyst":
        tool_executor = text_tool_executor
    elif current_worker == "data_analyst":
        tool_executor = data_tool_executor
    elif current_worker == "sentiment_analyst":
        tool_executor = sentiment_tool_executor
    else:
        return {}
    
    for tool_call in last_message.tool_calls:
        try:
            tool_invocation = ToolInvocation(tool=tool_call["name"], tool_input=tool_call["args"])
            tool_output = tool_executor.invoke(tool_invocation)
            tool_results.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
            # Store results based on worker type
            if current_worker == "text_analyst":
                return {
                    "messages": tool_results,
                    "text_analysis": str(tool_output)
                }
            elif current_worker == "data_analyst":
                return {
                    "messages": tool_results,
                    "data_analysis": str(tool_output)
                }
            elif current_worker == "sentiment_analyst":
                return {
                    "messages": tool_results,
                    "sentiment_analysis": str(tool_output)
                }
                
        except Exception as e:
            error_msg = f"Error executing {tool_call['name']}: {e}"
            tool_results.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
    
    return {"messages": tool_results}

# ============================================================================
# PART 4: ROUTING LOGIC
# ============================================================================
print("\n🔀 PART 4: Supervisor-Worker Routing")
print("-" * 50)

def route_workflow(state: SupervisorWorkerState):
    """Routes between supervisor and workers based on current state."""
    assigned_workers = state.get("assigned_workers", [])
    current_worker = state.get("current_worker")
    workflow_complete = state.get("workflow_complete", False)
    
    # Check for pending tool calls
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and last_message.tool_calls:
        print(f"   Router: Tool call pending from {current_worker}. Routing to execute_tools_node")
        return "execute_tools_node"
    
    # Check if workflow is complete
    if workflow_complete:
        print("   Router: Workflow complete. Ending.")
        return "END"
    
    # If no workers assigned yet, go to supervisor
    if not assigned_workers:
        print("   Router: No workers assigned. Routing to supervisor_node")
        return "supervisor_node"
    
    # Check if all assigned work is done
    text_done = "text_analyst" not in assigned_workers or state.get("text_analysis")
    data_done = "data_analyst" not in assigned_workers or state.get("data_analysis")
    sentiment_done = "sentiment_analyst" not in assigned_workers or state.get("sentiment_analysis")
    
    if text_done and data_done and sentiment_done:
        print("   Router: All worker tasks complete. Routing to supervisor for final report")
        return "supervisor_node"
    
    # Route to next pending worker
    if "text_analyst" in assigned_workers and not state.get("text_analysis"):
        print("   Router: Routing to text_analyst_node")
        return "text_analyst_node"
    elif "data_analyst" in assigned_workers and not state.get("data_analysis"):
        print("   Router: Routing to data_analyst_node")
        return "data_analyst_node"
    elif "sentiment_analyst" in assigned_workers and not state.get("sentiment_analysis"):
        print("   Router: Routing to sentiment_analyst_node")
        return "sentiment_analyst_node"
    else:
        print("   Router: Routing to supervisor_node")
        return "supervisor_node"

def route_from_tools(state: SupervisorWorkerState):
    """Routes back to workflow after tool execution."""
    print("   Router: Tool execution complete. Returning to main workflow")
    return route_workflow(state)

print("✅ Supervisor-worker routing functions defined")

# ============================================================================
# PART 5: BUILD THE SUPERVISOR-WORKER GRAPH
# ============================================================================
print("\n🏗️  PART 5: Building Supervisor-Worker Graph")
print("-" * 50)

workflow = StateGraph(SupervisorWorkerState)

# Add nodes
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("text_analyst_node", text_analyst_node)
workflow.add_node("data_analyst_node", data_analyst_node)
workflow.add_node("sentiment_analyst_node", sentiment_analyst_node)
workflow.add_node("execute_tools_node", execute_tools_node)

# Set entry point
workflow.set_entry_point("supervisor_node")

# Add conditional routing
workflow.add_conditional_edges(
    "supervisor_node",
    route_workflow,
    {
        "text_analyst_node": "text_analyst_node",
        "data_analyst_node": "data_analyst_node", 
        "sentiment_analyst_node": "sentiment_analyst_node",
        "END": END
    }
)

workflow.add_conditional_edges(
    "text_analyst_node",
    route_workflow,
    {
        "data_analyst_node": "data_analyst_node",
        "sentiment_analyst_node": "sentiment_analyst_node",
        "supervisor_node": "supervisor_node",
        "execute_tools_node": "execute_tools_node"
    }
)

workflow.add_conditional_edges(
    "data_analyst_node",
    route_workflow,
    {
        "text_analyst_node": "text_analyst_node",
        "sentiment_analyst_node": "sentiment_analyst_node", 
        "supervisor_node": "supervisor_node",
        "execute_tools_node": "execute_tools_node"
    }
)

workflow.add_conditional_edges(
    "sentiment_analyst_node",
    route_workflow,
    {
        "text_analyst_node": "text_analyst_node",
        "data_analyst_node": "data_analyst_node",
        "supervisor_node": "supervisor_node",
        "execute_tools_node": "execute_tools_node"
    }
)

workflow.add_conditional_edges(
    "execute_tools_node",
    route_from_tools,
    {
        "supervisor_node": "supervisor_node",
        "text_analyst_node": "text_analyst_node",
        "data_analyst_node": "data_analyst_node",
        "sentiment_analyst_node": "sentiment_analyst_node",
        "END": END
    }
)

# Compile the graph
memory = SqliteSaver.from_conn_string(":memory:")
supervisor_worker_graph = workflow.compile(checkpointer=memory)

print("✅ Supervisor-worker graph compiled successfully!")

# ============================================================================
# PART 6: INTERACTIVE SESSION
# ============================================================================
print("\n🎮 PART 6: Interactive Supervisor-Worker Session")
print("-" * 50)

def run_supervisor_worker():
    """Run the supervisor-worker multi-agent system."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting supervisor-worker session: {session_id}")
    
    # Get request from user
    user_request = input("Enter content to analyze (e.g., business proposal, research data, feedback): ").strip()
    if not user_request:
        user_request = "Our company's AI initiative has shown positive results with 25% efficiency gains, though some employees express concerns about job security."
        print(f"No input provided. Using sample: {user_request[:100]}...")
    
    # Initial input
    initial_input = {
        "user_request": user_request,
        "assigned_workers": [],
        "current_worker": None,
        "text_analysis": None,
        "data_analysis": None,
        "sentiment_analysis": None,
        "supervisor_decision": None,
        "final_report": None,
        "workflow_complete": False,
        "messages": [HumanMessage(content=f"Analyze this content: {user_request}")]
    }
    
    print(f"\n--- Starting Supervisor-Worker Analysis ---")
    print("-" * 60)
    
    try:
        # Run the supervisor-worker workflow
        for event in supervisor_worker_graph.stream(initial_input, config, stream_mode="values"):
            
            # Show supervisor decisions
            if event.get("supervisor_decision") and event.get("assigned_workers"):
                workers = ", ".join(event["assigned_workers"])
                print(f"\n📋 Supervisor assigned workers: {workers}")
            
            # Show worker progress
            if event.get("text_analysis"):
                print("✅ Text Analysis Complete")
            if event.get("data_analysis"):
                print("✅ Data Analysis Complete")
            if event.get("sentiment_analysis"):
                print("✅ Sentiment Analysis Complete")
            
            # Show final report
            if event.get("final_report"):
                print("✅ Final Report Generated")
        
        # Display final results
        final_state = supervisor_worker_graph.get_state(config)
        if final_state:
            values = final_state.values
            
            print("\n" + "="*60)
            print("📊 SUPERVISOR-WORKER ANALYSIS RESULTS")
            print("="*60)
            
            print(f"\n📝 Original Request:")
            print(values.get('user_request', 'N/A')[:200] + "...")
            
            print(f"\n👥 Workers Assigned:")
            workers = values.get('assigned_workers', [])
            for worker in workers:
                print(f"   • {worker.replace('_', ' ').title()}")
            
            if values.get('text_analysis'):
                print(f"\n📄 Text Analysis:")
                print(values['text_analysis'][:150] + "...")
            
            if values.get('data_analysis'):
                print(f"\n📊 Data Analysis:")
                print(values['data_analysis'][:150] + "...")
            
            if values.get('sentiment_analysis'):
                print(f"\n💭 Sentiment Analysis:")
                print(values['sentiment_analysis'][:150] + "...")
            
            if values.get('final_report'):
                print(f"\n📋 Final Report:")
                print(values['final_report'])
            
            print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Error during supervisor-worker session: {e}")

if __name__ == "__main__":
    run_supervisor_worker()
    print("\n" + "="*70)
    print("Supervisor-Worker Multi-Agent System Complete!")
    print("Key features demonstrated:")
    print("  ✓ Supervisor coordinates multiple specialized workers")
    print("  ✓ Dynamic task assignment based on content analysis")
    print("  ✓ Tool usage by specialized worker agents")
    print("  ✓ Conditional routing and intelligent decision making")
    print("  ✓ Comprehensive multi-perspective analysis")
    print("="*70)