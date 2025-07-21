#!/usr/bin/env python3
"""
LangGraph Advanced Agent Example 2: Adaptive Planning Agent

This script demonstrates an adaptive planning agent that can:
1. Create dynamic plans that evolve based on execution results
2. Adapt to unexpected situations and pivot strategies
3. Learn from failures and adjust approaches
4. Handle uncertainty and incomplete information

Scenario: Project Management Agent
- Takes a complex project goal (e.g., "Launch a mobile app")
- Creates initial plan with milestones and dependencies
- Executes plan steps while monitoring for blockers and risks
- Adapts plan dynamically when issues arise
- Learns from execution patterns to improve future planning
"""

import uuid
import time
import random
from typing import TypedDict, Annotated, List, Union, Optional, Dict, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 80)
print("LangGraph Advanced Example 2: Adaptive Planning Agent")
print("=" * 80)

# ============================================================================
# PART 1: DEFINE TOOLS FOR ADAPTIVE PLANNING
# ============================================================================
print("\n🔧 PART 1: Defining Adaptive Planning Tools")
print("-" * 60)

@tool
def execute_task(task_id: str, task_description: str) -> Dict[str, Union[str, bool, float]]:
    """
    Simulates task execution with variable outcomes and potential issues.
    
    Args:
        task_id: Unique identifier for the task
        task_description: Description of the task to execute
    
    Returns:
        Dictionary with execution results including success status, duration, and issues
    """
    print(f"   [Tool] execute_task: Executing '{task_description}' (ID: {task_id})")
    time.sleep(0.5)
    
    # Simulate variable execution outcomes
    success_rate = 0.7  # 70% base success rate
    
    # Different task types have different success rates and common issues
    if "design" in task_description.lower():
        success_rate = 0.8
        potential_issues = ["stakeholder feedback needed", "design iteration required", "brand guidelines unclear"]
    elif "development" in task_description.lower() or "code" in task_description.lower():
        success_rate = 0.6
        potential_issues = ["technical debt discovered", "API changes needed", "performance optimization required", "third-party dependency issue"]
    elif "testing" in task_description.lower():
        success_rate = 0.75
        potential_issues = ["bugs found", "edge cases discovered", "compatibility issues", "performance bottlenecks"]
    elif "deployment" in task_description.lower():
        success_rate = 0.85
        potential_issues = ["infrastructure scaling needed", "configuration issues", "rollback required"]
    else:
        potential_issues = ["resource constraints", "dependency delays", "scope clarification needed"]
    
    is_successful = random.random() < success_rate
    duration = random.uniform(0.5, 2.0)  # Task duration multiplier
    
    if is_successful:
        return {
            "task_id": task_id,
            "status": "completed",
            "success": True,
            "duration_multiplier": duration,
            "issues": [],
            "output": f"Task '{task_description}' completed successfully"
        }
    else:
        # Task encountered issues
        issue = random.choice(potential_issues)
        return {
            "task_id": task_id,
            "status": "blocked",
            "success": False,
            "duration_multiplier": duration * 1.5,  # Issues take longer
            "issues": [issue],
            "output": f"Task '{task_description}' encountered issue: {issue}"
        }

@tool
def risk_assessment(current_plan: str, execution_history: str) -> Dict[str, Union[str, List[str], float]]:
    """
    Assesses risks in current plan based on execution history.
    
    Args:
        current_plan: Current plan description
        execution_history: History of task executions and outcomes
    
    Returns:
        Risk assessment with identified risks, probability, and mitigation suggestions
    """
    print(f"   [Tool] risk_assessment: Analyzing risks in current plan")
    time.sleep(0.3)
    
    # Analyze execution history for patterns
    high_risk_factors = []
    medium_risk_factors = []
    
    if "blocked" in execution_history:
        high_risk_factors.append("Pattern of blocked tasks detected")
    if "technical debt" in execution_history:
        medium_risk_factors.append("Technical debt accumulation risk")
    if "dependency" in execution_history:
        high_risk_factors.append("External dependency risks")
    if "performance" in execution_history:
        medium_risk_factors.append("Performance optimization needs")
    
    # Calculate overall risk score
    risk_score = len(high_risk_factors) * 0.3 + len(medium_risk_factors) * 0.15
    risk_score = min(risk_score, 1.0)  # Cap at 1.0
    
    mitigation_strategies = [
        "Add buffer time to critical path tasks",
        "Implement parallel execution where possible", 
        "Create contingency plans for high-risk tasks",
        "Increase communication frequency with stakeholders",
        "Consider alternative approaches for blocked tasks"
    ]
    
    return {
        "overall_risk_score": risk_score,
        "high_risk_factors": high_risk_factors,
        "medium_risk_factors": medium_risk_factors,
        "recommended_mitigations": mitigation_strategies[:3],  # Top 3 recommendations
        "assessment": f"Risk assessment complete. Overall risk: {'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW'}"
    }

@tool
def resource_optimizer(task_list: List[str], constraints: str = "standard") -> Dict[str, Union[List[str], str]]:
    """
    Optimizes task scheduling and resource allocation.
    
    Args:
        task_list: List of tasks to optimize
        constraints: Resource constraints (standard, limited, extended)
    
    Returns:
        Optimized task sequence and resource allocation recommendations
    """
    print(f"   [Tool] resource_optimizer: Optimizing {len(task_list)} tasks with {constraints} constraints")
    time.sleep(0.4)
    
    # Simulate resource optimization
    if constraints == "limited":
        optimization = "Sequential execution recommended due to resource constraints"
        parallel_groups = [task_list]  # All sequential
    elif constraints == "extended":
        optimization = "Aggressive parallel execution possible with extended resources"
        # Group tasks for parallel execution
        parallel_groups = [task_list[i:i+3] for i in range(0, len(task_list), 3)]
    else:  # standard
        optimization = "Balanced approach with moderate parallelization"
        parallel_groups = [task_list[i:i+2] for i in range(0, len(task_list), 2)]
    
    return {
        "optimized_sequence": parallel_groups,
        "optimization_strategy": optimization,
        "estimated_time_savings": "15-30% compared to purely sequential execution",
        "resource_recommendations": [
            "Monitor bottleneck tasks closely",
            "Prepare contingency resources for critical path",
            "Consider outsourcing non-core tasks if needed"
        ]
    }

tools = [execute_task, risk_assessment, resource_optimizer]
tool_executor = ToolExecutor(tools)

print(f"✅ Defined {len(tools)} adaptive planning tools")

# ============================================================================
# PART 2: DEFINE ADAPTIVE PLANNING STATE
# ============================================================================
print("\n📊 PART 2: Adaptive Planning State Definition")
print("-" * 60)

class AdaptivePlanningState(TypedDict):
    messages: Annotated[List, add_messages]
    project_goal: str
    current_plan: Optional[List[Dict[str, Union[str, int]]]]  # List of tasks with metadata
    execution_history: List[Dict[str, Union[str, bool, float]]]  # History of task executions
    adaptation_count: int                                    # Number of plan adaptations made
    max_adaptations: int                                     # Maximum allowed adaptations
    current_phase: Literal["planning", "executing", "adapting", "monitoring", "completed"]
    risk_profile: Optional[Dict[str, Union[str, float, List[str]]]]  # Current risk assessment
    learning_insights: List[str]                            # Insights learned from execution
    blockers: List[Dict[str, str]]                          # Current blockers and issues
    success_metrics: Dict[str, float]                       # Track success metrics

print("✅ Defined AdaptivePlanningState with comprehensive planning and execution tracking")

# ============================================================================
# PART 3: DEFINE ADAPTIVE PLANNING NODES
# ============================================================================
print("\n🤖 PART 3: Adaptive Planning Node Functions")
print("-" * 60)

# Specialized LLMs for different planning phases
planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
executor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1).bind_tools(tools)
adaptor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
monitor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).bind_tools(tools)

def initial_planning_node(state: AdaptivePlanningState):
    """Creates initial project plan with tasks and dependencies."""
    print("   [Node] Initial Planning: Creating comprehensive project plan...")
    
    goal = state["project_goal"]
    adaptation_count = state.get("adaptation_count", 0)
    learning_insights = state.get("learning_insights", [])
    
    if adaptation_count > 0:
        # This is a re-planning scenario
        prompt = f"""You are re-planning a project that has encountered issues.
Original Goal: {goal}
Previous Adaptations: {adaptation_count}
Learning Insights: {learning_insights}

Create an improved plan that addresses previous issues. Structure as:
PHASE 1: [Phase Name]
- Task 1: [Description] (Priority: High/Medium/Low, Dependencies: [list])
- Task 2: [Description] (Priority: High/Medium/Low, Dependencies: [list])

PHASE 2: [Phase Name]
- Task 3: [Description] (Priority: High/Medium/Low, Dependencies: [list])

Include risk mitigation strategies based on learning insights.
"""
    else:
        # Initial planning
        prompt = f"""You are a project planning specialist creating a comprehensive plan.
Project Goal: {goal}

Create a detailed project plan with phases and tasks. Consider:
1. Task dependencies and critical path
2. Risk factors and mitigation strategies
3. Resource requirements and constraints
4. Success metrics and milestones

Structure the plan as:
PHASE 1: [Phase Name]
- Task 1: [Description] (Priority: High/Medium/Low, Dependencies: [list])
- Task 2: [Description] (Priority: High/Medium/Low, Dependencies: [list])

Continue with additional phases as needed. Include 5-10 total tasks.
"""
    
    response = planner_llm.invoke([HumanMessage(content=prompt)])
    
    # Parse response into structured plan (simplified parsing for demo)
    plan_text = response.content
    tasks = []
    task_id = 1
    
    for line in plan_text.split('\n'):
        if line.strip().startswith('- Task') or (line.strip().startswith('-') and 'Priority:' in line):
            # Extract task description (simplified)
            task_desc = line.split(':')[1].split('(')[0].strip() if ':' in line else line.strip('- ')
            priority = "Medium"  # Default priority
            if "Priority: High" in line:
                priority = "High"
            elif "Priority: Low" in line:
                priority = "Low"
            
            tasks.append({
                "id": f"task_{task_id}",
                "description": task_desc,
                "priority": priority,
                "status": "pending",
                "phase": f"phase_{(task_id-1)//3 + 1}"  # Group tasks into phases
            })
            task_id += 1
    
    return {
        "current_plan": tasks,
        "current_phase": "executing",
        "messages": [AIMessage(content=f"Initial plan created with {len(tasks)} tasks:\n{response.content}")],
        "success_metrics": {"planned_tasks": len(tasks), "completed_tasks": 0, "adaptation_efficiency": 1.0}
    }

def execution_node(state: AdaptivePlanningState):
    """Executes the next pending task in the current plan."""
    print("   [Node] Execution: Executing next planned task...")
    
    current_plan = state.get("current_plan", [])
    execution_history = state.get("execution_history", [])
    
    # Find next pending task
    pending_tasks = [task for task in current_plan if task["status"] == "pending"]
    
    if not pending_tasks:
        print("   No pending tasks found. Moving to monitoring phase.")
        return {
            "current_phase": "monitoring",
            "messages": [AIMessage(content="All planned tasks have been attempted. Moving to monitoring phase.")]
        }
    
    # Execute highest priority task first
    priority_order = {"High": 3, "Medium": 2, "Low": 1}
    next_task = max(pending_tasks, key=lambda t: priority_order.get(t["priority"], 1))
    
    prompt = f"""You are executing a project task.
Task: {next_task['description']}
Priority: {next_task['priority']}
Project Context: {state['project_goal']}

Use the execute_task tool to perform this task. The tool will simulate execution with realistic outcomes including potential issues.
"""
    
    response = executor_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [response],
        "current_phase": "executing"
    }

def adaptation_node(state: AdaptivePlanningState):
    """Adapts the plan based on execution results and identified issues."""
    print("   [Node] Adaptation: Analyzing execution results and adapting plan...")
    
    current_plan = state.get("current_plan", [])
    execution_history = state.get("execution_history", [])
    blockers = state.get("blockers", [])
    adaptation_count = state.get("adaptation_count", 0)
    
    # Analyze recent execution results
    recent_failures = [h for h in execution_history[-3:] if not h.get("success", True)]
    recent_issues = []
    for failure in recent_failures:
        recent_issues.extend(failure.get("issues", []))
    
    prompt = f"""You are adapting a project plan based on execution results.
Project Goal: {state['project_goal']}
Current Plan: {len(current_plan)} tasks
Recent Issues: {recent_issues}
Current Blockers: {blockers}
Adaptation Count: {adaptation_count}

Analyze the situation and decide on adaptations:
1. Should we modify existing tasks?
2. Add new tasks to address issues?
3. Remove/postpone low-priority tasks?
4. Change task priorities or dependencies?

Provide specific adaptation recommendations and update the plan accordingly.
Use risk_assessment and resource_optimizer tools if needed for analysis.
"""
    
    response = adaptor_llm.invoke([HumanMessage(content=prompt)])
    
    # Update adaptation count and add learning insights
    new_insights = [f"Adaptation {adaptation_count + 1}: Addressed issues with {', '.join(recent_issues[:2])}"]
    
    return {
        "messages": [response],
        "adaptation_count": adaptation_count + 1,
        "learning_insights": state.get("learning_insights", []) + new_insights,
        "current_phase": "executing"  # Return to execution after adaptation
    }

def monitoring_node(state: AdaptivePlanningState):
    """Monitors overall project progress and decides next actions."""
    print("   [Node] Monitoring: Assessing project progress and health...")
    
    current_plan = state.get("current_plan", [])
    execution_history = state.get("execution_history", [])
    success_metrics = state.get("success_metrics", {})
    
    completed_tasks = len([task for task in current_plan if task["status"] == "completed"])
    total_tasks = len(current_plan)
    success_rate = len([h for h in execution_history if h.get("success", False)]) / max(len(execution_history), 1)
    
    prompt = f"""You are monitoring project progress and health.
Project Goal: {state['project_goal']}
Progress: {completed_tasks}/{total_tasks} tasks completed
Success Rate: {success_rate:.2%}
Execution History: {len(execution_history)} task attempts

Use risk_assessment tool to analyze current situation.
Determine if we should:
1. Continue with current plan
2. Adapt the plan due to issues
3. Complete the project (if goals are met)

Provide analysis and recommendation for next phase.
"""
    
    response = monitor_llm.invoke([HumanMessage(content=prompt)])
    
    # Update success metrics
    updated_metrics = success_metrics.copy()
    updated_metrics.update({
        "completed_tasks": completed_tasks,
        "success_rate": success_rate,
        "progress_percentage": (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    })
    
    return {
        "messages": [response],
        "success_metrics": updated_metrics,
        "current_phase": "monitoring"
    }

def execute_tool_node(state: AdaptivePlanningState):
    """Executes tools called by other nodes and updates state accordingly."""
    print("   [Tool Execution] Processing adaptive planning tool calls...")
    
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {}
    
    tool_results = []
    current_plan = state.get("current_plan", []).copy()
    execution_history = state.get("execution_history", []).copy()
    blockers = state.get("blockers", []).copy()
    
    for tool_call in last_message.tool_calls:
        print(f"   Executing {tool_call['name']} with args {tool_call['args']}")
        
        try:
            tool_invocation = ToolInvocation(tool=tool_call["name"], tool_input=tool_call["args"])
            tool_output = tool_executor.invoke(tool_invocation)
            tool_results.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
            # Process tool-specific results
            if tool_call["name"] == "execute_task":
                # Update task status in plan and add to execution history
                task_id = tool_call["args"].get("task_id")
                if task_id:
                    for task in current_plan:
                        if task["id"] == task_id:
                            if tool_output.get("success"):
                                task["status"] = "completed"
                            else:
                                task["status"] = "blocked"
                                # Add to blockers list
                                blockers.append({
                                    "task_id": task_id,
                                    "issue": tool_output.get("issues", ["Unknown issue"])[0],
                                    "description": task["description"]
                                })
                
                execution_history.append(tool_output)
                
            elif tool_call["name"] == "risk_assessment":
                # Store risk profile
                return {
                    "messages": tool_results,
                    "risk_profile": tool_output,
                    "current_plan": current_plan,
                    "execution_history": execution_history,
                    "blockers": blockers
                }
                
        except Exception as e:
            error_msg = f"Error executing {tool_call['name']}: {e}"
            tool_results.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
    
    return {
        "messages": tool_results,
        "current_plan": current_plan,
        "execution_history": execution_history,
        "blockers": blockers
    }

# ============================================================================
# PART 4: ADAPTIVE ROUTING LOGIC
# ============================================================================
print("\n🔀 PART 4: Adaptive Routing Logic")
print("-" * 60)

def route_adaptive_flow(state: AdaptivePlanningState):
    """Routes between planning phases based on current state and conditions."""
    current_phase = state.get("current_phase", "planning")
    adaptation_count = state.get("adaptation_count", 0)
    max_adaptations = state.get("max_adaptations", 3)
    blockers = state.get("blockers", [])
    
    # Check for pending tool calls
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and last_message.tool_calls:
        print("   Router: Tool call pending. Routing to execute_tool_node")
        return "execute_tool_node"
    
    if current_phase == "planning":
        print("   Router: Planning complete. Starting execution")
        return "execution_node"
        
    elif current_phase == "executing":
        # Check if we have blockers that need adaptation
        if blockers and adaptation_count < max_adaptations:
            print(f"   Router: {len(blockers)} blockers detected. Routing to adaptation")
            return "adaptation_node"
        else:
            # Continue execution or move to monitoring
            current_plan = state.get("current_plan", [])
            pending_tasks = [task for task in current_plan if task["status"] == "pending"]
            
            if pending_tasks:
                print("   Router: More tasks to execute")
                return "execution_node"
            else:
                print("   Router: All tasks attempted. Moving to monitoring")
                return "monitoring_node"
                
    elif current_phase == "adapting":
        print("   Router: Adaptation complete. Returning to execution")
        return "execution_node"
        
    elif current_phase == "monitoring":
        # Monitoring decides whether to continue, adapt, or complete
        risk_profile = state.get("risk_profile", {})
        success_metrics = state.get("success_metrics", {})
        
        # Check completion criteria
        progress = success_metrics.get("progress_percentage", 0)
        success_rate = success_metrics.get("success_rate", 0)
        
        if progress >= 80 and success_rate >= 0.6:
            print("   Router: Project goals achieved. Completing")
            return "END"
        elif blockers and adaptation_count < max_adaptations:
            print("   Router: Issues detected during monitoring. Routing to adaptation")
            return "adaptation_node"
        elif adaptation_count >= max_adaptations:
            print("   Router: Max adaptations reached. Completing project")
            return "END"
        else:
            print("   Router: Continuing execution from monitoring")
            return "execution_node"
    
    else:  # completed or unknown
        print("   Router: Project completed")
        return "END"

def route_from_tools(state: AdaptivePlanningState):
    """Routes back to appropriate node after tool execution."""
    current_phase = state.get("current_phase", "planning")
    print(f"   Router: Tool execution complete. Returning to {current_phase} phase")
    
    if current_phase == "executing":
        return "execution_node"
    elif current_phase == "adapting":
        return "adaptation_node"
    elif current_phase == "monitoring":
        return "monitoring_node"
    else:
        return "initial_planning_node"

print("✅ Adaptive routing functions defined")

# ============================================================================
# PART 5: BUILD THE ADAPTIVE PLANNING GRAPH
# ============================================================================
print("\n🏗️  PART 5: Building Adaptive Planning Graph")
print("-" * 60)

memory = SqliteSaver.from_conn_string(":memory:")
workflow = StateGraph(AdaptivePlanningState)

# Add planning nodes
workflow.add_node("initial_planning_node", initial_planning_node)
workflow.add_node("execution_node", execution_node)
workflow.add_node("adaptation_node", adaptation_node)
workflow.add_node("monitoring_node", monitoring_node)
workflow.add_node("execute_tool_node", execute_tool_node)

# Set entry point
workflow.set_entry_point("initial_planning_node")

# Add conditional edges for adaptive flow
workflow.add_conditional_edges(
    "initial_planning_node",
    route_adaptive_flow,
    {
        "execution_node": "execution_node",
        "execute_tool_node": "execute_tool_node"
    }
)

workflow.add_conditional_edges(
    "execution_node",
    route_adaptive_flow,
    {
        "execution_node": "execution_node",
        "adaptation_node": "adaptation_node", 
        "monitoring_node": "monitoring_node",
        "execute_tool_node": "execute_tool_node",
        "END": END
    }
)

workflow.add_conditional_edges(
    "adaptation_node",
    route_adaptive_flow,
    {
        "execution_node": "execution_node",
        "execute_tool_node": "execute_tool_node"
    }
)

workflow.add_conditional_edges(
    "monitoring_node", 
    route_adaptive_flow,
    {
        "execution_node": "execution_node",
        "adaptation_node": "adaptation_node",
        "execute_tool_node": "execute_tool_node",
        "END": END
    }
)

workflow.add_conditional_edges(
    "execute_tool_node",
    route_from_tools,
    {
        "initial_planning_node": "initial_planning_node",
        "execution_node": "execution_node",
        "adaptation_node": "adaptation_node",
        "monitoring_node": "monitoring_node"
    }
)

# Compile the graph
adaptive_graph = workflow.compile(checkpointer=memory)
print("✅ Adaptive planning graph compiled successfully!")

# ============================================================================
# PART 6: INTERACTIVE SESSION FOR ADAPTIVE PLANNING
# ============================================================================
print("\n🎮 PART 6: Interactive Adaptive Planning Session")
print("-" * 60)

def run_adaptive_planning_session():
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting adaptive planning session: {session_id}")
    project_goal = input("Enter project goal (e.g., 'Launch a mobile app', 'Organize a conference'): ").strip()
    
    if not project_goal:
        print("No goal provided. Using default: 'Launch a mobile app'")
        project_goal = "Launch a mobile app"

    initial_input = {
        "project_goal": project_goal,
        "messages": [HumanMessage(content=f"Project goal: {project_goal}")],
        "current_phase": "planning",
        "adaptation_count": 0,
        "max_adaptations": 3,
        "execution_history": [],
        "learning_insights": [],
        "blockers": [],
        "success_metrics": {"planned_tasks": 0, "completed_tasks": 0}
    }

    print(f"\n--- Starting Adaptive Planning Agent ---")
    print(f"Goal: {project_goal}")
    print("-" * 60)
    
    step_count = 0
    max_steps = 25  # Prevent infinite loops
    
    try:
        for event in adaptive_graph.stream(initial_input, config, stream_mode="values"):
            step_count += 1
            if step_count > max_steps:
                print("\n⚠️  Maximum steps reached. Ending session.")
                break
                
            current_phase = event.get("current_phase", "unknown")
            adaptation_count = event.get("adaptation_count", 0)
            
            print(f"\n[Step {step_count}] Phase: {current_phase.title()} | Adaptations: {adaptation_count}")
            
            # Show plan status
            if event.get("current_plan"):
                plan = event["current_plan"]
                completed = len([t for t in plan if t["status"] == "completed"])
                blocked = len([t for t in plan if t["status"] == "blocked"])
                pending = len([t for t in plan if t["status"] == "pending"])
                print(f"📋 Plan Status: {completed} completed, {blocked} blocked, {pending} pending")
            
            # Show blockers
            if event.get("blockers"):
                blockers = event["blockers"]
                if blockers:
                    print(f"🚫 Current Blockers: {len(blockers)}")
                    for blocker in blockers[-2:]:  # Show last 2 blockers
                        print(f"   - {blocker.get('issue', 'Unknown issue')}")
            
            # Show success metrics
            if event.get("success_metrics"):
                metrics = event["success_metrics"]
                if "progress_percentage" in metrics:
                    print(f"📊 Progress: {metrics['progress_percentage']:.1f}% | Success Rate: {metrics.get('success_rate', 0):.1%}")
            
            # Show learning insights
            if event.get("learning_insights"):
                insights = event["learning_insights"]
                if insights and len(insights) > len(initial_input.get("learning_insights", [])):
                    print(f"💡 New Insight: {insights[-1]}")
            
            # Show agent messages
            if event.get("messages"):
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content and not last_msg.tool_calls:
                    print(f"🤖 Agent: {last_msg.content[:150]}...")
                elif isinstance(last_msg, ToolMessage):
                    tool_result = eval(last_msg.content) if last_msg.content.startswith('{') else last_msg.content
                    if isinstance(tool_result, dict):
                        if "status" in tool_result:
                            status = tool_result["status"]
                            success = "✅" if tool_result.get("success") else "❌"
                            print(f"🔧 Task Result: {success} {status}")
                            if tool_result.get("issues"):
                                print(f"   Issue: {tool_result['issues'][0]}")
                        else:
                            print(f"🔧 Tool: {str(tool_result)[:100]}...")
            
            time.sleep(0.3)  # Pause for readability

        print("\n" + "="*30 + " ADAPTIVE PLANNING COMPLETE " + "="*30)
        
        # Get final state
        final_state = adaptive_graph.get_state(config)
        if final_state:
            values = final_state.values
            final_metrics = values.get("success_metrics", {})
            adaptations = values.get("adaptation_count", 0)
            insights = values.get("learning_insights", [])
            
            print(f"\n📈 FINAL PROJECT SUMMARY:")
            print(f"Goal: {project_goal}")
            print(f"Progress: {final_metrics.get('progress_percentage', 0):.1f}%")
            print(f"Success Rate: {final_metrics.get('success_rate', 0):.1%}")
            print(f"Adaptations Made: {adaptations}")
            print(f"Key Insights: {len(insights)} learning insights captured")
            
            if insights:
                print(f"\n🎓 Learning Insights:")
                for insight in insights[-3:]:  # Show last 3 insights
                    print(f"   • {insight}")
                    
    except Exception as e:
        print(f"\n❌ Error during adaptive planning session: {e}")
        import traceback
        traceback.print_exc()
        
        # Get final state
        final_state = adaptive_graph.get_state(config)
        if final_state:
            values = final_state.values
            final_metrics = values.get("success_metrics", {})
            adaptations = values.get("adaptation_count", 0)
            insights = values.get("learning_insights", [])
            
            print(f"\n📈 FINAL PROJECT SUMMARY:")
            print(f"Goal: {project_goal}")
            print(f"Progress: {final_metrics.get('progress_percentage', 0):.1f}%")
            print(f"Success Rate: {final_metrics.get('success_rate', 0):.1%}")
            print(f"Adaptations Made: {adaptations}")
            print(f"Key Insights: {len(insights)} learning insights captured")
            
            if insights:
                print(f"\n🎓 Learning Insights:")
                for insight in insights[-3:]:  # Show last 3 insights
                    print(f"   • {insight}")
                    
    except Exception as e:
        print(f"\n❌ Error during adaptive planning session: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_adaptive_planning_session()
    print("\n" + "="*80)
    print("Adaptive Planning Agent Complete!")
    print("Key features demonstrated:")
    print("  ✓ Dynamic plan creation and modification")
    print("  ✓ Real-time adaptation to blockers and issues")
    print("  ✓ Learning from execution patterns")
    print("  ✓ Risk assessment and mitigation")
    print("  ✓ Resource optimization and scheduling")
    print("  ✓ Progress monitoring and success metrics")
    print("="*80)