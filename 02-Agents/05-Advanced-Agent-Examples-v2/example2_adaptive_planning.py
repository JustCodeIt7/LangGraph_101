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
