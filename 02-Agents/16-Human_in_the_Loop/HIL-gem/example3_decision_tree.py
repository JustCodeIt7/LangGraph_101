"""
Example 3: Multi-Step Decision Tree Pattern
===========================================

This example demonstrates a complex HITL workflow where humans make multiple
decisions throughout a multi-step process. Each decision point can lead to
different branches of execution, creating a dynamic decision tree.

Key Features:
- Multiple decision points in sequence
- Dynamic branching based on human choices
- Context-aware decision options
- Complex state management across decisions
- Rollback and retry capabilities
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime


class DecisionTreeState(TypedDict):
    """State for multi-step decision tree workflow"""
    scenario_type: str
    current_step: str
    decision_history: List[Dict[str, Any]]
    available_options: List[Dict[str, Any]]
    context_data: Dict[str, Any]
    human_decisions: Dict[str, Any]
    workflow_complete: bool
    current_path: List[str]
    rollback_requested: bool
    user_id: str


def initialize_scenario(state: DecisionTreeState) -> DecisionTreeState:
    """
    Initialize the decision tree scenario with context
    """
    scenario = state["scenario_type"]
    
    print(f"\n🎯 Initializing {scenario} Decision Workflow")
    print("=" * 60)
    
    if scenario == "project_planning":
        context = {
            "project_name": "AI Content Platform",
            "budget_range": "$50,000 - $200,000",
            "timeline": "6 months",
            "team_size": "5-8 people",
            "priority_factors": ["speed", "quality", "cost"]
        }
        initial_step = "budget_decision"
        
    elif scenario == "hiring_process":
        context = {
            "position": "Senior Python Developer",
            "candidates": 3,
            "urgency": "high",
            "requirements": ["5+ years Python", "ML experience", "Team leadership"],
            "budget": "$120,000 - $160,000"
        }
        initial_step = "candidate_screening"
        
    elif scenario == "product_launch":
        context = {
            "product": "Smart Home Assistant",
            "market_readiness": "75%",
            "competitor_analysis": "favorable",
            "resources_available": ["engineering", "marketing", "support"],
            "risk_factors": ["market timing", "technical complexity"]
        }
        initial_step = "launch_timing"
        
    else:
        context = {"scenario": scenario}
        initial_step = "initial_decision"
    
    print(f"📋 Scenario Context:")
    for key, value in context.items():
        print(f"   {key}: {value}")
    
    return {
        **state,
        "current_step": initial_step,
        "context_data": context,
        "current_path": [initial_step],
        "decision_history": [],
        "human_decisions": {}
    }


def present_decision_options(state: DecisionTreeState) -> DecisionTreeState:
    """
    Present available options for the current decision step
    """
    step = state["current_step"]
    scenario = state["scenario_type"]
    context = state["context_data"]
    
    print(f"\n🤔 Decision Point: {step.replace('_', ' ').title()}")
    print("-" * 40)
    
    # Define options based on current step and scenario
    if scenario == "project_planning":
        if step == "budget_decision":
            options = [
                {"id": "low_budget", "title": "Conservative Budget ($50k-$80k)", "description": "Lower risk, longer timeline, basic features"},
                {"id": "med_budget", "title": "Moderate Budget ($80k-$120k)", "description": "Balanced approach, standard timeline, core features"},
                {"id": "high_budget", "title": "Premium Budget ($120k-$200k)", "description": "Fast execution, advanced features, higher risk"}
            ]
        elif step == "technology_stack":
            options = [
                {"id": "proven_tech", "title": "Proven Technologies", "description": "Stable, well-documented, lower risk"},
                {"id": "modern_tech", "title": "Modern Stack", "description": "Latest tools, better performance, moderate risk"},
                {"id": "cutting_edge", "title": "Cutting-edge Tech", "description": "Innovative, competitive advantage, higher risk"}
            ]
        elif step == "team_structure":
            options = [
                {"id": "internal_team", "title": "Internal Team Only", "description": "Full control, higher cost, longer ramp-up"},
                {"id": "hybrid_team", "title": "Hybrid Team", "description": "Mix of internal and external, balanced approach"},
                {"id": "outsourced", "title": "Primarily Outsourced", "description": "Cost-effective, faster start, less control"}
            ]
        else:
            options = [{"id": "continue", "title": "Continue", "description": "Proceed with current settings"}]
    
    elif scenario == "hiring_process":
        if step == "candidate_screening":
            options = [
                {"id": "fast_track", "title": "Fast-track Top Candidate", "description": "Skip some steps, risk of missing issues"},
                {"id": "standard_process", "title": "Standard Interview Process", "description": "Thorough evaluation, standard timeline"},
                {"id": "extended_evaluation", "title": "Extended Evaluation", "description": "Additional technical tests, longer process"}
            ]
        elif step == "offer_strategy":
            options = [
                {"id": "competitive_offer", "title": "Competitive Market Offer", "description": "Standard market rate, moderate attraction"},
                {"id": "premium_offer", "title": "Premium Offer", "description": "Above market rate, high attraction, higher cost"},
                {"id": "package_offer", "title": "Comprehensive Package", "description": "Lower base, high benefits, equity options"}
            ]
        else:
            options = [{"id": "proceed", "title": "Proceed", "description": "Continue with hiring process"}]
    
    elif scenario == "product_launch":
        if step == "launch_timing":
            options = [
                {"id": "immediate_launch", "title": "Launch Immediately", "description": "Capture current market, accept current readiness level"},
                {"id": "delayed_launch", "title": "Delay for Polish", "description": "Improve product quality, risk market timing"},
                {"id": "phased_launch", "title": "Phased Launch", "description": "Gradual rollout, controlled risk, slower adoption"}
            ]
        elif step == "marketing_strategy":
            options = [
                {"id": "aggressive_marketing", "title": "Aggressive Marketing", "description": "High budget, wide reach, fast awareness"},
                {"id": "targeted_marketing", "title": "Targeted Marketing", "description": "Focused campaigns, efficient spend, slower growth"},
                {"id": "organic_growth", "title": "Organic Growth", "description": "Minimal marketing, word-of-mouth, very slow start"}
            ]
        else:
            options = [{"id": "finalize", "title": "Finalize Launch", "description": "Complete the launch process"}]
    
    else:
        options = [
            {"id": "option_a", "title": "Option A", "description": "First choice"},
            {"id": "option_b", "title": "Option B", "description": "Second choice"}
        ]
    
    # Display options
    print("Available options:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option['title']}")
        print(f"   {option['description']}")
    
    print(f"\n{len(options) + 1}. 🔄 Rollback to previous decision")
    print(f"{len(options) + 2}. ❌ Cancel workflow")
    
    return {
        **state,
        "available_options": options
    }


def get_human_decision(state: DecisionTreeState) -> DecisionTreeState:
    """
    Get human decision for the current step
    """
    options = state["available_options"]
    step = state["current_step"]
    
    print(f"\n👤 Your Decision Required:")
    choice = input(f"Please choose an option (1-{len(options) + 2}): ").strip()
    
    try:
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(options):
            # Valid option selected
            selected_option = options[choice_num - 1]
            
            # Record decision
            decision_record = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "option_selected": selected_option,
                "choice_number": choice_num
            }
            
            print(f"✅ Selected: {selected_option['title']}")
            
            return {
                **state,
                "human_decisions": {**state["human_decisions"], step: selected_option},
                "decision_history": state["decision_history"] + [decision_record],
                "rollback_requested": False
            }
        
        elif choice_num == len(options) + 1:
            # Rollback requested
            print("🔄 Rollback requested")
            return {
                **state,
                "rollback_requested": True
            }
        
        elif choice_num == len(options) + 2:
            # Cancel workflow
            print("❌ Workflow cancelled")
            return {
                **state,
                "workflow_complete": True,
                "rollback_requested": False
            }
        
        else:
            print(f"❌ Invalid choice: {choice_num}")
            return state
    
    except ValueError:
        print(f"❌ Invalid input: {choice}")
        return state


def determine_next_step(state: DecisionTreeState) -> DecisionTreeState:
    """
    Determine the next step based on the current decision
    """
    if state.get("rollback_requested", False):
        return handle_rollback(state)
    
    if state.get("workflow_complete", False):
        return state
    
    current_step = state["current_step"]
    scenario = state["scenario_type"]
    last_decision = state["human_decisions"].get(current_step)
    
    if not last_decision:
        return state
    
    # Determine next step based on scenario and decision
    next_step = None
    
    if scenario == "project_planning":
        if current_step == "budget_decision":
            next_step = "technology_stack"
        elif current_step == "technology_stack":
            next_step = "team_structure"
        elif current_step == "team_structure":
            next_step = "final_review"
    
    elif scenario == "hiring_process":
        if current_step == "candidate_screening":
            next_step = "offer_strategy"
        elif current_step == "offer_strategy":
            next_step = "final_approval"
    
    elif scenario == "product_launch":
        if current_step == "launch_timing":
            next_step = "marketing_strategy"
        elif current_step == "marketing_strategy":
            next_step = "launch_execution"
    
    if next_step:
        print(f"\n➡️ Moving to next step: {next_step.replace('_', ' ').title()}")
        return {
            **state,
            "current_step": next_step,
            "current_path": state["current_path"] + [next_step]
        }
    else:
        print(f"\n🏁 Workflow complete!")
        return {
            **state,
            "workflow_complete": True
        }


def handle_rollback(state: DecisionTreeState) -> DecisionTreeState:
    """
    Handle rollback to previous decision point
    """
    current_path = state["current_path"]
    
    if len(current_path) <= 1:
        print("❌ Cannot rollback further - already at the beginning")
        return {
            **state,
            "rollback_requested": False
        }
    
    # Remove last step from path and decisions
    previous_step = current_path[-2]
    new_path = current_path[:-1]
    
    # Remove the last decision
    new_decisions = {k: v for k, v in state["human_decisions"].items() 
                    if k != state["current_step"]}
    
    print(f"🔄 Rolling back to: {previous_step.replace('_', ' ').title()}")
    
    return {
        **state,
        "current_step": previous_step,
        "current_path": new_path,
        "human_decisions": new_decisions,
        "rollback_requested": False
    }


def should_continue_workflow(state: DecisionTreeState) -> Literal["present_options", "end"]:
    """
    Determine if workflow should continue
    """
    return "end" if state.get("workflow_complete", False) else "present_options"


def create_decision_tree_graph():
    """
    Create the multi-step decision tree workflow graph
    """
    workflow = StateGraph(DecisionTreeState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_scenario)
    workflow.add_node("present_options", present_decision_options)
    workflow.add_node("get_decision", get_human_decision)
    workflow.add_node("next_step", determine_next_step)
    
    # Define the workflow
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "present_options")
    workflow.add_edge("present_options", "get_decision")
    workflow.add_edge("get_decision", "next_step")
    
    # Conditional routing after determining next step
    workflow.add_conditional_edges(
        "next_step",
        should_continue_workflow,
        {
            "present_options": "present_options",
            "end": END
        }
    )
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def demo_project_planning():
    """
    Demo: Project planning decision tree
    """
    print("=" * 60)
    print("DEMO 1: Project Planning Decision Tree")
    print("=" * 60)
    
    app = create_decision_tree_graph()
    
    initial_state = DecisionTreeState(
        scenario_type="project_planning",
        current_step="",
        decision_history=[],
        available_options=[],
        context_data={},
        human_decisions={},
        workflow_complete=False,
        current_path=[],
        rollback_requested=False,
        user_id="pm123"
    )
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Run the workflow
    final_state = app.invoke(initial_state, config)
    
    print(f"\n📊 Workflow Summary:")
    print(f"   Total decisions made: {len(final_state['decision_history'])}")
    print(f"   Path taken: {' → '.join(final_state['current_path'])}")
    print(f"   Completed: {final_state['workflow_complete']}")


def demo_hiring_process():
    """
    Demo: Hiring process decision tree
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Hiring Process Decision Tree")
    print("=" * 60)
    
    app = create_decision_tree_graph()
    
    initial_state = DecisionTreeState(
        scenario_type="hiring_process",
        current_step="",
        decision_history=[],
        available_options=[],
        context_data={},
        human_decisions={},
        workflow_complete=False,
        current_path=[],
        rollback_requested=False,
        user_id="hr456"
    )
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Run the workflow
    final_state = app.invoke(initial_state, config)
    
    print(f"\n📊 Workflow Summary:")
    print(f"   Total decisions made: {len(final_state['decision_history'])}")
    print(f"   Path taken: {' → '.join(final_state['current_path'])}")
    print(f"   Completed: {final_state['workflow_complete']}")


if __name__ == "__main__":
    print("🔄 LangGraph Human-in-the-Loop: Multi-Step Decision Tree")
    print("=" * 60)
    print("This example demonstrates complex decision workflows with")
    print("multiple decision points, branching, and rollback capabilities.")
    print("=" * 60)
    
    # Run demos
    demo_project_planning()
    demo_hiring_process()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("Decision trees allow complex, context-aware workflows.")
    print("=" * 60)
