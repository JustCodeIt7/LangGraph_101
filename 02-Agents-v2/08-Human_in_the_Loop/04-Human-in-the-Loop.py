#!/usr/bin/env python3
"""
LangGraph Human-in-the-Loop Agents Tutorial

This interactive script demonstrates:
1. Creating agents that require human approval for critical actions
2. Implementing interrupt mechanisms in LangGraph workflows
3. Human oversight for tool execution and decision making
4. Managing conversation flow with human intervention points
5. Building approval workflows for sensitive operations

Topics Covered:
- Graph interrupts and checkpoints
- Human approval nodes
- Conditional routing based on human input
- Persistent state management with human oversight
- Interactive agent workflows
"""

import json
import operator
from datetime import datetime
from typing import TypedDict, Annotated, List, Union, Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 70)
print("LangGraph Human-in-the-Loop Agents Tutorial")
print("=" * 70)

# ============================================================================
# PART 1: DEFINE TOOLS THAT REQUIRE HUMAN APPROVAL
# ============================================================================

print("\n🔧 PART 1: Defining Tools with Different Risk Levels")
print("-" * 50)

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Send an email to a recipient (REQUIRES HUMAN APPROVAL).
    
    Args:
        recipient: Email address of the recipient
        subject: Email subject line
        body: Email body content
    
    Returns:
        Confirmation message
    """
    return f"EMAIL SENT:\nTo: {recipient}\nSubject: {subject}\nBody: {body[:100]}..."

@tool
def delete_file(file_path: str) -> str:
    """
    Delete a file from the system (REQUIRES HUMAN APPROVAL).
    
    Args:
        file_path: Path to the file to delete
    
    Returns:
        Deletion confirmation
    """
    return f"FILE DELETED: {file_path}"

@tool
def make_purchase(item: str, amount: float) -> str:
    """
    Make a purchase (REQUIRES HUMAN APPROVAL for amounts > $50).
    
    Args:
        item: Item to purchase
        amount: Purchase amount in USD
    
    Returns:
        Purchase confirmation
    """
    return f"PURCHASE MADE: {item} for ${amount:.2f}"

@tool
def get_weather(location: str) -> str:
    """
    Get weather information (NO APPROVAL NEEDED).
    
    Args:
        location: Location to get weather for
    
    Returns:
        Weather information
    """
    import random
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    temp = random.randint(15, 30)
    condition = random.choice(conditions)
    return f"Weather in {location}: {condition.title()}, {temp}°C"

@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> str:
    """
    Calculate tip amount (NO APPROVAL NEEDED).
    
    Args:
        bill_amount: Total bill amount
        tip_percentage: Tip percentage (default 15%)
    
    Returns:
        Tip calculation
    """
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return f"Bill: ${bill_amount:.2f}, Tip ({tip_percentage}%): ${tip:.2f}, Total: ${total:.2f}"

# Categorize tools by approval requirement
APPROVAL_REQUIRED_TOOLS = {"send_email", "delete_file", "make_purchase"}
SAFE_TOOLS = {"get_weather", "calculate_tip"}

tools = [send_email, delete_file, make_purchase, get_weather, calculate_tip]
tool_executor = ToolExecutor(tools)

print(f"✅ Created {len(tools)} tools:")
print(f"   🔒 Require approval: {list(APPROVAL_REQUIRED_TOOLS)}")
print(f"   ✅ Safe to execute: {list(SAFE_TOOLS)}")

# ============================================================================
# PART 2: DEFINE AGENT STATE WITH HUMAN OVERSIGHT
# ============================================================================

print("\n📊 PART 2: Agent State with Human Oversight")
print("-" * 50)

class HumanInLoopState(TypedDict):
    messages: Annotated[List, add_messages]
    user_name: str
    pending_actions: List[dict]  # Actions waiting for approval
    approved_actions: List[dict]  # Actions approved by human
    denied_actions: List[dict]   # Actions denied by human
    session_stats: dict
    awaiting_human: bool         # Flag to indicate if waiting for human input

print("✅ Defined HumanInLoopState with:")
print("   - messages: Conversation history")
print("   - user_name: Current user's name")
print("   - pending_actions: Actions awaiting approval")
print("   - approved_actions: Approved actions log")
print("   - denied_actions: Denied actions log")
print("   - session_stats: Session statistics")
print("   - awaiting_human: Human input required flag")

# ============================================================================
# PART 3: DEFINE AGENT NODES WITH HUMAN CHECKPOINTS
# ============================================================================

print("\n🤖 PART 3: Agent Nodes with Human Checkpoints")
print("-" * 50)

def initialize_human_loop_state(state: HumanInLoopState):
    """Initialize the human-in-the-loop agent state."""
    print("   🔄 Initializing human-in-the-loop state...")
    
    if not state.get("user_name"):
        state["user_name"] = "User"
    if not state.get("pending_actions"):
        state["pending_actions"] = []
    if not state.get("approved_actions"):
        state["approved_actions"] = []
    if not state.get("denied_actions"):
        state["denied_actions"] = []
    if not state.get("session_stats"):
        state["session_stats"] = {
            "total_requests": 0,
            "approved": 0,
            "denied": 0,
            "auto_executed": 0
        }
    if state.get("awaiting_human") is None:
        state["awaiting_human"] = False
    
    return state

def call_llm_with_oversight(state: HumanInLoopState):
    """Call LLM with human oversight capabilities."""
    print("   🧠 Calling LLM with oversight capabilities...")
    
    messages = state["messages"]
    user_name = state.get("user_name", "User")
    
    # Create system message with oversight information
    system_msg = f"""You are a helpful AI assistant with human oversight capabilities.
Current user: {user_name}

IMPORTANT TOOL USAGE RULES:
- Tools requiring approval: {list(APPROVAL_REQUIRED_TOOLS)}
- Safe tools (auto-execute): {list(SAFE_TOOLS)}

For tools requiring approval, always inform the user that approval will be needed.
Available tools: {', '.join([tool.name for tool in tools])}

Session stats: {state.get('session_stats', {})}"""

    full_messages = [SystemMessage(content=system_msg)] + messages
    
    # Create LLM with tools
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3
    ).bind_tools(tools)
    
    response = llm.invoke(full_messages)
    print(f"   💭 LLM Response: {response.content}")
    
    if response.tool_calls:
        print(f"   🔧 LLM requested {len(response.tool_calls)} tool call(s)")
        for tool_call in response.tool_calls:
            approval_needed = tool_call['name'] in APPROVAL_REQUIRED_TOOLS
            print(f"      - {tool_call['name']} {'(APPROVAL REQUIRED)' if approval_needed else '(AUTO-EXECUTE)'}")
    
    return {"messages": [response]}

def analyze_tool_requests(state: HumanInLoopState):
    """Analyze tool requests and categorize them by approval needs."""
    print("   🔍 Analyzing tool requests...")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("   ℹ️  No tool calls to analyze")
        return state
    
    pending_actions = []
    stats = state.get("session_stats", {})
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        action = {
            "tool_call": tool_call,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "requires_approval": tool_name in APPROVAL_REQUIRED_TOOLS,
            "timestamp": datetime.now().isoformat()
        }
        
        # Special handling for purchase amounts
        if tool_name == "make_purchase" and tool_args.get("amount", 0) <= 50:
            action["requires_approval"] = False
            print(f"   💰 Purchase under $50 - auto-approving: {tool_args}")
        
        if action["requires_approval"]:
            pending_actions.append(action)
            print(f"   ⏳ Added to pending: {tool_name} with {tool_args}")
        else:
            print(f"   ✅ Auto-execute approved: {tool_name}")
        
        stats["total_requests"] = stats.get("total_requests", 0) + 1
    
    awaiting_human = len(pending_actions) > 0
    
    return {
        "pending_actions": state.get("pending_actions", []) + pending_actions,
        "session_stats": stats,
        "awaiting_human": awaiting_human
    }

def request_human_approval(state: HumanInLoopState):
    """Request human approval for pending actions."""
    print("   👤 Requesting human approval...")
    
    pending = state.get("pending_actions", [])
    if not pending:
        return state
    
    print(f"\n{'='*60}")
    print("🚨 HUMAN APPROVAL REQUIRED")
    print(f"{'='*60}")
    
    for i, action in enumerate(pending, 1):
        print(f"\n📋 Action {i}: {action['tool_name']}")
        print(f"   Arguments: {action['tool_args']}")
        print(f"   Timestamp: {action['timestamp']}")
    
    print(f"\n{state.get('user_name', 'User')}, please review these actions.")
    print("Commands: 'approve all', 'deny all', 'approve <number>', 'deny <number>', 'details <number>'")
    
    while True:
        decision = input("\nYour decision: ").strip().lower()
        
        if decision == "approve all":
            approved = state.get("approved_actions", []) + pending
            stats = state.get("session_stats", {})
            stats["approved"] = stats.get("approved", 0) + len(pending)
            
            return {
                "approved_actions": approved,
                "pending_actions": [],
                "session_stats": stats,
                "awaiting_human": False
            }
        
        elif decision == "deny all":
            denied = state.get("denied_actions", []) + pending
            stats = state.get("session_stats", {})
            stats["denied"] = stats.get("denied", 0) + len(pending)
            
            # Add denial message to conversation
            denial_msg = "I understand. I've cancelled all pending actions as requested."
            
            return {
                "denied_actions": denied,
                "pending_actions": [],
                "session_stats": stats,
                "awaiting_human": False,
                "messages": [AIMessage(content=denial_msg)]
            }
        
        elif decision.startswith("approve "):
            try:
                idx = int(decision.split()[1]) - 1
                if 0 <= idx < len(pending):
                    action = pending.pop(idx)
                    approved = state.get("approved_actions", []) + [action]
                    stats = state.get("session_stats", {})
                    stats["approved"] = stats.get("approved", 0) + 1
                    
                    print(f"✅ Approved: {action['tool_name']}")
                    
                    return {
                        "approved_actions": approved,
                        "pending_actions": pending,
                        "session_stats": stats,
                        "awaiting_human": len(pending) > 0
                    }
                else:
                    print("❌ Invalid action number")
            except (ValueError, IndexError):
                print("❌ Invalid format. Use 'approve <number>'")
        
        elif decision.startswith("deny "):
            try:
                idx = int(decision.split()[1]) - 1
                if 0 <= idx < len(pending):
                    action = pending.pop(idx)
                    denied = state.get("denied_actions", []) + [action]
                    stats = state.get("session_stats", {})
                    stats["denied"] = stats.get("denied", 0) + 1
                    
                    print(f"❌ Denied: {action['tool_name']}")
                    
                    return {
                        "denied_actions": denied,
                        "pending_actions": pending,
                        "session_stats": stats,
                        "awaiting_human": len(pending) > 0
                    }
                else:
                    print("❌ Invalid action number")
            except (ValueError, IndexError):
                print("❌ Invalid format. Use 'deny <number>'")
        
        elif decision.startswith("details "):
            try:
                idx = int(decision.split()[1]) - 1
                if 0 <= idx < len(pending):
                    action = pending[idx]
                    print(f"\n📋 Detailed view of action {idx + 1}:")
                    print(f"   Tool: {action['tool_name']}")
                    print(f"   Arguments: {json.dumps(action['tool_args'], indent=2)}")
                    print(f"   Requires approval: {action['requires_approval']}")
                    print(f"   Timestamp: {action['timestamp']}")
                else:
                    print("❌ Invalid action number")
            except (ValueError, IndexError):
                print("❌ Invalid format. Use 'details <number>'")
        
        else:
            print("❌ Invalid command. Use: 'approve all', 'deny all', 'approve <number>', 'deny <number>', 'details <number>'")

def execute_approved_tools(state: HumanInLoopState):
    """Execute approved tools and auto-approved safe tools."""
    print("   ⚙️  Executing approved and safe tools...")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("   ℹ️  No tools to execute")
        return state
    
    approved_actions = state.get("approved_actions", [])
    tool_messages = []
    stats = state.get("session_stats", {})
    
    # Get approved tool call IDs
    approved_tool_ids = {action["tool_call"]["id"] for action in approved_actions}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Check if this tool should be executed
        should_execute = (
            tool_id in approved_tool_ids or  # Human approved
            tool_name in SAFE_TOOLS or      # Safe tool
            (tool_name == "make_purchase" and tool_args.get("amount", 0) <= 50)  # Small purchase
        )
        
        if should_execute:
            print(f"   🔧 Executing {tool_name} with args: {tool_args}")
            
            try:
                tool_invocation = ToolInvocation(
                    tool=tool_name,
                    tool_input=tool_args
                )
                tool_result = tool_executor.invoke(tool_invocation)
                
                print(f"   ✅ Tool result: {tool_result}")
                
                tool_message = ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_id
                )
                tool_messages.append(tool_message)
                
                stats["auto_executed"] = stats.get("auto_executed", 0) + 1
                
            except Exception as e:
                print(f"   ❌ Tool execution failed: {e}")
                error_message = ToolMessage(
                    content=f"Error executing {tool_name}: {str(e)}",
                    tool_call_id=tool_id
                )
                tool_messages.append(error_message)
        else:
            print(f"   🚫 Skipping {tool_name} - not approved or denied")
    
    # Clear approved actions after execution
    return {
        "messages": tool_messages,
        "approved_actions": [],
        "session_stats": stats
    }

def generate_response_with_oversight(state: HumanInLoopState):
    """Generate final response incorporating human oversight results."""
    print("   📝 Generating response with oversight results...")
    
    messages = state["messages"]
    denied_actions = state.get("denied_actions", [])
    stats = state.get("session_stats", {})
    
    # Create LLM for response generation
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Build context about what happened
    context_parts = []
    if denied_actions:
        denied_tools = [action["tool_name"] for action in denied_actions]
        context_parts.append(f"The user denied permission for: {', '.join(denied_tools)}")
    
    context_parts.append(f"Session stats: {stats}")
    
    context = " ".join(context_parts) if context_parts else "All requested actions were processed."
    
    system_msg = f"""Based on the conversation and tool execution results, provide a helpful response.
Context: {context}
Be understanding if any actions were denied and suggest alternatives if appropriate."""
    
    full_messages = [SystemMessage(content=system_msg)] + messages
    
    response = llm.invoke(full_messages)
    print(f"   💬 Final response: {response.content}")
    
    # Clear denied actions after generating response
    return {
        "messages": [response],
        "denied_actions": []
    }

# ============================================================================
# PART 4: ROUTING LOGIC WITH HUMAN CHECKPOINTS
# ============================================================================

print("\n🔀 PART 4: Routing Logic with Human Checkpoints")
print("-" * 50)

def should_request_approval(state: HumanInLoopState) -> str:
    """Determine if human approval is needed."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("   🏁 No tool calls - routing to end")
        return "end"
    
    # Check if any tools require approval
    needs_approval = any(
        tool_call["name"] in APPROVAL_REQUIRED_TOOLS and
        not (tool_call["name"] == "make_purchase" and tool_call["args"].get("amount", 0) <= 50)
        for tool_call in last_message.tool_calls
    )
    
    if needs_approval:
        print("   👤 Tools need approval - routing to approval")
        return "approval"
    else:
        print("   ✅ All tools safe - routing to execution")
        return "execute"

def should_execute_tools(state: HumanInLoopState) -> str:
    """Determine if we should execute tools."""
    if state.get("awaiting_human", False):
        print("   ⏳ Still awaiting human input")
        return "approval"
    else:
        print("   ⚙️  Routing to tool execution")
        return "execute"

def should_generate_response(state: HumanInLoopState) -> str:
    """Route to response generation."""
    print("   📝 Routing to response generation")
    return "response"

print("✅ Routing functions defined:")
print("   - should_request_approval: Routes to approval or execution")
print("   - should_execute_tools: Manages approval waiting state")
print("   - should_generate_response: Routes to final response")

# ============================================================================
# PART 5: BUILD THE HUMAN-IN-THE-LOOP GRAPH
# ============================================================================

print("\n🏗️  PART 5: Building the Human-in-the-Loop Graph")
print("-" * 50)

# Create the state graph
workflow = StateGraph(HumanInLoopState)

# Add nodes
workflow.add_node("initialize", initialize_human_loop_state)
workflow.add_node("agent", call_llm_with_oversight)
workflow.add_node("analyze", analyze_tool_requests)
workflow.add_node("approval", request_human_approval)
workflow.add_node("execute", execute_approved_tools)
workflow.add_node("response", generate_response_with_oversight)

# Define the flow
workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "agent")
workflow.add_edge("agent", "analyze")

# Conditional routing from analyze
workflow.add_conditional_edges(
    "analyze",
    should_request_approval,
    {
        "approval": "approval",
        "execute": "execute",
        "end": END
    }
)

# After approval, check if we should execute or wait
workflow.add_conditional_edges(
    "approval",
    should_execute_tools,
    {
        "execute": "execute",
        "approval": "approval"  # Stay in approval if still waiting
    }
)

# After execution, generate response
workflow.add_conditional_edges(
    "execute",
    should_generate_response,
    {
        "response": "response"
    }
)

# End after response
workflow.add_edge("response", END)

# For demonstration, we'll compile without a checkpointer
# In production, you'd use: checkpointer=SqliteSaver.from_conn_string(":memory:")
human_loop_agent = workflow.compile()

print("✅ Human-in-the-loop graph compiled successfully!")
print("   Flow: initialize → agent → analyze → [approval?] → execute → response → END")

# ============================================================================
# PART 6: DEMONSTRATION FUNCTIONS
# ============================================================================

print("\n🎭 PART 6: Demonstration Functions")
print("-" * 50)

def demonstrate_approval_scenarios():
    """Demonstrate different approval scenarios."""
    print("\n🔍 Approval Scenarios Demonstration")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "Safe Tool Only",
            "message": "What's the weather in New York?",
            "expected": "Auto-executed (no approval needed)"
        },
        {
            "name": "Approval Required Tool",
            "message": "Send an email to john@example.com with subject 'Meeting' and body 'Let's meet tomorrow'",
            "expected": "Requires human approval"
        },
        {
            "name": "Mixed Tools",
            "message": "Check the weather in Paris and then send an email to sarah@example.com about the forecast",
            "expected": "Weather auto-executed, email requires approval"
        },
        {
            "name": "Small Purchase",
            "message": "Buy a coffee for $4.50",
            "expected": "Auto-executed (under $50 threshold)"
        },
        {
            "name": "Large Purchase",
            "message": "Buy a laptop for $1200",
            "expected": "Requires human approval (over $50)"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Message: {scenario['message']}")
        print(f"   Expected: {scenario['expected']}")

def run_interactive_example(user_input: str, user_name: str = "Demo User", auto_approve: bool = False):
    """Run an interactive example with optional auto-approval."""
    print(f"\n🤖 Human-in-the-Loop Example: '{user_input}'")
    print("=" * 70)
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_name": user_name,
        "pending_actions": [],
        "approved_actions": [],
        "denied_actions": [],
        "session_stats": {"total_requests": 0, "approved": 0, "denied": 0, "auto_executed": 0},
        "awaiting_human": False
    }
    
    print(f"Input: {user_input}")
    print(f"User: {user_name}")
    print(f"Auto-approve: {auto_approve}")
    print("\n--- Graph Execution ---")
    
    # If auto_approve is True, we'll simulate automatic approval
    if auto_approve:
        print("🤖 Running with simulated automatic approval...")
    
    try:
        final_state = None
        for event in human_loop_agent.stream(initial_state, stream_mode="values"):
            final_state = event
            
            # Simulate auto-approval if requested
            if auto_approve and final_state.get("awaiting_human", False):
                print("   🤖 Auto-approving all pending actions...")
                pending = final_state.get("pending_actions", [])
                if pending:
                    stats = final_state.get("session_stats", {})
                    stats["approved"] = stats.get("approved", 0) + len(pending)
                    
                    final_state.update({
                        "approved_actions": final_state.get("approved_actions", []) + pending,
                        "pending_actions": [],
                        "session_stats": stats,
                        "awaiting_human": False
                    })
        
        if final_state and "messages" in final_state:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, 'content'):
                print(f"\n🎯 Final Response: {last_message.content}")
            
            print(f"\n📊 Session Statistics:")
            stats = final_state.get("session_stats", {})
            print(f"   - Total requests: {stats.get('total_requests', 0)}")
            print(f"   - Approved: {stats.get('approved', 0)}")
            print(f"   - Denied: {stats.get('denied', 0)}")
            print(f"   - Auto-executed: {stats.get('auto_executed', 0)}")
    
    except Exception as e:
        print(f"❌ Error running example: {e}")

# ============================================================================
# PART 7: INTERACTIVE MAIN FUNCTION
# ============================================================================

def interactive_mode():
    """Run the human-in-the-loop agent in interactive mode."""
    print("\n🎮 Interactive Human-in-the-Loop Mode")
    print("=" * 50)
    
    user_name = input("What's your name? ").strip() or "User"
    print(f"\nHello {user_name}! I'm your AI assistant with human oversight.")
    
    print("\n🔒 Tools requiring your approval:")
    for tool_name in APPROVAL_REQUIRED_TOOLS:
        tool_obj = next(t for t in tools if t.name == tool_name)
        print(f"  • {tool_name}: {tool_obj.description.split('.')[0]}")
    
    print("\n✅ Tools that auto-execute:")
    for tool_name in SAFE_TOOLS:
        tool_obj = next(t for t in tools if t.name == tool_name)
        print(f"  • {tool_name}: {tool_obj.description.split('.')[0]}")
    
    print("\nSpecial rules:")
    print("  • Purchases under $50 are auto-approved")
    print("  • All other purchases require your approval")
    
    print("\nCommands: 'exit' to quit, 'demo' for examples, 'stats' for statistics")
    
    # Initialize persistent state
    current_state = {
        "messages": [],
        "user_name": user_name,
        "pending_actions": [],
        "approved_actions": [],
        "denied_actions": [],
        "session_stats": {"total_requests": 0, "approved": 0, "denied": 0, "auto_executed": 0},
        "awaiting_human": False
    }
    
    while True:
        user_input = input(f"\n{user_name}: ").strip()
        
        if user_input.lower() == 'exit':
            print(f"\nGoodbye {user_name}! 👋")
            print("\n📊 Final Session Statistics:")
            stats = current_state.get("session_stats", {})
            print(f"   - Total requests: {stats.get('total_requests', 0)}")
            print(f"   - Approved: {stats.get('approved', 0)}")
            print(f"   - Denied: {stats.get('denied', 0)}")
            print(f"   - Auto-executed: {stats.get('auto_executed', 0)}")
            break
        elif user_input.lower() == 'demo':
            demonstrate_approval_scenarios()
            continue
        elif user_input.lower() == 'stats':
            print(f"\n📊 Current session statistics:")
            stats = current_state.get("session_stats", {})
            print(f"   - Total requests: {stats.get('total_requests', 0)}")
            print(f"   - Approved: {stats.get('approved', 0)}")
            print(f"   - Denied: {stats.get('denied', 0)}")
            print(f"   - Auto-executed: {stats.get('auto_executed', 0)}")
            print(f"   - Messages in history: {len(current_state.get('messages', []))}")
            continue
        elif not user_input:
            print("Please enter a message or 'exit' to quit.")
            continue
        
        # Add user message to state
        current_state["messages"].append(HumanMessage(content=user_input))
        
        print("\n--- Processing your request ---")
        
        # Run the agent
        try:
            final_state = None
            for event in human_loop_agent.stream(current_state, stream_mode="values"):
                final_state = event
                current_state.update(final_state)
            
            if final_state and "messages" in final_state:
                # Show AI response
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"\nAI: {last_message.content}")
        
        except Exception as e:
            print(f"❌ Error processing request: {e}")

def demonstrate_examples():
    """Demonstrate various human-in-the-loop scenarios."""
    print("\n🎭 Human-in-the-Loop Examples")
    print("=" * 50)
    
    examples = [
        {
            "name": "Safe Tool Example",
            "input": "What's the weather like in Tokyo?",
            "auto_approve": True
        },
        {
            "name": "Email Approval Example",
            "input": "Send an email to team@company.com with subject 'Project Update' and body 'The project is on track'",
            "auto_approve": True  # Will simulate approval
        },
        {
            "name": "Mixed Tools Example",
            "input": "Check the weather in London and calculate a 20% tip on a $85 bill",
            "auto_approve": True
        }
    ]
    
    for example in examples:
        print(f"\n{'='*60}")
        print(f"Example: {example['name']}")
        print(f"{'='*60}")
        run_interactive_example(
            example["input"], 
            "Demo User", 
            example["auto_approve"]
        )
        
        # Pause between examples
        input("\nPress Enter to continue to next example...")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 LangGraph Human-in-the-Loop Agent Tutorial")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Run interactive mode")
        print("2. View demonstration scenarios")
        print("3. Run automated examples")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            demonstrate_approval_scenarios()
        elif choice == "3":
            demonstrate_examples()
        elif choice == "4":
            print("\nThank you for trying the Human-in-the-Loop tutorial! 👋")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")