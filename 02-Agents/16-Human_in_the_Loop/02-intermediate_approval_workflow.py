#!/usr/bin/env python3
"""
Intermediate Human-in-the-Loop LangGraph Example

This example demonstrates:
1. Multiple tools with different risk levels
2. Conditional approval based on parameters
3. Batch approval for multiple actions
4. Basic audit logging
5. Different approval workflows

This builds on the basic example with more sophisticated approval logic.
"""

import json
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation

print("=" * 70)
print("Intermediate Human-in-the-Loop LangGraph Example")
print("=" * 70)

# ============================================================================
# STEP 1: DEFINE TOOLS WITH DIFFERENT RISK LEVELS
# ============================================================================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email (ALWAYS requires approval).
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content
    
    Returns:
        Confirmation message
    """
    return f"✉️ EMAIL SENT to {to}\nSubject: {subject}\nBody: {body[:50]}..."

@tool
def make_purchase(item: str, amount: float, category: str = "general") -> str:
    """
    Make a purchase (conditional approval based on amount).
    
    Args:
        item: Item to purchase
        amount: Purchase amount in USD
        category: Purchase category (office, travel, equipment, etc.)
    
    Returns:
        Purchase confirmation
    """
    return f"💳 PURCHASED: {item} for ${amount:.2f} (Category: {category})"

@tool
def modify_database(table: str, action: str, record_id: str = None) -> str:
    """
    Modify database records (ALWAYS requires approval for delete/update).
    
    Args:
        table: Database table name
        action: Action to perform (select, insert, update, delete)
        record_id: Record ID for update/delete operations
    
    Returns:
        Database operation result
    """
    if action.lower() == "select":
        return f"📊 SELECTED data from {table} table"
    else:
        return f"🗄️ {action.upper()} operation on {table} table (Record: {record_id})"

@tool
def get_system_info(info_type: str = "general") -> str:
    """
    Get system information (SAFE - no approval needed).
    
    Args:
        info_type: Type of info (general, disk, memory, cpu)
    
    Returns:
        System information
    """
    import random
    if info_type == "disk":
        return f"💾 Disk Usage: {random.randint(30, 80)}% used"
    elif info_type == "memory":
        return f"🧠 Memory Usage: {random.randint(40, 90)}% used"
    elif info_type == "cpu":
        return f"⚡ CPU Usage: {random.randint(10, 70)}%"
    else:
        return f"🖥️ System Status: Online, Uptime: {random.randint(1, 100)} hours"

@tool
def backup_files(source: str, destination: str, compress: bool = True) -> str:
    """
    Backup files (conditional approval for system directories).
    
    Args:
        source: Source directory
        destination: Destination directory
        compress: Whether to compress the backup
    
    Returns:
        Backup operation result
    """
    compression_note = " (compressed)" if compress else ""
    return f"💾 BACKUP: {source} → {destination}{compression_note}"

# Define approval rules
RISK_LEVELS = {
    "send_email": "HIGH",  # Always needs approval
    "make_purchase": "CONDITIONAL",  # Depends on amount
    "modify_database": "CONDITIONAL",  # Depends on action
    "get_system_info": "SAFE",  # Never needs approval
    "backup_files": "CONDITIONAL"  # Depends on source path
}

# Set up tools
tools = [send_email, make_purchase, modify_database, get_system_info, backup_files]
tool_executor = ToolExecutor(tools)

print("✅ Created 5 tools with risk levels:")
for tool_name, risk in RISK_LEVELS.items():
    emoji = "🔴" if risk == "HIGH" else "🟡" if risk == "CONDITIONAL" else "🟢"
    print(f"   {emoji} {tool_name}: {risk}")

# ============================================================================
# STEP 2: DEFINE INTERMEDIATE STATE
# ============================================================================

class IntermediateHumanLoopState(TypedDict):
    messages: Annotated[List, add_messages]
    pending_actions: List[Dict[str, Any]]  # Actions waiting for approval
    approved_actions: List[Dict[str, Any]]  # Approved actions
    denied_actions: List[Dict[str, Any]]    # Denied actions
    audit_log: List[Dict[str, str]]         # Audit trail
    approval_session_id: str                # Current approval session
    needs_batch_approval: bool              # Multiple approvals needed

print("\n📊 State includes:")
print("   - messages: conversation history")
print("   - pending_actions: actions awaiting approval") 
print("   - approved_actions: approved actions list")
print("   - denied_actions: denied actions list")
print("   - audit_log: complete audit trail")
print("   - approval_session_id: session tracking")
print("   - needs_batch_approval: batch processing flag")

# ============================================================================
# STEP 3: APPROVAL LOGIC FUNCTIONS
# ============================================================================

def assess_action_risk(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the risk level of a tool action."""
    risk_info = {
        "tool_name": tool_name,
        "risk_level": RISK_LEVELS.get(tool_name, "UNKNOWN"),
        "requires_approval": False,
        "reason": ""
    }
    
    if tool_name == "send_email":
        risk_info["requires_approval"] = True
        risk_info["reason"] = "Email sending always requires approval"
    
    elif tool_name == "make_purchase":
        amount = tool_args.get("amount", 0)
        category = tool_args.get("category", "general")
        
        if amount > 100:
            risk_info["requires_approval"] = True
            risk_info["reason"] = f"Purchase amount ${amount} exceeds $100 limit"
        elif category.lower() in ["equipment", "software"]:
            risk_info["requires_approval"] = True
            risk_info["reason"] = f"Category '{category}' requires approval"
        else:
            risk_info["reason"] = f"Purchase ${amount} under threshold and safe category"
    
    elif tool_name == "modify_database":
        action = tool_args.get("action", "").lower()
        
        if action in ["delete", "update", "drop"]:
            risk_info["requires_approval"] = True
            risk_info["reason"] = f"Database {action} operation is destructive"
        else:
            risk_info["reason"] = f"Database {action} operation is safe"
    
    elif tool_name == "backup_files":
        source = tool_args.get("source", "")
        
        if any(sys_path in source.lower() for sys_path in ["/system", "c:\\windows", "/etc"]):
            risk_info["requires_approval"] = True
            risk_info["reason"] = "Backing up system directories requires approval"
        else:
            risk_info["reason"] = "Backing up user directories is safe"
    
    elif tool_name == "get_system_info":
        risk_info["reason"] = "System information queries are always safe"
    
    return risk_info

# ============================================================================
# STEP 4: DEFINE WORKFLOW NODES
# ============================================================================

def initialize_session(state: IntermediateHumanLoopState):
    """Initialize the approval session."""
    print("   🔄 Initializing approval session...")
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize state if needed
    updates = {}
    if not state.get("pending_actions"):
        updates["pending_actions"] = []
    if not state.get("approved_actions"):
        updates["approved_actions"] = []
    if not state.get("denied_actions"):
        updates["denied_actions"] = []
    if not state.get("audit_log"):
        updates["audit_log"] = []
    
    updates["approval_session_id"] = session_id
    updates["needs_batch_approval"] = False
    
    # Log session start
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "session_start",
        "details": f"Started approval session {session_id}"
    }
    updates["audit_log"] = state.get("audit_log", []) + [audit_entry]
    
    print(f"   📝 Session ID: {session_id}")
    return updates

def call_agent_with_context(state: IntermediateHumanLoopState):
    """Call the AI agent with risk assessment context."""
    print("   🤖 Agent analyzing request with risk context...")
    
    messages = state["messages"]
    
    system_msg = f"""You are a helpful AI assistant with access to various tools.

TOOLS AND RISK LEVELS:
- send_email: HIGH RISK (always requires approval)
- make_purchase: CONDITIONAL (>$100 or equipment/software category needs approval)
- modify_database: CONDITIONAL (delete/update operations need approval)
- get_system_info: SAFE (never needs approval)
- backup_files: CONDITIONAL (system directories need approval)

Current session: {state.get('approval_session_id', 'unknown')}

When using tools that might need approval, inform the user about the approval process.
Available tools: {', '.join([tool.name for tool in tools])}"""

    full_messages = [SystemMessage(content=system_msg)] + messages
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3).bind_tools(tools)
    response = llm.invoke(full_messages)
    
    print(f"   💭 Agent response: {response.content}")
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"   🔧 Agent requested {len(response.tool_calls)} tool calls")
    
    return {"messages": [response]}

def analyze_and_categorize_actions(state: IntermediateHumanLoopState):
    """Analyze tool calls and categorize by approval needs."""
    print("   🔍 Analyzing and categorizing actions...")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("   ℹ️  No tool calls to analyze")
        return state
    
    pending_actions = []
    audit_entries = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Assess risk
        risk_assessment = assess_action_risk(tool_name, tool_args)
        
        action = {
            "tool_call": tool_call,
            "risk_assessment": risk_assessment,
            "timestamp": datetime.now().isoformat(),
            "session_id": state.get("approval_session_id")
        }
        
        # Log the risk assessment
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "risk_assessment",
            "details": f"{tool_name}: {risk_assessment['risk_level']} - {risk_assessment['reason']}"
        }
        audit_entries.append(audit_entry)
        
        if risk_assessment["requires_approval"]:
            pending_actions.append(action)
            print(f"   ⏳ Pending approval: {tool_name} - {risk_assessment['reason']}")
        else:
            print(f"   ✅ Auto-approved: {tool_name} - {risk_assessment['reason']}")
    
    needs_batch_approval = len(pending_actions) > 0
    
    return {
        "pending_actions": state.get("pending_actions", []) + pending_actions,
        "audit_log": state.get("audit_log", []) + audit_entries,
        "needs_batch_approval": needs_batch_approval
    }

def request_batch_approval(state: IntermediateHumanLoopState):
    """Request human approval for all pending actions."""
    print("   👤 Requesting batch approval...")
    
    pending = state.get("pending_actions", [])
    if not pending:
        return {"needs_batch_approval": False}
    
    print(f"\n{'='*70}")
    print(f"🚨 BATCH APPROVAL REQUEST - Session: {state.get('approval_session_id')}")
    print(f"{'='*70}")
    
    # Display all pending actions
    for i, action in enumerate(pending, 1):
        risk = action["risk_assessment"]
        tool_call = action["tool_call"]
        
        print(f"\n📋 Action {i}: {risk['tool_name']}")
        print(f"   Risk Level: {risk['risk_level']}")
        print(f"   Arguments: {json.dumps(tool_call['args'], indent=2)}")
        print(f"   Reason: {risk['reason']}")
        print(f"   Timestamp: {action['timestamp']}")
    
    print(f"\nAvailable commands:")
    print("• 'approve all' - approve all pending actions")
    print("• 'deny all' - deny all pending actions")
    print("• 'approve [1,2,3]' - approve specific actions by number")
    print("• 'deny [1,2,3]' - deny specific actions by number")
    print("• 'details [number]' - view detailed info for an action")
    
    approved_actions = []
    denied_actions = []
    audit_entries = []
    
    while pending:
        decision = input(f"\nDecision ({len(pending)} pending): ").strip().lower()
        
        if decision == "approve all":
            approved_actions.extend(pending)
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "batch_approve_all",
                "details": f"Approved {len(pending)} actions"
            }
            audit_entries.append(audit_entry)
            pending.clear()
            print(f"✅ Approved all {len(approved_actions)} actions!")
            
        elif decision == "deny all":
            denied_actions.extend(pending)
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "batch_deny_all",
                "details": f"Denied {len(pending)} actions"
            }
            audit_entries.append(audit_entry)
            pending.clear()
            print(f"❌ Denied all {len(denied_actions)} actions!")
            
        elif decision.startswith("approve [") and decision.endswith("]"):
            try:
                numbers_str = decision[9:-1]  # Extract numbers from "approve [1,2,3]"
                numbers = [int(x.strip()) for x in numbers_str.split(",")]
                
                approved_indices = []
                for num in numbers:
                    if 1 <= num <= len(pending):
                        approved_indices.append(num - 1)  # Convert to 0-based
                
                # Remove approved actions (in reverse order to maintain indices)
                for idx in sorted(approved_indices, reverse=True):
                    action = pending.pop(idx)
                    approved_actions.append(action)
                    print(f"✅ Approved: {action['risk_assessment']['tool_name']}")
                
                if approved_indices:
                    audit_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "selective_approve",
                        "details": f"Approved actions: {numbers}"
                    }
                    audit_entries.append(audit_entry)
                
            except (ValueError, IndexError) as e:
                print(f"❌ Invalid format. Use: approve [1,2,3]")
        
        elif decision.startswith("deny [") and decision.endswith("]"):
            try:
                numbers_str = decision[5:-1]  # Extract numbers from "deny [1,2,3]"
                numbers = [int(x.strip()) for x in numbers_str.split(",")]
                
                denied_indices = []
                for num in numbers:
                    if 1 <= num <= len(pending):
                        denied_indices.append(num - 1)  # Convert to 0-based
                
                # Remove denied actions (in reverse order to maintain indices)
                for idx in sorted(denied_indices, reverse=True):
                    action = pending.pop(idx)
                    denied_actions.append(action)
                    print(f"❌ Denied: {action['risk_assessment']['tool_name']}")
                
                if denied_indices:
                    audit_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "selective_deny",
                        "details": f"Denied actions: {numbers}"
                    }
                    audit_entries.append(audit_entry)
                
            except (ValueError, IndexError) as e:
                print(f"❌ Invalid format. Use: deny [1,2,3]")
        
        elif decision.startswith("details "):
            try:
                num = int(decision.split()[1])
                if 1 <= num <= len(pending):
                    action = pending[num - 1]
                    print(f"\n📋 Detailed view of action {num}:")
                    print(f"   Tool: {action['risk_assessment']['tool_name']}")
                    print(f"   Risk Level: {action['risk_assessment']['risk_level']}")
                    print(f"   Arguments: {json.dumps(action['tool_call']['args'], indent=4)}")
                    print(f"   Risk Reason: {action['risk_assessment']['reason']}")
                    print(f"   Session: {action['session_id']}")
                    print(f"   Timestamp: {action['timestamp']}")
                else:
                    print("❌ Invalid action number")
            except (ValueError, IndexError):
                print("❌ Invalid format. Use: details [number]")
        
        else:
            print("❌ Invalid command. See available commands above.")
    
    return {
        "pending_actions": [],
        "approved_actions": state.get("approved_actions", []) + approved_actions,
        "denied_actions": state.get("denied_actions", []) + denied_actions,
        "audit_log": state.get("audit_log", []) + audit_entries,
        "needs_batch_approval": False
    }

def execute_approved_and_safe_tools(state: IntermediateHumanLoopState):
    """Execute approved tools and safe tools."""
    print("   ⚙️  Executing approved and safe tools...")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("   ℹ️  No tools to execute")
        return {"messages": []}
    
    approved_actions = state.get("approved_actions", [])
    tool_messages = []
    audit_entries = []
    
    # Get approved tool call IDs
    approved_tool_ids = {action["tool_call"]["id"] for action in approved_actions}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_id = tool_call["id"]
        
        # Assess if this tool should be executed
        risk_assessment = assess_action_risk(tool_name, tool_call["args"])
        should_execute = (
            tool_id in approved_tool_ids or  # Human approved
            not risk_assessment["requires_approval"]  # Safe tool
        )
        
        if should_execute:
            print(f"   🔧 Executing: {tool_name}")
            
            try:
                tool_invocation = ToolInvocation(
                    tool=tool_name,
                    tool_input=tool_call["args"]
                )
                result = tool_executor.invoke(tool_invocation)
                
                tool_message = ToolMessage(
                    content=result,
                    tool_call_id=tool_id
                )
                tool_messages.append(tool_message)
                
                # Log execution
                audit_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "action": "tool_executed",
                    "details": f"Executed {tool_name}: {result[:100]}..."
                }
                audit_entries.append(audit_entry)
                
                print(f"   ✅ Result: {result}")
                
            except Exception as e:
                print(f"   ❌ Execution error: {e}")
                error_message = ToolMessage(
                    content=f"Error executing {tool_name}: {str(e)}",
                    tool_call_id=tool_id
                )
                tool_messages.append(error_message)
        else:
            print(f"   🚫 Skipping {tool_name} - not approved")
            denial_message = ToolMessage(
                content=f"Action {tool_name} was not approved by user.",
                tool_call_id=tool_id
            )
            tool_messages.append(denial_message)
    
    return {
        "messages": tool_messages,
        "approved_actions": [],  # Clear after execution
        "audit_log": state.get("audit_log", []) + audit_entries
    }

def generate_contextual_response(state: IntermediateHumanLoopState):
    """Generate final response with context about approvals."""
    print("   📝 Generating contextual response...")
    
    messages = state["messages"]
    denied_actions = state.get("denied_actions", [])
    audit_log = state.get("audit_log", [])
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Build context about what happened
    context_parts = []
    if denied_actions:
        denied_tools = [action["risk_assessment"]["tool_name"] for action in denied_actions]
        context_parts.append(f"User denied: {', '.join(denied_tools)}")
    
    context_parts.append(f"Session audit entries: {len(audit_log)}")
    context = " | ".join(context_parts) if context_parts else "All actions processed successfully."
    
    system_msg = f"""Based on the conversation and tool execution results, provide a helpful response.
    
Context: {context}

If any actions were denied, acknowledge this respectfully and suggest alternatives.
Be concise but informative about what was accomplished."""
    
    full_messages = [SystemMessage(content=system_msg)] + messages
    response = llm.invoke(full_messages)
    
    print(f"   💬 Final response generated")
    
    # Final audit log entry
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "session_complete",
        "details": f"Session {state.get('approval_session_id')} completed"
    }
    
    return {
        "messages": [response],
        "denied_actions": [],  # Clear after response
        "audit_log": state.get("audit_log", []) + [audit_entry]
    }

# ============================================================================
# STEP 5: ROUTING LOGIC
# ============================================================================

def should_request_approval(state: IntermediateHumanLoopState) -> str:
    """Route based on whether batch approval is needed."""
    if state.get("needs_batch_approval", False):
        print("   🔀 Routing to batch approval")
        return "approval"
    else:
        print("   🔀 Routing to tool execution")
        return "execute"

def continue_after_approval(state: IntermediateHumanLoopState) -> str:
    """Always route to execution after approval."""
    print("   🔀 Routing to tool execution")
    return "execute"

# ============================================================================
# STEP 6: BUILD THE GRAPH
# ============================================================================

print("\n🏗️  Building intermediate approval workflow...")

workflow = StateGraph(IntermediateHumanLoopState)

# Add nodes
workflow.add_node("initialize", initialize_session)
workflow.add_node("agent", call_agent_with_context)
workflow.add_node("analyze", analyze_and_categorize_actions)
workflow.add_node("approval", request_batch_approval)
workflow.add_node("execute", execute_approved_and_safe_tools)
workflow.add_node("respond", generate_contextual_response)

# Define flow
workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "agent")
workflow.add_edge("agent", "analyze")

# Conditional routing from analyze
workflow.add_conditional_edges(
    "analyze",
    should_request_approval,
    {
        "approval": "approval",
        "execute": "execute"
    }
)

# Route from approval to execution
workflow.add_conditional_edges(
    "approval",
    continue_after_approval,
    {
        "execute": "execute"
    }
)

# Always generate response after execution
workflow.add_edge("execute", "respond")
workflow.add_edge("respond", END)

# Compile the graph
intermediate_agent = workflow.compile()

print("✅ Intermediate workflow compiled successfully!")
print("   Flow: initialize → agent → analyze → [batch_approval?] → execute → respond → END")

# ============================================================================
# STEP 7: DEMONSTRATION FUNCTIONS
# ============================================================================

def show_audit_log(state: IntermediateHumanLoopState):
    """Display the audit log."""
    audit_log = state.get("audit_log", [])
    
    if not audit_log:
        print("📋 No audit entries found.")
        return
    
    print(f"\n📋 Audit Log ({len(audit_log)} entries):")
    print("=" * 60)
    
    for entry in audit_log:
        timestamp = entry["timestamp"]
        action = entry["action"]
        details = entry["details"]
        print(f"{timestamp} | {action.upper()} | {details}")

def run_example(user_input: str, show_audit: bool = True):
    """Run a single example with optional audit display."""
    print(f"\n{'='*70}")
    print(f"Example: {user_input}")
    print(f"{'='*70}")
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "pending_actions": [],
        "approved_actions": [],
        "denied_actions": [],
        "audit_log": [],
        "approval_session_id": "",
        "needs_batch_approval": False
    }
    
    print("Processing...")
    
    try:
        final_state = None
        for event in intermediate_agent.stream(initial_state, stream_mode="values"):
            final_state = event
        
        if show_audit and final_state:
            show_audit_log(final_state)
        
        print("\n✅ Example completed!")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def interactive_mode():
    """Run in interactive mode."""
    print(f"\n{'='*70}")
    print("Interactive Mode - Intermediate Approval Workflow")
    print(f"{'='*70}")
    
    print("\nAvailable tools and their approval requirements:")
    print("🔴 send_email - Always requires approval")
    print("🟡 make_purchase - Conditional (>$100 or equipment/software category)")
    print("🟡 modify_database - Conditional (delete/update operations)")
    print("🟢 get_system_info - Never requires approval")
    print("🟡 backup_files - Conditional (system directories only)")
    
    print("\nCommands: 'audit' to see audit log, 'exit' to quit")
    
    # Initialize persistent state
    current_state = {
        "messages": [],
        "pending_actions": [],
        "approved_actions": [],
        "denied_actions": [],
        "audit_log": [],
        "approval_session_id": "",
        "needs_batch_approval": False
    }
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye! 👋")
            show_audit_log(current_state)
            break
        elif user_input.lower() == 'audit':
            show_audit_log(current_state)
            continue
        elif not user_input:
            continue
        
        # Add user message to state
        current_state["messages"].append(HumanMessage(content=user_input))
        
        print("\n--- Processing your request ---")
        
        try:
            final_state = None
            for event in intermediate_agent.stream(current_state, stream_mode="values"):
                final_state = event
                current_state.update(final_state)
            
            if final_state and "messages" in final_state:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"\nAI: {last_message.content}")
        
        except Exception as e:
            print(f"❌ Error processing request: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Intermediate Human-in-the-Loop Demo")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Run interactive mode")
        print("2. Run demo examples")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            examples = [
                "Get system disk usage information",
                "Send an email to admin@company.com about system status",
                "Purchase office supplies for $25 and backup user documents",
                "Buy new development software for $200 and update the user database",
                "Backup the /etc directory and delete old logs from system table"
            ]
            
            for example in examples:
                run_example(example)
                input("\nPress Enter to continue...")
        elif choice == "3":
            print("Thank you for trying the intermediate example! 👋")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")