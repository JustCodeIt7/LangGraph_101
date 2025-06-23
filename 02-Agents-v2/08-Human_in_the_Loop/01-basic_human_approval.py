#!/usr/bin/env python3
"""
Basic Human-in-the-Loop LangGraph Example

This is the simplest possible human-in-the-loop implementation.
It demonstrates:
1. A single tool that requires human approval
2. Basic interrupt mechanism
3. Simple approve/deny workflow

This example is perfect for understanding the core concepts.
"""

import json
from typing import TypedDict, Annotated, List, Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation

print("=" * 60)
print("Basic Human-in-the-Loop LangGraph Example")
print("=" * 60)

# ============================================================================
# STEP 1: DEFINE A DANGEROUS TOOL
# ============================================================================

@tool
def delete_important_file(filename: str) -> str:
    """
    Delete an important file (DANGEROUS - requires human approval).
    
    Args:
        filename: Name of the file to delete
    
    Returns:
        Deletion confirmation message
    """
    return f"🗑️ DELETED: {filename} - This action cannot be undone!"

@tool
def list_files() -> str:
    """
    List files in the directory (SAFE - no approval needed).
    
    Returns:
        List of files
    """
    return "📁 Files: document.txt, backup.zip, important_data.xlsx, notes.md"

# Set up tools
tools = [delete_important_file, list_files]
tool_executor = ToolExecutor(tools)

print("✅ Created 2 tools:")
print("   🔒 delete_important_file (requires approval)")
print("   ✅ list_files (safe)")

# ============================================================================
# STEP 2: DEFINE SIMPLE STATE
# ============================================================================

class BasicHumanLoopState(TypedDict):
    messages: Annotated[List, add_messages]
    needs_approval: bool
    approved: bool

print("\n📊 State contains:")
print("   - messages: conversation history")
print("   - needs_approval: flag if human input needed")
print("   - approved: human decision result")

# ============================================================================
# STEP 3: DEFINE NODES
# ============================================================================

def call_agent(state: BasicHumanLoopState):
    """Call the AI agent to generate a response."""
    print("   🤖 Agent is thinking...")
    
    messages = state["messages"]
    
    system_msg = """You are a helpful assistant with access to file operations.
    
Available tools:
- delete_important_file: Deletes files (DANGEROUS - requires approval)
- list_files: Lists files (safe)

When using delete_important_file, warn the user that approval will be needed."""

    full_messages = [SystemMessage(content=system_msg)] + messages
    
    # Create LLM with tools
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools(tools)
    
    response = llm.invoke(full_messages)
    print(f"   💭 Agent response: {response.content}")
    
    # Check if dangerous tool was called
    needs_approval = False
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'delete_important_file':
                needs_approval = True
                print("   ⚠️  Dangerous tool detected - approval needed!")
    
    return {
        "messages": [response],
        "needs_approval": needs_approval
    }

def request_approval(state: BasicHumanLoopState):
    """Request human approval for dangerous actions."""
    print("   👤 Requesting human approval...")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"approved": False}
    
    # Find the dangerous tool call
    dangerous_call = None
    for tool_call in last_message.tool_calls:
        if tool_call['name'] == 'delete_important_file':
            dangerous_call = tool_call
            break
    
    if not dangerous_call:
        return {"approved": False}
    
    print(f"\n{'='*50}")
    print("🚨 APPROVAL REQUIRED")
    print(f"{'='*50}")
    print(f"Action: Delete file '{dangerous_call['args']['filename']}'")
    print("⚠️  This action cannot be undone!")
    
    while True:
        decision = input("\nDo you approve this action? (yes/no): ").strip().lower()
        
        if decision in ['yes', 'y']:
            print("✅ Action approved!")
            return {"approved": True, "needs_approval": False}
        elif decision in ['no', 'n']:
            print("❌ Action denied!")
            return {"approved": False, "needs_approval": False}
        else:
            print("Please enter 'yes' or 'no'")

def execute_tools(state: BasicHumanLoopState):
    """Execute tools based on approval status."""
    print("   ⚙️  Executing tools...")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}
    
    tool_messages = []
    approved = state.get("approved", False)
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        
        # Execute safe tools always, dangerous tools only if approved
        if tool_name == 'list_files' or (tool_name == 'delete_important_file' and approved):
            print(f"   🔧 Executing: {tool_name}")
            
            try:
                tool_invocation = ToolInvocation(
                    tool=tool_name,
                    tool_input=tool_call['args']
                )
                result = tool_executor.invoke(tool_invocation)
                
                tool_message = ToolMessage(
                    content=result,
                    tool_call_id=tool_call['id']
                )
                tool_messages.append(tool_message)
                print(f"   ✅ Result: {result}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                error_message = ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call['id']
                )
                tool_messages.append(error_message)
        
        elif tool_name == 'delete_important_file' and not approved:
            print(f"   🚫 Skipping {tool_name} - not approved")
            denial_message = ToolMessage(
                content="Action was not approved by user.",
                tool_call_id=tool_call['id']
            )
            tool_messages.append(denial_message)
    
    return {"messages": tool_messages}

def generate_final_response(state: BasicHumanLoopState):
    """Generate final response to user."""
    print("   📝 Generating final response...")
    
    messages = state["messages"]
    
    # Create a response based on what happened
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    system_msg = """Based on the conversation and tool results, provide a helpful final response.
    If any action was denied, acknowledge this respectfully and suggest alternatives if appropriate."""
    
    full_messages = [SystemMessage(content=system_msg)] + messages
    response = llm.invoke(full_messages)
    
    print(f"   💬 Final response: {response.content}")
    return {"messages": [response]}

# ============================================================================
# STEP 4: ROUTING LOGIC
# ============================================================================

def should_request_approval(state: BasicHumanLoopState) -> str:
    """Route to approval if needed, otherwise execute directly."""
    if state.get("needs_approval", False):
        print("   🔀 Routing to approval request")
        return "approval"
    else:
        print("   🔀 Routing to tool execution")
        return "execute"

def should_execute_after_approval(state: BasicHumanLoopState) -> str:
    """Always route to execution after approval."""
    print("   🔀 Routing to tool execution")
    return "execute"

# ============================================================================
# STEP 5: BUILD THE GRAPH
# ============================================================================

print("\n🏗️  Building the graph...")

workflow = StateGraph(BasicHumanLoopState)

# Add nodes
workflow.add_node("agent", call_agent)
workflow.add_node("approval", request_approval)
workflow.add_node("execute", execute_tools)
workflow.add_node("respond", generate_final_response)

# Define flow
workflow.set_entry_point("agent")

# Route from agent
workflow.add_conditional_edges(
    "agent",
    should_request_approval,
    {
        "approval": "approval",
        "execute": "execute"
    }
)

# Route from approval to execution
workflow.add_conditional_edges(
    "approval",
    should_execute_after_approval,
    {
        "execute": "execute"
    }
)

# Always generate response after execution
workflow.add_edge("execute", "respond")
workflow.add_edge("respond", END)

# Compile the graph
basic_agent = workflow.compile()

print("✅ Graph compiled successfully!")
print("   Flow: agent → [approval?] → execute → respond → END")

# ============================================================================
# STEP 6: DEMONSTRATION
# ============================================================================

def run_example(user_input: str):
    """Run a single example."""
    print(f"\n{'='*60}")
    print(f"Example: {user_input}")
    print(f"{'='*60}")
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "needs_approval": False,
        "approved": False
    }
    
    print("Processing...")
    
    try:
        for event in basic_agent.stream(initial_state, stream_mode="values"):
            pass  # Just process through the graph
        
        print("\n✅ Example completed!")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def interactive_mode():
    """Run in interactive mode."""
    print(f"\n{'='*60}")
    print("Interactive Mode - Basic Human-in-the-Loop")
    print(f"{'='*60}")
    
    print("\nAvailable commands:")
    print("• 'list files' - see available files (safe)")
    print("• 'delete [filename]' - delete a file (requires approval)")
    print("• 'exit' - quit")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye! 👋")
            break
        
        if not user_input:
            continue
        
        print("\n--- Processing ---")
        run_example(user_input)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Basic Human-in-the-Loop Demo")
    print("=" * 40)
    
    while True:
        print("\nChoose an option:")
        print("1. Run interactive mode")
        print("2. Run demo examples")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            # Demo examples
            examples = [
                "List the files in the directory",
                "Delete the file called document.txt",
                "List files and then delete important_data.xlsx"
            ]
            
            for example in examples:
                run_example(example)
                input("\nPress Enter to continue...")
        elif choice == "3":
            print("Thank you for trying the basic example! 👋")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")