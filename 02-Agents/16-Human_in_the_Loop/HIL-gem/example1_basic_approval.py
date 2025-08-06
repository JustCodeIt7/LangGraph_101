"""
Example 1: Basic Human Approval Pattern
========================================

This example demonstrates a simple yes/no approval workflow where an AI agent
requests human permission before executing sensitive actions like sending emails,
making purchases, or deleting files.

Key Features:
- Simple binary approval (approve/reject)
- State persistence during human interaction
- Clear action description for human review
- Graceful handling of rejections
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid


class BasicApprovalState(TypedDict):
    """State for basic approval workflow"""
    action_description: str
    action_type: str
    parameters: dict
    human_approval: Optional[bool]
    execution_result: Optional[str]
    user_id: str


def propose_action(state: BasicApprovalState) -> BasicApprovalState:
    """
    Agent proposes an action that requires human approval
    """
    print(f"\n🤖 Agent: I need to perform the following action:")
    print(f"   Action: {state['action_description']}")
    print(f"   Type: {state['action_type']}")
    print(f"   Parameters: {state['parameters']}")
    print(f"\n⏳ Waiting for human approval...")
    
    return state


def request_human_approval(state: BasicApprovalState) -> BasicApprovalState:
    """
    Request human input for approval - this is where we pause for human interaction
    """
    # In a real application, this would integrate with your UI/frontend
    # For demo purposes, we'll simulate user input
    print(f"\n👤 Human Decision Required:")
    print(f"   Do you approve this action? (y/n)")
    
    # Simulate human input (in production, this would come from your UI)
    user_input = input("Your decision: ").strip().lower()
    
    approval = user_input in ['y', 'yes', 'approve', '1', 'true']
    
    print(f"   Human decision: {'✅ APPROVED' if approval else '❌ REJECTED'}")
    
    return {
        **state,
        "human_approval": approval
    }


def execute_action(state: BasicApprovalState) -> BasicApprovalState:
    """
    Execute the approved action
    """
    if state["human_approval"]:
        print(f"\n🚀 Executing approved action: {state['action_description']}")
        
        # Simulate action execution based on type
        if state["action_type"] == "email":
            result = f"Email sent to {state['parameters']['recipient']}"
        elif state["action_type"] == "purchase":
            result = f"Purchase completed: {state['parameters']['item']} for ${state['parameters']['amount']}"
        elif state["action_type"] == "file_operation":
            result = f"File operation completed: {state['parameters']['operation']} on {state['parameters']['filename']}"
        else:
            result = f"Action completed: {state['action_description']}"
        
        print(f"   ✅ Result: {result}")
        
        return {
            **state,
            "execution_result": result
        }
    else:
        print(f"\n❌ Action rejected by human: {state['action_description']}")
        return {
            **state,
            "execution_result": "Action rejected by human"
        }


def should_execute(state: BasicApprovalState) -> Literal["execute", "end"]:
    """
    Conditional router based on human approval
    """
    return "execute" if state.get("human_approval") else "end"


def create_basic_approval_graph():
    """
    Create the basic approval workflow graph
    """
    # Create the graph
    workflow = StateGraph(BasicApprovalState)
    
    # Add nodes
    workflow.add_node("propose", propose_action)
    workflow.add_node("request_approval", request_human_approval)
    workflow.add_node("execute", execute_action)
    
    # Define the workflow
    workflow.add_edge(START, "propose")
    workflow.add_edge("propose", "request_approval")
    workflow.add_conditional_edges(
        "request_approval",
        should_execute,
        {
            "execute": "execute",
            "end": END
        }
    )
    workflow.add_edge("execute", END)
    
    # Compile with memory for persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


def demo_email_approval():
    """
    Demo: Email sending with human approval
    """
    print("=" * 60)
    print("DEMO 1: Email Approval Workflow")
    print("=" * 60)
    
    app = create_basic_approval_graph()
    
    initial_state = BasicApprovalState(
        action_description="Send marketing email to customer list",
        action_type="email",
        parameters={
            "recipient": "customers@company.com",
            "subject": "Monthly Newsletter",
            "template": "newsletter_march_2024"
        },
        human_approval=None,
        execution_result=None,
        user_id="user123"
    )
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Run the workflow
    final_state = app.invoke(initial_state, config)
    
    print(f"\n📊 Final State:")
    print(f"   Approval: {final_state['human_approval']}")
    print(f"   Result: {final_state['execution_result']}")


def demo_purchase_approval():
    """
    Demo: Purchase approval workflow
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Purchase Approval Workflow")
    print("=" * 60)
    
    app = create_basic_approval_graph()
    
    initial_state = BasicApprovalState(
        action_description="Purchase software license for development team",
        action_type="purchase",
        parameters={
            "item": "IntelliJ IDEA Ultimate License",
            "amount": 599.00,
            "vendor": "JetBrains",
            "quantity": 10
        },
        human_approval=None,
        execution_result=None,
        user_id="admin456"
    )
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Run the workflow
    final_state = app.invoke(initial_state, config)
    
    print(f"\n📊 Final State:")
    print(f"   Approval: {final_state['human_approval']}")
    print(f"   Result: {final_state['execution_result']}")


def demo_file_operation_approval():
    """
    Demo: File operation approval workflow
    """
    print("\n" + "=" * 60)
    print("DEMO 3: File Operation Approval Workflow")
    print("=" * 60)
    
    app = create_basic_approval_graph()
    
    initial_state = BasicApprovalState(
        action_description="Delete old backup files from server",
        action_type="file_operation",
        parameters={
            "operation": "delete",
            "filename": "/backups/old_backups_2023/*.tar.gz",
            "estimated_space_freed": "15.2 GB"
        },
        human_approval=None,
        execution_result=None,
        user_id="sysadmin789"
    )
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Run the workflow
    final_state = app.invoke(initial_state, config)
    
    print(f"\n📊 Final State:")
    print(f"   Approval: {final_state['human_approval']}")
    print(f"   Result: {final_state['execution_result']}")


if __name__ == "__main__":
    print("🔄 LangGraph Human-in-the-Loop: Basic Approval Pattern")
    print("=" * 60)
    print("This example demonstrates a simple approval workflow where")
    print("an AI agent requests human permission before executing actions.")
    print("=" * 60)
    
    # Run demos
    demo_email_approval()
    demo_purchase_approval()
    demo_file_operation_approval()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)
