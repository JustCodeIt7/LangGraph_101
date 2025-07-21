#!/usr/bin/env python3
"""
LangGraph Multi-Agent Tutorial - v2
Example 2: Supervisor with Delegated Tasks

This script demonstrates a common multi-agent pattern where a supervisor agent
delegates tasks to specialized worker agents based on the input or current state.

Scenario: Customer Support System
- User submits a query.
- Supervisor Agent: Analyzes the query and routes it to the appropriate department (worker agent).
- Technical Support Agent: Handles technical questions (e.g., using a mock tool).
- Billing Support Agent: Handles billing-related questions.
- General Inquiry Agent: Handles general questions.

Concepts Demonstrated:
- A supervisor agent for routing and delegation.
- Specialized worker agents for specific tasks.
- Conditional routing based on the supervisor's decision.
- Dynamic invocation of worker agents.
- Shared state for context and results.
"""

import uuid
import time
from typing import TypedDict, Annotated, List, Optional, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 70)
print("Multi-Agent Example 2: Supervisor with Delegated Tasks")
print("(Customer Support System)")
print("=" * 70)

# ============================================================================
# PART 1: TOOLS (for Technical Support Agent)
# ============================================================================
print("\n🔧 PART 1: Defining Tools")
print("-" * 50)

@tool
def check_system_status(service_name: str) -> str:
    """
    Checks the status of a simulated system or service.
    Args:
        service_name: The name of the service to check (e.g., 'auth_service', 'payment_gateway').
    Returns:
        A string indicating the status of the service.
    """
    print(f"   [Tool Executed] check_system_status: Checking '{service_name}'")
    time.sleep(0.5) # Simulate check
    known_services = {
        "auth_service": "Operational",
        "payment_gateway": "Experiencing intermittent issues",
        "user_dashboard": "Operational",
        "api_service": "Undergoing scheduled maintenance"
    }
    if service_name.lower() in known_services:
        return f"Status of {service_name}: {known_services[service_name.lower()]}."
    return f"Status of {service_name}: Unknown service. Please check the service name."

tech_support_tools = [check_system_status]
tech_support_tool_executor = ToolExecutor(tech_support_tools)
print(f"✅ Defined tool for Technical Support: {check_system_status.name}")

# ============================================================================
# PART 2: DEFINE AGENT STATE
# ============================================================================
print("\n📊 PART 2: Defining Shared Agent State")
print("-" * 50)

class CustomerSupportState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    department: Optional[str] # 'technical', 'billing', 'general', or 'resolved'
    resolution: Optional[str]
    error_message: Optional[str]

print("✅ Defined CustomerSupportState for tracking query, department, and resolution.")

# ============================================================================
# PART 3: DEFINE INDIVIDUAL AGENT LOGIC (NODES)
# ============================================================================
print("\n🤖 PART 3: Defining Agent Logic (Supervisor & Workers)")
print("-" * 50)

# LLM Configurations
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
technical_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3).bind_tools(tech_support_tools)
billing_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
general_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# --- Supervisor Agent ---
def supervisor_agent_node(state: CustomerSupportState):
    """Supervisor: Analyzes query and routes to the correct department."""
    print("   [Supervisor Agent Running]")
    user_query = state["user_query"]
    
    if state.get("error_message"): # If a worker failed catastrophically
        return {"department": "resolved", "resolution": f"Unable to resolve due to error: {state['error_message']}"}

    # If already resolved by a worker, just pass through to end.
    if state.get("department") == "resolved":
        print("   Supervisor: Query already resolved by a worker. Finalizing.")
        return {}

    prompt = f"""You are a customer support supervisor.
Analyze the following user query and determine the appropriate department to handle it.
User Query: "{user_query}"
Possible departments: 'technical', 'billing', 'general'.
If the query is very simple and you can answer it directly, you can set department to 'resolved' and provide the answer.
Respond with the department name only (e.g., "technical").
If you resolve it directly, respond "resolved" and I will take your message as the resolution.
"""
    # Pass previous messages for context if any (e.g. if it's a follow-up from a worker)
    history = state.get("messages", [])
    full_prompt = [SystemMessage(content=prompt)]
    if history: # Add user query as last message if not already there or if history is just supervisor prompts
        if not history or history[-1].content != user_query :
             full_prompt.append(HumanMessage(content=user_query))

    response = supervisor_llm.invoke(full_prompt)
    decision = response.content.strip().lower()

    if decision in ["technical", "billing", "general"]:
        print(f"   Supervisor decided department: {decision}")
        return {"department": decision, "messages": [AIMessage(content=f"Routing to {decision} department.")]}
    elif decision == "resolved" or "resolved" in decision : # Supervisor resolves directly
         # This part is tricky, as the supervisor's response *is* the resolution.
         # For now, let's assume if supervisor says "resolved", the content of its message is the resolution.
         # A more robust way would be for supervisor to output a structured {department: "resolved", resolution_text: "..."}
        resolution_text = response.content # Taking the whole response as resolution
        if resolution_text.lower().startswith("resolved"): # clean up if it prepends "resolved"
            resolution_text = resolution_text[len("resolved"):].strip(": ")

        print(f"   Supervisor resolved directly: {resolution_text}")
        return {"department": "resolved", "resolution": resolution_text, "messages": [AIMessage(content=resolution_text)]}
    else: # Fallback if LLM doesn't give one of the expected
        print(f"   Supervisor: Unclear decision ('{decision}'). Defaulting to 'general'.")
        return {"department": "general", "messages": [AIMessage(content="Routing to general inquiries.")]}


# --- Worker Agents ---
def technical_support_agent_node(state: CustomerSupportState):
    """Technical Support Agent: Handles technical queries, may use tools."""
    print("   [Technical Support Agent Running]")
    user_query = state["user_query"]
    prompt = f"""You are a Technical Support Specialist.
Address the following technical query: "{user_query}"
Use the 'check_system_status' tool if relevant to diagnose issues.
Provide a clear, concise answer or resolution steps.
If you use a tool, explain its output to the user.
"""
    response = technical_llm.invoke([HumanMessage(content=user_query)])
    
    # If tool is called, the graph will route to tool executor then back to this agent or a summarizer.
    # For this example, we assume one pass: if tool call, it's in response.tool_calls.
    # If no tool call, response.content is the answer.
    
    if response.tool_calls:
        print(f"   Technical Support proposed tool call: {response.tool_calls[0]['name']}")
        # The graph needs to handle tool execution. State will be updated with tool result.
        # This node's output will be the AIMessage containing the tool_call.
        # The actual resolution will come after tool execution.
        return {"messages": [response]} # Supervisor will re-evaluate after tool execution
    else:
        resolution_text = response.content
        print(f"   Technical Support resolved: {resolution_text[:100]}...")
        return {"department": "resolved", "resolution": resolution_text, "messages": [response]}

def execute_tech_tool_node(state: CustomerSupportState):
    print("   [Tool Execution] Technical Support Tool")
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {} # Should not happen if routed here

    tool_call = last_message.tool_calls[0]
    try:
        tool_output = tech_support_tool_executor.invoke(tool_call)
        tool_result_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        # After tool execution, the technical agent needs to process this.
        # We'll add the tool message and let the technical_support_agent_node run again.
        # This requires the technical_support_agent_node to handle ToolMessages in its history.
        print(f"   Tech Tool executed. Result: {str(tool_output)[:100]}...")
        return {"messages": [tool_result_msg]} # Next state will re-enter technical_support_agent_node
    except Exception as e:
        error_msg = f"Error executing tech tool: {e}"
        print(f"   {error_msg}")
        return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])], 
                "error_message": error_msg, "department": "supervisor"} # Route back to supervisor on tool error


def billing_support_agent_node(state: CustomerSupportState):
    """Billing Support Agent: Handles billing queries."""
    print("   [Billing Support Agent Running]")
    user_query = state["user_query"]
    prompt = f"""You are a Billing Support Specialist.
Address the following billing query: "{user_query}"
Provide a clear and helpful answer. For example, explain charges, process refunds (simulated), or update payment methods (simulated).
Example response for a refund query: "I've processed a refund of $XX.XX for order #YYYY. It should appear in your account within 3-5 business days."
"""
    response = billing_llm.invoke([HumanMessage(content=user_query)])
    resolution_text = response.content
    print(f"   Billing Support resolved: {resolution_text[:100]}...")
    return {"department": "resolved", "resolution": resolution_text, "messages": [response]}

def general_inquiry_agent_node(state: CustomerSupportState):
    """General Inquiry Agent: Handles general questions."""
    print("   [General Inquiry Agent Running]")
    user_query = state["user_query"]
    prompt = f"""You are a General Inquiry Support Specialist.
Address the following general query: "{user_query}"
Provide a helpful and informative answer.
"""
    response = general_llm.invoke([HumanMessage(content=user_query)])
    resolution_text = response.content
    print(f"   General Inquiry resolved: {resolution_text[:100]}...")
    return {"department": "resolved", "resolution": resolution_text, "messages": [response]}

print("✅ Defined logic for Supervisor and Worker Agents (Technical, Billing, General).")

# ============================================================================
# PART 4: BUILD THE SUPERVISOR-WORKER GRAPH
# ============================================================================
print("\n🏗️  PART 4: Building the Graph")
print("-" * 50)

memory = SqliteSaver.from_conn_string(":memory:")
workflow = StateGraph(CustomerSupportState)

# Add nodes
workflow.add_node("supervisor", supervisor_agent_node)
workflow.add_node("technical_support", technical_support_agent_node)
workflow.add_node("execute_tech_tool", execute_tech_tool_node)
workflow.add_node("billing_support", billing_support_agent_node)
workflow.add_node("general_inquiry", general_inquiry_agent_node)

# Define entry point
workflow.set_entry_point("supervisor")

# Routing from Supervisor
def route_from_supervisor(state: CustomerSupportState):
    department = state.get("department")
    print(f"   Router (from Supervisor): Next department is '{department}'")
    if department == "technical":
        return "technical_support"
    elif department == "billing":
        return "billing_support"
    elif department == "general":
        return "general_inquiry"
    elif department == "resolved": # Supervisor or worker resolved it
        return END
    return END # Default to end if department is unclear

workflow.add_conditional_edges("supervisor", route_from_supervisor)

# Routing from Technical Support (might call a tool or resolve)
def route_from_technical_support(state: CustomerSupportState):
    if state.get("error_message"): # Tool execution error
        return "supervisor" # Let supervisor handle tool errors for now
    
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and last_message.tool_calls:
        print("   Router (from Tech Support): Tool call pending. Routing to 'execute_tech_tool'.")
        return "execute_tech_tool"
    
    # If no tool call, or if a tool result was just processed by tech agent, it should set department to "resolved"
    print("   Router (from Tech Support): Resolution provided or no tool call. Routing to 'supervisor' for review/finalization.")
    return "supervisor" # Worker agents always route back to supervisor after their attempt

workflow.add_conditional_edges("technical_support", route_from_technical_support)

# Routing after tech tool execution
workflow.add_edge("execute_tech_tool", "technical_support") # Tech agent processes tool result

# Worker agents route back to supervisor (or END if they resolve directly and supervisor agrees)
workflow.add_edge("billing_support", "supervisor")
workflow.add_edge("general_inquiry", "supervisor")


# Compile the graph
supervisor_graph = workflow.compile(checkpointer=memory)
print("✅ Supervisor-Worker multi-agent graph compiled successfully!")

# ============================================================================
# PART 5: INTERACTIVE SESSION
# ============================================================================
print("\n🎮 PART 5: Interactive Customer Support Session")
print("-" * 50)

def run_customer_support_session():
    session_id = f"csmg-{str(uuid.uuid4())}"
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting new customer support session: {session_id}")
    
    while True:
        user_query = input("\nYour query (or type 'exit' to end): ").strip()
        if user_query.lower() == 'exit':
            print("Ending session.")
            break
        if not user_query:
            continue

        initial_input = {
            "user_query": user_query,
            "messages": [HumanMessage(content=user_query)] # Start with the user query
        }
        # For follow-up turns in a session, the 'messages' would accumulate via SqliteSaver.
        # For this interactive script, each input is treated as a new thread for simplicity of demonstration,
        # unless we load existing state for session_id.
        # For true multi-turn, we'd use the same config and let messages accumulate.
        # The current setup will create a new thread for each query.
        # To make it a single continuous session, we'd initialize `current_state` outside the loop
        # and pass `current_state["messages"]` to `initial_input`.

        print("\n--- Processing Your Query ---")
        
        final_event_part = None
        try:
            for event_part in supervisor_graph.stream(initial_input, config, stream_mode="values"):
                print(f"\n   --- Agent Turn ---")
                if "supervisor" in event_part: print("    Active: Supervisor")
                if "technical_support" in event_part: print("    Active: Technical Support")
                if "execute_tech_tool" in event_part: print("    Active: Executing Tech Tool")
                if "billing_support" in event_part: print("    Active: Billing Support")
                if "general_inquiry" in event_part: print("    Active: General Inquiry")

                if event_part.get("messages"):
                    last_msg = event_part["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        print(f"    AI: {last_msg.content[:150]}...")
                    elif isinstance(last_msg, ToolMessage):
                        print(f"    Tool Result: {last_msg.content[:150]}...")
                
                final_event_part = event_part
            
            print("\n--- Query Processing Complete ---")
            if final_event_part and final_event_part.get("resolution"):
                print(f"\n✅ Resolution Provided:")
                print(final_event_part["resolution"])
            elif final_event_part and final_event_part.get("error_message"):
                print(f"\n❌ Error: {final_event_part['error_message']}")
            elif final_event_part and final_event_part.get("messages"):
                 # If it ended without explicit resolution, show last AI message
                last_ai_msg = next((m for m in reversed(final_event_part["messages"]) if isinstance(m, AIMessage)), None)
                if last_ai_msg:
                    print(f"\nℹ️ Final Agent Message: {last_ai_msg.content}")
                else:
                    print("\n⚠️ No clear resolution or final message from the agent system.")
            else:
                print("\n⚠️ No final state or resolution found.")

        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("Example Queries to try:")
    print("  - 'My internet is down, can you help?' (Technical)")
    print("  - 'What is the status of the payment_gateway?' (Technical with tool)")
    print("  - 'I have a question about my last invoice.' (Billing)")
    print("  - 'What are your opening hours?' (General)")
    print("  - 'My account is locked.' (Technical, might need tool or specific steps)")
    run_customer_support_session()
    print("\n" + "="*70)
    print("Supervisor-Worker Multi-Agent Example Complete!")
    print("Key takeaways:")
    print("  - A supervisor can route tasks to specialized worker agents.")
    print("  - Conditional edges are key for implementing supervisor logic.")
    print("  - Workers can have their own tools and internal logic.")
    print("  - State needs to track who should act next or the outcome of delegations.")
    print("="*70)
