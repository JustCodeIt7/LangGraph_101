#!/usr/bin/env python3
"""
LangGraph Human-in-the-Loop Agents Tutorial - v2

This interactive script demonstrates a streamlined approach to:
1. Creating agents that pause for human input at critical junctures.
2. Using LangGraph's checkpointing feature for persistent state.
3. Resuming graph execution after human feedback.
4. A clear, focused example of human oversight in an agent workflow.

Topics Covered:
- Graph interrupts for human feedback (`interrupt_before_nodes`)
- SqliteSaver for persistent checkpoints
- Modifying state based on human input
- Resuming graph execution from a checkpoint
- A simplified, interactive human-in-the-loop workflow
"""

import uuid
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver

print("=" * 70)
print("LangGraph Human-in-the-Loop Agents Tutorial - v2")
print("=" * 70)

# ============================================================================
# PART 1: DEFINE A TOOL THAT MIGHT REQUIRE HUMAN CONFIRMATION
# ============================================================================
print("\n🔧 PART 1: Defining a Tool for the Agent")
print("-" * 50)

@tool
def schedule_meeting(topic: str, time: str, attendees: List[str]) -> str:
    """
    Schedules a meeting. This action might require human confirmation
    before actually being finalized by the system.

    Args:
        topic: The subject or topic of the meeting.
        time: The proposed time for the meeting (e.g., "Tomorrow at 2 PM").
        attendees: A list of email addresses or names of attendees.

    Returns:
        A confirmation string that the meeting scheduling process has been initiated.
    """
    attendees_str = ", ".join(attendees)
    confirmation_message = (
        f"Meeting scheduling initiated for '{topic}' at {time} "
        f"with {attendees_str}. Waiting for final confirmation."
    )
    # In a real system, this might just create a draft or a pending calendar event.
    print(f"   [Tool Executed] schedule_meeting: {confirmation_message}")
    return confirmation_message

tools = [schedule_meeting]
tool_executor = ToolExecutor(tools)

print(f"✅ Defined tool: {schedule_meeting.name} - {schedule_meeting.description}")

# ============================================================================
# PART 2: DEFINE AGENT STATE
# ============================================================================
print("\n📊 PART 2: Agent State Definition")
print("-" * 50)

class MeetingSchedulerAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    # 'human_approved' will store the human's decision (True/False) or None if not yet reviewed.
    human_approved: Union[bool, None]
    # Store the proposed meeting details if a tool call is made
    proposed_meeting_details: Union[dict, None]

print("✅ Defined MeetingSchedulerAgentState with:")
print("   - messages: Conversation history")
print("   - human_approved: Stores human's decision (True/False/None)")
print("   - proposed_meeting_details: Stores details of the meeting if proposed by LLM")

# ============================================================================
# PART 3: DEFINE AGENT NODES
# ============================================================================
print("\n🤖 PART 3: Agent Node Functions")
print("-" * 50)

def llm_propose_meeting_node(state: MeetingSchedulerAgentState):
    """
    The LLM node. It proposes actions (like scheduling a meeting)
    or responds to the user.
    """
    print("   [Node Running] llm_propose_meeting_node: Calling LLM...")
    
    system_prompt = """You are a helpful assistant that can schedule meetings.
If the user asks to schedule a meeting, use the 'schedule_meeting' tool.
Clearly state the meeting details (topic, time, attendees) in your response
before calling the tool.
If a meeting has been proposed and is awaiting confirmation, inform the user.
If a meeting was confirmed or denied by the human, acknowledge that.
"""
    
    current_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools(tools)
    response_message = llm.invoke(current_messages)
    
    proposed_details = None
    if response_message.tool_calls:
        # For simplicity, assuming one tool call for meeting scheduling
        tool_call = response_message.tool_calls[0]
        if tool_call['name'] == schedule_meeting.name:
            proposed_details = tool_call['args']
            print(f"   LLM proposed tool call: {tool_call['name']} with args {tool_call['args']}")
    
    return {
        "messages": [response_message],
        "proposed_meeting_details": proposed_details,
        "human_approved": None # Reset approval status for new proposals
    }

def execute_tool_node(state: MeetingSchedulerAgentState):
    """
    Executes the tool call if one was made by the LLM and approved by human.
    This node will only run if human_approved is True.
    """
    print("   [Node Running] execute_tool_node: Attempting tool execution...")
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        print("   No tool calls from LLM. Skipping tool execution.")
        return {}

    # We assume human_approved was set to True for this node to be reached
    # (as per graph logic defined later)
    
    tool_call = last_message.tool_calls[0] # Assuming one tool call
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"   Executing approved tool: {tool_name} with args: {tool_args}")
    
    try:
        tool_invocation = ToolInvocation(tool=tool_name, tool_input=tool_args)
        tool_output = tool_executor.invoke(tool_invocation)
        tool_result_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        print(f"   Tool execution successful. Output: {tool_output}")
        return {"messages": [tool_result_message]}
    except Exception as e:
        error_msg = f"Error executing tool {tool_name}: {e}"
        print(f"   {error_msg}")
        # Return an error message as a ToolMessage
        return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call["id"])]}

def final_response_node(state: MeetingSchedulerAgentState):
    """
    Generates a final response to the user after tool execution or if no tool was called.
    """
    print("   [Node Running] final_response_node: Generating final response...")
    
    # If the last message was a ToolMessage, the LLM needs to formulate a response based on it.
    # If the LLM didn't call a tool, its last message is already the response.
    
    if isinstance(state["messages"][-1], ToolMessage) or state["human_approved"] is not None :
        # If a tool was executed OR human interaction occurred, let LLM summarize.
        system_prompt = "Summarize the outcome of the recent actions or provide a response based on the tool's output. If a meeting was just confirmed by human, state that. If denied, state that."
        
        current_messages = [SystemMessage(content=system_prompt)] + state["messages"]
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        response = llm.invoke(current_messages)
        print(f"   LLM final response: {response.content}")
        return {"messages": [response]}
    else:
        # If no tool was called and no human interaction, the LLM's previous message is the final one.
        print("   No tool execution or human interaction, LLM's last message is final.")
        return {}

# ============================================================================
# PART 4: ROUTING LOGIC
# ============================================================================
print("\n🔀 PART 4: Routing Logic")
print("-" * 50)

def should_proceed_to_human_or_execute(state: MeetingSchedulerAgentState):
    """
    Router: Decides whether to ask for human approval, execute directly, or go to final response.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # A tool has been proposed by the LLM
        print("   Router: Tool proposed by LLM. Routing to 'human_approval_check'.")
        return "human_approval_check" # This will be an interruption point
    
    print("   Router: No tool proposed by LLM. Routing to 'final_response_node'.")
    return "final_response_node" # No tool call, just respond

def after_human_approval_router(state: MeetingSchedulerAgentState):
    """
    Router: After human interaction, decides whether to execute the tool or just respond.
    """
    if state["human_approved"] is True and state["proposed_meeting_details"]:
        print("   Router: Human approved. Routing to 'execute_tool_node'.")
        return "execute_tool_node"
    else:
        # Human denied, or no action was pending that needed approval
        denial_reason = "Meeting scheduling denied by user." if state["human_approved"] is False else "No action taken."
        print(f"   Router: Human denied or no action. Routing to 'final_response_node'. Reason: {denial_reason}")
        # Add a message indicating denial if applicable
        if state["human_approved"] is False:
            return {
                "messages": [AIMessage(content="Okay, I will not schedule the meeting as per your instruction.")],
                "human_approved": None, # Reset for next turn
                "proposed_meeting_details": None
            } 
        return "final_response_node"

print("✅ Routing functions defined.")

# ============================================================================
# PART 5: BUILD THE GRAPH WITH CHECKPOINTS AND INTERRUPTS
# ============================================================================
print("\n🏗️  PART 5: Building the Graph with Interrupts")
print("-" * 50)

# Using SqliteSaver for checkpointing. In a real app, use a persistent DB.
memory = SqliteSaver.from_conn_string(":memory:")

workflow = StateGraph(MeetingSchedulerAgentState)

workflow.add_node("llm_propose_meeting_node", llm_propose_meeting_node)
workflow.add_node("execute_tool_node", execute_tool_node)
workflow.add_node("final_response_node", final_response_node)

workflow.set_entry_point("llm_propose_meeting_node")

workflow.add_conditional_edges(
    "llm_propose_meeting_node",
    should_proceed_to_human_or_execute,
    {
        "human_approval_check": END, # END here signifies an interruption point
        "final_response_node": "final_response_node"
    }
)

workflow.add_conditional_edges(
    # This edge effectively starts from where the graph was interrupted
    # We'll manually route to this after human input.
    # For the graph structure, we can imagine an implicit node "human_interaction_complete"
    # that leads here.
    "execute_tool_node", # This is a bit of a conceptual leap for graphviz,
                         # but represents the flow after human approval.
                         # In practice, we re-invoke with updated state.
    after_human_approval_router, # This router is called *after* human input is processed
                                 # and state is updated.
    {
        "execute_tool_node": "execute_tool_node", # If approved, execute
        "final_response_node": "final_response_node"  # If denied, or other cases
    }
)
# This edge is for the case where human approval leads to tool execution
workflow.add_edge("execute_tool_node", "final_response_node")
# This edge is for the case where no tool is called, or human denies
workflow.add_edge("final_response_node", END)


# Compile the graph with checkpointing and specify where to interrupt.
# The graph will pause BEFORE the "execute_tool_node" if a tool is called.
# We will name this interruption point "human_approval_check".
# The key here is that "human_approval_check" is a conceptual state,
# the graph actually ENDS at the conditional edge if it routes to "human_approval_check".
# We then handle the human input and resume.

# The `interrupt_before_nodes` argument will make the graph stop if it's
# about to run any of the listed nodes. We'll use this to pause before
# `execute_tool_node` if a tool was proposed.
# For this v2, we will handle interruption more manually based on routing to END.

agent_graph = workflow.compile(
    checkpointer=memory,
    # interrupt_before_nodes=["execute_tool_node"] # Alternative way to interrupt
)

print("✅ Graph compiled with SqliteSaver for checkpoints.")
print("   Interrupts will be handled based on routing to END.")

# ============================================================================
# PART 6: INTERACTIVE LOOP
# ============================================================================
print("\n🎮 PART 6: Interactive Human-in-the-Loop Simulation")
print("-" * 50)

def run_interactive_session():
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"Starting new session: {session_id}")
    print("Ask to schedule a meeting (e.g., 'Schedule a team sync for tomorrow at 10 AM with alice@example.com and bob@example.com')")
    print("Type 'exit' to end.")

    current_state = None

    while True:
        user_input = input(f"\nYour input: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting session.")
            break

        if current_state and current_state.get("human_approved") is None and current_state.get("proposed_meeting_details"):
            # This means we are in a state where human approval is pending
            print("\n--- HUMAN APPROVAL STEP ---")
            print(f"Proposed meeting: {current_state['proposed_meeting_details']}")
            approval = input("Approve scheduling this meeting? (yes/no): ").strip().lower()
            
            human_approved_decision = approval == 'yes'
            
            # Update the state with human's decision
            # We need to pass the *entire current state* plus the new decision.
            # LangGraph's `update_state` takes the config and the values to update.
            
            # To resume, we need to construct the input for the next step.
            # The graph ended, so we are "re-injecting" into a conceptual continuation.
            # We'll manually decide the next node based on approval.
            
            if human_approved_decision:
                print("You approved. The meeting scheduling tool will now run.")
                # The state should reflect approval, then we call the graph again.
                # The graph's `after_human_approval_router` should pick this up.
                # We need to ensure the `messages` list includes the AI's proposal.
                # The `execute_tool_node` expects to find tool_calls on the last message.
                
                # Re-invoke the graph, starting from a point that makes sense.
                # Since our graph is simple, we can re-invoke with the updated state.
                # The `after_human_approval_router` is not directly connected in a way
                # that `agent_graph.invoke` can start from it after an END.
                # This is a limitation of simple `invoke` after interrupt.
                # A more robust way involves specific entry points or modifying the graph.

                # For this v2, we'll simplify: if approved, we directly call execute_tool_node's logic
                # then final_response_node's logic. This bypasses complex graph resumption logic
                # for a clearer tutorial on the human-in-the-loop part itself.

                # Simulate graph continuation:
                current_state_dict = current_state.dict() # Get plain dict
                current_state_dict["human_approved"] = True
                
                # Manually call the nodes for clarity in this tutorial version
                tool_execution_result = execute_tool_node(current_state_dict)
                current_state_dict["messages"].extend(tool_execution_result.get("messages", []))
                
                final_response_result = final_response_node(current_state_dict)
                current_state_dict["messages"].extend(final_response_result.get("messages", []))
                
                ai_response = current_state_dict["messages"][-1].content
                print(f"AI: {ai_response}")
                
                # Persist this manually constructed state
                current_state = MeetingSchedulerAgentState(**current_state_dict)
                memory.put(config, current_state)

            else:
                print("You denied. The meeting will not be scheduled.")
                # Update state and let LLM formulate a response
                current_state_dict = current_state.dict()
                current_state_dict["human_approved"] = False
                current_state_dict["messages"].append(AIMessage(content="Okay, I have cancelled the meeting scheduling process as per your request."))
                
                ai_response = current_state_dict["messages"][-1].content
                print(f"AI: {ai_response}")

                current_state = MeetingSchedulerAgentState(**current_state_dict)
                memory.put(config, current_state)

            # Reset for next user input cycle
            current_state.human_approved = None 
            current_state.proposed_meeting_details = None
            continue # Go to next user input

        # Regular interaction with the graph
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # Stream events to see the state evolve
        print("--- Graph Execution ---")
        for event_part in agent_graph.stream(inputs, config, stream_mode="values"):
            current_state = event_part # Keep the latest state
            # print(f"   Event: {event_part}") # Can be verbose

        # After stream, current_state holds the final state of this run
        # This might be an interruption point or a final response.

        if current_state.get("proposed_meeting_details") and current_state.get("human_approved") is None:
            # This means the graph interrupted because a tool was proposed and needs approval
            print("\n--- PENDING HUMAN APPROVAL ---")
            print(f"AI proposed to schedule a meeting: {current_state['proposed_meeting_details']}")
            print("Please provide your input next to approve or deny.")
            # The loop will now prompt for approval in the next iteration.
        elif current_state["messages"]:
            ai_response = current_state["messages"][-1].content
            print(f"AI: {ai_response}")
        else:
            print("AI did not return a message.")

if __name__ == "__main__":
    run_interactive_session()
    print("\n" + "="*70)
    print("Human-in-the-Loop v2 Tutorial Complete!")
    print("Key takeaways:")
    print("  - Graphs can be designed to pause for human input.")
    print("  - Checkpointing (like SqliteSaver) persists state during pauses.")
    print("  - Human decisions can modify the agent's state and influence its next actions.")
    print("  - Resuming requires careful state management and graph invocation.")
    print("This v2 simplified resumption for clarity on the HIL concept.")
    print("="*70)
