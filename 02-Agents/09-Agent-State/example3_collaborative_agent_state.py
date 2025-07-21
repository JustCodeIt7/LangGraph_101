"""
Example 3: Collaborative Agent State
====================================

This example demonstrates how to use LangGraph agent state for coordinating
multiple specialized agents working together on complex tasks with shared
knowledge, work delegation, and consensus building.

Key Features:
- Multi-agent coordination and communication
- Shared workspace and knowledge base
- Work delegation and task assignment
- Consensus building and conflict resolution
- Agent specialization and expertise areas
- Collaborative decision making
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from enum import Enum

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    CRITIC = "critic"

class TaskStatus(Enum):
    PROPOSED = "proposed"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REQUIRES_REVIEW = "requires_review"
    NEEDS_COLLABORATION = "needs_collaboration"

class AgentInfo(TypedDict):
    role: AgentRole
    expertise: List[str]
    current_task: Optional[str]
    workload: int  # 0-10 scale
    collaboration_history: List[str]
    performance_score: float

class CollaborativeTask(TypedDict):
    id: str
    title: str
    description: str
    assigned_agents: List[AgentRole]
    status: TaskStatus
    priority: int
    created_at: str
    deadline: Optional[str]
    dependencies: List[str]
    outputs: List[Dict[str, any]]

class SharedKnowledge(TypedDict):
    facts: List[str]
    insights: List[str]
    decisions: List[Dict[str, any]]
    resources: List[str]
    constraints: List[str]

class CollaborativeAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    active_agents: Dict[AgentRole, AgentInfo]
    current_coordinator: AgentRole
    shared_workspace: Dict[str, any]
    knowledge_base: SharedKnowledge
    active_tasks: Dict[str, CollaborativeTask]
    collaboration_queue: List[str]  # Task IDs needing collaboration
    consensus_items: List[Dict[str, any]]  # Items requiring group consensus
    meeting_notes: List[str]
    project_timeline: List[Dict[str, any]]

def coordinator_node(state: CollaborativeAgentState):
    """
    Coordinates work between agents, assigns tasks, and manages workflow.
    """
    print("---COORDINATOR MANAGING WORKFLOW---")
    
    messages = state["messages"]
    if not messages:
        return state
    
    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return state
    
    user_request = latest_message.content
    
    # Analyze request and create collaborative task
    task_id = f"task_{len(state['active_tasks']) + 1}"
    
    # Simple task analysis (in real implementation, use NLP)
    complexity_keywords = ["complex", "research", "analyze", "creative", "technical"]
    complexity_score = sum(1 for keyword in complexity_keywords if keyword in user_request.lower())
    
    # Determine which agents should collaborate
    required_agents = [AgentRole.COORDINATOR]  # Always include coordinator
    
    if "research" in user_request.lower() or "find" in user_request.lower():
        required_agents.append(AgentRole.RESEARCHER)
    if "analyze" in user_request.lower() or "data" in user_request.lower():
        required_agents.append(AgentRole.ANALYST)
    if "creative" in user_request.lower() or "design" in user_request.lower():
        required_agents.append(AgentRole.CREATIVE)
    if "technical" in user_request.lower() or "code" in user_request.lower():
        required_agents.append(AgentRole.TECHNICAL)
    
    # If complex task, add critic for review
    if complexity_score >= 2:
        required_agents.append(AgentRole.CRITIC)
    
    new_task: CollaborativeTask = {
        "id": task_id,
        "title": f"Collaborative Task: {user_request[:50]}...",
        "description": user_request,
        "assigned_agents": required_agents,
        "status": TaskStatus.ASSIGNED,
        "priority": 5,  # High priority for user requests
        "created_at": datetime.now().isoformat(),
        "deadline": None,
        "dependencies": [],
        "outputs": []
    }
    
    updated_tasks = state["active_tasks"].copy()
    updated_tasks[task_id] = new_task
    
    # Add to collaboration queue
    updated_queue = state["collaboration_queue"].copy()
    updated_queue.append(task_id)
    
    print(f"Created collaborative task: {task_id}")
    print(f"Assigned agents: {[agent.value for agent in required_agents]}")
    
    return {
        "active_tasks": updated_tasks,
        "collaboration_queue": updated_queue
    }

def multi_agent_collaboration_node(state: CollaborativeAgentState):
    """
    Simulates multiple agents working together on the current task.
    """
    print("---AGENTS COLLABORATING---")
    
    if not state["collaboration_queue"]:
        return state
    
    current_task_id = state["collaboration_queue"][0]
    current_task = state["active_tasks"][current_task_id]
    
    # Simulate each agent contributing their expertise
    agent_contributions = {}
    
    for agent_role in current_task["assigned_agents"]:
        contribution = f"[{agent_role.value.upper()}] "
        
        if agent_role == AgentRole.RESEARCHER:
            contribution += "I've gathered relevant background information and identified key data sources. "
        elif agent_role == AgentRole.ANALYST:
            contribution += "Based on the data patterns, I've identified three key insights and potential approaches. "
        elif agent_role == AgentRole.CREATIVE:
            contribution += "I've brainstormed innovative solutions and alternative perspectives on this challenge. "
        elif agent_role == AgentRole.TECHNICAL:
            contribution += "I've assessed technical feasibility and identified implementation strategies. "
        elif agent_role == AgentRole.CRITIC:
            contribution += "I've reviewed the proposed approaches and identified potential risks and improvements. "
        elif agent_role == AgentRole.COORDINATOR:
            contribution += "I'm synthesizing all inputs and coordinating the collaborative effort. "
        
        agent_contributions[agent_role] = contribution
        print(f"Agent {agent_role.value} contributed to task {current_task_id}")
    
    # Update task with agent outputs
    updated_tasks = state["active_tasks"].copy()
    updated_tasks[current_task_id]["outputs"] = [
        {"agent": agent.value, "contribution": contrib}
        for agent, contrib in agent_contributions.items()
    ]
    updated_tasks[current_task_id]["status"] = TaskStatus.REQUIRES_REVIEW
    
    # Update shared knowledge base
    updated_knowledge = state["knowledge_base"].copy()
    
    # Add insights from collaboration
    new_insights = [
        f"Task {current_task_id}: Multi-agent collaboration with {len(agent_contributions)} specialists",
        f"Expertise combined: {', '.join([agent.value for agent in current_task['assigned_agents']])}",
    ]
    
    updated_knowledge["insights"] = updated_knowledge["insights"] + new_insights
    
    return {
        "active_tasks": updated_tasks,
        "knowledge_base": updated_knowledge
    }

def consensus_builder_node(state: CollaborativeAgentState):
    """
    Builds consensus from agent contributions and creates final response.
    """
    print("---BUILDING CONSENSUS---")
    
    if not state["collaboration_queue"]:
        return state
    
    current_task_id = state["collaboration_queue"][0]
    current_task = state["active_tasks"][current_task_id]
    
    if current_task["status"] != TaskStatus.REQUIRES_REVIEW:
        return state
    
    # Synthesize agent contributions
    all_contributions = []
    for output in current_task["outputs"]:
        all_contributions.append(output["contribution"])
    
    # Create synthesis prompt
    synthesis_prompt = f"""
    You are synthesizing input from multiple specialized agents working on this task:
    TASK: {current_task['description']}
    
    AGENT CONTRIBUTIONS:
    {chr(10).join(all_contributions)}
    
    Please create a comprehensive response that:
    1. Integrates insights from all agents
    2. Addresses the original request thoroughly
    3. Highlights key findings and recommendations
    4. Shows how different perspectives complement each other
    5. Provides actionable next steps
    
    Present this as a collaborative team response.
    """
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    messages = state["messages"]
    synthesis_messages = [
        SystemMessage(content=synthesis_prompt),
        HumanMessage(content=current_task["description"])
    ]
    
    response = llm.invoke(synthesis_messages)
    
    # Update task status
    updated_tasks = state["active_tasks"].copy()
    updated_tasks[current_task_id]["status"] = TaskStatus.COMPLETED
    
    # Remove from collaboration queue
    updated_queue = state["collaboration_queue"].copy()
    updated_queue.remove(current_task_id)
    
    # Add meeting notes
    meeting_note = f"COLLABORATION COMPLETED - Task {current_task_id}: {len(current_task['assigned_agents'])} agents collaborated successfully"
    updated_meeting_notes = state["meeting_notes"].copy()
    updated_meeting_notes.append(meeting_note)
    
    print(f"Consensus built for task {current_task_id}")
    
    return {
        "messages": [AIMessage(content=response.content)],
        "active_tasks": updated_tasks,
        "collaboration_queue": updated_queue,
        "meeting_notes": updated_meeting_notes
    }

# Build the graph
workflow = StateGraph(CollaborativeAgentState)

workflow.add_node("coordinator", coordinator_node)
workflow.add_node("collaborators", multi_agent_collaboration_node)
workflow.add_node("consensus", consensus_builder_node)

# Set up the collaboration flow
workflow.set_entry_point("coordinator")
workflow.add_edge("coordinator", "collaborators")
workflow.add_edge("collaborators", "consensus")
workflow.add_edge("consensus", END)

# Compile the graph
collaborative_agent = workflow.compile()

if __name__ == "__main__":
    print("Starting Collaborative Multi-Agent System...")
    print("Multiple specialized agents will work together on your requests.")
    
    # Initialize the collaborative state
    initial_state: CollaborativeAgentState = {
        "messages": [],
        "active_agents": {
            AgentRole.COORDINATOR: {
                "role": AgentRole.COORDINATOR,
                "expertise": ["project management", "workflow coordination"],
                "current_task": None,
                "workload": 3,
                "collaboration_history": [],
                "performance_score": 8.5
            },
            AgentRole.RESEARCHER: {
                "role": AgentRole.RESEARCHER,
                "expertise": ["information gathering", "fact-checking", "data collection"],
                "current_task": None,
                "workload": 2,
                "collaboration_history": [],
                "performance_score": 9.0
            },
            AgentRole.ANALYST: {
                "role": AgentRole.ANALYST,
                "expertise": ["data analysis", "pattern recognition", "insights"],
                "current_task": None,
                "workload": 4,
                "collaboration_history": [],
                "performance_score": 8.8
            },
            AgentRole.CREATIVE: {
                "role": AgentRole.CREATIVE,
                "expertise": ["brainstorming", "innovation", "design thinking"],
                "current_task": None,
                "workload": 1,
                "collaboration_history": [],
                "performance_score": 9.2
            },
            AgentRole.TECHNICAL: {
                "role": AgentRole.TECHNICAL,
                "expertise": ["implementation", "architecture", "technical solutions"],
                "current_task": None,
                "workload": 5,
                "collaboration_history": [],
                "performance_score": 8.7
            },
            AgentRole.CRITIC: {
                "role": AgentRole.CRITIC,
                "expertise": ["quality assurance", "risk assessment", "improvement"],
                "current_task": None,
                "workload": 2,
                "collaboration_history": [],
                "performance_score": 9.1
            }
        },
        "current_coordinator": AgentRole.COORDINATOR,
        "shared_workspace": {},
        "knowledge_base": {
            "facts": [],
            "insights": [],
            "decisions": [],
            "resources": [],
            "constraints": []
        },
        "active_tasks": {},
        "collaboration_queue": [],
        "consensus_items": [],
        "meeting_notes": ["Multi-agent collaboration system initialized"],
        "project_timeline": []
    }
    
    current_state = initial_state
    
    print("\nAvailable agent specializations:")
    for role, info in current_state["active_agents"].items():
        print(f"• {role.value.title()}: {', '.join(info['expertise'])}")
    
    print("\nThe system will automatically assign relevant agents based on your request.")
    print("Type 'status' to see current tasks, or 'exit' to quit.\n")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        
        if user_input.lower() == "status":
            print(f"\n--- COLLABORATION STATUS ---")
            print(f"Active Tasks: {len(current_state['active_tasks'])}")
            print(f"Collaboration Queue: {len(current_state['collaboration_queue'])}")
            print(f"Completed Collaborations: {len(current_state['meeting_notes']) - 1}")
            
            for task_id, task in current_state["active_tasks"].items():
                agents = [agent.value for agent in task["assigned_agents"]]
                print(f"Task {task_id}: {task['status'].value} | Agents: {', '.join(agents)}")
            continue
        
        # Add user message to state
        current_state["messages"] = [HumanMessage(content=user_input)]
        
        print("\n🤝 Agents collaborating on your request...")
        
        # Run the collaborative workflow
        result = collaborative_agent.invoke(current_state)
        
        # Update current state with results
        current_state.update(result)
        
        # Display collaborative response
        if "messages" in result and result["messages"]:
            ai_message = result["messages"][-1]
            if isinstance(ai_message, AIMessage):
                print(f"\n🎯 COLLABORATIVE RESPONSE:\n{ai_message.content}")
    
    print(f"\n--- COLLABORATION SESSION SUMMARY ---")
    print(f"Total Tasks Completed: {len(current_state['active_tasks'])}")
    print(f"Agents Participated: {len(current_state['active_agents'])}")
    print(f"Knowledge Insights Generated: {len(current_state['knowledge_base']['insights'])}")
    print(f"Meeting Notes: {len(current_state['meeting_notes'])}")
    print("\nThank you for using the Collaborative Multi-Agent System!")