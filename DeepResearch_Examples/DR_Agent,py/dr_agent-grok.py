# Deep Research Agent using LangGraph, LangChain, Ollama, and Brave Search
# This script creates a simple agent that performs deep research on a query.
# It uses Ollama for local LLM, Brave Search for web searching, and LangGraph for workflow.
# The agent can iteratively search and reason until it generates a final report.
# Total lines: ~150 (including comments and blanks for readability)

import os
import requests
from typing import Dict, TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
# load .env
from dotenv import load_dotenv
load_dotenv()
# Step 1: Set up environment variables
# Get your free Brave Search API key from https://api.brave.com/
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")  # Set this in your environment
if not BRAVE_API_KEY:
    raise ValueError("Please set BRAVE_API_KEY environment variable")

# Step 2: Define the Brave Search Tool
@tool
def brave_search(query: str) -> str:
    """Search the web using Brave Search API and return summarized results."""
    url = "https://api.brave.com/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {
        "q": query,
        "count": 5,  # Limit to 5 results for simplicity
        "safesearch": "moderate"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"
    
    data = response.json()
    results = []
    if "web" in data and "results" in data["web"]:
        for result in data["web"]["results"]:
            title = result.get("title", "No title")
            description = result.get("description", "No description")
            url = result.get("url", "No URL")
            results.append(f"Title: {title}\nDescription: {description}\nURL: {url}\n")
    return "\n".join(results) if results else "No results found."

# Step 3: Set up the LLM (using Ollama with llama3 model - ensure Ollama is running)
llm = ChatOllama(model="llama3.2")  # Assumes Ollama is installed and running with llama3

# Bind tools to LLM
tools = [brave_search]
llm_with_tools = llm.bind_tools(tools)

# Step 4: Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], "append"]
    query: str
    research_depth: int  # To limit depth and prevent infinite loops

# Step 5: Define the agent node
def agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    research_depth = state.get("research_depth", 0)
    
    # System prompt for deep research
    system_prompt = SystemMessage(content=(
        "You are a deep research agent. Your goal is to research the query thoroughly. "
        "Use the brave_search tool to gather information. Iterate if needed to deepen research. "
        "Limit to 3 iterations max. When ready, output a final report starting with 'FINAL REPORT:'."
    ))
    
    # Invoke LLM
    response = llm_with_tools.invoke([system_prompt] + messages)
    
    # Append response
    new_messages = messages + [response]
    
    # Check if final report
    if "FINAL REPORT:" in response.content:
        return {"messages": new_messages, "research_depth": research_depth}
    
    # Increment depth
    return {"messages": new_messages, "research_depth": research_depth + 1}

# Step 6: Define the should_continue function
def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If tool calls, go to tools
    if last_message.tool_calls:
        return "tools"
    
    # If final report or max depth, end
    if "FINAL REPORT:" in last_message.content or state["research_depth"] >= 3:
        return "end"
    
    # Otherwise, continue to agent
    return "agent"

# Step 7: Build the graph
workflow = StateGraph(state_schema=AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "agent": "agent", "end": END}
)
workflow.add_conditional_edges(
    "tools",
    should_continue,
    {"agent": "agent", "end": END}
)

# Compile the graph with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Step 8: Function to run the research agent
def run_research(query: str, thread_id: str = "research_thread"):
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "research_depth": 0
    }
    config = {"configurable": {"thread_id": thread_id}}
    
    for event in app.stream(initial_state, config, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print(last_msg.content)
            if "FINAL REPORT:" in last_msg.content:
                break

# Example usage
if __name__ == "__main__":
    # Ensure you have set BRAVE_API_KEY
    query = "Latest advancements in AI agents as of 2025"
    run_research(query)