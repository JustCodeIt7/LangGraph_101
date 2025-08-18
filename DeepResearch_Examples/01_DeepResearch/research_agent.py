import asyncio
import operator
from typing import Annotated, List, Dict, Any, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Research state definition
class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    research_data: List[Dict[str, Any]]
    current_step: str
    is_research_complete: bool

# Dummy research tools (in a real implementation, these would connect to actual APIs)
@tool
def search_web(query: str) -> List[Dict[str, str]]:
    """Simulate web search for research topic"""
    # In a real implementation, this would call an API like Google Search, Bing, etc.
    return [
        {"title": f"Result 1 for {query}", "url": "https://example.com/1", "snippet": "This is a sample search result."},
        {"title": f"Result 2 for {query}", "url": "https://example.com/2", "snippet": "Another relevant search result."},
        {"title": f"Result 3 for {query}", "url": "https://example.com/3", "snippet": "Additional information about the topic."}
    ]

@tool
def get_article_content(url: str) -> str:
    """Simulate retrieving full article content"""
    # In a real implementation, this would fetch and parse the actual webpage
    return f"This is the full content of the article from {url}. In a real implementation, this would contain the actual article text."

# Research functions
def initiate_research(state: ResearchState) -> Dict[str, Any]:
    """Start the research process by identifying the topic"""
    user_message = state["messages"][-1].content
    return {
        "topic": user_message,
        "current_step": "researching",
        "messages": [AIMessage(content=f"Starting research on: {user_message}")]
    }

def conduct_research(state: ResearchState) -> Dict[str, Any]:
    """Conduct research on the topic using web search"""
    topic = state["topic"]
    
    # Search for relevant information
    search_results = search_web(topic)
    
    # Gather detailed information from top results
    detailed_research = []
    for result in search_results[:2]:  # Limit to top 2 results for demo
        content = get_article_content(result["url"])
        detailed_research.append({
            "title": result["title"],
            "url": result["url"],
            "snippet": result["snippet"],
            "content": content
        })
    
    return {
        "research_data": detailed_research,
        "current_step": "summarizing",
        "messages": [AIMessage(content=f"Found {len(detailed_research)} relevant sources for '{topic}'")]
    }

def summarize_research(state: ResearchState) -> Dict[str, Any]:
    """Create a summary of the research findings"""
    research_data = state["research_data"]
    topic = state["topic"]
    
    summary_points = []
    for data in research_data:
        summary_points.append(f"- {data['title']}: {data['snippet']}")
    
    summary = f"Research Summary for '{topic}':\n" + "\n".join(summary_points)
    
    return {
        "is_research_complete": True,
        "current_step": "complete",
        "messages": [AIMessage(content=summary)]
    }

def route_research(state: ResearchState) -> str:
    """Determine the next step in the research process"""
    if state["current_step"] == "initiating":
        return "researching"
    elif state["current_step"] == "researching":
        return "summarizing"
    elif state["current_step"] == "summarizing":
        return "complete"
    else:
        return END

# Create the research graph
def create_research_graph():
    """Create and compile the research workflow graph"""
    # Initialize the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("initiate", initiate_research)
    workflow.add_node("research", conduct_research)
    workflow.add_node("summarize", summarize_research)
    
    # Add edges
    workflow.add_edge("initiate", "research")
    workflow.add_edge("research", "summarize")
    workflow.add_edge("summarize", END)
    
    # Set entry point
    workflow.set_entry_point("initiate")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Chat interface
async def chat_with_research_agent():
    """Run a chat interface with the research agent"""
    # Create the research graph
    research_app = create_research_graph()
    
    # Initial state
    state = ResearchState(
        messages=[],
        topic="",
        research_data=[],
        current_step="initiating",
        is_research_complete=False
    )
    
    print("Deep Research Agent (type 'quit' to exit)")
    print("What topic would you like me to research?")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
            
        # Add user message to state
        state["messages"] = [HumanMessage(content=user_input)]
        state["current_step"] = "initiating"
        state["research_data"] = []
        state["is_research_complete"] = False
        state["topic"] = ""
        
        # Process through the research graph
        final_state = research_app.invoke(state)
        
        # Print the final response
        last_message = final_state["messages"][-1]
        print(f"\nAgent: {last_message.content}")

if __name__ == "__main__":
    # For demo purposes, we'll run the chat
    asyncio.run(chat_with_research_agent())