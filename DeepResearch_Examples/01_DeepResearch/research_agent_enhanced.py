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

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Research state definition
class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    research_data: List[Dict[str, Any]]
    current_step: str
    is_research_complete: bool

# Research tools (examples for connecting to real APIs are commented below)
@tool
def search_web(query: str) -> List[Dict[str, str]]:
    """Simulate web search for research topic"""
    # In a real implementation, this would call an API like Google Search, Bing, etc.
    # Example with Tavily API (uncomment and install tavily-python):
    # from tavily import TavilyClient
    # client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    # results = client.search(query, max_results=5)
    # return [{"title": r["title"], "url": r["url"], "snippet": r["content"][:200]} for r in results["results"]]
    
    return [
        {"title": f"Comprehensive Guide to {query}", "url": "https://example.com/guide", "snippet": f"An in-depth look at {query} covering history, current state, and future prospects."},
        {"title": f"Latest Developments in {query}", "url": "https://example.com/latest", "snippet": f"Recent breakthroughs and news related to {query} in 2024."},
        {"title": f"{query} Applications and Use Cases", "url": "https://example.com/applications", "snippet": f"How {query} is being applied across different industries and domains."}
    ]

@tool
def get_article_content(url: str) -> str:
    """Simulate retrieving full article content"""
    # In a real implementation, this would fetch and parse the actual webpage
    # Example with BeautifulSoup (uncomment and install beautifulsoup4, requests):
    # import requests
    # from bs4 import BeautifulSoup
    # response = requests.get(url)
    # soup = BeautifulSoup(response.content, 'html.parser')
    # content = soup.get_text()
    # return content[:2000]  # Limit content length
    
    return f"""
    This is a comprehensive article about the topic. In a real implementation, 
    this would contain detailed information extracted from {url}.
    
    The content would include several key sections:
    1. Introduction and background
    2. Historical development
    3. Current state of the field
    4. Recent advancements
    5. Challenges and limitations
    6. Future prospects and opportunities
    
    This detailed content would be used by the research agent to understand
    the topic thoroughly and provide comprehensive answers to user queries.
    """

# Research functions
def initiate_research(state: ResearchState) -> Dict[str, Any]:
    """Start the research process by identifying the topic"""
    user_message = state["messages"][-1].content
    
    # Use LLM to refine the research topic if needed
    prompt = f"""
    The user wants to research: "{user_message}"
    
    Please identify the main topic for research. Extract only the core subject.
    Topic:
    """
    
    response = llm.invoke(prompt)
    refined_topic = response.content.strip()
    
    return {
        "topic": refined_topic,
        "current_step": "researching",
        "messages": [AIMessage(content=f"Starting research on: {refined_topic}")]
    }

def conduct_research(state: ResearchState) -> Dict[str, Any]:
    """Conduct research on the topic using web search"""
    topic = state["topic"]
    
    # Search for relevant information
    search_results = search_web(topic)
    
    # Gather detailed information from top results
    detailed_research = []
    for result in search_results[:3]:  # Limit to top 3 results
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
    
    # Prepare content for LLM summarization
    content_to_summarize = "\n\n".join([
        f"Source: {data['title']}\nContent: {data['content']}" 
        for data in research_data
    ])
    
    # Use LLM to create a comprehensive summary
    prompt = f"""
    Based on the following research content about '{topic}', create a comprehensive summary.
    
    Include these sections:
    1. Overview of the topic
    2. Key findings
    3. Recent developments
    4. Applications and use cases
    5. Future prospects
    
    Research Content:
    {content_to_summarize}
    
    Summary:
    """
    
    response = llm.invoke(prompt)
    summary = response.content.strip()
    
    return {
        "is_research_complete": True,
        "current_step": "complete",
        "messages": [AIMessage(content=summary)]
    }

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