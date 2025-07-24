"""
Deep Research Agent - Core Logic Implementation
A LangGraph-based research agent that performs iterative web research and generates comprehensive reports.
"""

import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from tavily import TavilyClient
import logging
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_ITERATIONS = 3  # Maximum research iterations to prevent infinite loops
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://eos-james.local:11434')
if not TAVILY_API_KEY:
    logger.warning('TAVILY_API_KEY not found in environment variables')


# Agent State Definition
class ResearchState(TypedDict):
    """State structure for the deep research agent."""
    research_topic: str
    search_results: List[Dict[str, Any]]
    processed_information: str
    report_content: str
    num_iterations: int
    search_queries: List[str]
    current_query: Optional[str]


# Tavily Search Tool
@tool
def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily API for research information.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        List of search results with title, content, url, and score
    """
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
            include_raw_content=True
        )
        
        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", ""),
                "score": result.get("score", 0),
                "raw_content": result.get("raw_content", "")
            })
        
        logger.info(f"Tavily search completed for query: {query}")
        return results
    
    except Exception as e:
        logger.error(f"Error in Tavily search: {str(e)}")
        return []


# Initialize LLM with Ollama
llm = ChatOllama(model='qwen3', temperature=0.7, base_url=OLLAMA_BASE_URL)


# Node Functions
def research_initiator(state: ResearchState) -> ResearchState:
    """
    Initialize the research process with the given topic.
    
    Args:
        state: Current research state
    
    Returns:
        Updated state with initial values
    """
    logger.info("Starting research process")
    
    return {
        **state,
        "search_results": [],
        "processed_information": "",
        "report_content": "",
        "num_iterations": 0,
        "search_queries": [state["research_topic"]],
        "current_query": state["research_topic"]
    }


def search_executor(state: ResearchState) -> ResearchState:
    """
    Execute web searches based on the current query.
    
    Args:
        state: Current research state
    
    Returns:
        Updated state with new search results
    """
    current_query = state.get("current_query", state["research_topic"])
    logger.info(f"Executing search for query: {current_query}")
    
    # Perform search using Tavily
    search_results = tavily_search(current_query)
    
    # Combine new results with existing ones
    all_results = state.get("search_results", []) + search_results
    
    return {
        **state,
        "search_results": all_results
    }


def information_processor(state: ResearchState) -> ResearchState:
    """
    Process search results and extract key information using LLM.
    
    Args:
        state: Current research state
    
    Returns:
        Updated state with processed information and potential new queries
    """
    logger.info("Processing search results")
    
    # Prepare context for LLM
    search_context = "\n\n".join([
        f"Source: {result['title']}\nURL: {result['url']}\nContent: {result['content']}"
        for result in state["search_results"]
    ])
    
    # Create processing prompt
    processing_prompt = f"""
    You are a research analyst. Your task is to process the following search results and extract key insights about: {state['research_topic']}
    
    Search Results:
    {search_context}
    
    Please provide:
    1. A comprehensive summary of the key findings
    2. Identification of any gaps or areas that need further research
    3. Suggestions for specific follow-up search queries (if needed)
    
    Format your response as:
    SUMMARY: [Your comprehensive summary]
    
    GAPS: [List any knowledge gaps or areas needing more research]
    
    FOLLOW_UP_QUERIES: [List specific search queries, one per line, or "None" if sufficient]
    """
    
    # Process with LLM
    response = llm.invoke([HumanMessage(content=processing_prompt)])
    content = response.content
    
    # Parse response
    summary = ""
    gaps = ""
    follow_up_queries = []
    
    lines = content.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('SUMMARY:'):
            current_section = 'summary'
            summary = line[8:].strip()
        elif line.startswith('GAPS:'):
            current_section = 'gaps'
            gaps = line[5:].strip()
        elif line.startswith('FOLLOW_UP_QUERIES:'):
            current_section = 'queries'
            queries_text = line[18:].strip()
            if queries_text and queries_text != "None":
                follow_up_queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        elif current_section == 'summary' and line:
            summary += ' ' + line
        elif current_section == 'gaps' and line:
            gaps += ' ' + line
    
    # Update processed information
    processed_info = f"Summary: {summary}\n\nIdentified Gaps: {gaps}"
    
    return {
        **state,
        "processed_information": processed_info,
        "search_queries": state.get("search_queries", []) + follow_up_queries,
        "current_query": follow_up_queries[0] if follow_up_queries else None,
        "num_iterations": state.get("num_iterations", 0) + 1
    }


def decide_next_step(state: ResearchState) -> str:
    """
    Decide whether to continue researching or generate the final report.
    
    Args:
        state: Current research state
    
    Returns:
        Next node to transition to: "search_executor" or "report_generator"
    """
    num_iterations = state.get("num_iterations", 0)
    has_follow_up = state.get("current_query") is not None
    
    logger.info(f"Deciding next step - Iterations: {num_iterations}, Has follow-up: {has_follow_up}")
    
    if num_iterations < MAX_ITERATIONS and has_follow_up:
        return "search_executor"
    else:
        return "report_generator"


def report_generator(state: ResearchState) -> ResearchState:
    """
    Generate the final comprehensive research report.
    
    Args:
        state: Current research state
    
    Returns:
        Updated state with the final report
    """
    logger.info("Generating final research report")
    
    # Prepare context for report generation
    all_information = state.get("processed_information", "")
    search_results = state.get("search_results", [])
    
    # Create report prompt
    report_prompt = f"""
    You are a professional research report writer. Create a comprehensive research report on: {state['research_topic']}
    
    Based on the following processed information and search results, create a well-structured, detailed report.
    
    Processed Information:
    {all_information}
    
    Search Results Summary:
    {len(search_results)} sources were analyzed during the research process.
    
    Your report should include:
    1. Executive Summary
    2. Introduction
    3. Main Findings (organized by themes)
    4. Detailed Analysis
    5. Conclusions
    6. References (include URLs from search results)
    
    Make the report professional, comprehensive, and well-cited.
    """
    
    # Generate report with LLM
    response = llm.invoke([HumanMessage(content=report_prompt)])
    report = response.content
    
    return {
        **state,
        "report_content": report
    }


# Create the LangGraph workflow
def create_research_graph() -> StateGraph:
    """
    Create and configure the LangGraph research workflow.
    
    Returns:
        Configured StateGraph instance
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("research_initiator", research_initiator)
    workflow.add_node("search_executor", search_executor)
    workflow.add_node("information_processor", information_processor)
    workflow.add_node("report_generator", report_generator)
    
    # Add edges
    workflow.set_entry_point("research_initiator")
    
    # Research loop
    workflow.add_edge("research_initiator", "search_executor")
    workflow.add_edge("search_executor", "information_processor")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "information_processor",
        decide_next_step,
        {
            "search_executor": "search_executor",
            "report_generator": "report_generator"
        }
    )
    
    # End edge
    workflow.add_edge("report_generator", END)
    
    return workflow


# Main execution function
def run_research_agent(research_topic: str) -> Dict[str, Any]:
    """
    Run the deep research agent with a given topic.
    
    Args:
        research_topic: The topic to research
    
    Returns:
        Final research state with report
    """
    logger.info(f"Starting research on topic: {research_topic}")
    
    # Create and compile the graph
    workflow = create_research_graph()
    app = workflow.compile()
    
    # Initialize state
    initial_state = {
        "research_topic": research_topic,
        "search_results": [],
        "processed_information": "",
        "report_content": "",
        "num_iterations": 0,
        "search_queries": [],
        "current_query": None
    }
    
    # Run the agent
    final_state = app.invoke(initial_state)
    
    logger.info("Research completed successfully")
    return final_state


# Entry point - only run Streamlit app when called directly
if __name__ == "__main__":
    # This will only run when the file is executed directly
    # Streamlit will handle the actual app execution
    pass


# Streamlit UI Implementation
def main():
    """Main Streamlit application for the Deep Research Agent."""
    
    # Page configuration
    st.set_page_config(
        page_title="Deep Research Agent",
        page_icon="🔍",
        layout="wide"
    )

    # Title and description
    st.title("🔍 Deep Research Agent")
    st.markdown("""
    Welcome to the Deep Research Agent! This AI-powered tool performs comprehensive web research
    on any topic and generates detailed, well-cited reports.
    """)

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.info("""
        This agent uses:
        - **Tavily API** for web search
        - **Ollama qwen3** for analysis and report generation
        - **LangGraph** for orchestrating the research workflow
        
        The agent performs iterative research with up to 3 iterations to ensure comprehensive coverage.
        """)

        st.header("Usage Tips")
        st.markdown("""
        - Be specific with your research topic
        - Include key aspects you want to explore
        - Wait for the full report generation
        """)

    # Main interface
    st.header("Chat with Research Agent")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_research" not in st.session_state:
        st.session_state.current_research = None
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me to research any topic..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process research request
        with st.chat_message("assistant"):
            with st.spinner("🔍 Researching your topic..."):
                try:
                    # Run research agent
                    result = run_research_agent(prompt.strip())
                    
                    # Format response
                    report_content = result.get("report_content", "No report generated")
                    response = f"## Research Report: {prompt}\n\n{report_content}"
                    
                    # Add research summary
                    summary = f"""
                    \n\n---
                    **Research Summary:**
                    - **Sources analyzed:** {len(result.get('search_results', []))}
                    - **Search queries:** {len(result.get('search_queries', []))}
                    - **Iterations:** {result.get('num_iterations', 0)}
                    """
                    
                    full_response = response + summary
                    
                    # Display assistant response
                    st.markdown(full_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    
                    # Store current research
                    st.session_state.current_research = result
                    
                except Exception as e:
                    error_msg = f"❌ Research error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Sidebar with additional controls
    with st.sidebar:
        st.header("Chat Controls")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.current_research = None
            st.rerun()
        
        if st.session_state.current_research:
            with st.expander("📊 Latest Research Details"):
                research = st.session_state.current_research
                st.markdown(f"""
                **Search Queries Used:**
                {chr(10).join(f"- {query}" for query in research.get('search_queries', []))}
                
                **Sources Found:** {len(research.get('search_results', []))}
                **Iterations:** {research.get('num_iterations', 0)}
                """)

    # Footer
    st.markdown("---")
    st.markdown('*Powered by LangGraph, Ollama, and Tavily*')


# TODO Rename this here and in `main`
def _extracted_from_main_57(research_topic):
    # Run the research agent
    result = run_research_agent(research_topic.strip())

    # Display results
    st.success("✅ Research completed successfully!")

    # Report section
    st.header("📊 Research Report")

    # Display the report
    report_content = result.get("report_content", "No report generated")
    st.markdown(report_content)

    # Additional details in expander
    with st.expander("📈 Research Summary"):
        st.markdown(f"""
                        **Research Details:**
                        - **Total iterations:** {result.get('num_iterations', 0)}
                        - **Search queries used:** {len(result.get('search_queries', []))}
                        - **Total sources analyzed:** {len(result.get('search_results', []))}
                        
                        **Search queries executed:**
                        {chr(10).join(f"- {query}" for query in result.get('search_queries', []))}
                        """)


# Entry point for Streamlit
if __name__ == "__main__":
    # When run with Streamlit, this will be handled by Streamlit itself
    # The main() function will be called automatically by Streamlit
    main()
