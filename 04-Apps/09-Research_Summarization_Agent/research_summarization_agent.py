"""
Research & Summarization Agent
=============================

This agent performs research by scraping web pages and scholarly papers. 
It can synthesize information from multiple sources, include citations, 
and generate concise reports or executive summaries.

Features:
- Web scraping for information gathering
- Scholarly paper search and extraction
- Information synthesis with citations
- Multiple output formats (concise report, executive summary, bullet points)
- LangGraph for agent orchestration

Implementation Notes:
- The current implementation uses mock functions for web scraping and scholarly paper search
- In a production environment, these would be replaced with actual implementations
- The agent uses LangGraph for orchestration, with nodes for each step of the research process
- The agent uses Ollama with the llama3.2 model for LLM functionality

Usage:
- Run the script directly to start an interactive session
- Enter research requests like "Research the impact of AI on healthcare"
- Specify research scope ("web", "scholarly", "all") and output format in your request
- See README_USAGE.md for more detailed instructions

Author: LangGraph_101 Team
Date: 2025-07-21
"""

import os
import json
import re
import datetime
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM

# For web scraping (to be implemented)
import requests
from bs4 import BeautifulSoup

# For scholarly paper search (to be implemented)
# Would use specific APIs like Semantic Scholar, CORE, etc.

# Data Models
class ResearchScope(str, Enum):
    WEB = "web"
    SCHOLARLY = "scholarly"
    ALL = "all"

class OutputFormat(str, Enum):
    CONCISE_REPORT = "concise_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    BULLET_POINTS = "bullet_points"

@dataclass
class ContentSource:
    text: str
    url: str
    author: Optional[str] = None
    date: Optional[str] = None
    title: Optional[str] = None

@dataclass
class Finding:
    content: str
    source: ContentSource
    relevance_score: float = 1.0

@dataclass
class ResearchRequest:
    topic: str
    research_scope: ResearchScope = ResearchScope.ALL
    output_format: OutputFormat = OutputFormat.CONCISE_REPORT

# Agent State
class ResearchAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    request: ResearchRequest
    web_sources: List[str]
    scholarly_sources: List[Dict[str, Any]]
    scraped_content: List[ContentSource]
    key_findings: List[Finding]
    synthesized_information: str
    final_report: str
    current_step: str

# Mock functions for development - to be replaced with actual implementations
def mock_web_search(topic: str) -> List[str]:
    """Mock function for web search."""
    return [
        f"https://example.com/article-about-{topic.replace(' ', '-')}-1",
        f"https://example.com/article-about-{topic.replace(' ', '-')}-2",
        f"https://example.com/article-about-{topic.replace(' ', '-')}-3",
    ]

def mock_scholarly_search(topic: str) -> List[Dict[str, Any]]:
    """Mock function for scholarly search."""
    return [
        {
            "title": f"Research on {topic}: A Comprehensive Review",
            "authors": ["Smith, J.", "Johnson, A."],
            "year": 2023,
            "url": f"https://example.com/scholarly-paper-{topic.replace(' ', '-')}-1",
            "abstract": f"This paper provides a comprehensive review of {topic}..."
        },
        {
            "title": f"Advances in {topic} Research",
            "authors": ["Brown, R.", "Davis, M."],
            "year": 2022,
            "url": f"https://example.com/scholarly-paper-{topic.replace(' ', '-')}-2",
            "abstract": f"Recent advances in {topic} have shown promising results..."
        }
    ]

def mock_web_scrape(url: str) -> ContentSource:
    """Mock function for web scraping."""
    topic = re.search(r"article-about-(.+)-\d+", url)
    topic_text = topic.group(1).replace('-', ' ') if topic else "unknown topic"
    
    return ContentSource(
        text=f"This is mock content about {topic_text}. It contains information that would be scraped from a real webpage. The content discusses various aspects of {topic_text} including its history, current developments, and future prospects.",
        url=url,
        author="John Doe",
        date="2023-07-15",
        title=f"Article about {topic_text}"
    )

def mock_extract_paper_text(paper_metadata: Dict[str, Any]) -> ContentSource:
    """Mock function for extracting text from scholarly papers."""
    title = paper_metadata.get("title", "Unknown Title")
    authors = ", ".join(paper_metadata.get("authors", ["Unknown Author"]))
    year = paper_metadata.get("year", 2023)
    url = paper_metadata.get("url", "https://example.com")
    abstract = paper_metadata.get("abstract", "No abstract available")
    
    text = f"""
    Title: {title}
    Authors: {authors}
    Year: {year}
    
    Abstract:
    {abstract}
    
    Introduction:
    This is mock content for a scholarly paper about {title}. It would contain the full text of the paper including methodology, results, discussion, and conclusion.
    
    Methodology:
    The research methodology involved comprehensive analysis of existing literature and experimental validation.
    
    Results:
    The results indicate significant findings in the area of study.
    
    Conclusion:
    This research contributes to the advancement of knowledge in the field.
    """
    
    return ContentSource(
        text=text,
        url=url,
        author=authors,
        date=str(year),
        title=title
    )

# Research Tools
@tool("identify_web_sources")
def identify_web_sources_tool(topic: str) -> str:
    """
    Identify relevant web sources for a given topic.
    
    Args:
        topic: The research topic
    """
    # In a real implementation, this would use search engine APIs
    urls = mock_web_search(topic)
    return json.dumps(urls)

@tool("scrape_web_pages")
def scrape_web_pages_tool(urls_json: str) -> str:
    """
    Scrape content from web pages.
    
    Args:
        urls_json: JSON string of URLs to scrape
    """
    try:
        urls = json.loads(urls_json)
        scraped_content = []
        
        for url in urls:
            # In a real implementation, this would use BeautifulSoup or similar
            content = mock_web_scrape(url)
            scraped_content.append(asdict(content))
        
        return json.dumps(scraped_content)
    except Exception as e:
        return f"Error scraping web pages: {str(e)}"

@tool("search_scholarly_papers")
def search_scholarly_papers_tool(topic: str) -> str:
    """
    Search for scholarly papers on a given topic.
    
    Args:
        topic: The research topic
    """
    # In a real implementation, this would use academic APIs
    papers = mock_scholarly_search(topic)
    return json.dumps(papers)

@tool("extract_text_from_papers")
def extract_text_from_papers_tool(papers_json: str) -> str:
    """
    Extract text from scholarly papers.
    
    Args:
        papers_json: JSON string of paper metadata
    """
    try:
        papers = json.loads(papers_json)
        extracted_content = []
        
        for paper in papers:
            # In a real implementation, this would download and parse PDFs
            content = mock_extract_paper_text(paper)
            extracted_content.append(asdict(content))
        
        return json.dumps(extracted_content)
    except Exception as e:
        return f"Error extracting text from papers: {str(e)}"

@tool("extract_key_findings")
def extract_key_findings_tool(content_json: str, topic: str) -> str:
    """
    Extract key findings from content.
    
    Args:
        content_json: JSON string of content sources
        topic: The research topic
    """
    try:
        content_sources = json.loads(content_json)
        findings = []
        
        for source in content_sources:
            # In a real implementation, this would use NLP techniques
            # For now, we'll create a simple mock finding
            finding = Finding(
                content=f"Key finding about {topic} from {source.get('url', 'unknown source')}",
                source=ContentSource(
                    text=source.get("text", ""),
                    url=source.get("url", ""),
                    author=source.get("author", ""),
                    date=source.get("date", ""),
                    title=source.get("title", "")
                ),
                relevance_score=0.8
            )
            findings.append(asdict(finding))
        
        return json.dumps(findings)
    except Exception as e:
        return f"Error extracting key findings: {str(e)}"

@tool("synthesize_information")
def synthesize_information_tool(findings_json: str) -> str:
    """
    Synthesize information from key findings.
    
    Args:
        findings_json: JSON string of key findings
    """
    try:
        findings = json.loads(findings_json)
        
        # In a real implementation, this would use an LLM to synthesize
        # For now, we'll create a simple mock synthesis
        synthesis = f"Synthesis of {len(findings)} key findings:\n\n"
        
        for i, finding in enumerate(findings):
            content = finding.get("content", "No content")
            source = finding.get("source", {})
            url = source.get("url", "unknown source")
            
            synthesis += f"{i+1}. {content} [Source: {url}]\n"
        
        return synthesis
    except Exception as e:
        return f"Error synthesizing information: {str(e)}"

@tool("generate_report")
def generate_report_tool(synthesized_info: str, topic: str, output_format: str) -> str:
    """
    Generate a report from synthesized information.
    
    Args:
        synthesized_info: Synthesized information
        topic: The research topic
        output_format: The desired output format
    """
    try:
        # In a real implementation, this would use an LLM to generate the report
        # For now, we'll create a simple mock report
        
        if output_format == OutputFormat.CONCISE_REPORT:
            report = f"""
            # Concise Report: {topic}
            
            ## Introduction
            This report provides a concise overview of research on {topic}.
            
            ## Key Findings
            {synthesized_info}
            
            ## Conclusion
            Based on the research, {topic} shows significant importance in its field.
            
            ## References
            [List of references would be generated from the sources]
            """
        
        elif output_format == OutputFormat.EXECUTIVE_SUMMARY:
            report = f"""
            # Executive Summary: {topic}
            
            This executive summary provides a high-level overview of {topic}.
            
            {synthesized_info}
            
            In conclusion, {topic} demonstrates notable significance and warrants further attention.
            """
        
        elif output_format == OutputFormat.BULLET_POINTS:
            report = f"""
            # {topic}: Key Points
            
            {synthesized_info}
            """
        
        else:
            report = f"Invalid output format: {output_format}"
        
        return report
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Agent Nodes
def parse_request_node(state: ResearchAgentState) -> Dict[str, Any]:
    """Parse the user's research request."""
    messages = state["messages"]
    if not messages:
        return {"current_step": "error", "final_report": "No messages provided."}
    
    last_message = messages[-1].content
    
    llm = ChatLiteLLM(
        model="ollama/llama3.2",
        api_base="http://localhost:11434",
        temperature=0.3
    )
    
    extraction_prompt = f"""
    Extract research request details from this message: "{last_message}"
    
    Please identify:
    1. Research topic
    2. Research scope (web, scholarly, or all)
    3. Output format (concise_report, executive_summary, or bullet_points)
    
    If any information is missing, use these defaults:
    - Scope: all
    - Output format: concise_report
    
    Format your response as: topic|scope|format
    """
    
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    
    try:
        parts = response.content.strip().split('|')
        topic = parts[0].strip()
        
        scope = ResearchScope.ALL
        if len(parts) > 1 and parts[1].strip().lower() in [e.value for e in ResearchScope]:
            scope = ResearchScope(parts[1].strip().lower())
        
        output_format = OutputFormat.CONCISE_REPORT
        if len(parts) > 2 and parts[2].strip().lower() in [e.value for e in OutputFormat]:
            output_format = OutputFormat(parts[2].strip().lower())
        
        request = ResearchRequest(
            topic=topic,
            research_scope=scope,
            output_format=output_format
        )
        
        return {
            "request": request,
            "current_step": "request_parsed",
            "messages": [AIMessage(content=f"I'll research '{topic}' with scope '{scope}' and provide a {output_format.replace('_', ' ')}.")]
        }
    except:
        return {
            "current_step": "error",
            "messages": [AIMessage(content="I couldn't understand your research request. Please specify a clear topic.")]
        }

def identify_sources_node(state: ResearchAgentState) -> Dict[str, Any]:
    """Identify sources based on the research scope."""
    request = state["request"]
    topic = request.topic
    scope = request.research_scope
    
    web_sources = []
    scholarly_sources = []
    
    # Update the current step
    current_step = "sources_identified"
    
    # Identify web sources if scope is web or all
    if scope == ResearchScope.WEB or scope == ResearchScope.ALL:
        web_sources_json = identify_web_sources_tool.invoke({"topic": topic})
        try:
            web_sources = json.loads(web_sources_json)
        except:
            web_sources = []
    
    # Identify scholarly sources if scope is scholarly or all
    if scope == ResearchScope.SCHOLARLY or scope == ResearchScope.ALL:
        scholarly_sources_json = search_scholarly_papers_tool.invoke({"topic": topic})
        try:
            scholarly_sources = json.loads(scholarly_sources_json)
        except:
            scholarly_sources = []
    
    # Check if any sources were found
    if not web_sources and not scholarly_sources:
        return {
            "current_step": "error",
            "messages": [AIMessage(content=f"No relevant information found for '{topic}'.")]
        }
    
    # Prepare a message about the sources found
    source_message = f"I found {len(web_sources)} web sources and {len(scholarly_sources)} scholarly papers about '{topic}'. Now gathering content..."
    
    return {
        "web_sources": web_sources,
        "scholarly_sources": scholarly_sources,
        "current_step": current_step,
        "messages": [AIMessage(content=source_message)]
    }

def scrape_content_node(state: ResearchAgentState) -> Dict[str, Any]:
    """Scrape content from identified sources."""
    web_sources = state["web_sources"]
    scholarly_sources = state["scholarly_sources"]
    
    all_content = []
    
    # Scrape web content
    if web_sources:
        web_content_json = scrape_web_pages_tool.invoke({"urls_json": json.dumps(web_sources)})
        try:
            web_content = json.loads(web_content_json)
            all_content.extend(web_content)
        except:
            pass
    
    # Extract text from scholarly papers
    if scholarly_sources:
        papers_content_json = extract_text_from_papers_tool.invoke({"papers_json": json.dumps(scholarly_sources)})
        try:
            papers_content = json.loads(papers_content_json)
            all_content.extend(papers_content)
        except:
            pass
    
    # Check if any content was scraped
    if not all_content:
        return {
            "current_step": "error",
            "messages": [AIMessage(content="Failed to extract content from the identified sources.")]
        }
    
    # Convert dictionary content to ContentSource objects
    content_sources = []
    for content_dict in all_content:
        try:
            content_source = ContentSource(
                text=content_dict.get("text", ""),
                url=content_dict.get("url", ""),
                author=content_dict.get("author", None),
                date=content_dict.get("date", None),
                title=content_dict.get("title", None)
            )
            content_sources.append(content_source)
        except:
            continue
    
    return {
        "scraped_content": content_sources,
        "current_step": "content_scraped",
        "messages": [AIMessage(content=f"Successfully gathered content from {len(content_sources)} sources. Now extracting key findings...")]
    }

def extract_findings_node(state: ResearchAgentState) -> Dict[str, Any]:
    """Extract key findings from the scraped content."""
    request = state["request"]
    topic = request.topic
    content_sources = state["scraped_content"]
    
    # Convert ContentSource objects to dictionaries for the tool
    content_dicts = [asdict(source) for source in content_sources]
    
    # Extract key findings
    findings_json = extract_key_findings_tool.invoke({
        "content_json": json.dumps(content_dicts),
        "topic": topic
    })
    
    try:
        findings_dicts = json.loads(findings_json)
        
        # Convert dictionary findings to Finding objects
        findings = []
        for finding_dict in findings_dicts:
            try:
                source_dict = finding_dict.get("source", {})
                source = ContentSource(
                    text=source_dict.get("text", ""),
                    url=source_dict.get("url", ""),
                    author=source_dict.get("author", None),
                    date=source_dict.get("date", None),
                    title=source_dict.get("title", None)
                )
                
                finding = Finding(
                    content=finding_dict.get("content", ""),
                    source=source,
                    relevance_score=finding_dict.get("relevance_score", 1.0)
                )
                findings.append(finding)
            except:
                continue
        
        if not findings:
            return {
                "current_step": "error",
                "messages": [AIMessage(content=f"Could not extract key findings about '{topic}' from the sources.")]
            }
        
        return {
            "key_findings": findings,
            "current_step": "findings_extracted",
            "messages": [AIMessage(content=f"Extracted {len(findings)} key findings about '{topic}'. Now synthesizing information...")]
        }
    except:
        return {
            "current_step": "error",
            "messages": [AIMessage(content="Failed to extract key findings from the content.")]
        }

def synthesize_information_node(state: ResearchAgentState) -> Dict[str, Any]:
    """Synthesize information from key findings."""
    findings = state["key_findings"]
    
    # Convert Finding objects to dictionaries for the tool
    findings_dicts = [asdict(finding) for finding in findings]
    
    # Synthesize information
    synthesized_info = synthesize_information_tool.invoke({
        "findings_json": json.dumps(findings_dicts)
    })
    
    if not synthesized_info or "Error" in synthesized_info:
        return {
            "current_step": "error",
            "messages": [AIMessage(content="Failed to synthesize information from the findings.")]
        }
    
    return {
        "synthesized_information": synthesized_info,
        "current_step": "information_synthesized",
        "messages": [AIMessage(content="Information synthesized successfully. Generating final report...")]
    }

def generate_report_node(state: ResearchAgentState) -> Dict[str, Any]:
    """Generate the final report."""
    request = state["request"]
    topic = request.topic
    output_format = request.output_format
    synthesized_info = state["synthesized_information"]
    
    # Generate report
    report = generate_report_tool.invoke({
        "synthesized_info": synthesized_info,
        "topic": topic,
        "output_format": output_format.value
    })
    
    if not report or "Error" in report:
        return {
            "current_step": "error",
            "messages": [AIMessage(content="Failed to generate the final report.")]
        }
    
    return {
        "final_report": report,
        "current_step": "report_generated",
        "messages": [AIMessage(content=f"Here's your {output_format.value.replace('_', ' ')} on '{topic}':\n\n{report}")]
    }

# Routing Logic
def route_next_step(state: ResearchAgentState) -> str:
    """Route to the appropriate node based on current step."""
    current_step = state.get("current_step", "")
    
    if current_step == "error":
        return "end"
    elif current_step == "request_parsed":
        return "identify_sources"
    elif current_step == "sources_identified":
        return "scrape_content"
    elif current_step == "content_scraped":
        return "extract_findings"
    elif current_step == "findings_extracted":
        return "synthesize_information"
    elif current_step == "information_synthesized":
        return "generate_report"
    elif current_step == "report_generated":
        return "end"
    else:
        return "end"

# Create the Research Agent Graph
def create_research_agent():
    """Create the research agent with LangGraph."""
    
    # Create the graph
    workflow = StateGraph(ResearchAgentState)
    
    # Add nodes
    workflow.add_node("parse_request", parse_request_node)
    workflow.add_node("identify_sources", identify_sources_node)
    workflow.add_node("scrape_content", scrape_content_node)
    workflow.add_node("extract_findings", extract_findings_node)
    workflow.add_node("synthesize_information", synthesize_information_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # Set entry point
    workflow.set_entry_point("parse_request")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "parse_request",
        route_next_step,
        {
            "identify_sources": "identify_sources",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "identify_sources",
        route_next_step,
        {
            "scrape_content": "scrape_content",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "scrape_content",
        route_next_step,
        {
            "extract_findings": "extract_findings",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "extract_findings",
        route_next_step,
        {
            "synthesize_information": "synthesize_information",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "synthesize_information",
        route_next_step,
        {
            "generate_report": "generate_report",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_report",
        route_next_step,
        {
            "end": END
        }
    )
    
    return workflow.compile()

def main():
    """Main function to run the research agent."""
    print("📚 Welcome to the Research & Summarization Agent!")
    print("=" * 60)
    print("I can help you research topics by gathering information from:")
    print("🌐 Web sources")
    print("📖 Scholarly papers")
    print("\nI can provide results as:")
    print("📝 Concise reports")
    print("📊 Executive summaries")
    print("🔍 Bullet point lists")
    print("\nType 'exit' to quit, 'help' for more information.")
    print("=" * 60)
    
    # Create the agent
    agent = create_research_agent()
    
    # Conversation loop
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if user_input.lower() == 'exit':
            print("\n👋 Thanks for using the Research Agent!")
            break
        
        if user_input.lower() == 'help':
            print("\n📋 Example requests:")
            print("• 'Research the impact of artificial intelligence on healthcare'")
            print("• 'Find scholarly articles about climate change and provide an executive summary'")
            print("• 'Give me bullet points about renewable energy from web sources'")
            continue
        
        if not user_input:
            continue
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "request": None,
            "web_sources": [],
            "scholarly_sources": [],
            "scraped_content": [],
            "key_findings": [],
            "synthesized_information": "",
            "final_report": "",
            "current_step": ""
        }
        
        try:
            # Run the agent
            result = agent.invoke(initial_state)
            
            # Get the response
            if result["messages"]:
                response = result["messages"][-1].content
                print(f"\n🤖 Agent: {response}")
            else:
                print("\n🤖 Agent: I'm sorry, I couldn't process your request.")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for assistance.")

if __name__ == "__main__":
    main()