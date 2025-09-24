# research_agent.py
# This script implements a LangGraph-based research agent using Brave Search API for web searches.
# It autonomously conducts research on a given topic, formulates questions, searches, synthesizes information,
# evaluates credibility, handles gaps, and generates a structured report.
# Incorporates a simple knowledge graph using NetworkX for concept relationships and spaCy for semantic analysis.

# Required packages and versions (for reproducibility as of 2025-07-21):
# - langgraph==0.2.5 (latest stable as per recent checks)
# - langchain==0.3.0
# - langchain-core==0.3.0
# - requests==2.32.3
# - networkx==3.3 (for knowledge graph)
# - spacy==3.7.5 (for semantic analysis)
# - en-core-web-sm==3.7.1 (spaCy model, install via: python -m spacy download en_core_web_sm)
# - beautifulsoup4==4.12.3 (for web page extraction)
# - python-dotenv==1.0.1 (for loading API keys from .env)

# Note: You need a Brave Search API key. Sign up at https://api.search.brave.com/ and set it in a .env file as BRAVE_API_KEY=your_key
# Rate limiting: Brave API has limits (e.g., 1000 queries/month free tier). This script implements basic rate limiting with time.sleep.
# Cost management: Tracks API calls and stops if a configurable limit is reached.

import os
import time
import requests
from bs4 import BeautifulSoup
import networkx as nx
import spacy
from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
# from langchain_openai import ChatOllama  # Assuming OpenAI LLM; replace with your preferred LLM
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()

# Load spaCy model for semantic analysis
nlp = spacy.load("en_core_web_sm")

# Configuration class
class ResearchConfig:
    def __init__(self, topic: str, report_length: str = "medium", acceptable_sources: List[str] = ["all"],
                 max_api_calls: int = 10, llm_model: str = "llama3.2"):
        self.topic = topic
        self.report_length = report_length  # short, medium, long (affects depth)
        self.acceptable_sources = acceptable_sources  # e.g., ["edu", "gov", "org"] or "all"
        self.max_api_calls = max_api_calls
        self.llm = ChatOllama(model=llm_model, temperature=0.3)  # Low temperature for objectivity
        self.api_calls = 0

# State definition for LangGraph
class ResearchState(TypedDict):
    config: ResearchConfig
    research_questions: List[str]
    search_queries: List[str]
    retrieved_data: Dict[str, List[Dict[str, Any]]]  # query -> list of {url, content, credibility}
    knowledge_graph: nx.Graph
    synthesized_info: str
    gaps: List[str]
    report: str
    api_calls: int
    errors: List[str]

# Helper functions

def brave_search(query: str, config: ResearchConfig) -> List[Dict[str, Any]]:
    """Perform search using Brave API with rate limiting and error handling."""
    if config.api_calls >= config.max_api_calls:
        raise ValueError("API call limit reached for cost management.")
    
    time.sleep(1)  # Basic rate limiting: 1 second delay between calls
    
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": os.getenv("BRAVE_API_KEY")}
    params = {"q": query, "count": 10, "safesearch": "strict"}  # Adjust as needed
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        config.api_calls += 1
        results = response.json().get("web", {}).get("results", [])
        return [{"url": res["url"], "title": res["title"], "description": res["description"]} for res in results]
    except requests.RequestException as e:
        return [{"error": str(e)}]

def extract_content(url: str) -> str:
    """Extract main content from a web page using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Simple extraction: get paragraphs
        paragraphs = [p.text for p in soup.find_all("p")]
        return " ".join(paragraphs)[:2000]  # Limit to 2000 chars for efficiency
    except Exception as e:
        return f"Error extracting content: {str(e)}"

def evaluate_credibility(url: str, content: str, config: ResearchConfig) -> float:
    """Simple credibility score: check domain, length, keywords. Returns 0-1 score."""
    domain = url.split("//")[-1].split("/")[0].split(".")[-1]
    if config.acceptable_sources != ["all"] and domain not in config.acceptable_sources:
        return 0.0
    score = 1.0 if len(content) > 500 else 0.5
    if "reputable" in content.lower() or "study" in content.lower():  # Placeholder; improve with LLM
        score += 0.2
    return min(score, 1.0)

def build_knowledge_graph(content: str, graph: nx.Graph) -> nx.Graph:
    """Use spaCy to extract entities and relations for knowledge graph."""
    doc = nlp(content)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]]
    for i, ent1 in enumerate(entities):
        for ent2 in entities[i+1:]:
            if ent1 != ent2:  # Simple co-occurrence relation
                graph.add_edge(ent1, ent2, weight=1)
    return graph

# LangGraph nodes

def formulate_questions(state: ResearchState) -> ResearchState:
    """Formulate initial research questions based on topic."""
    prompt = PromptTemplate.from_template("Given the topic '{topic}', generate 5 key research questions.")
    questions = state["config"].llm.invoke(prompt.format(topic=state["config"].topic)).content.split("\n")
    state["research_questions"] = [q.strip() for q in questions if q.strip()]
    return state

def generate_queries(state: ResearchState) -> ResearchState:
    """Generate search queries from questions."""
    queries = []
    for q in state["research_questions"]:
        prompt = PromptTemplate.from_template("Generate a precise web search query for: {question}")
        query = state["config"].llm.invoke(prompt.format(question=q)).content
        queries.append(query)
    state["search_queries"] = queries
    return state

def perform_searches(state: ResearchState) -> ResearchState:
    """Search using Brave and extract content."""
    data = {}
    graph = nx.Graph()
    for query in state["search_queries"]:
        results = brave_search(query, state["config"])
        processed = []
        for res in results:
            if "error" in res:
                state["errors"].append(res["error"])
                continue
            content = extract_content(res["url"])
            credibility = evaluate_credibility(res["url"], content, state["config"])
            if credibility > 0.5:  # Threshold for inclusion
                processed.append({"url": res["url"], "content": content, "credibility": credibility})
                graph = build_knowledge_graph(content, graph)
        data[query] = processed
    state["retrieved_data"] = data
    state["knowledge_graph"] = graph
    state["api_calls"] = state["config"].api_calls
    return state

def synthesize_information(state: ResearchState) -> ResearchState:
    """Synthesize info, handle contradictions, use knowledge graph."""
    all_content = "\n".join([item["content"] for query_data in state["retrieved_data"].values() for item in query_data])
    # Semantic analysis with spaCy
    doc = nlp(all_content[:5000])  # Limit for performance
    summary = "Key entities: " + ", ".join([ent.text for ent in doc.ents[:10]])
    
    prompt = PromptTemplate.from_template(
        "Synthesize the following information objectively, handling contradictions: {content}\nKnowledge graph insights: {graph}"
    )
    graph_insights = f"Top connections: {list(state['knowledge_graph'].edges(data=True))[:5]}"
    synthesized = state["config"].llm.invoke(prompt.format(content=all_content[:10000], graph=graph_insights)).content
    state["synthesized_info"] = synthesized + "\n" + summary
    return state

def identify_gaps(state: ResearchState) -> ResearchState:
    """Identify knowledge gaps and refine if needed."""
    prompt = PromptTemplate.from_template("Identify gaps or contradictions in: {info}")
    gaps = state["config"].llm.invoke(prompt.format(info=state["synthesized_info"])).content.split("\n")
    state["gaps"] = [g.strip() for g in gaps if g.strip()]
    
    # Refine: If gaps, add new questions (simple loop simulation)
    if state["gaps"]:
        new_questions = state["gaps"][:2]  # Limit to 2 for brevity
        state["research_questions"].extend(new_questions)
    return state

def generate_report(state: ResearchState) -> ResearchState:
    """Produce the final report."""
    length_map = {"short": 500, "medium": 1500, "long": 3000}
    max_len = length_map.get(state["config"].report_length, 1500)
    
    citations = [item["url"] for query_data in state["retrieved_data"].values() for item in query_data]
    prompt = PromptTemplate.from_template(
        "Generate a research report on {topic}. Summary: {summary}\nEvidence: {evidence}\nLimitations: {gaps}\nFurther investigation: {further}\nCitations: {citations}\nKeep under {max_len} words."
    )
    further = ", ".join(state["gaps"])  # Use gaps for further avenues
    report = state["config"].llm.invoke(prompt.format(
        topic=state["config"].topic, summary=state["synthesized_info"][:max_len//2],
        evidence=state["synthesized_info"], gaps=", ".join(state["gaps"]),
        further=further, citations=", ".join(citations), max_len=max_len
    )).content
    state["report"] = report
    return state

# Build the graph
workflow = StateGraph(state_schema=ResearchState)

workflow.add_node("formulate_questions", formulate_questions)
workflow.add_node("generate_queries", generate_queries)
workflow.add_node("perform_searches", perform_searches)
workflow.add_node("synthesize_information", synthesize_information)
workflow.add_node("identify_gaps", identify_gaps)
workflow.add_node("generate_report", generate_report)

# Edges (simple linear with refinement loop)
workflow.add_edge("formulate_questions", "generate_queries")
workflow.add_edge("generate_queries", "perform_searches")
workflow.add_edge("perform_searches", "synthesize_information")
workflow.add_edge("synthesize_information", "identify_gaps")
workflow.add_conditional_edges(
    "identify_gaps",
    lambda state: "generate_queries" if state["gaps"] else "generate_report"  # Loop if gaps
)
workflow.add_edge("generate_report", "__end__")

# Set entry point
workflow.set_entry_point("formulate_questions")

# Compile the graph
app = workflow.compile()

# Function to run the agent
def run_research_agent(config: ResearchConfig) -> str:
    initial_state = {
        "config": config,
        "research_questions": [],
        "search_queries": [],
        "retrieved_data": {},
        "knowledge_graph": nx.Graph(),
        "synthesized_info": "",
        "gaps": [],
        "report": "",
        "api_calls": 0,
        "errors": []
    }
    final_state = app.invoke(initial_state)
    if final_state["errors"]:
        print("Errors encountered:", final_state["errors"])
    return final_state["report"]

# Example usage
if __name__ == "__main__":
    config = ResearchConfig(topic="Impact of AI on climate change", report_length="medium", acceptable_sources=["edu", "gov"])
    report = run_research_agent(config)
    print(report)