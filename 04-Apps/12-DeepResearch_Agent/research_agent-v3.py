import os
import json
from typing import List, Dict, Annotated
from langgraph.graph import StateGraph
from pydantic import BaseModel
import networkx as nx
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import BraveSearch
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from bs4 import BeautifulSoup
import logging
import re  # For simple parsing in KG
from langchain_ollama import ChatOllama
from rich import print
# Set up logging
logging.basicConfig(level=logging.INFO)

# LLM Setup (using OpenAI; replace with other providers if needed)
# llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOllama(model="gemma3:12b", base_url='100.95.122.242:11434')

# Brave Search Setup using LangChain
brave_tool = BraveSearch.from_api_key(
    api_key=os.getenv("BRAVE_API_KEY"),
    search_kwargs={"count": 5}
)

# State Definition
class ResearchState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    topic: str
    questions: List[str] = []
    queries: List[str] = []
    results: Dict[str, List[Dict]] = {}  # query -> list of {url, title, description, content, credibility}
    knowledge_graph: nx.Graph = nx.Graph()
    synthesized_info: str = ""
    gaps: List[str] = []
    report: str = ""
    iterations: int = 0
    max_iterations: int = 5
    report_length: int = 1000
    acceptable_sources: List[str] = [".org", ".edu", ".gov"]
    usage_counter: int = 0  # Track API calls for cost management

# Node Functions

def formulate_questions(state: ResearchState) -> dict:
    prompt = PromptTemplate.from_template("Generate 5 sub-questions for the research topic: {topic}")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"topic": state.topic})
    questions = [q.strip() for q in response.split("\n") if q.strip()]
    logging.info(f"Formulated questions: {questions}")
    return {"questions": questions}

def generate_queries(state: ResearchState) -> dict:
    prompt = PromptTemplate.from_template("Generate a precise web search query for the question: {question}")
    chain = prompt | llm | StrOutputParser()
    queries = [chain.invoke({"question": q}) for q in state.questions]
    logging.info(f"Generated queries: {queries}")
    return {"queries": queries}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@sleep_and_retry
@limits(calls=1, period=1)  # 1 call per second
def search_brave(state: ResearchState) -> dict:
    if state.usage_counter >= 900:  # Near free tier limit (1000/month)
        logging.warning("Approaching Brave API limit; halting searches.")
        return {}

    results = {}
    usage_counter = state.usage_counter
    for query in state.queries:
        try:
            results_str = brave_tool.run(query)
            # Parse the string as JSON
            parsed_results = json.loads(results_str)
            # Filter and map keys for consistency
            filtered = [
                {
                    "url": r['link'],
                    "title": r['title'],
                    "description": r.get('snippet', '')
                }
                for r in parsed_results if any(s in r['link'] for s in state.acceptable_sources)
            ]
            results[query] = filtered
            usage_counter += 1
            logging.info(f"Search results for '{query}': {len(filtered)} filtered sources")
        except Exception as e:
            logging.error(f"Error searching '{query}': {e}")
    return {"results": results, "usage_counter": usage_counter}

def extract_content(state: ResearchState) -> dict:
    results = state.results.copy()  # Copy to avoid mutating original
    for query, res_list in results.items():
        for res in res_list:
            try:
                response = requests.get(res['url'], timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                res['content'] = ' '.join(soup.get_text().split())[:2000]  # Truncate and clean
                logging.info(f"Extracted content from {res['url']}")
            except Exception as e:
                res['content'] = f"Error fetching content: {str(e)}"
                logging.error(f"Error extracting from {res['url']}: {e}")
    return {"results": results}

def evaluate_credibility(state: ResearchState) -> dict:
    prompt = PromptTemplate.from_template(
        "Rate the credibility of the source {url} on a scale of 1-10 based on domain authority, recency, and content quality. "
        "Provide only the number: {content_snippet}"
    )
    chain = prompt | llm | StrOutputParser()
    results = state.results.copy()  # Copy to avoid mutating original
    for res_list in results.values():
        for res in res_list:
            if 'content' in res:
                try:
                    score = int(chain.invoke({"url": res['url'], "content_snippet": res['content'][:500]}))
                    res['credibility'] = score
                except ValueError:
                    res['credibility'] = 0
                logging.info(f"Credibility for {res['url']}: {res.get('credibility')}")
    return {"results": results}

def synthesize(state: ResearchState) -> dict:
    # Build Knowledge Graph (simple entity-relation extraction)
    extract_prompt = PromptTemplate.from_template(
        "Extract key entities and relations from this text in the format: Entity1 - Relation - Entity2\nText: {content}"
    )
    extract_chain = extract_prompt | llm | StrOutputParser()

    knowledge_graph = state.knowledge_graph.copy()  # Copy to avoid mutating original directly
    for res_list in state.results.values():
        for res in res_list:
            if res.get('credibility', 0) > 7 and 'content' in res:  # Threshold for inclusion
                extractions = extract_chain.invoke({"content": res['content'][:1000]})
                for line in extractions.split("\n"):
                    match = re.match(r"(.+?)\s*-\s*(.+?)\s*-\s*(.+)", line.strip())
                    if match:
                        ent1, rel, ent2 = match.groups()
                        knowledge_graph.add_edge(ent1.strip(), ent2.strip(), relation=rel.strip(), source=res['url'])

    # Synthesize info, detect gaps/contradictions
    all_content = "\n".join([res.get('content', '') for res_list in state.results.values() for res in res_list])
    synth_prompt = PromptTemplate.from_template(
        "Synthesize the following information objectively, highlighting key findings, contradictions, and knowledge gaps: {content}"
    )
    synth_chain = synth_prompt | llm | StrOutputParser()
    synthesized_info = synth_chain.invoke({"content": all_content[:5000]})  # Limit input size

    # Simple gap extraction (placeholder; improve with better parsing)
    gaps_prompt = PromptTemplate.from_template("List knowledge gaps from: {synth}")
    gaps_chain = gaps_prompt | llm | StrOutputParser()
    gaps_response = gaps_chain.invoke({"synth": synthesized_info})
    gaps = [g.strip() for g in gaps_response.split("\n") if g.strip()]

    logging.info(f"Synthesized info length: {len(synthesized_info)}, Gaps: {gaps}")

    updates = {
        "knowledge_graph": knowledge_graph,
        "synthesized_info": synthesized_info,
        "gaps": gaps
    }

    if gaps and state.iterations < state.max_iterations:
        updates["iterations"] = state.iterations + 1
        logging.info(f"Iteration {updates['iterations']}: Refining based on gaps")
        # Refine queries based on gaps (simple: append gap as new query)
        updates["queries"] = gaps  # Loop back by re-running search, etc.

    return updates

def generate_report(state: ResearchState) -> dict:
    prompt = PromptTemplate.from_template(
        "Write a comprehensive research report of approximately {length} words on the topic '{topic}'.\n"
        "Structure:\n- Executive Summary: Key findings from {synth}\n"
        "- Supporting Evidence: Cite sources with URLs and credibility scores\n{evidence}\n"
        "- Limitations and Contradictions: {gaps}\n"
        "- Avenues for Further Investigation: Suggest based on gaps\n"
        "Ensure objectivity, include markdown citations like [Source](url), and handle any contradictory information."
    )
    chain = prompt | llm | StrOutputParser()

    evidence = "\n".join([
        f"- [{res['title']}]({res['url']}) (Credibility: {res.get('credibility', 'N/A')}): {res.get('description', '')[:200]}"
        for res_list in state.results.values() for res in res_list
    ])
    gaps_str = ", ".join(state.gaps)

    report = chain.invoke({
        "length": state.report_length,
        "topic": state.topic,
        "synth": state.synthesized_info,
        "evidence": evidence,
        "gaps": gaps_str
    })
    logging.info("Report generated.")
    return {"report": report}

# Build the Graph
workflow = StateGraph(state_schema=ResearchState)

workflow.add_node("formulate_questions", formulate_questions)
workflow.add_node("generate_queries", generate_queries)
workflow.add_node("search_brave", search_brave)
workflow.add_node("extract_content", extract_content)
workflow.add_node("evaluate_credibility", evaluate_credibility)
workflow.add_node("synthesize", synthesize)
workflow.add_node("generate_report", generate_report)

# Edges
workflow.add_edge("formulate_questions", "generate_queries")
workflow.add_edge("generate_queries", "search_brave")
workflow.add_edge("search_brave", "extract_content")
workflow.add_edge("extract_content", "evaluate_credibility")
workflow.add_edge("evaluate_credibility", "synthesize")

# For refinement loop: Add conditional edge from synthesize
from langgraph.graph import END

def should_refine(state: ResearchState):
    if state.gaps and state.iterations < state.max_iterations:
        return "generate_queries"  # Loop back to generate new queries
    return "generate_report"  # Proceed to report

workflow.add_conditional_edges("synthesize", should_refine)

# Entry and finish
workflow.set_entry_point("formulate_questions")
workflow.set_finish_point("generate_report")

# Compile
graph = workflow.compile()

# Example Invocation
if __name__ == "__main__":
    initial_state = ResearchState(
        topic="Impact of AI on climate change",
        report_length=800,
        acceptable_sources=[".edu", ".gov", ".org"],
        max_iterations=2
    )
    output = graph.invoke(initial_state)
    print("\nGenerated Report:\n")
    print(output)
    # Optionally, print knowledge graph nodes
    print("\nKnowledge Graph Nodes:", list(output.knowledge_graph.nodes))
