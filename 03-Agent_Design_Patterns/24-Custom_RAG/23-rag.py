#!/usr/bin/env python3
"""
Agentic RAG (LangGraph) in a single, teachable script.

Features (mirrors the LangGraph "agentic-rag" tutorial):
- Load docs (URLs or local .txt/.md)
- Chunk + embed into an in-memory vector store
- LLM decides: answer directly OR call a retriever tool
- If retrieved context seems irrelevant, rewrite the question and retry
- Otherwise, generate a grounded answer
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from rich import print
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
import base64
from IPython.display import HTML, display
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

load_dotenv()

#%%
################################ Data Ingestion ################################

DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# Set a default user agent for web scraping
os.environ.setdefault("USER_AGENT", "agentic-rag-demo/1.0 (contact: you@example.com)")

def load_url_docs(urls: List[str]) -> List[Document]:
    # Fetch content from a list of URLs
    docs_nested = [WebBaseLoader(u).load() for u in urls]
    return [d for sub in docs_nested for d in sub] # Flatten nested list of documents


def load_path_docs(paths: List[str]) -> List[Document]:
    # Load local text and markdown files from provided paths
    out: List[Document] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            # Recursively find all supported text files in the directory
            for fp in path.rglob("*"):
                if fp.suffix.lower() in {".txt", ".md"}:
                    out.extend(TextLoader(str(fp), encoding="utf-8").load())
        elif path.is_file():
            out.extend(TextLoader(str(path), encoding="utf-8").load())
    return out


def split_docs(
    docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 100
) -> List[Document]:
    # Break large documents into smaller, overlapping chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

#%%
################################ Retriever Tool ################################

def build_retriever_tool(doc_splits: List[Document], k: int = 4):
    # Initialize a vector store in memory using Ollama embeddings
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text:latest", base_url=os.getenv("OLLAMA_BASE_URL")),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    @tool
    def retrieve(query: str) -> str:
        """Semantic search over the indexed documents. Returns relevant text snippets with sources."""
        docs = retriever.invoke(query)
        parts = []
        # Format the retrieved chunks for the LLM to process
        for d in docs:
            src = d.metadata.get("source") or d.metadata.get("url") or "unknown"
            parts.append(f"SOURCE: {src}\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    return retrieve, vectorstore

#%%
################################ Graph Nodes ################################

def latest_user_question(messages: List[BaseMessage]) -> str:
    # Identify the most recent question asked by the user in the thread
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return messages[0].content if messages else ""


class GradeDocuments(BaseModel):
    """Binary relevance grade for retrieved context."""
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if context is relevant, else 'no'"
    )


# Define system prompts for different stages of the agentic flow
GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved context to a user question.\n"
    "Question:\n{question}\n\n"
    "Retrieved context:\n{context}\n\n"
    "If the context contains information that would help answer the question, respond with 'yes'. "
    "Otherwise respond with 'no'."
)

REWRITE_PROMPT = (
    "Rewrite the question to improve retrieval.\n"
    "Original question:\n{question}\n\n"
    "Rewritten question (be specific, include key terms):"
)

GENERATE_PROMPT = (
    "You are a helpful assistant for question-answering.\n"
    "Use the retrieved context to answer the question.\n"
    "If the answer is not in the context, say you don't know.\n"
    "Keep the answer concise (max 3 sentences).\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}"
)


def build_graph(retriever_tool):
    # Initialize chat and grader models with deterministic settings
    chat_model =  ChatOllama(model='gpt-oss:20b' , base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0.1)
    grader_model = ChatOllama(model='llama3.2' , temperature=0)

    def generate_query_or_respond(state: MessagesState):
        """LLM decides: answer directly, OR call the retriever tool."""
        # Enable tool calling capabilities for the main assistant
        response = chat_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Route based on whether retrieved context seems relevant."""
        question = latest_user_question(state["messages"])
        context = state["messages"][-1].content  # Extract content from tool output
        prompt = GRADE_PROMPT.format(question=question, context=context)
        # Evaluate context quality using structured output
        verdict = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        return "generate_answer" if verdict.binary_score == "yes" else "rewrite_question"

    def rewrite_question(state: MessagesState):
        """Ask the model to produce a better retrieval query."""
        question = latest_user_question(state["messages"])
        prompt = REWRITE_PROMPT.format(question=question)
        rewritten = chat_model.invoke([{"role": "user", "content": prompt}]).content
        return {"messages": [HumanMessage(content=rewritten)]}

    def generate_answer(state: MessagesState):
        """Produce the final grounded answer."""
        question = latest_user_question(state["messages"])
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = chat_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    def fallback_retrieve(state: MessagesState):
        """Retrieve docs directly when the model skips tool calling."""
        question = latest_user_question(state["messages"])
        context = retriever_tool.invoke({"query": question})
        return {
            "messages": [ToolMessage(content=context, tool_call_id="retriever_fallback")]
        }

    def route_after_generate(state: MessagesState) -> Literal["retrieve", "fallback_retrieve"]:
        # Check if the model generated tool calls or a direct response
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        return "retrieve" if tool_calls else "fallback_retrieve"

    # Define the state machine structure
    workflow = StateGraph(MessagesState)

    # Register processing nodes
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("fallback_retrieve", fallback_retrieve)
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    # Determine if a tool call was made or if we should fall back
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        route_after_generate,
        {"retrieve": "retrieve", "fallback_retrieve": "fallback_retrieve"},
    )

    # Route logic based on context relevance after retrieval
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_conditional_edges("fallback_retrieve", grade_documents)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()

#%%
################################ Simple CLI App ################################

def parse_args():
    # Configure command line arguments for flexible usage
    p = argparse.ArgumentParser()
    p.add_argument("--urls", nargs="*", default=None, help="URLs to index")
    p.add_argument("--paths", nargs="*", default=None, help="Local .txt/.md files or directories to index")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--k", type=int, default=4, help="Top-K retrieved chunks")
    p.add_argument("--stream", action="store_true", help="Print per-node updates like the tutorial")
    
    return p.parse_args()



def draw_mermaid_png(agent, height="100vh", width="min(90vw, 500px)"):
    """Render a Mermaid graph using the Mermaid.ink API for notebook display.

    The diagram keeps its aspect ratio, shrinks if it would exceed the viewport
    height, and otherwise stays readable without manual tuning.
    """
    png_bytes = agent.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    
    b64_img = base64.b64encode(png_bytes).decode()
    
    # save to file for debugging
    with open("graph.png", "wb") as f:
        f.write(png_bytes)

def main():
    args = parse_args()

    # Determine which sources to load based on arguments
    urls = args.urls if args.urls else DEFAULT_URLS
    path_docs = load_path_docs(args.paths) if args.paths else []
    url_docs = load_url_docs(urls) if urls else []

    docs = url_docs + path_docs
    if not docs:
        print("No documents loaded. Provide --urls and/or --paths.", file=sys.stderr)
        sys.exit(1)

    # Process documents and initialize the graph
    doc_splits = split_docs(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever_tool, vectorstore = build_retriever_tool(doc_splits, k=args.k)
    graph = build_graph(retriever_tool)

    draw_mermaid_png(graph) 

    messages: List[BaseMessage] = []
    print("Agentic RAG ready. Type '/exit' to quit.\n")
    print("Type @/path/to/file.txt to index a file dynamically.\n")

    # Start the interactive chat loop
    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if not user_text:
            continue

        # Check for dynamic file indexing syntax
        if file_matches := re.findall(r"@(\S+)", user_text):
            try:
                if new_docs := load_path_docs(file_matches):
                    new_splits = split_docs(new_docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                    vectorstore.add_documents(new_splits) # Update the existing vector store
                    print(f"\n[System] Indexed {len(file_matches)} file(s) ({len(new_splits)} chunks).")
                else:
                    print(f"\n[System] Warning: No readable content found in {file_matches}")
            except Exception as e:
                print(f"\n[System] Error loading files: {e}")

        messages.append(HumanMessage(content=user_text))

        # Handle streaming vs batch execution
        if args.stream:
            last_state = None
            # Iterate through graph updates as they happen
            for chunk in graph.stream({"messages": messages}):
                # Extract and print updates from each node as they execute
                for node, update in chunk.items():
                    last = update["messages"][-1]
                    print(f"\n--- update from {node} ---")
                    last.pretty_print()
                    last_state = update
            if last_state:
                messages = last_state["messages"] # Sync message history
                print()
        else:
            # Execute the full graph logic and retrieve final state
            out = graph.invoke({"messages": messages})
            messages = out["messages"]
            print()
            messages[-1].pretty_print()
            print()

    print("Bye.")


if __name__ == "__main__":
    main()
