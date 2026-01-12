#!/usr/bin/env python3
"""
Agentic RAG Graph - Core Logic

This module contains all the core components for building an agentic RAG system:
1. Document ingestion and processing
2. Vector store and retriever tool
3. LangGraph state machine with intelligent routing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.tools import tool

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode


# =============================================================================
# SECTION 1: Document Loading and Processing
# =============================================================================

def load_url_docs(urls: List[str]) -> List[Document]:
    """
    Load documents from a list of URLs.

    Args:
        urls: List of web URLs to scrape

    Returns:
        List of Document objects containing the scraped content
    """
    docs_nested = [WebBaseLoader(u).load() for u in urls]
    return [d for sub in docs_nested for d in sub]


def load_path_docs(paths: List[str]) -> List[Document]:
    """
    Load local text and markdown files from provided paths.

    Args:
        paths: List of file paths or directories to load

    Returns:
        List of Document objects containing the file contents
    """
    out: List[Document] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            # Recursively find all supported text files
            for fp in path.rglob("*"):
                if fp.suffix.lower() in {".txt", ".md"}:
                    out.extend(TextLoader(str(fp), encoding="utf-8").load())
        elif path.is_file():
            out.extend(TextLoader(str(path), encoding="utf-8").load())
    return out


def split_docs(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.

    Args:
        docs: List of documents to split
        chunk_size: Maximum size of each chunk (in tokens)
        chunk_overlap: Number of overlapping tokens between chunks

    Returns:
        List of chunked documents
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# =============================================================================
# SECTION 2: Retriever Tool Setup
# =============================================================================

def build_retriever_tool(doc_splits: List[Document], k: int = 4):
    """
    Create a vector store and retriever tool for semantic search.

    Args:
        doc_splits: List of document chunks to index
        k: Number of top results to return

    Returns:
        Tuple of (retriever_tool, vectorstore)
    """
    # Initialize vector store with Ollama embeddings
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url=os.getenv("OLLAMA_BASE_URL")
        ),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    @tool
    def retrieve(query: str) -> str:
        """Semantic search over indexed documents. Returns relevant text snippets with sources."""
        docs = retriever.invoke(query)
        parts = []
        for d in docs:
            src = d.metadata.get("source") or d.metadata.get("url") or "unknown"
            parts.append(f"SOURCE: {src}\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    return retrieve, vectorstore


# =============================================================================
# SECTION 3: Graph Components - Models and Prompts
# =============================================================================

class GradeDocuments(BaseModel):
    """Schema for grading retrieved document relevance."""
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if context is relevant, else 'no'"
    )


# System prompts for different stages
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


# =============================================================================
# SECTION 4: Graph Building - Nodes and Routing
# =============================================================================

def latest_user_question(messages: List[BaseMessage]) -> str:
    """Extract the most recent user question from message history."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return messages[0].content if messages else ""


def build_graph(retriever_tool):
    """
    Build the agentic RAG graph with intelligent routing.

    The graph follows this flow:
    1. LLM decides whether to use retriever or answer directly
    2. If retrieved, grade the relevance of context
    3. If irrelevant, rewrite question and retry
    4. If relevant, generate final answer

    Args:
        retriever_tool: The retriever tool function

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize models
    chat_model = ChatOllama(
        model='gpt-oss:20b',
        base_url=os.getenv("OLLAMA_BASE_URL"),
        temperature=0.1
    )
    grader_model = ChatOllama(
        model='llama3.2',
        temperature=0
    )

    # -------------------------------------------------------------------------
    # Node 1: Initial Query Processing
    # -------------------------------------------------------------------------
    def generate_query_or_respond(state: MessagesState):
        """LLM decides: answer directly OR call the retriever tool."""
        response = chat_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    # -------------------------------------------------------------------------
    # Node 2: Grade Retrieved Documents
    # -------------------------------------------------------------------------
    def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Route based on whether retrieved context is relevant."""
        question = latest_user_question(state["messages"])
        context = state["messages"][-1].content
        prompt = GRADE_PROMPT.format(question=question, context=context)

        verdict = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        return "generate_answer" if verdict.binary_score == "yes" else "rewrite_question"

    # -------------------------------------------------------------------------
    # Node 3: Rewrite Question
    # -------------------------------------------------------------------------
    def rewrite_question(state: MessagesState):
        """Rewrite the question to improve retrieval quality."""
        question = latest_user_question(state["messages"])
        prompt = REWRITE_PROMPT.format(question=question)
        rewritten = chat_model.invoke([{"role": "user", "content": prompt}]).content
        return {"messages": [HumanMessage(content=rewritten)]}

    # -------------------------------------------------------------------------
    # Node 4: Generate Final Answer
    # -------------------------------------------------------------------------
    def generate_answer(state: MessagesState):
        """Generate the final grounded answer using retrieved context."""
        question = latest_user_question(state["messages"])
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = chat_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    # -------------------------------------------------------------------------
    # Node 5: Fallback Retrieval
    # -------------------------------------------------------------------------
    def fallback_retrieve(state: MessagesState):
        """Retrieve docs directly when model doesn't call tool."""
        question = latest_user_question(state["messages"])
        context = retriever_tool.invoke({"query": question})
        return {
            "messages": [ToolMessage(content=context, tool_call_id="retriever_fallback")]
        }

    # -------------------------------------------------------------------------
    # Routing Logic
    # -------------------------------------------------------------------------
    def route_after_generate(state: MessagesState) -> Literal["retrieve", "fallback_retrieve"]:
        """Determine if LLM made a tool call or needs fallback."""
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        return "retrieve" if tool_calls else "fallback_retrieve"

    # -------------------------------------------------------------------------
    # Build the Graph
    # -------------------------------------------------------------------------
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("fallback_retrieve", fallback_retrieve)
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    # Add edges
    workflow.add_edge(START, "generate_query_or_respond")

    workflow.add_conditional_edges(
        "generate_query_or_respond",
        route_after_generate,
        {"retrieve": "retrieve", "fallback_retrieve": "fallback_retrieve"},
    )

    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_conditional_edges("fallback_retrieve", grade_documents)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
