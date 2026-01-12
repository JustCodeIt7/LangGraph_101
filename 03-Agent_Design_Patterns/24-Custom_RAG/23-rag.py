#!/usr/bin/env python3
"""
Agentic RAG (LangGraph) in a single, teachable script.

Features (mirrors the LangGraph "agentic-rag" tutorial):
- Load docs (URLs or local .txt/.md)
- Chunk + embed into an in-memory vector store
- LLM decides: answer directly OR call a retriever tool
- If retrieved context seems irrelevant, rewrite the question and retry
- Otherwise, generate a grounded answer

Run:
  pip install -U langgraph "langchain[openai]" langchain-community langchain-text-splitters bs4
  export OPENAI_API_KEY=...
  python agentic_rag.py --stream

Optional:
  python agentic_rag.py --urls https://example.com/a https://example.com/b
  python agentic_rag.py --paths ./docs ./more_docs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------- Data ingestion ----------------------------

DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

os.environ.setdefault("USER_AGENT", "agentic-rag-demo/1.0 (contact: you@example.com)")
def load_url_docs(urls: List[str]) -> List[Document]:
    docs_nested = [WebBaseLoader(u).load() for u in urls]
    return [d for sub in docs_nested for d in sub]


def load_path_docs(paths: List[str]) -> List[Document]:
    out: List[Document] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for fp in path.rglob("*"):
                if fp.suffix.lower() in {".txt", ".md"}:
                    out.extend(TextLoader(str(fp), encoding="utf-8").load())
        elif path.is_file():
            out.extend(TextLoader(str(path), encoding="utf-8").load())
    return out


def split_docs(
    docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 100
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# ---------------------------- Retriever tool ----------------------------

def build_retriever_tool(doc_splits: List[Document], k: int = 4):
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small")),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    @tool
    def retrieve(query: str) -> str:
        """Semantic search over the indexed documents. Returns relevant text snippets with sources."""
        docs = retriever.invoke(query)
        parts = []
        for d in docs:
            src = d.metadata.get("source") or d.metadata.get("url") or "unknown"
            parts.append(f"SOURCE: {src}\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    return retrieve


# ---------------------------- Graph nodes ----------------------------

def latest_user_question(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return messages[0].content if messages else ""


class GradeDocuments(BaseModel):
    """Binary relevance grade for retrieved context."""
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if context is relevant, else 'no'"
    )


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
    chat_model = init_chat_model(os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    grader_model = init_chat_model(os.getenv("GRADER_MODEL", "gpt-4o-mini"), temperature=0)

    def generate_query_or_respond(state: MessagesState):
        """LLM decides: answer directly, OR call the retriever tool."""
        response = chat_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Route based on whether retrieved context seems relevant."""
        question = latest_user_question(state["messages"])
        context = state["messages"][-1].content  # tool output
        prompt = GRADE_PROMPT.format(question=question, context=context)
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

    workflow = StateGraph(MessagesState)

    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    # If the model called a tool -> "retrieve"; else end (direct answer)
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )

    # After retrieval, grade; if bad -> rewrite -> back to start; if good -> answer -> end
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ---------------------------- Simple CLI app ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--urls", nargs="*", default=None, help="URLs to index")
    p.add_argument("--paths", nargs="*", default=None, help="Local .txt/.md files or directories to index")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--chunk-overlap", type=int, default=100)
    p.add_argument("--k", type=int, default=4, help="Top-K retrieved chunks")
    p.add_argument("--stream", action="store_true", help="Print per-node updates like the tutorial")
    return p.parse_args()


def main():
    args = parse_args()

    urls = args.urls if args.urls else DEFAULT_URLS
    path_docs = load_path_docs(args.paths) if args.paths else []
    url_docs = load_url_docs(urls) if urls else []

    docs = url_docs + path_docs
    if not docs:
        print("No documents loaded. Provide --urls and/or --paths.", file=sys.stderr)
        sys.exit(1)

    doc_splits = split_docs(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever_tool = build_retriever_tool(doc_splits, k=args.k)
    graph = build_graph(retriever_tool)

    messages: List[BaseMessage] = []
    print("Agentic RAG ready. Type '/exit' to quit.\n")

    while True:
        user_text = input("You> ").strip()
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if not user_text:
            continue

        messages.append(HumanMessage(content=user_text))

        if args.stream:
            last_state = None
            for chunk in graph.stream({"messages": messages}):
                # chunk: {node_name: {"messages": [...]}}
                for node, update in chunk.items():
                    last = update["messages"][-1]
                    print(f"\n--- update from {node} ---")
                    last.pretty_print()
                    last_state = update
            if last_state:
                messages = last_state["messages"]
                print()
        else:
            out = graph.invoke({"messages": messages})
            messages = out["messages"]
            print()
            messages[-1].pretty_print()
            print()

    print("Bye.")


if __name__ == "__main__":
    main()