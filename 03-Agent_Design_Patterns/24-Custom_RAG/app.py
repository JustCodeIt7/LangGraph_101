#!/usr/bin/env python3
"""
Agentic RAG Application - Interactive CLI

This is the main application that provides an interactive command-line interface
for the agentic RAG system. Users can ask questions and dynamically add files.
"""

import argparse
import os
import re
import sys
from typing import List

from dotenv import load_dotenv
from rich import print

from langchain_core.messages import BaseMessage, HumanMessage

from graph import (
    load_url_docs,
    load_path_docs,
    split_docs,
    build_retriever_tool,
    build_graph,
)

# Load environment variables
load_dotenv()

# Set default user agent for web scraping
os.environ.setdefault("USER_AGENT", "agentic-rag-demo/1.0 (contact: you@example.com)")


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


# =============================================================================
# CLI Argument Parsing
# =============================================================================

def parse_args():
    """
    Parse command line arguments for configuring the RAG system.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Agentic RAG - Interactive question-answering with retrieval"
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=None,
        help="URLs to index (default: LLM blog posts)"
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Local .txt/.md files or directories to index"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of document chunks (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of top-K chunks to retrieve (default: 4)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print per-node updates during execution"
    )
    return parser.parse_args()


# =============================================================================
# Main Application
# =============================================================================

def main():
    """
    Main application loop for the agentic RAG system.

    This function:
    1. Loads and processes documents from URLs or local files
    2. Builds the retriever tool and LangGraph workflow
    3. Runs an interactive chat loop where users can ask questions
    4. Supports dynamic file indexing during runtime with @/path/to/file
    """
    args = parse_args()

    # -------------------------------------------------------------------------
    # Step 1: Load Documents
    # -------------------------------------------------------------------------
    print("[bold cyan]Loading documents...[/bold cyan]")

    urls = args.urls if args.urls else DEFAULT_URLS
    path_docs = load_path_docs(args.paths) if args.paths else []
    url_docs = load_url_docs(urls) if urls else []

    docs = url_docs + path_docs
    if not docs:
        print("[bold red]Error:[/bold red] No documents loaded. Provide --urls and/or --paths.", file=sys.stderr)
        sys.exit(1)

    print(f"[green]Loaded {len(docs)} document(s)[/green]")

    # -------------------------------------------------------------------------
    # Step 2: Process and Index Documents
    # -------------------------------------------------------------------------
    print("[bold cyan]Chunking and indexing documents...[/bold cyan]")

    doc_splits = split_docs(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    retriever_tool, vectorstore = build_retriever_tool(doc_splits, k=args.k)

    print(f"[green]Created {len(doc_splits)} chunks[/green]")

    # -------------------------------------------------------------------------
    # Step 3: Build Graph
    # -------------------------------------------------------------------------
    print("[bold cyan]Building agentic RAG graph...[/bold cyan]")
    graph = build_graph(retriever_tool)
    print("[green]Graph ready![/green]\n")

    # -------------------------------------------------------------------------
    # Step 4: Interactive Chat Loop
    # -------------------------------------------------------------------------
    messages: List[BaseMessage] = []

    print("[bold yellow]Agentic RAG ready![/bold yellow]")
    print("Type your questions and I'll use retrieval to help answer them.")
    print("Type [bold]@/path/to/file.txt[/bold] to index files dynamically.")
    print("Type [bold]/exit[/bold] or [bold]/quit[/bold] to exit.\n")

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[yellow]Interrupted[/yellow]")
            break

        if user_text.lower() in {"/exit", "/quit"}:
            break

        if not user_text:
            continue

        # ---------------------------------------------------------------------
        # Handle Dynamic File Indexing
        # ---------------------------------------------------------------------
        if file_matches := re.findall(r"@(\S+)", user_text):
            try:
                if new_docs := load_path_docs(file_matches):
                    new_splits = split_docs(
                        new_docs,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap
                    )
                    vectorstore.add_documents(new_splits)
                    print(f"[green]Indexed {len(file_matches)} file(s) ({len(new_splits)} chunks)[/green]")
                else:
                    print(f"[yellow]Warning: No readable content found in {file_matches}[/yellow]")
            except Exception as e:
                print(f"[red]Error loading files: {e}[/red]")

        # ---------------------------------------------------------------------
        # Process Question with Graph
        # ---------------------------------------------------------------------
        messages.append(HumanMessage(content=user_text))

        if args.stream:
            # Stream mode: show each node's output as it executes
            last_state = None
            for chunk in graph.stream({"messages": messages}):
                for node, update in chunk.items():
                    last = update["messages"][-1]
                    print(f"\n[bold blue]--- Update from {node} ---[/bold blue]")
                    last.pretty_print()
                    last_state = update

            if last_state:
                messages = last_state["messages"]
                print()
        else:
            # Batch mode: just show final result
            out = graph.invoke({"messages": messages})
            messages = out["messages"]
            print()
            messages[-1].pretty_print()
            print()

    print("\n[bold yellow]Goodbye![/bold yellow]")


if __name__ == "__main__":
    main()
