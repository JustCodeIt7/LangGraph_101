#!/usr/bin/env python3
"""
LangGraph CLI Chat Application
Chat with documents and directories using Ollama Llama3.2 and ChromaDB
"""

import click
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from dotenv import load_dotenv

from src.chat_agent import ChatAgent
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

# Load environment variables
load_dotenv()

console = Console()

@click.group()
def cli():
    """LangGraph CLI Chat - Chat with your documents using AI"""
    pass

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--collection', '-c', default='default', help='ChromaDB collection name')
@click.option('--model', '-m', default='llama3.2', help='Ollama model name')
@click.option('--chunk-size', default=1000, help='Document chunk size')
@click.option('--chunk-overlap', default=200, help='Document chunk overlap')
def chat(path, collection, model, chunk_size, chunk_overlap):
    """Start a chat session with documents from PATH"""
    
    console.print(Panel.fit(
        "[bold blue]LangGraph Document Chat[/bold blue]\n"
        f"Path: {path}\n"
        f"Model: {model}\n"
        f"Collection: {collection}",
        title="Starting Chat Session"
    ))
    
    try:
        # Initialize components
        with console.status("[bold green]Initializing components..."):
            doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vector_store = VectorStoreManager(collection_name=collection)
            chat_agent = ChatAgent(model_name=model, vector_store=vector_store)
        
        # Process documents
        with console.status("[bold green]Processing documents..."):
            documents = doc_processor.process_path(path)
            if not documents:
                console.print("[red]No supported documents found![/red]")
                return
            
            vector_store.add_documents(documents)
            console.print(f"[green]Processed {len(documents)} document chunks[/green]")
        
        # Start chat loop
        console.print("\n[bold cyan]Chat started! Type 'quit' or 'exit' to end the session.[/bold cyan]")
        console.print("[dim]Type 'help' for available commands.[/dim]\n")
        
        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if user_input.lower() == 'help':
                    show_help()
                    continue
                
                if user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                # Get response from chat agent
                with console.status("[bold green]Thinking..."):
                    response = chat_agent.chat(user_input)
                
                # Display response
                console.print(Panel(
                    response,
                    title="[bold green]Assistant[/bold green]",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--collection', '-c', default='default', help='ChromaDB collection name')
def index(path, collection):
    """Index documents from PATH without starting chat"""
    
    console.print(Panel.fit(
        f"[bold blue]Indexing Documents[/bold blue]\n"
        f"Path: {path}\n"
        f"Collection: {collection}",
        title="Document Indexing"
    ))
    
    try:
        with console.status("[bold green]Processing documents..."):
            doc_processor = DocumentProcessor()
            vector_store = VectorStoreManager(collection_name=collection)
            
            documents = doc_processor.process_path(path)
            if not documents:
                console.print("[red]No supported documents found![/red]")
                return
            
            vector_store.add_documents(documents)
            console.print(f"[green]Successfully indexed {len(documents)} document chunks[/green]")
    
    except Exception as e:
        console.print(f"[red]Failed to index documents: {e}[/red]")

@cli.command()
def list_collections():
    """List all ChromaDB collections"""
    try:
        vector_store = VectorStoreManager()
        collections = vector_store.list_collections()
        
        if collections:
            console.print("[bold blue]Available Collections:[/bold blue]")
            for collection in collections:
                console.print(f"  • {collection}")
        else:
            console.print("[yellow]No collections found[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error listing collections: {e}[/red]")

def show_help():
    """Display help information"""
    help_text = """
[bold blue]Available Commands:[/bold blue]

• [bold]help[/bold] - Show this help message
• [bold]clear[/bold] - Clear the screen
• [bold]quit/exit[/bold] - Exit the chat session

[bold blue]Tips:[/bold blue]

• Ask questions about the content of your documents
• Reference specific sections or topics
• Ask for summaries or explanations
• Request code examples or explanations (for code files)
"""
    console.print(Panel(help_text, title="Help"))

if __name__ == '__main__':
    cli()