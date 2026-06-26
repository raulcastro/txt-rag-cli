#!/usr/bin/env python3
"""
Chat module - Interactive RAG chat with rich logging and colors.
"""

import os
import sys
import requests
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table
from rich import box


console = Console()
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"


def log_step(step_num: int, title: str, details: str = ""):
    """Log a step with colors."""
    console.print(f"\n[bold cyan]Step {step_num}:[/bold cyan] [bold]{title}[/bold]")
    if details:
        console.print(f"[dim]{details}[/dim]")


def log_success(message: str):
    """Log success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def log_info(label: str, value: str):
    """Log info pair."""
    console.print(f"  [yellow]{label}:[/yellow] {value}")


def log_chunk_preview(chunk: Dict[str, Any], index: int):
    """Display chunk with formatting."""
    text = chunk.get('text', '')[:300]
    distance = chunk.get('_distance', 0)
    relevance = (1 - distance) * 100
    
    panel = Panel(
        f"[dim]{text}...[/dim]",
        title=f"[bold]Chunk {index}[/bold] | Relevance: [green]{relevance:.1f}%[/green]",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print(panel)


def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL
) -> str:
    """Generate a response using retrieved context."""
    log_step(3, "Building Prompt", f"Using {len(context_chunks)} chunks as context")
    
    # Build context from chunks
    context_text = "\n\n".join([
        f"[Excerpt {i+1}]: {chunk.get('text', '')[:500]}..."
        for i, chunk in enumerate(context_chunks)
    ])
    
    # Show context being sent to LLM
    console.print("\n[dim]Context sent to LLM:[/dim]")
    for i, chunk in enumerate(context_chunks, 1):
        preview = chunk.get('text', '')[:100].replace('\n', ' ')
        console.print(f"  [dim]{i}. {preview}...[/dim]")
    
    # Build prompt
    prompt = f"""You are a helpful assistant answering questions based on the provided context.

Context from the document:
{context_text}

Question: {query}

Answer based on the context above in Spanish. If the context doesn't contain the answer, say "No tengo suficiente información para responder eso."

Answer:"""
    
    log_step(4, "Generating Response", f"Model: {model}")
    console.print(f"  [dim]Sending request to Ollama...[/dim]")
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500
                }
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        log_success(f"Generated {len(data['response'])} characters")
        return data["response"].strip()
    except requests.exceptions.ConnectionError:
        console.print("[bold red]Error:[/bold red] Cannot connect to Ollama")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


def chat_query(
    query: str,
    db_path: str = "data/vectors.lance",
    table_name: str = "chunks",
    top_k: int = 3,
    model: str = DEFAULT_MODEL
) -> None:
    """Full RAG chat: search + generate response."""
    from query import search, get_embedding
    
    # Step 1: Embed query
    log_step(1, "Embedding Query", f"Query: '{query}'")
    console.print(f"  [dim]Using model: {DEFAULT_MODEL}[/dim]")
    
    # Step 2: Search
    log_step(2, "Semantic Search", f"Database: {db_path}")
    
    import lancedb
    db = lancedb.connect(db_path)
    
    available_tables = db.list_tables()
    if hasattr(available_tables, 'tables'):
        table_names = available_tables.tables
    else:
        table_names = available_tables
    
    if table_name not in table_names:
        console.print(f"[red]Error: Table '{table_name}' not found[/red]")
        return
    
    tbl = db.open_table(table_name)
    query_vector = get_embedding(query)
    results = tbl.search(query_vector).limit(top_k).to_list()
    
    log_success(f"Found {len(results)} relevant chunks")
    
    # Show chunks found
    console.print("\n[bold magenta]Retrieved Chunks:[/bold magenta]")
    for i, result in enumerate(results, 1):
        log_chunk_preview(result, i)
    
    if not results:
        console.print("[yellow]No relevant context found.[/yellow]")
        return
    
    # Step 3-4: Generate response
    response = generate_response(query, results, model)
    
    # Display answer
    console.print("\n" + "=" * 60)
    console.print(Panel(
        response,
        title="[bold green]Respuesta[/bold green]",
        border_style="green",
        box=box.DOUBLE
    ))
    console.print("=" * 60)
    
    # Show sources table
    table = Table(title="Fuentes", box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Archivo", style="yellow")
    table.add_column("Relevancia", style="green")
    
    for i, result in enumerate(results, 1):
        source = result.get('source', 'Unknown')
        distance = result.get('_distance', 0)
        relevance = f"{(1 - distance) * 100:.1f}%"
        table.add_row(str(i), source, relevance)
    
    console.print(table)


def interactive_chat():
    """Run interactive chat session."""
    console.print(Panel.fit(
        "[bold blue]txt-rag-cli Chat[/bold blue]\n"
        "[dim]RAG Pipeline: Chunk → Embed → Search → Generate[/dim]\n"
        "[dim]Model: qwen2.5:7b | Database: data/vectors.lance[/dim]\n\n"
        "[yellow]Escribe 'exit' o 'quit' para salir[/yellow]",
        border_style="blue",
        box=box.DOUBLE
    ))
    
    while True:
        console.print()
        query = console.input("[bold green]Tú:[/bold green] ")
        
        if query.lower() in ('exit', 'quit', 'salir', 'q'):
            console.print("\n[dim]¡Hasta luego! 👋[/dim]")
            break
        
        if not query.strip():
            continue
        
        try:
            chat_query(query)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrumpido. Escribe 'exit' para salir.[/dim]")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chat with your documents using RAG'
    )
    parser.add_argument(
        'query',
        nargs='?',
        help='Your question (optional, starts interactive mode if omitted)'
    )
    parser.add_argument(
        '--db', '-d',
        default='data/vectors.lance',
        help='LanceDB path (default: data/vectors.lance)'
    )
    parser.add_argument(
        '--table', '-t',
        default='chunks',
        help='Table name (default: chunks)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=3,
        help='Number of context chunks (default: 3)'
    )
    parser.add_argument(
        '--model', '-m',
        default=DEFAULT_MODEL,
        help=f'LLM model (default: {DEFAULT_MODEL})'
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        chat_query(args.query, args.db, args.table, args.top_k, args.model)
    else:
        # Interactive mode
        interactive_chat()


if __name__ == '__main__':
    main()
