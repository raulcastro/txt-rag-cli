#!/usr/bin/env python3
"""
Chat with Memory module - RAG chat that remembers conversation history.

This extends the basic chat by maintaining context across multiple turns.
The memory is persisted to a JSON file so conversations can be resumed later.
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


console = Console()
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"
MEMORY_FILE = "data/conversation_memory.json"


class ConversationMemory:
    """
    Stores conversation history with persistence to JSON file.
    
    Attributes:
        max_history: Maximum number of exchanges to remember
        history: List of conversation exchanges
        memory_file: Path to JSON file for persistence
    """
    
    def __init__(self, max_history: int = 5, memory_file: str = MEMORY_FILE):
        self.max_history = max_history
        self.memory_file = memory_file
        self.history = []
        self._load()
    
    def _load(self):
        """Load conversation history from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    console.print(f"[dim]Memoria cargada: {len(self.history)} turnos previos[/dim]")
            except Exception as e:
                console.print(f"[dim]No se pudo cargar memoria previa: {e}[/dim]")
    
    def _save(self):
        """Save conversation history to file."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            data = {
                'history': self.history,
                'last_updated': datetime.now().isoformat(),
                'total_turns': len(self.history)
            }
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[dim]No se pudo guardar memoria: {e}[/dim]")
    
    def add_exchange(self, user_message: str, assistant_response: str):
        """Add a conversation exchange to memory and save."""
        self.history.append({
            'user': user_message,
            'assistant': assistant_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Save to file
        self._save()
    
    def get_context(self) -> str:
        """Format conversation history as context string."""
        if not self.history:
            return ""
        
        context_parts = []
        for i, exchange in enumerate(self.history, 1):
            context_parts.append(f"Turno {i}:")
            context_parts.append(f"Usuario: {exchange['user']}")
            context_parts.append(f"Asistente: {exchange['assistant']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history and delete file."""
        self.history = []
        if os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
            except:
                pass
        console.print("[dim]Memoria borrada del disco.[/dim]")
    
    def __len__(self):
        return len(self.history)


def log_step(step_num: int, title: str, details: str = ""):
    """Log a step with colors."""
    console.print(f"\n[bold cyan]Paso {step_num}:[/bold cyan] [bold]{title}[/bold]")
    if details:
        console.print(f"[dim]{details}[/dim]")


def log_success(message: str):
    """Log success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: str,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Generate a response using document context + conversation history.
    
    Args:
        query: Current user question
        context_chunks: Retrieved document chunks
        conversation_history: Previous conversation turns
        model: LLM model name
    
    Returns:
        Generated response
    """
    # Build document context
    doc_context = "\n\n".join([
        f"[Excerpt {i+1}]: {chunk.get('text', '')[:400]}..."
        for i, chunk in enumerate(context_chunks)
    ])
    
    # Build full prompt with memory
    if conversation_history:
        prompt = f"""You are a helpful assistant having a conversation about a document.

Previous conversation:
{conversation_history}

Document context:
{doc_context}

User's new question: {query}

Answer in Spanish, maintaining continuity with the previous conversation. Reference prior context when relevant. If the document doesn't contain the answer, say "No tengo suficiente información en el documento para responder eso."

Answer:"""
    else:
        prompt = f"""You are a helpful assistant answering questions based on the provided document.

Document context:
{doc_context}

Question: {query}

Answer in Spanish based on the document. If the context doesn't contain the answer, say "No tengo suficiente información para responder eso."

Answer:"""
    
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
        return data["response"].strip()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


def chat_with_memory(memory: ConversationMemory):
    """Run interactive chat with memory."""
    from query import get_embedding
    import lancedb
    
    # Header
    console.print(Panel.fit(
        "[bold blue]txt-rag-cli Chat con Memoria Persistente[/bold blue]\n"
        "[dim]Mantiene contexto entre turnos y sesiones[/dim]\n"
        "[dim]Modelo: qwen2.5:7b | Memoria: últimos 5 turnos[/dim]\n"
        "[dim]Archivo: data/conversation_memory.json[/dim]\n\n"
        "[yellow]Comandos:[/yellow]\n"
        "  [dim]exit, quit, salir[/dim] - Terminar\n"
        "  [dim]clear, limpiar[/dim] - Borrar memoria",
        border_style="blue",
        box=box.DOUBLE
    ))
    
    # Connect to DB once
    db = lancedb.connect("data/vectors.lance")
    tbl = db.open_table("chunks")
    
    while True:
        console.print()
        query = console.input("[bold green]Tú:[/bold green] ").strip()
        
        # Handle commands
        if query.lower() in ('exit', 'quit', 'salir', 'q'):
            console.print("\n[dim]¡Hasta luego! 👋[/dim]")
            break
        
        if query.lower() in ('clear', 'limpiar', 'reset'):
            memory.clear()
            continue
        
        if not query:
            continue
        
        try:
            # Step 1: Show memory status
            if memory.history:
                log_step(1, "Cargando Memoria", f"Recordando {len(memory)} turnos previos")
                console.print(f"[dim]Contexto previo incluido en el prompt[/dim]")
            else:
                log_step(1, "Nueva Conversación", "Sin historial previo")
            
            # Step 2: Embed and search
            log_step(2, "Búsqueda Semántica", f"Query: '{query}'")
            query_vector = get_embedding(query)
            results = tbl.search(query_vector).limit(3).to_list()
            log_success(f"Encontrados {len(results)} chunks relevantes")
            
            # Step 3: Generate with memory
            log_step(3, "Generando Respuesta", f"Usando documento + historial")
            conversation_context = memory.get_context()
            response = generate_response(query, results, conversation_context)
            
            # Display response
            console.print("\n" + "=" * 60)
            console.print(Panel(
                response,
                title="[bold green]Asistente[/bold green]",
                border_style="green",
                box=box.DOUBLE
            ))
            console.print("=" * 60)
            
            # Show sources
            table = Table(title="Fuentes del Documento", box=box.SIMPLE)
            table.add_column("#", style="cyan", justify="center")
            table.add_column("Relevancia", style="green")
            
            for i, result in enumerate(results, 1):
                distance = result.get('_distance', 0)
                relevance = f"{(1 - distance) * 100:.1f}%"
                table.add_row(str(i), relevance)
            
            console.print(table)
            
            # Step 4: Store in memory
            memory.add_exchange(query, response)
            log_success(f"Guardado en memoria (total: {len(memory)} turnos)")
            console.print(f"[dim]Persistido en: {memory.memory_file}[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Interrumpido.[/dim]")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chat with memory - remembers conversation context across sessions'
    )
    parser.add_argument(
        '--memory-size', '-m',
        type=int,
        default=5,
        help='Number of conversation turns to remember (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Initialize memory (loads from file if exists)
    memory = ConversationMemory(max_history=args.memory_size)
    
    # Start chat
    chat_with_memory(memory)


if __name__ == '__main__':
    main()
