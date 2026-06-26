#!/usr/bin/env python3
"""
Chat module - Generate responses using retrieved context with LLM.
"""

import os
import sys
import requests
from typing import List, Dict, Any


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"


def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL
) -> str:
    """
    Generate a response using retrieved context.
    
    Args:
        query: User question
        context_chunks: Retrieved chunks from search
        model: LLM model name
    
    Returns:
        Generated response
    """
    # Build context from chunks
    context_text = "\n\n".join([
        f"[Excerpt {i+1}]: {chunk.get('text', '')[:500]}..."
        for i, chunk in enumerate(context_chunks)
    ])
    
    # Build prompt
    prompt = f"""You are a helpful assistant answering questions based on the provided context.

Context from the document:
{context_text}

Question: {query}

Answer based on the context above. If the context doesn't contain the answer, say "I don't have enough information to answer that."

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
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {OLLAMA_URL}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating response: {e}")
        raise


def chat(
    query: str,
    db_path: str = "data/vectors.lance",
    table_name: str = "chunks",
    top_k: int = 3,
    model: str = DEFAULT_MODEL
) -> None:
    """
    Full RAG chat: search + generate response.
    
    Args:
        query: User question
        db_path: Path to LanceDB
        table_name: Table name
        top_k: Number of context chunks
        model: LLM model name
    """
    from query import search
    
    print(f"\nSearching for relevant context...")
    results = search(query, db_path, table_name, top_k)
    
    if not results:
        print("No relevant context found.")
        return
    
    print(f"Found {len(results)} relevant excerpts. Generating answer...\n")
    
    response = generate_response(query, results, model)
    
    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    # Show sources
    print("\nSources:")
    for i, result in enumerate(results, 1):
        source = result.get('source', 'Unknown')
        distance = result.get('_distance', 0)
        print(f"  {i}. {source} (relevance: {1-distance:.2%})")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chat with your documents using RAG'
    )
    parser.add_argument('query', help='Your question')
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
    
    print(f"txt-rag-cli Chat")
    print(f"================")
    print(f"Model: {args.model}")
    print(f"Query: {args.query}")
    
    chat(args.query, args.db, args.table, args.top_k, args.model)


if __name__ == '__main__':
    main()
