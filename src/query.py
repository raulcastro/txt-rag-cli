#!/usr/bin/env python3
"""
Query module - Search embeddings using semantic similarity.
"""

import os
import sys
import requests
from typing import List, Dict, Any
import lancedb
import numpy as np


OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "mxbai-embed-large"


def get_embedding(text: str, model: str = DEFAULT_MODEL) -> List[float]:
    """Generate embedding for query text."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {OLLAMA_URL}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def search(
    query: str,
    db_path: str = "data/vectors.lance",
    table_name: str = "chunks",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks.
    
    Args:
        query: Search query
        db_path: Path to LanceDB
        table_name: Table name
        top_k: Number of results
    
    Returns:
        List of results with text and similarity score
    """
    # Connect to DB
    db = lancedb.connect(db_path)
    
    available_tables = db.list_tables()
    # Handle both old list format and new object format
    if hasattr(available_tables, 'tables'):
        table_names = available_tables.tables
    else:
        table_names = available_tables
    
    if table_name not in table_names:
        print(f"Error: Table '{table_name}' not found in {db_path}")
        print(f"Available tables: {table_names}")
        print("Run embedder first: python src/embedder.py <file>")
        sys.exit(1)
    
    tbl = db.open_table(table_name)
    
    # Generate query embedding
    print(f"Embedding query: '{query}'...")
    query_vector = get_embedding(query)
    
    # Search
    print(f"Searching top {top_k} results...")
    results = tbl.search(query_vector).limit(top_k).to_list()
    
    return results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Search embeddings using semantic similarity'
    )
    parser.add_argument('query', help='Search query')
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
        default=5,
        help='Number of results (default: 5)'
    )
    
    args = parser.parse_args()
    
    print(f"txt-rag-cli Query")
    print(f"=================")
    print()
    
    results = search(args.query, args.db, args.table, args.top_k)
    
    print(f"\nFound {len(results)} results:\n")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        # LanceDB returns _distance, lower is better
        distance = result.get('_distance', 0)
        text = result.get('text', 'N/A')
        source = result.get('source', 'N/A')
        
        # Show preview of text (first 200 chars)
        preview = text[:300].replace('\n', ' ')
        
        print(f"\nResult {i} (distance: {distance:.4f})")
        print(f"Source: {source}")
        print(f"-" * 60)
        print(f"{preview}...")
        print()


if __name__ == '__main__':
    main()
