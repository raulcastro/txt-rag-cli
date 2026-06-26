#!/usr/bin/env python3
"""
Embedder module - Generates embeddings using Ollama and stores in LanceDB.
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any
import lancedb
import numpy as np
import pyarrow as pa


OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "mxbai-embed-large"


def get_embedding(text: str, model: str = DEFAULT_MODEL) -> List[float]:
    """
    Generate embedding for a text using Ollama.
    
    Args:
        text: The text to embed
        model: Ollama model name (default: mxbai-embed-large)
    
    Returns:
        List of floats (embedding vector)
    """
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
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def get_embeddings_batch(texts: List[str], model: str = DEFAULT_MODEL) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed
        model: Ollama model name
    
    Returns:
        List of embedding vectors
    """
    embeddings = []
    for i, text in enumerate(texts, 1):
        print(f"  Embedding chunk {i}/{len(texts)}...", end='\r')
        embedding = get_embedding(text, model)
        embeddings.append(embedding)
    print(f"  Embedded {len(texts)} chunks ✓")
    return embeddings


def init_lancedb(db_path: str = "data/vectors.lance") -> lancedb.DBConnection:
    """
    Initialize LanceDB connection.
    
    Args:
        db_path: Path to LanceDB directory
    
    Returns:
        LanceDB connection
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return lancedb.connect(db_path)


def store_chunks(
    chunks: List[str],
    source_file: str,
    db_path: str = "data/vectors.lance",
    table_name: str = "chunks"
) -> None:
    """
    Store chunks with embeddings in LanceDB.
    
    Args:
        chunks: List of text chunks
        source_file: Source file name for metadata
        db_path: Path to LanceDB
        table_name: Name of the table
    """
    print(f"\nInitializing LanceDB at {db_path}...")
    db = init_lancedb(db_path)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = get_embeddings_batch(chunks)
    
    # Prepare data for LanceDB
    data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        data.append({
            "id": i,
            "text": chunk,
            "source": source_file,
            "vector": embedding
        })
    
    # Create or replace table
    print(f"Storing in table '{table_name}'...")
    
    # Convert to PyArrow table
    vectors = np.array([d["vector"] for d in data])
    texts = [d["text"] for d in data]
    sources = [d["source"] for d in data]
    ids = [d["id"] for d in data]
    
    table = pa.table({
        "id": ids,
        "text": texts,
        "source": sources,
        "vector": vectors.tolist()
    })
    
    # Create table with vector index
    if table_name in db.table_names():
        db.drop_table(table_name)
    
    tbl = db.create_table(table_name, table)
    
    print(f"✓ Stored {len(chunks)} chunks in '{table_name}'")
    print(f"  Database: {db_path}")


def main():
    """CLI entry point."""
    import argparse
    from chunker import chunk_file
    
    parser = argparse.ArgumentParser(
        description='Embed chunks from a text file using Ollama'
    )
    parser.add_argument('file', help='Path to the text file')
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
        '--size', '-s',
        type=int,
        default=500,
        help='Chunk size (default: 500)'
    )
    parser.add_argument(
        '--overlap', '-o',
        type=int,
        default=100,
        help='Overlap (default: 100)'
    )
    
    args = parser.parse_args()
    
    print(f"txt-rag-cli Embedder")
    print(f"====================")
    print(f"File: {args.file}")
    print(f"Chunk size: {args.size}")
    print(f"Overlap: {args.overlap}")
    print()
    
    # Chunk the file
    print("Step 1: Chunking...")
    chunks = chunk_file(args.file, args.size, args.overlap)
    print(f"Generated {len(chunks)} chunks")
    
    # Store with embeddings
    print("\nStep 2: Embedding...")
    store_chunks(chunks, os.path.basename(args.file), args.db, args.table)
    
    print("\n✓ Done! You can now query with: python src/query.py")


if __name__ == '__main__':
    main()
