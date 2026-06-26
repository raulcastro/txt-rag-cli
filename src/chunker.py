#!/usr/bin/env python3
"""
Chunker module - Splits text files into overlapping chunks.
"""

import os
import re
from typing import List
import chardet


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: The input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence ending
            next_period = text.find('. ', end - 50, end + 50)
            if next_period != -1 and next_period < len(text):
                end = next_period + 1
            else:
                # Break at word boundary
                while end < len(text) and text[end] not in ' \n':
                    end += 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start forward by chunk_size minus overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= end:
            start = end
    
    return chunks


def chunk_file(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Read a file and split it into chunks.
    
    Args:
        file_path: Path to the text file
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Detect encoding automatically
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding'] or 'utf-8'
    
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        text = f.read()
    
    return chunk_text(text, chunk_size, overlap)


def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chunk a text file into overlapping segments'
    )
    parser.add_argument('file', help='Path to the text file')
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=500,
        help='Chunk size in characters (default: 500)'
    )
    parser.add_argument(
        '--overlap', '-o',
        type=int,
        default=100,
        help='Overlap between chunks in characters (default: 100)'
    )
    parser.add_argument(
        '--output', '-out',
        help='Output file for chunks (optional)'
    )
    
    args = parser.parse_args()
    
    print(f"Chunking: {args.file}")
    print(f"Chunk size: {args.size}")
    print(f"Overlap: {args.overlap}")
    print("-" * 50)
    
    chunks = chunk_file(args.file, args.size, args.overlap)
    
    print(f"Generated {len(chunks)} chunks")
    print()
    
    # Show first 3 chunks as preview
    for i, chunk in enumerate(chunks[:3], 1):
        preview = chunk[:100].replace('\n', ' ')
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"  {preview}...")
        print()
    
    if len(chunks) > 3:
        print(f"... and {len(chunks) - 3} more chunks")
    
    # Save to file if requested
    if args.output:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"=== CHUNK {i} ===\n")
                f.write(chunk)
                f.write("\n\n")
        print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
