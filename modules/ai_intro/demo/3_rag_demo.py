#!/usr/bin/env python3
"""
RAG Demo
========
Demonstrates a complete Retrieval-Augmented Generation pipeline.

Topics covered:
- Loading documents
- Creating embeddings
- Semantic search / retrieval
- Context augmentation
- Generating grounded answers
"""

import os
import ollama
import numpy as np
from typing import List, Tuple
from pathlib import Path

# Models
LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text:v1.5"

# Chunk size for splitting documents
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60 + "\n")


def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for a text."""
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(response['embeddings'][0])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by paragraph."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


class SimpleVectorStore:
    """A simple in-memory vector store for demonstration."""
    
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
    
    def add(self, text: str):
        """Add a document to the store."""
        embedding = get_embedding(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
    
    def add_batch(self, texts: List[str]):
        """Add multiple documents."""
        for text in texts:
            self.add(text)
            print(f"  ✓ Added chunk ({len(text)} chars)")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for most similar documents."""
        query_embedding = get_embedding(query)
        
        similarities = []
        for doc, emb in zip(self.documents, self.embeddings):
            sim = cosine_similarity(query_embedding, emb)
            similarities.append((doc, sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def load_documents() -> str:
    """Load sample documents from knowledge folder."""
    knowledge_path = Path(__file__).parent / "knowledge" / "sample_docs.txt"
    
    if not knowledge_path.exists():
        print(f"Warning: {knowledge_path} not found. Using sample text.")
        return """
        Large Language Models (LLMs) are neural networks trained on text.
        RAG connects LLMs to external knowledge sources.
        Agents combine LLMs with reasoning and tools.
        MCP is the Model Context Protocol for AI integration.
        """
    
    return knowledge_path.read_text()


def demo_rag_pipeline():
    """Run the complete RAG pipeline."""
    
    # Step 1: Load Documents
    section("Step 1: Load Documents")
    raw_text = load_documents()
    print(f"Loaded {len(raw_text)} characters of text")
    
    # Step 2: Chunk Documents
    section("Step 2: Chunk Documents")
    chunks = chunk_text(raw_text)
    print(f"Split into {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks[:3], 1):
        preview = chunk[:80].replace('\n', ' ')
        print(f"  Chunk {i}: \"{preview}...\"")
    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more")
    
    # Step 3: Create Vector Store
    section("Step 3: Create Vector Store & Embed")
    store = SimpleVectorStore()
    print("Adding chunks to vector store:")
    store.add_batch(chunks)
    print(f"\n✓ Vector store contains {len(store.documents)} documents")
    
    # Step 4: Query and Retrieve
    section("Step 4: Query & Retrieve")
    
    queries = [
        "What is the ReAct pattern?",
        "How do embeddings work?",
        "What does MCP stand for?",
    ]
    
    for query in queries:
        print(f"\n📝 Query: \"{query}\"")
        print("-" * 50)
        
        results = store.search(query, top_k=2)
        
        print("Retrieved context:")
        for i, (doc, score) in enumerate(results, 1):
            preview = doc[:150].replace('\n', ' ')
            print(f"  [{score:.3f}] {preview}...")
        
        # Step 5: Generate with Context
        context = "\n---\n".join([doc for doc, _ in results])
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer concisely in 1-2 sentences:"""
        
        print("\n🤖 Generating answer...")
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": 0.3}
        )
        print(f"\n💬 Answer: {response['response']}")


def main():
    print("\n" + "📚 RAG DEMO".center(60))
    print(f"LLM: {LLM_MODEL} | Embeddings: {EMBED_MODEL}\n")
    
    # Verify models
    for model in [LLM_MODEL, EMBED_MODEL]:
        try:
            ollama.show(model)
        except ollama.ResponseError:
            print(f"Error: Model '{model}' not found.")
            print(f"Please run: ollama pull {model}")
            return
    
    demo_rag_pipeline()
    
    print("\n" + "="*60)
    print("✅ RAG Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
