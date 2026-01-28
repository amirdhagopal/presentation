#!/usr/bin/env python3
"""
Embeddings Demo
===============
Demonstrates how text embeddings work for semantic similarity.

Topics covered:
- Generating vector embeddings
- Computing cosine similarity
- Finding semantically similar texts
"""

import ollama
import numpy as np
from typing import List

# Embedding model
EMBED_MODEL = "nomic-embed-text:v1.5"


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


def demo_embedding_vectors():
    """Show what embeddings look like."""
    section("1. Embedding Vectors")
    
    text = "Machine learning is fascinating"
    embedding = get_embedding(text)
    
    print(f"Text: \"{text}\"")
    print(f"\nEmbedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10].round(4)}")
    print(f"Min: {embedding.min():.4f}, Max: {embedding.max():.4f}")


def demo_semantic_similarity():
    """Compare semantic similarity between texts."""
    section("2. Semantic Similarity")
    
    pairs = [
        ("The cat sat on the mat", "A feline rested on the rug"),
        ("The cat sat on the mat", "Python is a programming language"),
        ("I love programming", "Coding is my passion"),
        ("The weather is nice", "It's a beautiful day outside"),
    ]
    
    print("Comparing text pairs:\n")
    for text1, text2 in pairs:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        similarity = cosine_similarity(emb1, emb2)
        
        print(f"Text 1: \"{text1}\"")
        print(f"Text 2: \"{text2}\"")
        print(f"Similarity: {similarity:.4f}")
        print(f"{'→ Very similar!' if similarity > 0.7 else '→ Different topics'}")
        print()


def demo_find_similar():
    """Find most similar text from a collection."""
    section("3. Finding Similar Texts")
    
    # Knowledge base
    documents = [
        "Python is a popular programming language for data science.",
        "Machine learning models learn patterns from data.",
        "The Eiffel Tower is located in Paris, France.",
        "Neural networks are inspired by the human brain.",
        "Coffee is one of the most consumed beverages worldwide.",
        "Deep learning is a subset of machine learning.",
    ]
    
    # Pre-compute embeddings
    print("Documents in knowledge base:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    doc_embeddings = [get_embedding(doc) for doc in documents]
    
    # Query
    query = "How do computers learn?"
    print(f"\nQuery: \"{query}\"\n")
    
    query_embedding = get_embedding(query)
    
    # Find similarities
    similarities: List[tuple[float, int, str]] = []
    for i, (doc, doc_emb) in enumerate(zip(documents, doc_embeddings)):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((sim, i, doc))
    
    # Sort by similarity
    similarities.sort(reverse=True)
    
    print("Results ranked by similarity:")
    for sim, idx, doc in similarities:
        bar = "█" * int(sim * 20)
        print(f"  [{sim:.3f}] {bar}")
        print(f"          {doc}")


def main():
    print("\n" + "🔢 EMBEDDINGS DEMO".center(60))
    print(f"Using model: {EMBED_MODEL}\n")
    
    try:
        # Verify model is available
        ollama.show(EMBED_MODEL)
    except ollama.ResponseError as e:
        print(f"Error: Model '{EMBED_MODEL}' not found.")
        print(f"Please run: ollama pull {EMBED_MODEL}")
        return
    
    demo_embedding_vectors()
    demo_semantic_similarity()
    demo_find_similar()
    
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    main()
