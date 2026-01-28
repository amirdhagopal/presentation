"""
Embeddings comparison and search endpoints.
"""

from fastapi import APIRouter, Query

from .utils import get_embedding, cosine_similarity

router = APIRouter()


@router.get("/api/embeddings/compare")
async def embeddings_compare(
    text1: str = Query(...),
    text2: str = Query(...),
):
    """Compare similarity between two texts."""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    similarity = cosine_similarity(emb1, emb2)
    
    return {
        "text1": text1,
        "text2": text2,
        "similarity": round(similarity, 4),
        "interpretation": "Very similar" if similarity > 0.7 else "Somewhat related" if similarity > 0.5 else "Different topics"
    }


@router.get("/api/embeddings/search")
async def embeddings_search(
    query: str = Query(...),
):
    """Search through sample documents."""
    documents = [
        "Python is a popular programming language for data science.",
        "Machine learning models learn patterns from data.",
        "The Eiffel Tower is located in Paris, France.",
        "Neural networks are inspired by the human brain.",
        "Coffee is one of the most consumed beverages worldwide.",
        "Deep learning is a subset of machine learning.",
        "Large Language Models understand and generate text.",
        "RAG retrieves relevant knowledge for AI systems.",
    ]
    
    query_emb = get_embedding(query)
    
    results = []
    for doc in documents:
        doc_emb = get_embedding(doc)
        sim = cosine_similarity(query_emb, doc_emb)
        results.append({"document": doc, "similarity": round(sim, 4)})
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"query": query, "results": results[:5]}
