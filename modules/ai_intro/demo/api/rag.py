"""
RAG (Retrieval-Augmented Generation) endpoints.
"""

from pathlib import Path
from typing import AsyncGenerator

import ollama
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from .config import get_current_llm_model
from .utils import get_embedding, cosine_similarity

router = APIRouter()

# Knowledge base storage
KNOWLEDGE_PATH = Path(__file__).parent.parent / "knowledge" / "sample_docs.txt"
KNOWLEDGE_CHUNKS = []
KNOWLEDGE_EMBEDDINGS = []


def load_knowledge():
    """Load and embed knowledge base."""
    global KNOWLEDGE_CHUNKS, KNOWLEDGE_EMBEDDINGS
    
    if KNOWLEDGE_CHUNKS:
        return  # Already loaded
    
    if not KNOWLEDGE_PATH.exists():
        return
    
    text = KNOWLEDGE_PATH.read_text()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for para in paragraphs:
        if len(para) > 50:  # Skip very short paragraphs
            KNOWLEDGE_CHUNKS.append(para)
            KNOWLEDGE_EMBEDDINGS.append(get_embedding(para))


@router.get("/api/rag/query")
async def rag_query(
    question: str = Query(...),
    top_k: int = Query(2, ge=1, le=5),
):
    """Perform RAG query with streaming response."""
    load_knowledge()
    llm_model = get_current_llm_model()
    
    async def stream() -> AsyncGenerator[str, None]:
        try:
            # Retrieve
            yield f"data: [STEP] Searching knowledge base...\n\n"
            
            query_emb = get_embedding(question)
            similarities = []
            for i, (chunk, emb) in enumerate(zip(KNOWLEDGE_CHUNKS, KNOWLEDGE_EMBEDDINGS)):
                sim = cosine_similarity(query_emb, emb)
                similarities.append((sim, chunk))
            
            similarities.sort(reverse=True)
            top_chunks = similarities[:top_k]
            
            for sim, chunk in top_chunks:
                preview = chunk[:100].replace('\n', ' ')
                yield f"data: [CONTEXT] [{sim:.3f}] {preview}...\n\n"
            
            # Augment + Generate
            yield f"data: [STEP] Generating answer...\n\n"
            
            context = "\n---\n".join([chunk for _, chunk in top_chunks])
            prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""
            
            for chunk in ollama.generate(model=llm_model, prompt=prompt, stream=True, options={"temperature": 0.3}):
                yield f"data: {chunk['response']}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")
