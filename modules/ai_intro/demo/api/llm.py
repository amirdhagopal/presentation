"""
LLM generation endpoints.
"""

from typing import AsyncGenerator, Optional

import ollama
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from .config import get_current_llm_model

router = APIRouter()


@router.get("/api/llm/generate")
async def llm_generate(
    prompt: str = Query(..., description="The prompt to generate from"),
    temperature: float = Query(0.7, ge=0, le=2),
    model: Optional[str] = Query(None, description="Override LLM model"),
):
    """Stream LLM generation."""
    llm_model = model or get_current_llm_model()
    
    async def stream() -> AsyncGenerator[str, None]:
        try:
            for chunk in ollama.generate(
                model=llm_model,
                prompt=prompt,
                stream=True,
                options={"temperature": temperature}
            ):
                yield f"data: {chunk['response']}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/api/llm/chat")
async def llm_chat(
    message: str = Query(...),
    system: str = Query("You are a helpful assistant."),
    model: Optional[str] = Query(None),
):
    """Stream chat completion."""
    llm_model = model or get_current_llm_model()
    
    async def stream() -> AsyncGenerator[str, None]:
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ]
            for chunk in ollama.chat(model=llm_model, messages=messages, stream=True):
                if 'message' in chunk and 'content' in chunk['message']:
                    yield f"data: {chunk['message']['content']}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")
