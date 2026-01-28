"""
AI Agent with ReAct pattern and tools.
"""

import re
from datetime import datetime
from typing import AsyncGenerator

import ollama
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from .config import get_current_llm_model
from .utils import safe_eval

router = APIRouter()


# =============================================================================
# AGENT TOOLS
# =============================================================================

def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        result = safe_eval(expression)
        return f"Result: {result}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_current_time() -> str:
    """Get current time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def web_search(query: str) -> str:
    """Mock web search."""
    mock_results = {
        "weather": "Current weather: Sunny, 72°F. Partly cloudy afternoon.",
        "python": "Python 3.12 is the latest stable version.",
        "ai": "AI is transforming industries through automation and intelligence.",
    }
    for key, response in mock_results.items():
        if key in query.lower():
            return f"Search results: {response}"
    return f"No specific results for '{query}'."


TOOLS = {
    "calculator": {"fn": calculator, "desc": "Evaluate math expressions"},
    "get_current_time": {"fn": get_current_time, "desc": "Get current time"},
    "web_search": {"fn": web_search, "desc": "Search the web"},
}


def build_agent_prompt():
    """Build the agent system prompt with available tools."""
    tool_list = "\n".join([f"- {name}: {info['desc']}" for name, info in TOOLS.items()])
    return f"""You are a helpful AI assistant with tools.

Available tools:
{tool_list}

Format:
Thought: [your reasoning]
Action: [tool_name]
Action Input: [input]

After observation, continue or respond with:
Thought: I have enough information.
Final Answer: [your response]
"""


@router.get("/api/agent/run")
async def agent_run(query: str = Query(...)):
    """Run agent with ReAct pattern, streaming steps."""
    llm_model = get_current_llm_model()
    
    async def stream() -> AsyncGenerator[str, None]:
        try:
            system_prompt = build_agent_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            for step in range(5):
                yield f"data: [STEP] Step {step + 1}\n\n"
                
                response = ollama.chat(model=llm_model, messages=messages, options={"temperature": 0.2})
                content = response['message']['content']
                
                # Extract thought
                thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', content, re.DOTALL)
                if thought_match:
                    thought_text = thought_match.group(1).strip()
                    for line in thought_text.splitlines():
                        yield f"data: [THOUGHT] {line}\n\n"
                
                # Check for final answer
                final_match = re.search(r'Final Answer:\s*(.+)', content, re.DOTALL)
                if final_match:
                    answer_text = final_match.group(1).strip()
                    for line in answer_text.splitlines():
                        yield f"data: [ANSWER] {line}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # Extract and execute action
                action_match = re.search(r'Action:\s*(\w+)', content)
                input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', content)
                
                if action_match:
                    action = action_match.group(1).strip()
                    action_input = input_match.group(1).strip() if input_match else ""
                    
                    yield f"data: [ACTION] {action}({action_input})\n\n"
                    
                    if action in TOOLS:
                        if action == "get_current_time":
                            observation = TOOLS[action]["fn"]()
                        else:
                            observation = TOOLS[action]["fn"](action_input)
                        
                        for line in str(observation).splitlines():
                            yield f"data: [OBSERVATION] {line}\n\n"
                        
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": f"Observation: {observation}"})
                    else:
                        yield f"data: [ERROR] Unknown tool: {action}\n\n"
                        break
                else:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "Please provide your final answer."})
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            for line in str(e).splitlines():
                yield f"data: [ERROR] {line}\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")
