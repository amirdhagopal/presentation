#!/usr/bin/env python3
"""
Agent Demo
==========
Demonstrates an AI Agent with tool use and the ReAct pattern.

Topics covered:
- Defining custom tools
- ReAct loop: Thought → Action → Observation
- Visible reasoning process
- Multi-step problem solving
"""

import ollama
import re
import math
from typing import Callable, Any, Optional
from datetime import datetime

# Model for the agent
MODEL = "qwen3:8b"


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60 + "\n")


# =============================================================================
# TOOLS - Functions the agent can call
# =============================================================================

def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A math expression like "2 + 2" or "sqrt(16)"
    
    Returns:
        The result as a string
    """
    try:
        # Safe math evaluation
        allowed = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'pow': math.pow,
            'abs': abs,
            'round': round,
            'pi': math.pi,
            'e': math.e,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_current_time() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current date and time as a formatted string
    """
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def web_search(query: str) -> str:
    """
    Simulate a web search (mock for demo purposes).
    
    Args:
        query: The search query
    
    Returns:
        Mock search results
    """
    # Mock responses for demo
    mock_results = {
        "weather": "Current weather: Sunny, 72°F (22°C). Partly cloudy this afternoon.",
        "python": "Python is a high-level programming language. Latest version: 3.12.0",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "llm": "Large Language Models are neural networks trained on vast text data.",
        "default": f"No specific results found for '{query}'. Try a different search.",
    }
    
    query_lower = query.lower()
    for key, response in mock_results.items():
        if key in query_lower:
            return f"Search results for '{query}':\n{response}"
    
    return mock_results["default"]


def read_file(filename: str) -> str:
    """
    Read content from a file.
    
    Args:
        filename: Name of the file to read
    
    Returns:
        File contents or error message
    """
    # Mock file system for demo
    mock_files = {
        "notes.txt": "Meeting notes: Discussed Q4 targets. Revenue goal is $1.2M.",
        "config.json": '{"api_key": "xxx", "endpoint": "https://api.example.com"}',
        "todo.txt": "1. Review code\n2. Write tests\n3. Deploy to staging",
    }
    
    if filename in mock_files:
        return f"Contents of {filename}:\n{mock_files[filename]}"
    return f"Error: File '{filename}' not found."


# Tool registry
TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "Evaluate mathematical expressions. Input: math expression (e.g., '2+2', 'sqrt(16)')",
    },
    "get_current_time": {
        "function": get_current_time,
        "description": "Get the current date and time. No input needed.",
    },
    "web_search": {
        "function": web_search,
        "description": "Search the web for information. Input: search query",
    },
    "read_file": {
        "function": read_file,
        "description": "Read content from a file. Input: filename",
    },
}


# =============================================================================
# AGENT - ReAct Loop Implementation
# =============================================================================

def build_system_prompt() -> str:
    """Build the system prompt with tool descriptions."""
    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    ])
    
    return f"""You are a helpful AI assistant that can use tools to help answer questions.

Available tools:
{tool_descriptions}

Use this exact format for each step:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [input for the tool]

After receiving an observation, continue with another Thought.
When you have enough information to answer, respond with:

Thought: I now have enough information to answer.
Final Answer: [your final response to the user]

Important:
- Think step by step
- Only use one tool at a time
- Always wait for the observation before continuing
"""


def parse_agent_response(response: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse the agent's response to extract thought, action, input, or final answer.
    
    Returns:
        (thought, action, action_input, final_answer)
    """
    thought = None
    action = None
    action_input = None
    final_answer = None
    
    # Extract thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    
    # Check for final answer
    final_match = re.search(r'Final Answer:\s*(.+)', response, re.DOTALL)
    if final_match:
        final_answer = final_match.group(1).strip()
        return thought, None, None, final_answer
    
    # Extract action and input
    action_match = re.search(r'Action:\s*(\w+)', response)
    input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', response)
    
    if action_match:
        action = action_match.group(1).strip()
    if input_match:
        action_input = input_match.group(1).strip()
    
    return thought, action, action_input, final_answer


def execute_tool(action: str, action_input: str) -> str:
    """Execute a tool and return the observation."""
    if action not in TOOLS:
        return f"Error: Unknown tool '{action}'. Available tools: {', '.join(TOOLS.keys())}"
    
    tool_fn = TOOLS[action]["function"]
    
    if action == "get_current_time":
        return tool_fn()
    else:
        return tool_fn(action_input)


def run_agent(user_query: str, max_steps: int = 5, verbose: bool = True) -> str:
    """
    Run the ReAct agent loop.
    
    Args:
        user_query: The user's question
        max_steps: Maximum number of reasoning steps
        verbose: Whether to print the reasoning process
    
    Returns:
        The agent's final answer
    """
    system_prompt = build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    
    if verbose:
        print(f"🎯 User Query: {user_query}\n")
        print("-" * 50)
    
    for step in range(max_steps):
        if verbose:
            print(f"\n📍 Step {step + 1}")
            print("-" * 30)
        
        # Get agent response
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": 0.2}
        )
        agent_output = response['message']['content']
        
        # Parse the response
        thought, action, action_input, final_answer = parse_agent_response(agent_output)
        
        if verbose:
            if thought:
                print(f"💭 Thought: {thought}")
        
        # Check for final answer
        if final_answer:
            if verbose:
                print(f"\n✅ Final Answer: {final_answer}")
            return final_answer
        
        # Execute tool if action specified
        if action:
            if verbose:
                print(f"🔧 Action: {action}")
                print(f"📥 Input: {action_input}")
            
            observation = execute_tool(action, action_input or "")
            
            if verbose:
                print(f"👁️ Observation: {observation}")
            
            # Add to conversation
            messages.append({"role": "assistant", "content": agent_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # No action, try to get final answer
            messages.append({"role": "assistant", "content": agent_output})
            messages.append({
                "role": "user",
                "content": "Please provide your final answer based on the information gathered."
            })
    
    return "Agent reached maximum steps without a final answer."


def demo_agent():
    """Run agent demos with various queries."""
    
    queries = [
        "What is the square root of 144 plus 25?",
        "What time is it and what's the weather like?",
        "Read the notes.txt file and tell me about the revenue goal.",
    ]
    
    for i, query in enumerate(queries, 1):
        section(f"Demo {i}: {query[:40]}...")
        result = run_agent(query)
        print("\n")


def main():
    print("\n" + "🤖 AGENT DEMO (ReAct Pattern)".center(60))
    print(f"Using model: {MODEL}\n")
    
    try:
        ollama.show(MODEL)
    except ollama.ResponseError:
        print(f"Error: Model '{MODEL}' not found.")
        print(f"Please run: ollama pull {MODEL}")
        return
    
    # Show available tools
    print("Available Tools:")
    for name, info in TOOLS.items():
        print(f"  • {name}: {info['description'][:50]}...")
    
    demo_agent()
    
    print("="*60)
    print("✅ Agent Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
