#!/usr/bin/env python3
"""
LLM Basics Demo
===============
Demonstrates fundamental LLM interactions using Ollama.

Topics covered:
- Connecting to Ollama
- Simple text generation
- Temperature effects on creativity
- Streaming responses
"""

import ollama

# Model to use (change if you have a different model)
MODEL = "qwen3:8b"


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60 + "\n")


def demo_simple_generation():
    """Basic text generation."""
    section("1. Simple Text Generation")
    
    prompt = "Explain what a neural network is in one sentence."
    print(f"Prompt: {prompt}\n")
    
    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
    )
    print(f"Response: {response['response']}")


def demo_temperature_effects():
    """Show how temperature affects output creativity."""
    section("2. Temperature Effects")
    
    prompt = "Complete this sentence creatively: The robot walked into the bar and"
    print(f"Prompt: {prompt}\n")
    
    # Low temperature = more focused/deterministic
    print("Temperature 0.1 (deterministic):")
    response_low = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0.1}
    )
    print(f"  {response_low['response'][:200]}...\n")
    
    # High temperature = more creative/random
    print("Temperature 1.2 (creative):")
    response_high = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 1.2}
    )
    print(f"  {response_high['response'][:200]}...\n")


def demo_streaming():
    """Demonstrate streaming responses."""
    section("3. Streaming Response")
    
    prompt = "Write a haiku about programming."
    print(f"Prompt: {prompt}\n")
    print("Streaming response: ", end="", flush=True)
    
    stream = ollama.generate(
        model=MODEL,
        prompt=prompt,
        stream=True
    )
    
    for chunk in stream:
        print(chunk['response'], end="", flush=True)
    
    print("\n")


def demo_chat():
    """Demonstrate chat format with system message."""
    section("4. Chat with System Message")
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that responds in exactly 3 bullet points."
        },
        {
            "role": "user",
            "content": "What are the benefits of using AI agents?"
        }
    ]
    
    print(f"System: {messages[0]['content']}")
    print(f"User: {messages[1]['content']}\n")
    
    response = ollama.chat(
        model=MODEL,
        messages=messages
    )
    print(f"Assistant:\n{response['message']['content']}")


def main():
    print("\n" + "🤖 LLM BASICS DEMO".center(60))
    print(f"Using model: {MODEL}\n")
    
    try:
        # Verify model is available
        ollama.show(MODEL)
    except ollama.ResponseError as e:
        print(f"Error: Model '{MODEL}' not found.")
        print(f"Please run: ollama pull {MODEL}")
        return
    
    demo_simple_generation()
    demo_temperature_effects()
    demo_streaming()
    demo_chat()
    
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    main()
