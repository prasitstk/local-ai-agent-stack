"""
03_agent_loop.py — Interactive agent with persistent conversation and tool use.

Run this and have a conversation. The agent remembers context across turns
and can use tools at any point in the conversation.
"""

import json
import readline  # Enables arrow keys and history in the terminal
import requests

from multi_tools_lib import TOOLS, AVAILABLE_FUNCTIONS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e2b"


def process_tool_calls(messages: list, assistant_message: dict) -> str:
    """Handle tool calls and return the final model response."""
    messages.append(assistant_message)

    for tool_call in assistant_message.get("tool_calls", []):
        func_name = tool_call["function"]["name"]
        func_args = tool_call["function"]["arguments"]
        print(f"  [tool] {func_name}({json.dumps(func_args)})")

        if func_name in AVAILABLE_FUNCTIONS:
            result = AVAILABLE_FUNCTIONS[func_name](**func_args)
        else:
            result = json.dumps({"error": f"Unknown function: {func_name}"})

        print(f"  [result] {result}")
        messages.append({"role": "tool", "content": result})

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    })
    return response.json()["message"]["content"]


def main():
    print("Agent ready. Type 'quit' to exit.\n")
    print("Available tools: convert_currency, calculate_ema, system_health")
    print("-" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. You have tools for currency conversion, "
                "stock EMA calculation, and server health checks. Use them when relevant. "
                "Keep responses concise. Remember context from earlier in the conversation."
            ),
        },
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        })
        result = response.json()
        assistant_message = result["message"]

        if "tool_calls" in assistant_message:
            answer = process_tool_calls(messages, assistant_message)
        else:
            answer = assistant_message["content"]

        print(f"\nAssistant: {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
