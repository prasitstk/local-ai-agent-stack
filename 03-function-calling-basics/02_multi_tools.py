"""
02_multi_tools.py — Multiple tools the model can choose between.

The model decides which tool to use (or none) based on the question.
Tool definitions live in multi_tools_lib.py so 03_agent_loop.py can
reuse them without copy-paste.
"""

import json
import requests

from multi_tools_lib import TOOLS, AVAILABLE_FUNCTIONS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e2b"


def chat_with_tools(user_message: str) -> str:
    """Full tool-calling loop with multiple tools."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools for currency conversion, "
                "stock technical analysis, and server monitoring. Use the appropriate tool "
                "when the user's question requires real data. Answer directly when no tool is needed."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}")

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    })
    result = response.json()
    assistant_message = result["message"]

    if "tool_calls" not in assistant_message:
        print(f"Model (direct): {assistant_message['content']}")
        return assistant_message["content"]

    messages.append(assistant_message)

    for tool_call in assistant_message["tool_calls"]:
        func_name = tool_call["function"]["name"]
        func_args = tool_call["function"]["arguments"]
        print(f"  Tool: {func_name}({json.dumps(func_args, indent=2)})")

        if func_name in AVAILABLE_FUNCTIONS:
            func_result = AVAILABLE_FUNCTIONS[func_name](**func_args)
        else:
            func_result = json.dumps({"error": f"Unknown function: {func_name}"})

        print(f"  Result: {func_result}")
        messages.append({"role": "tool", "content": func_result})

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    })
    answer = response.json()["message"]["content"]
    print(f"Model: {answer}")
    return answer


if __name__ == "__main__":
    # Should use convert_currency
    chat_with_tools("How much is 1000 USD in Thai Baht?")

    # Should use calculate_ema
    chat_with_tools(
        "Calculate the 12-period EMA for these closing prices: "
        "45.2, 46.1, 45.8, 46.5, 47.0, 46.8, 47.2, 47.5, 47.1, "
        "47.8, 48.0, 47.6, 48.2, 48.5"
    )

    # Should use system_health
    chat_with_tools("Is Ollama running? How much memory is it using?")

    # Should answer directly — no tool needed
    chat_with_tools("What does EMA stand for in trading?")
