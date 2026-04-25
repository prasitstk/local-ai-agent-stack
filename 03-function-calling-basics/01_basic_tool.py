"""
01_basic_tool.py — Minimal function calling example.

Demonstrates the complete tool-call loop:
  1. Define a tool schema
  2. Send it to the model with a user prompt
  3. Detect when the model wants to call the tool
  4. Execute the function locally
  5. Return the result for the model to summarize
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e2b"


# --- Tool definition ---
# This tells the model what tools are available and how to call them.
# The schema follows a standard format: name, description, and parameters.

def get_current_time(timezone: str) -> str:
    """Return the current time for a given timezone."""
    from datetime import datetime, timezone as tz, timedelta

    offsets = {
        "Asia/Bangkok": 7,
        "America/New_York": -4,
        "Europe/London": 1,
        "Asia/Tokyo": 9,
        "UTC": 0,
    }
    offset = offsets.get(timezone, 0)
    dt = datetime.now(tz.utc) + timedelta(hours=offset)
    return json.dumps({
        "timezone": timezone,
        "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
    })


# The schema the model sees — it never sees your Python code
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in a specific timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name (e.g., Asia/Bangkok, UTC)",
                    }
                },
                "required": ["timezone"],
            },
        },
    }
]

# Map function names to actual Python functions
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
}


def chat_with_tools(user_message: str) -> str:
    """Send a message to the model with tool access and handle the full loop."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed to answer questions accurately."},
        {"role": "user", "content": user_message},
    ]

    # Step 1: Send the prompt with tool definitions
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

    # Step 2: Check if the model wants to call a tool
    if "tool_calls" not in assistant_message:
        # No tool call — model answered directly
        print(f"Model (direct): {assistant_message['content']}")
        return assistant_message["content"]

    # Step 3: Execute each tool call
    messages.append(assistant_message)  # Add model's response to history

    for tool_call in assistant_message["tool_calls"]:
        func_name = tool_call["function"]["name"]
        func_args = tool_call["function"]["arguments"]

        print(f"Tool call: {func_name}({json.dumps(func_args)})")

        # Look up and execute the function
        if func_name in AVAILABLE_FUNCTIONS:
            func_result = AVAILABLE_FUNCTIONS[func_name](**func_args)
            print(f"Tool result: {func_result}")
        else:
            func_result = json.dumps({"error": f"Unknown function: {func_name}"})

        # Add the tool result to the conversation
        messages.append({
            "role": "tool",
            "content": func_result,
        })

    # Step 4: Send the tool results back for the model to summarize
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    })
    final = response.json()
    answer = final["message"]["content"]
    print(f"Model (final): {answer}")
    return answer


if __name__ == "__main__":
    # This should trigger a tool call
    chat_with_tools("What time is it right now in Bangkok?")

    # This should answer directly (no tool needed)
    chat_with_tools("What is the capital of Thailand?")
