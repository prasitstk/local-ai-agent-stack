# 03 — Function Calling from Scratch with Gemma 4 E2B

Teach a local LLM to use external tools. This guide builds a Python agent that gives Gemma 4 E2B the ability to check the weather, convert currencies, and look up stock prices — all through function calling, no frameworks required.

> **Prerequisite:** Complete [Part 01](../01-gemma4-e2b-setup/). Ollama with Gemma 4 E2B must be running on your server.

## Why This Matters

A chatbot that can only generate text is useful but limited. An AI that can **take actions** — querying databases, calling APIs, running calculations — is an agent. Function calling is the bridge between the two.

Most agent frameworks hide this mechanism behind abstractions. Here, we build it manually so you understand exactly what happens when an LLM "uses a tool."

## How Function Calling Works

The flow is straightforward:

```
┌──────┐     1. Prompt + tool definitions     ┌──────────┐
│      │ ──────────────────────────────────►   │          │
│      │     2. Model returns tool_call        │  Gemma 4 │
│ Your │ ◄──────────────────────────────────   │   E2B    │
│ Code │     3. You execute the function       │          │
│      │     4. Send result back to model      │          │
│      │ ──────────────────────────────────►   │          │
│      │     5. Model gives final answer       │          │
│      │ ◄──────────────────────────────────   │          │
└──────┘                                       └──────────┘
```

The model never runs code itself. It outputs a structured request saying "call this function with these arguments." Your code runs the function and feeds the result back. The model then uses that result to compose a natural language answer.

## Step 1: Set Up the Python Environment

```bash
mkdir -p ~/function-calling && cd ~/function-calling

python3 -m venv venv
source venv/bin/activate
pip install requests
```

## Step 2: Your First Tool Call

Create `01_basic_tool.py`:

```python
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
```

Run it:

```bash
python 01_basic_tool.py
```

You should see the model requesting a tool call for the time question, your code executing the function, and the model composing a final answer using the result. The capital question should be answered directly without any tool call.

## Step 3: Multiple Tools

Now let's give the model several tools to choose from. Create `02_multi_tools.py`:

```python
"""
02_multi_tools.py — Multiple tools the model can choose between.

The model decides which tool to use (or none) based on the question.
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e2b"


# --- Tool implementations ---

def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies using approximate rates."""
    rates = {
        ("USD", "THB"): 34.5,
        ("THB", "USD"): 0.029,
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("USD", "JPY"): 157.0,
        ("JPY", "USD"): 0.0064,
        ("THB", "JPY"): 4.55,
    }
    key = (from_currency.upper(), to_currency.upper())
    if key in rates:
        result = amount * rates[key]
        return json.dumps({
            "amount": amount,
            "from": from_currency,
            "to": to_currency,
            "result": round(result, 2),
            "rate": rates[key],
            "note": "Approximate rate for demonstration purposes",
        })
    return json.dumps({"error": f"Rate not available for {from_currency} to {to_currency}"})


def calculate_ema(prices: list, period: int) -> str:
    """Calculate Exponential Moving Average for a list of prices."""
    if len(prices) < period:
        return json.dumps({"error": f"Need at least {period} prices, got {len(prices)}"})

    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period  # SMA as seed

    ema_values = []
    for i, price in enumerate(prices):
        if i < period:
            continue
        ema = (price - ema) * multiplier + ema
        ema_values.append(round(ema, 2))

    return json.dumps({
        "period": period,
        "latest_ema": ema_values[-1] if ema_values else round(ema, 2),
        "ema_series": ema_values[-5:],  # Last 5 values
    })


def system_health(service: str) -> str:
    """Check if a service is running (simulated)."""
    services = {
        "ollama": {"status": "running", "uptime": "3d 14h", "memory_mb": 2840},
        "open-webui": {"status": "running", "uptime": "3d 14h", "memory_mb": 512},
        "nginx": {"status": "stopped", "uptime": "0", "memory_mb": 0},
        "postgres": {"status": "running", "uptime": "10d 2h", "memory_mb": 156},
    }
    info = services.get(service.lower())
    if info:
        return json.dumps({"service": service, **info})
    return json.dumps({"error": f"Unknown service: {service}", "available": list(services.keys())})


# --- Tool schemas ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another. Supports USD, THB, EUR, JPY.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert"},
                    "from_currency": {"type": "string", "description": "Source currency code (e.g., USD)"},
                    "to_currency": {"type": "string", "description": "Target currency code (e.g., THB)"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ema",
            "description": "Calculate the Exponential Moving Average (EMA) for a list of stock prices. Useful for technical analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prices": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of closing prices, oldest first",
                    },
                    "period": {"type": "integer", "description": "EMA period (e.g., 12, 26)"},
                },
                "required": ["prices", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_health",
            "description": "Check the health status of a running service on the server. Returns status, uptime, and memory usage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name (e.g., ollama, open-webui, postgres)"},
                },
                "required": ["service"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "convert_currency": convert_currency,
    "calculate_ema": calculate_ema,
    "system_health": system_health,
}


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
```

Run it:

```bash
python 02_multi_tools.py
```

Watch which tool the model selects for each question. The key insight: the model reads the tool descriptions and decides which one (if any) matches the user's intent.

## Step 4: Conversational Agent with Tool Memory

Create `03_agent_loop.py` — a persistent chat session where the model remembers previous tool results:

```python
"""
03_agent_loop.py — Interactive agent with persistent conversation and tool use.

Run this and have a conversation. The agent remembers context across turns
and can use tools at any point in the conversation.
"""

import json
import readline  # Enables arrow keys and history in the terminal
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e2b"

# Reuse tools from 02_multi_tools.py
from multi_tools_lib import TOOLS, AVAILABLE_FUNCTIONS


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
```

Create the shared tool library `multi_tools_lib.py` so the agent loop can import it:

```bash
# Copy the TOOLS, AVAILABLE_FUNCTIONS, and function definitions
# from 02_multi_tools.py into multi_tools_lib.py
# (Extract everything except the __main__ block)
```

Run the interactive agent:

```bash
python 03_agent_loop.py
```

Try a multi-turn conversation:

```
You: How much is 500 USD in Thai Baht?
You: And how much would that be in Japanese Yen?
You: Is Ollama running ok on the server?
You: What is an EMA in stock trading?
```

Notice how the model keeps context — the second question references "that" (the Baht amount) from the first answer.

## Understanding the Mechanics

### What the model actually sees

When you send tool definitions, the model receives them as part of its prompt context. It doesn't "know" about your Python functions — it only sees the JSON schema describing what's available.

### What the model actually outputs

When it decides to call a tool, the raw response includes a structured `tool_calls` field instead of (or alongside) regular text content. Your code parses this structured output and maps it to real function calls.

### Why descriptions matter

The model chooses tools based on the `description` field in each schema. Vague descriptions lead to wrong tool selection. Be specific about what each tool does and when to use it.

### Error handling in production

The examples above are minimal for clarity. In a real system, you'd want to:

- Validate the model's arguments before calling functions
- Handle cases where the model calls a tool that doesn't exist
- Set timeouts on tool execution
- Limit the number of tool-call rounds to prevent infinite loops
- Log every tool call for debugging

## What I Learned

1. **Function calling is just structured output.** There's no magic. The model outputs JSON saying "call X with Y," and your code does the rest. Understanding this demystifies every agent framework.
2. **Tool descriptions are prompt engineering.** The quality of your function descriptions directly affects whether the model picks the right tool. This is where most bugs live.
3. **Small models can do tool selection.** Gemma 4 E2B at 2B parameters reliably picks the right tool for simple cases. It struggles with ambiguous requests or complex multi-tool chains — that's where larger models earn their cost.
4. **The conversation history grows fast.** Each tool call adds multiple messages. On a CPU-only server, long conversations with many tool calls will slow down noticeably.

## Files in This Repo

```
03-function-calling-basics/
├── README.md              # This guide
├── 01_basic_tool.py       # Single tool example
├── 02_multi_tools.py      # Multiple tools, model chooses
├── 03_agent_loop.py       # Interactive agent with memory
├── multi_tools_lib.py     # Shared tool definitions
└── requirements.txt       # Python dependencies
```

## Next

In [Part 04](../04-nanobot-local-agent/), we'll integrate everything into a proper agent framework with container isolation and security hardening — taking what we built manually and making it production-ready.
