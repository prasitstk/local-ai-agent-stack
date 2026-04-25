"""
multi_tools_lib.py — Shared tool definitions for 02_multi_tools.py and 03_agent_loop.py.

Three demo tools the model can choose between:
  - convert_currency: USD/THB/EUR/JPY conversion with hardcoded rates
  - calculate_ema:    Exponential Moving Average for a price series
  - system_health:    Simulated service status lookup

Each tool returns a JSON string. The TOOLS list below is the OpenAI-shaped
Chat Completions schema; AVAILABLE_FUNCTIONS maps tool names back to callables.
"""

import json


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
