"""
agent.py — Security-hardened agent with tool execution.

Loads config, registers tools, runs the agent loop with
prompt injection detection and resource limits.
"""

import json
import os
import re
import sys
import time

import requests
import yaml


def load_config(path: str = "config.yml") -> dict:
    """Load agent configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def check_prompt_injection(text: str, patterns: list[str]) -> bool:
    """Return True if the text contains a suspected prompt injection."""
    text_lower = text.lower()
    for pattern in patterns:
        if pattern.lower() in text_lower:
            return True
    return False


def sanitize_input(text: str, max_length: int) -> str:
    """Truncate and clean user input."""
    text = text[:max_length]
    # Remove null bytes and control characters (except newlines)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


# --- Tool implementations ---
# Each tool runs inside the container with limited permissions.

def tool_disk_usage(**kwargs) -> str:
    """Check disk usage — reads from /proc, no shell commands."""
    try:
        stat = os.statvfs("/workspace")
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bfree * stat.f_frsize
        used = total - free
        return json.dumps({
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage_percent": round((used / total) * 100, 1),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_service_status(service: str = "ollama") -> str:
    """Check if Ollama is reachable (the only service we should access)."""
    allowed_services = {"ollama": "http://ollama:11434"}
    if service not in allowed_services:
        return json.dumps({
            "error": f"Access denied. Can only check: {list(allowed_services.keys())}"
        })
    try:
        r = requests.get(allowed_services[service], timeout=5)
        return json.dumps({"service": service, "status": "running", "response_code": r.status_code})
    except requests.exceptions.ConnectionError:
        return json.dumps({"service": service, "status": "unreachable"})
    except requests.exceptions.Timeout:
        return json.dumps({"service": service, "status": "timeout"})


def tool_container_list(**kwargs) -> str:
    """List containers — reads from workspace log file, not Docker socket."""
    # Security: We do NOT mount the Docker socket into this container.
    # Instead, a separate sidecar writes container stats to a shared file.
    stats_file = "/workspace/container_stats.json"
    try:
        with open(stats_file) as f:
            return f.read()
    except FileNotFoundError:
        return json.dumps({"info": "Container stats not yet available. The stats collector may not be running."})


def tool_log_tail(service: str = "agent", lines: int = 20) -> str:
    """Read log lines — restricted to workspace directory only."""
    lines = min(lines, 50)  # Hard cap
    allowed_logs = {
        "agent": "/workspace/logs/agent.log",
    }
    log_path = allowed_logs.get(service)
    if not log_path:
        return json.dumps({"error": f"Can only read logs for: {list(allowed_logs.keys())}"})
    try:
        with open(log_path) as f:
            all_lines = f.readlines()
            return json.dumps({"service": service, "lines": all_lines[-lines:]})
    except FileNotFoundError:
        return json.dumps({"info": f"No log file found for {service}"})


# --- Tool registry ---

TOOL_REGISTRY = {
    "disk_usage": {
        "function": tool_disk_usage,
        "schema": {
            "type": "function",
            "function": {
                "name": "disk_usage",
                "description": "Check disk space usage on the server workspace.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
    "service_status": {
        "function": tool_service_status,
        "schema": {
            "type": "function",
            "function": {
                "name": "service_status",
                "description": "Check if Ollama is running and reachable.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "Service to check (currently only 'ollama')"},
                    },
                },
            },
        },
    },
    "container_list": {
        "function": tool_container_list,
        "schema": {
            "type": "function",
            "function": {
                "name": "container_list",
                "description": "List running containers and their resource usage.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
    "log_tail": {
        "function": tool_log_tail,
        "schema": {
            "type": "function",
            "function": {
                "name": "log_tail",
                "description": "Read the last N lines of a service log.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "Service name (currently only 'agent')"},
                        "lines": {"type": "integer", "description": "Number of lines to read (max 50)"},
                    },
                },
            },
        },
    },
}


def run_agent(config: dict):
    """Main agent loop."""
    agent_cfg = config["agent"]
    security_cfg = config["security"]

    # Build tool list from config
    enabled_tools = [t["name"] for t in config["tools"] if t.get("enabled", True)]
    tools = [TOOL_REGISTRY[name]["schema"] for name in enabled_tools if name in TOOL_REGISTRY]
    functions = {name: TOOL_REGISTRY[name]["function"] for name in enabled_tools if name in TOOL_REGISTRY}

    messages = [{"role": "system", "content": agent_cfg["system_prompt"]}]

    # Set up logging
    os.makedirs("/workspace/logs", exist_ok=True)
    log_file = open("/workspace/logs/agent.log", "a")

    def log(msg: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {msg}\n")
        log_file.flush()

    print(f"Agent '{agent_cfg['name']}' ready. Type 'quit' to exit.")
    print(f"Enabled tools: {', '.join(enabled_tools)}")
    print(f"Security: prompt injection detection ON, max input {security_cfg['max_input_length']} chars")
    print("-" * 60)

    turn_count = 0

    while turn_count < agent_cfg["max_turns"]:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if not user_input:
            continue

        # --- Security checks ---
        user_input = sanitize_input(user_input, security_cfg["max_input_length"])

        if check_prompt_injection(user_input, security_cfg["blocked_patterns"]):
            print("\nAssistant: I can't process that request.")
            log(f"BLOCKED prompt injection attempt: {user_input[:100]}")
            continue

        log(f"USER: {user_input[:200]}")
        messages.append({"role": "user", "content": user_input})
        turn_count += 1

        # --- Agent loop (with tool-call limit) ---
        for round_num in range(agent_cfg["max_tool_rounds"]):
            try:
                response = requests.post(
                    f"{agent_cfg['ollama_url']}/api/chat",
                    json={
                        "model": agent_cfg["model"],
                        "messages": messages,
                        "tools": tools,
                        "stream": False,
                    },
                    timeout=120,
                )
                result = response.json()
            except requests.exceptions.RequestException as e:
                print(f"\nAssistant: Sorry, I couldn't reach the model. Error: {e}")
                log(f"ERROR: Model request failed: {e}")
                break

            assistant_message = result["message"]

            if "tool_calls" not in assistant_message:
                # No tool call — deliver the response
                answer = assistant_message.get("content", "")
                print(f"\nAssistant: {answer}")
                messages.append({"role": "assistant", "content": answer})
                log(f"ASSISTANT: {answer[:200]}")
                break

            # Handle tool calls
            messages.append(assistant_message)

            for tool_call in assistant_message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]
                log(f"TOOL_CALL: {func_name}({json.dumps(func_args)})")

                if func_name in functions:
                    try:
                        tool_result = functions[func_name](**func_args)
                    except Exception as e:
                        tool_result = json.dumps({"error": f"Tool execution failed: {str(e)}"})
                        log(f"TOOL_ERROR: {func_name}: {e}")
                else:
                    tool_result = json.dumps({"error": f"Tool '{func_name}' not available"})
                    log(f"TOOL_UNKNOWN: {func_name}")

                messages.append({"role": "tool", "content": tool_result})
                log(f"TOOL_RESULT: {tool_result[:200]}")
        else:
            print("\nAssistant: I've reached the maximum number of tool calls for this request. Here's what I have so far.")
            log("WARNING: Max tool rounds reached")

    log("Agent session ended")
    log_file.close()
    print("\nSession ended.")


if __name__ == "__main__":
    config = load_config()
    run_agent(config)
