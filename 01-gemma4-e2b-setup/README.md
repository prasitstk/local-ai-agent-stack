# 01 — Getting Started with Gemma 4 E2B on a CPU-Only Server

Run Google's Gemma 4 E2B model on a DigitalOcean droplet with no GPU. This guide covers server setup, model serving with Ollama, and benchmarking inference performance so you know exactly what to expect.

## Why Gemma 4 E2B?

Gemma 4 E2B is Google DeepMind's edge-optimized model with an effective 2 billion parameters. It was designed to run on phones and laptops, which makes it ideal for a CPU-only cloud server. Despite its small size, it supports:

- **128K token context window** — enough for long documents
- **Native function calling** — the model can request tool use without hacks
- **Built-in thinking mode** — step-by-step reasoning when you need it
- **Multimodal input** — text, images, and audio
- **Apache 2.0 license** — use it commercially, no strings attached

The Q4_K_M quantized version is about 1.3 GB. That's small enough to leave plenty of room for your application stack on an 8 GB server.

## Prerequisites

- A DigitalOcean account (or any cloud provider — the steps are nearly identical)
- Basic comfort with SSH and the Linux command line
- About 30 minutes

## Step 1: Provision the Server

Create a droplet with the following specs:

| Setting | Value |
|---------|-------|
| Image | Ubuntu 24.04 LTS |
| Plan | Regular (CPU) — 8 GB / 4 vCPUs |
| Region | Closest to you |
| Auth | SSH key (recommended) |

SSH into your server:

```bash
ssh root@your-server-ip
```

Update the system and install essentials:

```bash
apt update && apt upgrade -y
apt install -y curl wget htop
```

## Step 2: Install Ollama

Ollama is the simplest way to serve open models. One command handles downloading the runtime, setting up the systemd service, and exposing an API.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the installation:

```bash
ollama --version
```

Ollama runs as a systemd service by default. Check that it's active:

```bash
systemctl status ollama
```

You should see `active (running)`. The API is now live on `http://localhost:11434`.

## Step 3: Pull and Run Gemma 4 E2B

Download the model:

```bash
ollama pull gemma4:e2b
```

This pulls the default quantization, which balances quality and performance for resource-constrained environments. On an 8 GB droplet, the download takes a few minutes.

Start an interactive chat to verify everything works:

```bash
ollama run gemma4:e2b
```

Try a few prompts:

```
>>> What is container isolation and why does it matter for AI agents?
>>> Explain the EMA crossover strategy in stock trading, keep it brief.
>>> Write a Python function that validates an email address.
```

Type `/bye` to exit.

## Step 4: Test the REST API

Ollama exposes a full REST API automatically. This is what your applications will use to communicate with the model.

**Basic chat completion:**

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "gemma4:e2b",
  "messages": [
    {"role": "user", "content": "Explain Docker in one sentence."}
  ],
  "stream": false
}' | python3 -m json.tool
```

**With a system prompt:**

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "gemma4:e2b",
  "messages": [
    {"role": "system", "content": "You are a DevOps assistant. Be concise."},
    {"role": "user", "content": "What is the difference between a container and a VM?"}
  ],
  "stream": false
}' | python3 -m json.tool
```

**Streaming response** (token by token, useful for chat UIs):

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4:e2b",
  "messages": [
    {"role": "user", "content": "List 3 benefits of infrastructure as code."}
  ],
  "stream": true
}'
```

## Step 5: Benchmark Performance

Understanding your baseline performance is important. On a CPU-only server, you need to know if the speed is acceptable for your use case.

Create a benchmark script:

```bash
cat > benchmark.sh << 'EOF'
#!/bin/bash

echo "=== Gemma 4 E2B — CPU Benchmark ==="
echo "Date: $(date)"
echo "Server: $(hostname)"
echo "CPU: $(nproc) vCPUs"
echo "RAM: $(free -h | awk '/Mem:/ {print $2}')"
echo ""

# Short prompt
echo "--- Test 1: Short prompt (simple question) ---"
time curl -s http://localhost:11434/api/chat -d '{
  "model": "gemma4:e2b",
  "messages": [{"role": "user", "content": "What is Kubernetes?"}],
  "stream": false
}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
msg = data['message']['content']
total = data.get('total_duration', 0) / 1e9
eval_count = data.get('eval_count', 0)
print(f'Tokens generated: {eval_count}')
print(f'Total time: {total:.2f}s')
if total > 0:
    print(f'Speed: {eval_count/total:.1f} tokens/sec')
print(f'Response: {msg[:200]}...')
"

echo ""

# Medium prompt
echo "--- Test 2: Medium prompt (code generation) ---"
time curl -s http://localhost:11434/api/chat -d '{
  "model": "gemma4:e2b",
  "messages": [{"role": "user", "content": "Write a Python function that reads a CSV file and returns the top 5 rows sorted by a given column. Include error handling."}],
  "stream": false
}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
msg = data['message']['content']
total = data.get('total_duration', 0) / 1e9
eval_count = data.get('eval_count', 0)
print(f'Tokens generated: {eval_count}')
print(f'Total time: {total:.2f}s')
if total > 0:
    print(f'Speed: {eval_count/total:.1f} tokens/sec')
"

echo ""

# Thinking mode
echo "--- Test 3: Reasoning with thinking mode ---"
time curl -s http://localhost:11434/api/chat -d '{
  "model": "gemma4:e2b",
  "messages": [
    {"role": "system", "content": "Think step by step before answering."},
    {"role": "user", "content": "A trader buys 100 shares at 50 THB each. The price drops 10%, they buy 100 more. What is their average cost per share?"}
  ],
  "stream": false
}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
total = data.get('total_duration', 0) / 1e9
eval_count = data.get('eval_count', 0)
print(f'Tokens generated: {eval_count}')
print(f'Total time: {total:.2f}s')
if total > 0:
    print(f'Speed: {eval_count/total:.1f} tokens/sec')
"

echo ""
echo "=== Benchmark complete ==="
EOF

chmod +x benchmark.sh
./benchmark.sh
```

### What to Expect

Based on testing with an 8 GB / 4 vCPU DigitalOcean droplet:

| Metric | Typical Range |
|--------|---------------|
| Tokens/sec | 5–15 tok/s |
| Short answer latency | 3–8 seconds |
| Code generation (100+ tokens) | 10–30 seconds |
| Memory usage (idle) | ~2–3 GB |
| Memory usage (inference) | ~3–5 GB |

This is fast enough for demos, personal tools, and async workflows. For real-time multi-user chat, you'd want a GPU or a larger model served via cloud API.

## Step 6: Configure Ollama for Production

A few settings to adjust before moving on to the next part.

**Allow remote API access** (needed when Open WebUI runs in Docker):

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d

cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

> **Security note:** This opens Ollama to all network interfaces. In [Part 02](../02-self-hosted-chatbot/), we'll restrict access using Docker networking so only the chat UI can reach it. For now, make sure your droplet's firewall blocks port 11434 from the public internet.

**Set up UFW firewall:**

```bash
ufw allow OpenSSH
ufw enable
# Do NOT allow 11434 publicly
```

## Step 7: Monitor Resource Usage

Keep an eye on your server while the model is running:

```bash
# Real-time process monitoring
htop

# Check Ollama's memory footprint
ps aux | grep ollama

# Watch memory during inference
watch -n 1 free -h
```

## What I Learned

1. **Quantization matters on CPU.** The default GGUF quantization Ollama uses works well. Going to full precision would be impractical on 8 GB RAM.
2. **First inference is slow.** The model takes a few seconds to load into memory on the first request. Subsequent requests are faster because the model stays cached.
3. **Context length affects speed.** Keeping prompts focused and concise gives noticeably faster responses. The 128K window is there when you need it, but don't use it by default.
4. **CPU-only is viable for single-user tools.** The 5–15 tok/s range is perfectly fine for personal assistants, automation scripts, and demos. It's not suitable for serving 10 concurrent users.

## Files in This Repo

```
01-gemma4-e2b-setup/
├── README.md          # This guide
└── benchmark.sh       # Performance benchmark script
```

## Next

In [Part 02](../02-self-hosted-chatbot/), we'll add a web-based chat interface using Open WebUI and Docker Compose, turning this into a self-hosted ChatGPT alternative.
