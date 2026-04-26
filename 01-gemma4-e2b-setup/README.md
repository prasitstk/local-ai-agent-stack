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

> `python3 -m json.tool` is Python's built-in JSON pretty-printer. It reads JSON on stdin and re-emits it indented and key-sorted (and errors out on malformed input, so it doubles as a validator). `jq .` is a more powerful alternative if you have it installed.

### Understanding the response

For a non-streaming request, Ollama returns a single JSON object shaped like this:

```json
{
    "model": "gemma4:e2b",
    "created_at": "2026-04-25T05:56:29Z",
    "message": {
        "role": "assistant",
        "content": "Docker is a platform that ...",
        "thinking": "1. Analyze the request: ..."
    },
    "done": true,
    "done_reason": "stop",
    "total_duration": 101741800079,
    "load_duration": 37802053344,
    "prompt_eval_count": 22,
    "prompt_eval_duration": 20382145576,
    "eval_count": 274,
    "eval_duration": 41017298062
}
```

The schema is defined by **Ollama's REST API**, not by the model or by `json.tool`. The model only emits raw tokens; Ollama wraps them and adds the timing metadata. Key fields:

| Field | Meaning |
|---|---|
| `message.content` | The user-visible reply. |
| `message.thinking` | Gemma 4's internal chain-of-thought (built-in thinking mode). Hide from end users; useful for debugging. |
| `done` / `done_reason` | Whether generation finished, and why (`stop`, `length`, `load`). |
| `prompt_eval_count` / `prompt_eval_duration` | Input tokens processed and time spent prefilling (nanoseconds). |
| `eval_count` / `eval_duration` | Output tokens generated and time spent decoding (nanoseconds). |
| `load_duration` | Time spent loading the model into RAM. Large on the first call, near zero while the model stays cached (see Step 6). |
| `total_duration` | Wall-clock time for the whole request (nanoseconds). |

Output speed (tokens/sec) = `eval_count / (eval_duration / 1e9)` — this is the number to compare against the benchmark table in the next step.

If you'd rather use the OpenAI Chat Completions schema, Ollama also exposes a compatible endpoint at `/v1/chat/completions`.

## Step 5: Benchmark Performance

Understanding your baseline performance is important. On a CPU-only server, you need to know if the speed is acceptable for your use case.

This repo ships a `benchmark.sh` next to this README. It runs four tests against the local Ollama API — a short factual question, code generation, multi-step reasoning, and JSON structured output — then prints tokens generated, prompt-eval time, generation time, and tokens/sec for each, plus a memory snapshot at the end.

Copy it onto your droplet (or `git clone` this repo) and run it:

```bash
chmod +x benchmark.sh
./benchmark.sh
```

The first run will be slow because of the cold-start `load_duration` (see Step 4). Run it a second time to get steady-state numbers.

### What to Expect

Measured on an 8 GB / 4 vCPU DigitalOcean droplet (DO-Regular CPUs) with the four-test `benchmark.sh`:

| Metric | Observed |
|--------|----------|
| Generation speed | **5.5–6.7 tok/s** (steady state across all four tests) |
| Prompt eval (warm) | 2–3 s for typical prompts |
| Cold-start overhead | +35–40 s on the very first request after the model unloads (the `load_duration` field) |
| Short factual answer (~280 tokens) | ~50 s warm; ~110 s cold |
| Code generation (~2000 tokens) | ~5.5 minutes |
| Memory while model is loaded | ~7.1 GB resident — this is why swap is required on an 8 GB droplet |
| Memory after `OLLAMA_KEEP_ALIVE` expires | ~150 MB (`ollama serve` only) |

A few takeaways:

- **Wall-clock time scales with response length, not prompt length.** At ~6 tok/s, every 100 output tokens costs ~16 seconds.
- **First call after idle is slow.** The 35–40 s `load_duration` only happens when the RAM cache is cold. Raise `OLLAMA_KEEP_ALIVE` (Step 6) if your workload is bursty.
- **Memory headroom is tight.** Once loaded, the model holds ~7 GB; the OS, Open WebUI, etc. share what's left. Without swap, you'll OOM under load.

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

**Tune model keep-alive (RAM cache):**

After the first inference, Ollama keeps the model loaded in RAM so subsequent requests skip the multi-second `load_duration` you saw in Step 4. The default keep-alive is 5 minutes — after that, the model unloads and the next request pays the load cost again.

For an always-on personal tool, raise it by adding another `Environment` line to the same override file:

```bash
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=24h"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Use `-1` to keep the model loaded forever, `0` to unload immediately after each request. You can also override per-request with a `keep_alive` field in the JSON body.

**Where Ollama caches the model:**

Two distinct caches, both managed by the `ollama serve` process:

| Cache | Where | Lifetime | Cost when missed |
|---|---|---|---|
| Disk — model files (populated by `ollama pull`) | `/usr/share/ollama/.ollama/models/` (`blobs/` holds content-addressed weights, `manifests/` maps tags to blobs) | Survives reboots | Re-download from the registry |
| RAM — loaded weights | `ollama serve` process memory | `OLLAMA_KEEP_ALIVE` (default `5m`) | `load_duration` ≈ 30–40 s on this droplet |

Inspect what's on disk with `ollama list` or `sudo ls -lh /usr/share/ollama/.ollama/models/blobs/`. If you run Ollama as a regular user instead of the systemd service, the disk path is `~/.ollama/models/`. Override with the `OLLAMA_MODELS` env var.

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

## Step 8: Tear It Down

Removes the Ollama service, its model cache, and the user/group the installer created. Don't run this if you're moving on to Parts 02–04 — they all reuse this Ollama instance. Tear those down first.

```bash
# Stop the service and remove its systemd unit + override
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm -rf /etc/systemd/system/ollama.service.d
sudo rm -f /etc/systemd/system/ollama.service
sudo systemctl daemon-reload

# Remove the binary, models, user, and per-user history
sudo rm -f /usr/local/bin/ollama
sudo rm -rf /usr/share/ollama
sudo userdel ollama 2>/dev/null
sudo groupdel ollama 2>/dev/null
rm -rf ~/.ollama
```

Reclaims the model size on disk (≈ 7.2 GB for `gemma4:e2b`). Leaves the base `apt` packages from Step 1 (`curl`, `wget`, `htop`) alone — they're harmless and likely useful elsewhere.

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
