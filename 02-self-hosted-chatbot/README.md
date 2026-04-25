# 02 — Self-Hosted AI Chatbot with Open WebUI

Build a ChatGPT-like web interface powered by Gemma 4 E2B running on your own server. No API keys, no subscriptions, no data leaving your infrastructure.

> **Prerequisite:** Complete [Part 01](../01-gemma4-e2b-setup/) first. You should have Ollama running with Gemma 4 E2B on your DigitalOcean droplet.

## What We're Building

```
┌──────────────┐  HTTPS  ┌──────────┐   ┌──────────────┐      ┌──────────────┐
│   Browser    │────────►│  Caddy   │──►│  Open WebUI  │─────►│   Ollama     │
│  (any device)│  443    │ (TLS +   │   │  (port 8080  │  API │  (port 11434)│
│              │◄────────│  proxy)  │◄──│   internal)  │◄─────│  Gemma 4 E2B │
└──────────────┘         └──────────┘   └──────────────┘      └──────────────┘
                          │       Docker network (internal)        │
```

Open WebUI gives you chat history, conversation management, model switching, prompt templates, and a clean responsive UI — all self-hosted.

## Step 1: Install Docker and Docker Compose

If Docker isn't already on your droplet:

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Add your user to the docker group (if not root)
sudo usermod -aG docker $USER

# Verify
docker --version
docker compose version
```

## Step 2: Configure the Shipped Compose

This part of the repo ships a production-ready `docker-compose.yml` and `Caddyfile`. Don't recreate them — `cd` into the directory and use what's there:

```bash
cd ~/local-ai-agent-stack/02-self-hosted-chatbot
ls
# Caddyfile  README.md  docker-compose.yml
```

The compose runs two services: **`open-webui`** (the chat UI, exposed only on the internal Docker network) and **`caddy`** (a reverse proxy that obtains and auto-renews a Let's Encrypt cert for your domain).

### What each setting does

| Setting | Purpose |
|---|---|
| `OLLAMA_BASE_URL` | Tells Open WebUI where Ollama is running. `host.docker.internal` reaches the host machine from inside the container. |
| `WEBUI_AUTH=true` | Requires login. The first user to register becomes the admin. |
| `ENABLE_SIGNUP=false` | Prevents random people from creating accounts after you've registered. |
| `expose: 8080` | Open WebUI is reachable only inside the Docker network — Caddy fronts it. |
| `caddy 80:80, 443:443` | Caddy terminates HTTPS and reverse-proxies to `open-webui:8080`. |
| `open-webui-data` | Persists chat history and settings across container restarts. |

### Set your domain

You'll need a domain pointing to your droplet's IP (an `A` record). If you don't have one, [DuckDNS](https://duckdns.org) gives free subdomains.

Edit the Caddyfile and replace `your-domain.com` with your actual domain:

```bash
nano Caddyfile
```

That's the only file edit required.

### Don't have a domain yet?

If you just want to verify Open WebUI works before setting up DNS, run *without* Caddy. Skip ahead to Step 3 with this override file:

```bash
cat > docker-compose.override.yml << 'EOF'
services:
  open-webui:
    ports:
      - "3000:8080"
  caddy:
    profiles: ["disabled"]
EOF
```

This exposes Open WebUI on `http://your-server-ip:3000` directly and disables the Caddy service via Compose profiles. Delete the override file (`rm docker-compose.override.yml`) and `docker compose up -d` again once your domain is set up.

## Step 3: Start the Stack

Open the firewall for HTTPS (only needed if you're running with Caddy):

```bash
sudo ufw allow 80
sudo ufw allow 443
```

Then start everything:

```bash
docker compose up -d
docker compose ps
docker compose logs -f open-webui
```

Wait until you see the startup complete message in the logs. This usually takes 30–60 seconds on the first run while it initializes the database. Caddy will obtain a Let's Encrypt cert in parallel — its logs (`docker compose logs caddy`) will tell you when it's done.

## Step 4: Initial Setup

1. Open your browser to `https://your-domain.com` (or `http://your-server-ip:3000` if you used the no-domain override).
2. Register your admin account (first registration only).
3. You should see Gemma 4 E2B listed as an available model.
4. Start chatting.

If the model doesn't appear, check the connection:

```bash
# From inside the container, can it reach Ollama?
docker exec open-webui curl -s http://host.docker.internal:11434/api/tags | python3 -m json.tool
```

## Step 5: Customize the Experience

### Set a default system prompt

In Open WebUI, go to **Settings → General** and set a default system prompt. Here's one I use:

```
You are a helpful AI assistant running locally on a private server.
Be concise and practical. When writing code, include brief comments
explaining the key parts. If you're unsure about something, say so.
```

### Create prompt templates

Open WebUI supports saved prompt templates. Some useful ones to create:

**Code Review:**
```
Review this code for bugs, security issues, and improvement opportunities.
Focus on practical issues, not style preferences.

```code
{paste code here}
```
```

**Explain Like a README:**
```
Explain the following concept as if you're writing a section
in a project README. Use clear headings and a code example.

Topic: {topic}
```

**DevOps Troubleshoot:**
```
I'm seeing this error in my deployment. Walk me through
diagnosing the root cause step by step.

Error: {paste error}
Environment: {e.g., Docker on Ubuntu 24.04}
```

## Step 6: Operational Basics

### Updating Open WebUI

```bash
cd ~/local-ai-agent-stack/02-self-hosted-chatbot
docker compose pull
docker compose up -d
```

### Backup chat history

The conversation database lives in the Docker volume. To back it up:

```bash
# Create a backup
docker run --rm \
  -v open-webui-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/open-webui-backup-$(date +%Y%m%d).tar.gz /data

# List backups
ls -la backups/
```

### Check resource usage

```bash
# Container stats
docker stats --no-stream

# Disk usage
docker system df
```

### Common issues

| Problem | Solution |
|---------|----------|
| Model not showing up | Run `docker exec open-webui curl http://host.docker.internal:11434/api/tags` to check connectivity |
| Slow first response | Normal — model loads into RAM on first request. Keep Ollama running to avoid cold starts. |
| Out of memory | Check `docker stats`. Open WebUI itself uses ~500 MB. Combined with Ollama (~3 GB), you need headroom. |
| Chat history lost | Make sure the `open-webui-data` volume is defined in `docker-compose.yml` |

## What I Learned

1. **Docker networking has nuances.** The `host.docker.internal` trick is the cleanest way to reach host services from containers without using `--network host`, which would bypass Docker's network isolation.
2. **Caddy is underrated.** Compared to Nginx + Certbot, Caddy handles HTTPS automatically with near-zero configuration. For small deployments like this, it's perfect.
3. **Open WebUI is surprisingly complete.** It supports multiple models, RAG (document upload), prompt templates, user management, and even image generation. We're only scratching the surface here.
4. **Backups are easy to forget.** The Docker volume approach makes it simple, but you have to actually set up the cron job. Don't lose your conversation history like I almost did.

## Files in This Repo

```
02-self-hosted-chatbot/
├── README.md            # This guide
├── docker-compose.yml   # Production compose with Caddy
└── Caddyfile            # Reverse proxy config (edit your domain)
```

## Next

In [Part 03](../03-function-calling-basics/), we'll go beyond chat. We'll teach the model to call external tools — the foundation of every AI agent.
