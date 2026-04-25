# 02 — Self-Hosted AI Chatbot with Open WebUI

Build a ChatGPT-like web interface powered by Gemma 4 E2B running on your own server. No API keys, no subscriptions, no data leaving your infrastructure.

> **Prerequisite:** Complete [Part 01](../01-gemma4-e2b-setup/) first. You should have Ollama running with Gemma 4 E2B on your DigitalOcean droplet.

## What We're Building

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Browser    │─────►│  Open WebUI  │─────►│   Ollama     │
│  (any device)│ HTTPS│  (port 3000) │  API │  (port 11434)│
│              │◄─────│              │◄─────│  Gemma 4 E2B │
└──────────────┘      └──────────────┘      └──────────────┘
                      │  Docker network (internal)  │
```

Open WebUI gives you chat history, conversation management, model switching, prompt templates, and a clean responsive UI — all self-hosted.

## Step 1: Install Docker and Docker Compose

If Docker isn't already on your droplet:

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Add your user to the docker group (if not root)
usermod -aG docker $USER

# Verify
docker --version
docker compose version
```

## Step 2: Create the Project Structure

```bash
mkdir -p ~/chatbot && cd ~/chatbot
```

Create the Docker Compose file:

```bash
cat > docker-compose.yml << 'EOF'
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - WEBUI_AUTH=true
      - WEBUI_NAME=My AI Assistant
      - ENABLE_SIGNUP=false
    volumes:
      - open-webui-data:/app/backend/data
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  open-webui-data:
EOF
```

### What each setting does

| Setting | Purpose |
|---------|---------|
| `OLLAMA_BASE_URL` | Tells Open WebUI where Ollama is running. `host.docker.internal` reaches the host machine from inside the container. |
| `WEBUI_AUTH=true` | Requires login. The first user to register becomes the admin. |
| `ENABLE_SIGNUP=false` | Prevents random people from creating accounts after you've registered. |
| `ports: 3000:8080` | Maps the container's internal port 8080 to port 3000 on the host. |
| `open-webui-data` | Persists chat history and settings across container restarts. |

## Step 3: Start the Stack

```bash
docker compose up -d
```

Check that it's running:

```bash
docker compose ps
docker compose logs -f open-webui
```

Wait until you see the startup complete message in the logs. This usually takes 30–60 seconds on the first run while it initializes the database.

## Step 4: Initial Setup

1. Open your browser to `http://your-server-ip:3000`
2. Register your admin account (first registration only)
3. You should see Gemma 4 E2B listed as an available model
4. Start chatting

If the model doesn't appear, check the connection:

```bash
# From inside the container, can it reach Ollama?
docker exec open-webui curl -s http://host.docker.internal:11434/api/tags | python3 -m json.tool
```

## Step 5: Secure with HTTPS (Caddy Reverse Proxy)

Running on plain HTTP is fine for testing but not for showing others. Caddy gives you automatic HTTPS with zero configuration.

**You'll need a domain name** pointing to your droplet's IP. If you don't have one, you can use a free subdomain from services like DuckDNS or freedns.afraid.org.

Add Caddy to your Docker Compose:

```bash
cat > docker-compose.yml << 'EOF'
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped
    expose:
      - "8080"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - WEBUI_AUTH=true
      - WEBUI_NAME=My AI Assistant
      - ENABLE_SIGNUP=false
    volumes:
      - open-webui-data:/app/backend/data
    extra_hosts:
      - "host.docker.internal:host-gateway"

  caddy:
    image: caddy:2-alpine
    container_name: caddy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data
      - caddy-config:/config
    depends_on:
      - open-webui

volumes:
  open-webui-data:
  caddy-data:
  caddy-config:
EOF
```

Create the Caddyfile:

```bash
cat > Caddyfile << 'EOF'
your-domain.com {
    reverse_proxy open-webui:8080
}
EOF
```

Replace `your-domain.com` with your actual domain.

Open the firewall and restart:

```bash
ufw allow 80
ufw allow 443
docker compose down
docker compose up -d
```

Caddy automatically obtains and renews TLS certificates from Let's Encrypt. Your chatbot is now accessible at `https://your-domain.com`.

## Step 6: Customize the Experience

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

## Step 7: Operational Basics

### Updating Open WebUI

```bash
cd ~/chatbot
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
