# Local AI Agent Stack

A four-part guide to running self-hosted AI agents using **Gemma 4 E2B** on a CPU-only cloud server — from first install to security-hardened containers.

Each part builds on the previous one. By the end, you'll have a fully working AI agent stack running on a single DigitalOcean droplet — no GPU required.

## Who This Is For

- Developers curious about running LLMs locally but unsure where to start
- DevOps engineers exploring AI agent infrastructure
- Anyone who wants to understand how agentic AI works under the hood before adopting frameworks

## What's Inside

| # | Part | What You'll Build | Key Skills |
|---|------|-------------------|------------|
| 01 | [Getting Started with Gemma 4 E2B](./01-gemma4-e2b-setup/) | Local LLM running on a cloud server | Ollama, model serving, benchmarking |
| 02 | [Self-Hosted AI Chatbot](./02-self-hosted-chatbot/) | ChatGPT-like web interface you own | Docker Compose, reverse proxy, Open WebUI |
| 03 | [Function Calling from Scratch](./03-function-calling-basics/) | AI that can use external tools | Python, tool schemas, structured output |
| 04 | [Building a Local AI Agent](./04-nanobot-local-agent/) | Security-hardened autonomous agent | Nanobot, container isolation, agent loops |

## Hardware Requirements

Everything runs on a single **DigitalOcean droplet**:

- **CPU**: 2+ vCPUs (4 recommended)
- **RAM**: 8 GB minimum
- **Disk**: 50 GB SSD
- **GPU**: None required
- **Cost**: ~$48/month (8 GB / 4 vCPU droplet)

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│              DigitalOcean Droplet            │
│                                             │
│  ┌──────────┐   ┌────────────────────────┐  │
│  │  Ollama   │◄──│  Open WebUI (Chat UI)  │  │
│  │  Server   │   └────────────────────────┘  │
│  │          │   ┌────────────────────────┐  │
│  │ Gemma 4  │◄──│  Python Agent Scripts  │  │
│  │  E2B     │   └────────────────────────┘  │
│  │          │   ┌────────────────────────┐  │
│  │          │◄──│  Nanobot Agent Engine   │  │
│  └──────────┘   └────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## My Background

I come from a DevOps background working with AWS, Azure, and trading systems. This series is part of a broader project exploring how open-source AI agents can replace commercial SaaS products with self-hosted alternatives. I'm documenting everything as I learn — mistakes included.

## License

MIT
