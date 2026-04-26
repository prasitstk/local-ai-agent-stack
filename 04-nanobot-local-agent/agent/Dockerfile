FROM python:3.12-slim

# Security: run as non-root user. UID/GID are pinned to 10001 so the
# host's bind-mounted ./workspace can be chowned to match (see Step 1
# in README) — otherwise the in-container `agent` user can't write to
# /workspace, since bind mounts preserve host ownership.
RUN groupadd -g 10001 agent && useradd -u 10001 -g agent -m -s /bin/bash agent

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy agent code
COPY --chown=agent:agent . /app

# Security: make application code read-only
RUN chmod -R a-w /app

# Workspace is the only writable location
RUN mkdir -p /workspace && chown agent:agent /workspace

USER agent
WORKDIR /app

ENTRYPOINT ["python", "agent.py"]
