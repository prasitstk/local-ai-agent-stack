#!/bin/bash
# security_test.sh — Verify security hardening of the agent container
# Run from the project root: ./scripts/security_test.sh

set -euo pipefail

echo "=== Agent Security Tests ==="
echo ""

PASS=0
FAIL=0

run_test() {
    local label="$1"
    local cmd="$2"
    local expect_fail="${3:-true}"

    printf "%-50s" "$label"

    if output=$(docker compose run --rm --no-TTY agent sh -c "$cmd" 2>&1); then
        if [ "$expect_fail" = "true" ]; then
            echo "FAIL (should have been denied)"
            FAIL=$((FAIL + 1))
        else
            echo "PASS"
            PASS=$((PASS + 1))
        fi
    else
        if [ "$expect_fail" = "true" ]; then
            echo "PASS"
            PASS=$((PASS + 1))
        else
            echo "FAIL (unexpected error)"
            FAIL=$((FAIL + 1))
        fi
    fi
}

# Filesystem tests
run_test "Read-only /etc" "touch /etc/test"
run_test "Read-only /app" "touch /app/test"
run_test "Read-only /usr" "touch /usr/test"
run_test "Writable /workspace" "touch /workspace/test && rm /workspace/test" "false"
run_test "Writable /tmp" "touch /tmp/test" "false"
run_test "No exec in /tmp" "cp /bin/echo /tmp/echo && /tmp/echo test"

# User tests
printf "%-50s" "Running as non-root"
uid=$(docker compose run --rm --no-TTY agent id -u 2>&1)
if [ "$uid" != "0" ]; then
    echo "PASS (uid=$uid)"
    PASS=$((PASS + 1))
else
    echo "FAIL (running as root!)"
    FAIL=$((FAIL + 1))
fi

# Network tests
run_test "No internet access" "python3 -c \"import urllib.request; urllib.request.urlopen('https://google.com', timeout=5)\""

printf "%-50s" "Can reach Ollama"
if docker compose run --rm --no-TTY agent python3 -c "import requests; r=requests.get('http://ollama:11434', timeout=5); print(r.status_code)" 2>&1 | grep -q "200"; then
    echo "PASS"
    PASS=$((PASS + 1))
else
    echo "FAIL (cannot reach Ollama)"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
