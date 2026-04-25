#!/bin/bash
# benchmark.sh — Measure Gemma 4 E2B inference performance on CPU
# Usage: ./benchmark.sh

set -euo pipefail

MODEL="gemma4:e2b"
API="http://localhost:11434/api/chat"

echo "=== Gemma 4 E2B — CPU Benchmark ==="
echo "Date:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Server: $(hostname)"
echo "CPU:    $(nproc) vCPUs — $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "RAM:    $(free -h | awk '/Mem:/ {print $2}') total, $(free -h | awk '/Mem:/ {print $7}') available"
echo ""

run_test() {
    local label="$1"
    local payload="$2"

    echo "--- $label ---"
    result=$(curl -s "$API" -d "$payload")

    echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
msg = data.get('message', {}).get('content', '')
total = data.get('total_duration', 0) / 1e9
prompt_eval = data.get('prompt_eval_duration', 0) / 1e9
eval_count = data.get('eval_count', 0)
eval_duration = data.get('eval_duration', 0) / 1e9

print(f'  Tokens generated : {eval_count}')
print(f'  Prompt eval      : {prompt_eval:.2f}s')
print(f'  Generation time  : {eval_duration:.2f}s')
print(f'  Total time       : {total:.2f}s')
if eval_duration > 0:
    print(f'  Generation speed : {eval_count/eval_duration:.1f} tokens/sec')
print(f'  Response preview : {msg[:150].strip()}...')
"
    echo ""
}

# Test 1: Short factual question
run_test "Test 1: Short factual question" '{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "What is Kubernetes? Answer in 2 sentences."}],
  "stream": false
}'

# Test 2: Code generation
run_test "Test 2: Code generation" '{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "Write a Python function that reads a CSV file and returns the top 5 rows sorted by a given column. Include error handling and type hints."}],
  "stream": false
}'

# Test 3: Reasoning
run_test "Test 3: Multi-step reasoning" '{
  "model": "'"$MODEL"'",
  "messages": [
    {"role": "system", "content": "Think step by step before answering."},
    {"role": "user", "content": "A trader buys 100 shares at 50 THB each. The price drops 10%, and they buy 100 more shares. What is their average cost per share?"}
  ],
  "stream": false
}'

# Test 4: Structured output
run_test "Test 4: JSON structured output" '{
  "model": "'"$MODEL"'",
  "messages": [
    {"role": "system", "content": "Respond only in valid JSON. No markdown, no explanation."},
    {"role": "user", "content": "List 3 popular container orchestration tools with name, description (one sentence), and license."}
  ],
  "stream": false
}'

# Memory snapshot
echo "--- Memory snapshot during idle ---"
ps aux --sort=-%mem | head -5
echo ""
echo "=== Benchmark complete ==="
