#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${LLM_BASE_URL:-http://localhost:8000}"
MODEL="${LLM_MODEL:-glm47-flash30b}"

curl "${BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Return the word OK only.\"}],
    \"temperature\": 0
  }"

echo

