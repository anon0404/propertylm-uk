#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"

echo "Checking API..."
curl --fail --silent \
  "${API_URL}/health" \
  | python -m json.tool

echo
echo "Testing matcher..."
curl --fail --silent \
  "${API_URL}/match" \
  -H "Content-Type: application/json" \
  -d '{"address":"16 Alesmore WS13 8FD","top_k":5}' \
  | python -m json.tool
