#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed but not running."
  exit 1
fi

mkdir -p data/qdrant

docker compose up -d --build

echo
echo "Services started."
echo "API health: http://127.0.0.1:8000/health"
echo "API docs:   http://127.0.0.1:8000/docs"
echo
echo "The matcher requires a restored uk_properties Qdrant snapshot."
