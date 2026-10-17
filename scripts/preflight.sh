#!/usr/bin/env bash
# Preflight checks for run_paper_experiments.sh — safe to run any time.
set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; echo "        fix: $2"; FAIL=$((FAIL+1)); }

echo "== PropertyLM preflight =="

# 1. Docker daemon
if docker info >/dev/null 2>&1; then
  ok "Docker daemon running"
else
  bad "Docker daemon not running" "open Docker Desktop and wait for the whale icon, then re-run"
fi

# 2. Stack containers
if docker compose ps --status running 2>/dev/null | grep -q qdrant; then
  ok "Qdrant container up"
else
  bad "Qdrant container not up" "./scripts/start_runtime.sh"
fi
if docker compose ps --status running 2>/dev/null | grep -q api; then
  ok "API container up"
else
  bad "API container not up" "./scripts/start_runtime.sh"
fi

# 3. API health
if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  ok "API healthy at 127.0.0.1:8000"
else
  bad "API not responding" "./scripts/start_runtime.sh, then wait ~1 min for models to load"
fi

# 4. Collection loaded with expected points
POINTS=$(curl -sf http://127.0.0.1:6333/collections/uk_properties 2>/dev/null \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])" 2>/dev/null)
if [ "${POINTS:-0}" -gt 18000000 ] 2>/dev/null; then
  ok "uk_properties collection loaded (${POINTS} points)"
elif [ -n "${POINTS:-}" ]; then
  bad "collection has only ${POINTS} points (expected ~18.2M)" "./scripts/restore_snapshot.sh"
else
  bad "uk_properties collection missing" "./scripts/restore_snapshot.sh"
fi

# 5. Python venv + pandas
if [ -f .venv/bin/activate ]; then
  ok ".venv exists"
  if .venv/bin/python -c "import pandas" 2>/dev/null; then
    ok "pandas available in .venv"
  else
    bad "pandas missing in .venv" ".venv/bin/pip install pandas"
  fi
else
  bad ".venv missing" "python3 -m venv .venv && .venv/bin/pip install pandas"
fi

# 6. Benchmark files present
[ -f evaluation/SX9090_benchmark_stratified_500.csv ] \
  && ok "stratified-500 benchmark present" \
  || bad "stratified-500 benchmark missing" "ask Claude to regenerate it"
[ -f ../addressbase_phase1/SX9090_gold_benchmark_bootstrap.csv ] \
  && ok "full 10k benchmark present" \
  || bad "full 10k benchmark missing at ../addressbase_phase1/" "set BENCH_FULL=/path/to/SX9090_gold_benchmark_bootstrap.csv"

# 7. One live smoke match
SMOKE=$(curl -sf -X POST http://127.0.0.1:8000/match \
  -H 'Content-Type: application/json' \
  -d '{"address":"80 latimer road exeter ex4 7jp","top_k":1}' 2>/dev/null \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['best']['status'])" 2>/dev/null)
if [ -n "${SMOKE:-}" ]; then
  ok "smoke match returned status: ${SMOKE}"
else
  bad "smoke match failed" "check API logs: docker compose logs api --tail 50"
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "All checks passed — run: ./scripts/run_paper_experiments.sh --full"
else
  echo "$FAIL check(s) failed — apply the fixes above, then re-run this script."
fi
