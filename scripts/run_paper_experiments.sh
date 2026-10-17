#!/usr/bin/env bash
# ICAIF paper experiments: full benchmark + ablations.
# Prereq: docker compose stack running (./scripts/start_runtime.sh), snapshot restored.
# Usage: ./scripts/run_paper_experiments.sh [--full]
#   default: ablations on the identical stratified-500 sample used in report_500
#   --full : additionally run the full 10,000-case benchmark on the main config (~2.5 h)
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH_500="evaluation/SX9090_benchmark_stratified_500.csv"
BENCH_FULL="${BENCH_FULL:-../addressbase_phase1/SX9090_gold_benchmark_bootstrap.csv}"
API="http://127.0.0.1:8000"

wait_healthy() {
  echo "waiting for API..."
  for _ in $(seq 1 120); do
    if curl -sf "$API/health" >/dev/null 2>&1; then echo "API up"; return 0; fi
    sleep 5
  done
  echo "API did not become healthy" >&2; exit 1
}

restart_api() { # restart_api [ENV overrides...]
  docker compose stop api >/dev/null
  env "$@" docker compose up -d --build api >/dev/null
  wait_healthy
}

run_eval() { # run_eval BENCHMARK OUTPUT
  python3 scripts/evaluate_addressbase_benchmark.py "$1" --output "$2"
}

wait_healthy

# --- Ablation A: no deterministic adjudication -------------------------------
echo "== Ablation: rules OFF =="
restart_api DISABLE_RULES=true
run_eval "$BENCH_500" "evaluation/ablation_rules_off_500.csv"

# --- Ablation B: no cross-encoder reranker -----------------------------------
echo "== Ablation: reranker OFF =="
restart_api DISABLE_RULES=false RERANK_MODEL=none
run_eval "$BENCH_500" "evaluation/ablation_reranker_off_500.csv"

# --- Restore main configuration ----------------------------------------------
echo "== Restore main config =="
restart_api

# --- Optional: fine-tuned-encoder ablation -----------------------------------
# Requires an index built with the same encoder as the queries. If you have a
# fine-tuned checkpoint AND a matching snapshot/collection, run e.g.:
#   QDRANT_COLLECTION=uk_properties_ft EMBED_MODEL=/models/propertylm-ft \
#     docker compose up -d --build api
#   python3 scripts/evaluate_addressbase_benchmark.py "$BENCH_500" \
#     --output evaluation/ablation_finetuned_500.csv

# --- Full benchmark on main config -------------------------------------------
if [[ "${1:-}" == "--full" ]]; then
  echo "== Full 10,000-case benchmark (main config) =="
  run_eval "$BENCH_FULL" "evaluation/SX9090_results_full_10000.csv"
fi

echo "Done. Results in evaluation/ — ready for report generation."
