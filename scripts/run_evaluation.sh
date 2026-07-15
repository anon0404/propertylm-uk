#!/usr/bin/env bash
set -euo pipefail

RESULTS="${1:-evaluation/SX9090_covered_results_500.csv}"
COVERAGE="${2:-evaluation/SX9090_coverage.csv}"
OUTPUT="${3:-evaluation/report_500}"

python scripts/build_evaluation_report.py \
  "$RESULTS" \
  --coverage "$COVERAGE" \
  --output-dir "$OUTPUT" \
  --review-per-failure 15

echo
echo "Report:"
echo "$OUTPUT/report.md"
