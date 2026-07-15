#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/match")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("evaluation/addressbase_results.csv"))
    args = parser.parse_args()

    benchmark = pd.read_csv(args.benchmark, dtype=str, keep_default_na=False)
    if args.limit:
        benchmark = benchmark.head(args.limit)

    records = []
    for i, row in benchmark.iterrows():
        started = time.perf_counter()
        error = ""
        response = {}
        try:
            response = post_json(
                args.api_url,
                {"address": row["query"], "top_k": args.top_k},
                args.timeout,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = (time.perf_counter() - started) * 1000
        matches = response.get("matches") or []
        best = response.get("best") or {}
        best_property = best.get("property") or {}
        predicted_uprn = str(best_property.get("uprn") or "")
        topk_uprns = [str((m.get("property") or {}).get("uprn") or "") for m in matches]
        expected_uprn = str(row["expected_uprn"])

        records.append(
            {
                **row.to_dict(),
                "predicted_uprn": predicted_uprn,
                "predicted_status": best.get("status", ""),
                "predicted_confidence": best.get("confidence", ""),
                "retrieval_mode": response.get("retrieval_mode", ""),
                "top1_correct": int(predicted_uprn == expected_uprn),
                "topk_correct": int(expected_uprn in topk_uprns),
                "latency_ms": round(latency_ms, 2),
                "error": error,
            }
        )

        if (i + 1) % 25 == 0:
            print(f"Evaluated {i + 1:,}/{len(benchmark):,}")

    results = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    valid = results[results["error"].eq("")]
    print(f"Rows: {len(results):,}")
    print(f"Errors: {(results['error'] != '').sum():,}")
    if len(valid):
        print(f"Top-1 accuracy: {valid['top1_correct'].mean():.4%}")
        print(f"Top-{args.top_k} recall: {valid['topk_correct'].mean():.4%}")
        print(f"Median latency: {valid['latency_ms'].median():.1f} ms")
        print(f"P95 latency: {valid['latency_ms'].quantile(.95):.1f} ms")
        print("\nBy test group:")
        grouped = valid.groupby("test_group").agg(
            rows=("benchmark_id", "size"),
            top1_accuracy=("top1_correct", "mean"),
            topk_recall=("topk_correct", "mean"),
            median_latency_ms=("latency_ms", "median"),
        )
        print(grouped.to_string())
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
