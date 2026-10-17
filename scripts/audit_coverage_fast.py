#!/usr/bin/env python3
"""Corpus-coverage audit via full payload stream (robust to payload types/indexes).

Scrolls the whole collection unfiltered, keeping only the `uprn` payload field,
and intersects with the benchmark's expected UPRNs locally. ~18.2M points in
batches of 50k ≈ a few minutes.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from qdrant_client import QdrantClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="uk_properties")
    parser.add_argument("--batch", type=int, default=50000)
    parser.add_argument("--output", default="evaluation/SX9090_coverage_current.csv")
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url, timeout=300)

    df = pd.read_csv(args.benchmark, dtype=str, keep_default_na=False)
    wanted = {u.strip() for u in df["expected_uprn"]}
    print(f"Looking for {len(wanted):,} unique UPRNs...")

    total = client.get_collection(args.collection).points_count
    found: set[str] = set()
    seen = 0
    offset = None
    started = time.time()

    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            limit=args.batch,
            offset=offset,
            with_payload=["uprn"],
            with_vectors=False,
        )
        for p in points:
            u = str((p.payload or {}).get("uprn", "") or "").strip()
            if u in wanted:
                found.add(u)
        seen += len(points)
        if seen % 1_000_000 < args.batch:
            rate = seen / max(time.time() - started, 1)
            eta = (total - seen) / max(rate, 1)
            print(
                f"  scanned {seen:,}/{total:,} "
                f"({seen / total:.0%}) found {len(found):,} "
                f"ETA {eta / 60:.1f} min"
            )
        if offset is None:
            break

    df["uprn_in_corpus"] = df["expected_uprn"].str.strip().isin(found).astype(int)
    df[["benchmark_id", "expected_uprn", "test_group", "uprn_in_corpus"]].to_csv(
        args.output, index=False
    )
    covered = int(df["uprn_in_corpus"].sum())
    print(f"\nCoverage: {covered:,}/{len(df):,} ({covered / len(df):.2%})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
