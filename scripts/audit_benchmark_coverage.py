#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


def uprn_exists(
    client: QdrantClient,
    collection: str,
    uprn: str,
) -> bool:
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="uprn",
                    match=MatchValue(value=str(uprn)),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    return bool(points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
    )
    parser.add_argument(
        "--collection",
        default="uk_properties",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        default="evaluation/SX9090_coverage.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.benchmark, dtype=str)

    if args.limit:
        df = df.head(args.limit).copy()

    client = QdrantClient(url=args.qdrant_url)

    results = []

    for index, row in df.iterrows():
        uprn = str(row["expected_uprn"]).strip()

        exists = uprn_exists(
            client,
            args.collection,
            uprn,
        )

        results.append({
            "benchmark_id": row["benchmark_id"],
            "expected_uprn": uprn,
            "query": row["query"],
            "test_group": row["test_group"],
            "uprn_in_collection": int(exists),
        })

        if (index + 1) % 25 == 0:
            print(f"Checked {index + 1}/{len(df)}")

    output = pd.DataFrame(results)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)

    present = int(output["uprn_in_collection"].sum())
    total = len(output)

    print()
    print("Rows:", total)
    print("UPRNs present:", present)
    print("UPRNs absent:", total - present)
    print("Coverage:", f"{present / total:.2%}")
    print("Wrote:", path)


if __name__ == "__main__":
    main()
