#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl


def load_uprns(path: Path) -> set[str]:
    if path.suffix.lower() == ".parquet":
        values = (
            pl.scan_parquet(path)
            .select(pl.col("uprn").cast(pl.String))
            .collect()["uprn"]
            .drop_nulls()
            .to_list()
        )
    else:
        values = (
            pd.read_csv(path, usecols=["uprn"], dtype=str)["uprn"]
            .dropna()
            .tolist()
        )

    return {str(value).strip() for value in values if str(value).strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("corpus")
    parser.add_argument(
        "--output",
        default="evaluation/coverage.csv",
    )
    args = parser.parse_args()

    benchmark = pd.read_csv(args.benchmark, dtype=str)
    corpus_uprns = load_uprns(Path(args.corpus))

    benchmark["uprn_in_corpus"] = (
        benchmark["expected_uprn"]
        .astype(str)
        .str.strip()
        .isin(corpus_uprns)
        .astype(int)
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output, index=False)

    present = int(benchmark["uprn_in_corpus"].sum())
    total = len(benchmark)

    print("Rows:", total)
    print("Present:", present)
    print("Absent:", total - present)
    print("Coverage:", f"{present / total:.2%}")
    print("Wrote:", output)


if __name__ == "__main__":
    main()
