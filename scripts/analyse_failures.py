#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd


def classify(row) -> str:
    if str(row.get("uprn_in_corpus", "0")) != "1":
        return "ABSENT_FROM_CORPUS"

    if str(row.get("topk_correct", "0")) == "1":
        return "EXPECTED_IN_TOP5_WRONG_RANK"

    mode = str(row.get("retrieval_mode", ""))

    if mode == "postcode_sector":
        return "POSTCODE_PARSE_OR_EXACT_FILTER_FAILURE"

    if mode == "postcode_district":
        return "POSTCODE_SECTOR_FAILURE"

    if mode == "global":
        return "POSTCODE_MISSING_OR_UNPARSED"

    if not str(row.get("predicted_uprn", "")).strip():
        return "NO_CANDIDATES"

    return "EXPECTED_NOT_IN_TOP5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("coverage")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results, dtype=str)
    coverage = pd.read_csv(
        args.coverage,
        dtype=str,
    )[["benchmark_id", "uprn_in_corpus"]]

    df = results.merge(
        coverage,
        on="benchmark_id",
        how="left",
    )

    df["failure_type"] = df.apply(classify, axis=1)
    df.to_csv(args.output, index=False)

    print(df["failure_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
