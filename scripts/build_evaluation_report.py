#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from propertylm.evaluation.failures import (
    add_failure_types,
)
from propertylm.evaluation.metrics import (
    calculate_summary,
    grouped_metrics,
)
from propertylm.evaluation.report import write_report
from propertylm.evaluation.review import (
    create_review_sample,
)


def load_inputs(
    results_path: Path,
    coverage_path: Path | None,
) -> pd.DataFrame:
    results = pd.read_csv(
        results_path,
        dtype=str,
    )

    if coverage_path is None:
        return results

    coverage = pd.read_csv(
        coverage_path,
        dtype=str,
    )

    coverage_columns = [
        column
        for column in [
            "benchmark_id",
            "uprn_in_corpus",
        ]
        if column in coverage.columns
    ]

    if "benchmark_id" not in coverage_columns:
        raise ValueError(
            "Coverage CSV must contain benchmark_id"
        )

    return results.merge(
        coverage[coverage_columns],
        on="benchmark_id",
        how="left",
    )


def simplify_group_metrics(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    grouped = grouped_metrics(
        frame,
        group_column,
    )

    if grouped.empty:
        return grouped

    wanted = [
        group_column,
        "rows",
        "top1_accuracy",
        "topk_recall",
        "median_latency_ms",
        "p95_latency_ms",
    ]

    available = [
        column
        for column in wanted
        if column in grouped.columns
    ]

    result = grouped[available].copy()

    for column in [
        "top1_accuracy",
        "topk_recall",
    ]:
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: f"{value:.2%}"
            )

    for column in [
        "median_latency_ms",
        "p95_latency_ms",
    ]:
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: round(float(value), 1)
            )

    return result


def build_failure_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        frame["failure_type"]
        .value_counts(dropna=False)
        .rename_axis("failure_type")
        .reset_index(name="rows")
    )

    counts["percentage"] = (
        counts["rows"] / len(frame)
    ).map(lambda value: f"{value:.2%}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument(
        "--coverage",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/report",
    )
    parser.add_argument(
        "--review-per-failure",
        type=int,
        default=15,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = load_inputs(
        Path(args.results),
        Path(args.coverage)
        if args.coverage
        else None,
    )

    analysed = add_failure_types(frame)

    analysed_path = (
        output_dir / "analysed_results.csv"
    )
    analysed.to_csv(
        analysed_path,
        index=False,
    )

    summary = calculate_summary(analysed)

    by_test_group = simplify_group_metrics(
        analysed,
        "test_group",
    )

    by_retrieval_mode = simplify_group_metrics(
        analysed,
        "retrieval_mode",
    )

    by_failure_type = build_failure_summary(
        analysed,
    )

    by_test_group.to_csv(
        output_dir / "metrics_by_test_group.csv",
        index=False,
    )

    by_retrieval_mode.to_csv(
        output_dir / "metrics_by_retrieval_mode.csv",
        index=False,
    )

    by_failure_type.to_csv(
        output_dir / "failure_distribution.csv",
        index=False,
    )

    review = create_review_sample(
        analysed,
        per_failure_type=args.review_per_failure,
    )

    review.to_csv(
        output_dir / "manual_review.csv",
        index=False,
    )

    write_report(
        output_dir / "report.md",
        summary,
        by_test_group,
        by_failure_type,
        by_retrieval_mode,
    )

    print("Rows:", summary["rows"])
    print(
        "Top-1 accuracy:",
        f"{summary['top1_accuracy']:.2%}",
    )
    print(
        "Top-k recall:",
        f"{summary['topk_recall']:.2%}",
    )
    print(
        "Median latency:",
        f"{summary['median_latency_ms']:.1f} ms",
    )
    print(
        "P95 latency:",
        f"{summary['p95_latency_ms']:.1f} ms",
    )
    print()
    print("Failure distribution:")
    print(
        by_failure_type.to_string(index=False)
    )
    print()
    print("Wrote:", output_dir)


if __name__ == "__main__":
    main()
