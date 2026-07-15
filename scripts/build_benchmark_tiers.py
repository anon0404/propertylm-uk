#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def stratified_sample(
    frame: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    if size >= len(frame):
        return frame.sample(
            frac=1,
            random_state=seed,
        ).reset_index(drop=True)

    if "test_group" not in frame.columns:
        return frame.sample(
            n=size,
            random_state=seed,
        ).reset_index(drop=True)

    groups = list(
        frame.groupby(
            "test_group",
            dropna=False,
        )
    )

    total_rows = len(frame)
    samples: list[pd.DataFrame] = []
    allocated = 0

    for group_index, (_, group) in enumerate(groups):
        if group_index == len(groups) - 1:
            group_size = size - allocated
        else:
            share = len(group) / total_rows
            group_size = max(
                1,
                round(size * share),
            )

        group_size = min(
            group_size,
            len(group),
        )

        samples.append(
            group.sample(
                n=group_size,
                random_state=seed + group_index,
            )
        )

        allocated += group_size

    result = pd.concat(
        samples,
        ignore_index=True,
    )

    if len(result) > size:
        result = result.sample(
            n=size,
            random_state=seed,
        )
    elif len(result) < size:
        remaining = frame[
            ~frame["benchmark_id"].isin(
                result["benchmark_id"]
            )
        ]

        needed = min(
            size - len(result),
            len(remaining),
        )

        if needed:
            result = pd.concat(
                [
                    result,
                    remaining.sample(
                        n=needed,
                        random_state=seed + 999,
                    ),
                ],
                ignore_index=True,
            )

    return (
        result.sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(drop=True)
    )


def write_set(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    frame.to_csv(
        path,
        index=False,
    )
    print(path, len(frame))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument(
        "--output-dir",
        default="evaluation/benchmarks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--regression-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--golden-size",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--manual-size",
        type=int,
        default=200,
    )
    args = parser.parse_args()

    source = pd.read_csv(
        args.benchmark,
        dtype=str,
    )

    source = source.drop_duplicates(
        subset=["benchmark_id"],
        keep="first",
    )

    output_dir = Path(args.output_dir)

    smoke = stratified_sample(
        source,
        min(args.smoke_size, len(source)),
        args.seed,
    )

    regression = stratified_sample(
        source,
        min(args.regression_size, len(source)),
        args.seed + 100,
    )

    golden = stratified_sample(
        source,
        min(args.golden_size, len(source)),
        args.seed + 200,
    )

    manual_source = golden.copy()

    manual = stratified_sample(
        manual_source,
        min(args.manual_size, len(manual_source)),
        args.seed + 300,
    )

    manual["human_review_status"] = ""
    manual["human_failure_type"] = ""
    manual["human_notes"] = ""
    manual["reviewed_by"] = ""
    manual["reviewed_at"] = ""

    write_set(
        smoke,
        output_dir / "smoke_100.csv",
    )

    write_set(
        regression,
        output_dir / "regression_1000.csv",
    )

    write_set(
        golden,
        output_dir / "golden_10000.csv",
    )

    write_set(
        manual,
        output_dir / "manual_review_200.csv",
    )


if __name__ == "__main__":
    main()
