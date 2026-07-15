#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


TIERS = {
    "smoke": "evaluation/benchmarks/smoke_100.csv",
    "regression": "evaluation/benchmarks/regression_1000.csv",
    "golden": "evaluation/benchmarks/golden_10000.csv",
    "manual": "evaluation/benchmarks/manual_review_200.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tier",
        choices=TIERS,
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/runs",
    )
    args = parser.parse_args()

    benchmark = Path(TIERS[args.tier])
    output_dir = Path(args.output_dir) / args.tier
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = output_dir / "results.csv"

    command = [
        "python",
        "scripts/evaluate_addressbase_benchmark.py",
        str(benchmark),
        "--output",
        str(results),
    ]

    print("Running:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
    )

    report_command = [
        "python",
        "scripts/build_evaluation_report.py",
        str(results),
        "--output-dir",
        str(output_dir / "report"),
        "--review-per-failure",
        "15",
    ]

    print()
    print("Building report:")
    print(" ".join(report_command))

    subprocess.run(
        report_command,
        check=True,
    )

    print()
    print("Results:", results)
    print(
        "Report:",
        output_dir / "report/report.md",
    )


if __name__ == "__main__":
    main()
