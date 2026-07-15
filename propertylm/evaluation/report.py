from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def percentage(value: float) -> str:
    return f"{value:.2%}"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data available._"

    return frame.to_markdown(index=False)


def write_report(
    output_path: Path,
    summary: dict[str, Any],
    by_test_group: pd.DataFrame,
    by_failure_type: pd.DataFrame,
    by_retrieval_mode: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# PropertyLM Evaluation Report",
        "",
        "## Overall metrics",
        "",
        f"- Rows: **{summary['rows']:,}**",
        f"- Evaluation errors: **{summary['errors']:,}**",
        (
            "- Top-1 accuracy: "
            f"**{percentage(summary['top1_accuracy'])}**"
        ),
        (
            "- Top-k recall: "
            f"**{percentage(summary['topk_recall'])}**"
        ),
        (
            "- Median latency: "
            f"**{summary['median_latency_ms']:.1f} ms**"
        ),
        (
            "- P95 latency: "
            f"**{summary['p95_latency_ms']:.1f} ms**"
        ),
        (
            "- Mean predicted confidence: "
            f"**{summary['mean_confidence']:.3f}**"
        ),
    ]

    if "corpus_coverage" in summary:
        lines.extend(
            [
                (
                    "- Corpus coverage: "
                    f"**{percentage(summary['corpus_coverage'])}**"
                ),
                (
                    "- Covered Top-1 accuracy: "
                    f"**{percentage(summary['covered_top1_accuracy'])}**"
                ),
                (
                    "- Covered Top-k recall: "
                    f"**{percentage(summary['covered_topk_recall'])}**"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Metrics by test group",
            "",
            markdown_table(by_test_group),
            "",
            "## Failure distribution",
            "",
            markdown_table(by_failure_type),
            "",
            "## Metrics by retrieval mode",
            "",
            markdown_table(by_retrieval_mode),
            "",
            "## Interpretation guide",
            "",
            "- `ABSENT_FROM_CORPUS`: expected UPRN is not searchable.",
            "- `EXPECTED_IN_TOP5_WRONG_RANK`: recall succeeded but ranking failed.",
            "- `EXACT_POSTCODE_RETRIEVAL_FAILURE`: exact postcode candidate set failed.",
            "- `MISSING_OR_UNPARSED_POSTCODE`: query used global fallback.",
            "- `WRONG_NUMBER` or `WRONG_UNIT`: deterministic address identity issue.",
            "- `EXPECTED_NOT_IN_TOP5`: candidate generation or semantic retrieval issue.",
            "",
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
