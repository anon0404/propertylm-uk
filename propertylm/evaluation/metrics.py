from __future__ import annotations

from typing import Any

import pandas as pd


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            [default] * len(frame),
            index=frame.index,
            dtype=float,
        )

    return (
        pd.to_numeric(frame[column], errors="coerce")
        .fillna(default)
        .astype(float)
    )


def calculate_summary(frame: pd.DataFrame) -> dict[str, Any]:
    rows = len(frame)

    top1 = numeric_series(frame, "top1_correct")
    topk = numeric_series(frame, "topk_correct")
    latency = numeric_series(frame, "latency_ms")

    errors = (
        frame.get("error", pd.Series(index=frame.index, dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    confidence = numeric_series(
        frame,
        "predicted_confidence",
    )

    summary: dict[str, Any] = {
        "rows": rows,
        "errors": int(errors.sum()),
        "top1_accuracy": float(top1.mean()) if rows else 0.0,
        "topk_recall": float(topk.mean()) if rows else 0.0,
        "median_latency_ms": (
            float(latency.median()) if rows else 0.0
        ),
        "p95_latency_ms": (
            float(latency.quantile(0.95)) if rows else 0.0
        ),
        "mean_confidence": (
            float(confidence.mean()) if rows else 0.0
        ),
    }

    if "uprn_in_corpus" in frame.columns:
        covered = numeric_series(frame, "uprn_in_corpus")
        covered_frame = frame[covered == 1]

        covered_top1 = numeric_series(
            covered_frame,
            "top1_correct",
        )
        covered_topk = numeric_series(
            covered_frame,
            "topk_correct",
        )

        summary.update(
            {
                "corpus_coverage": float(covered.mean()),
                "covered_rows": int((covered == 1).sum()),
                "covered_top1_accuracy": (
                    float(covered_top1.mean())
                    if len(covered_frame)
                    else 0.0
                ),
                "covered_topk_recall": (
                    float(covered_topk.mean())
                    if len(covered_frame)
                    else 0.0
                ),
            }
        )

    return summary


def grouped_metrics(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    if group_column not in frame.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for group, group_frame in frame.groupby(
        group_column,
        dropna=False,
    ):
        summary = calculate_summary(group_frame)
        rows.append(
            {
                group_column: str(group),
                **summary,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(group_column)
        .reset_index(drop=True)
    )
