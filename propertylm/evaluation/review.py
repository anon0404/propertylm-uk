from __future__ import annotations

import pandas as pd


DEFAULT_REVIEW_COLUMNS = [
    "benchmark_id",
    "query",
    "expected_uprn",
    "expected_address",
    "expected_postcode",
    "predicted_uprn",
    "predicted_status",
    "predicted_confidence",
    "retrieval_mode",
    "test_group",
    "failure_type",
]


def create_review_sample(
    frame: pd.DataFrame,
    per_failure_type: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    failures = frame[
        frame["failure_type"] != "CORRECT_TOP1"
    ].copy()

    samples: list[pd.DataFrame] = []

    for _, group in failures.groupby(
        "failure_type",
        dropna=False,
    ):
        count = min(per_failure_type, len(group))

        samples.append(
            group.sample(
                n=count,
                random_state=seed,
            )
        )

    if not samples:
        result = failures.head(0).copy()
    else:
        result = pd.concat(
            samples,
            ignore_index=True,
        )

    available_columns = [
        column
        for column in DEFAULT_REVIEW_COLUMNS
        if column in result.columns
    ]

    result = result[available_columns].copy()

    result["human_review_status"] = ""
    result["human_failure_type"] = ""
    result["human_notes"] = ""

    return result
