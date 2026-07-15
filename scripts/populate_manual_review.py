#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


POSTCODE_RE = re.compile(
    r"\b([A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2})\b",
    re.IGNORECASE,
)

UNIT_RE = re.compile(
    r"\b("
    r"FLAT|APARTMENT|APT|STUDIO|UNIT|SUITE|ROOM"
    r")\s*"
    r"([0-9]+[A-Z]?(?:\.[0-9]+)?|[A-Z])?"
    r"\b",
    re.IGNORECASE,
)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


def normalize_postcode(value: str) -> str:
    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )

    if len(compact) < 5:
        return compact

    return f"{compact[:-3]} {compact[-3:]}"


def extract_postcode(value: str) -> str:
    match = POSTCODE_RE.search(clean(value).upper())

    if not match:
        return ""

    return normalize_postcode(match.group(1))


def extract_unit(value: str) -> tuple[str, str] | None:
    match = UNIT_RE.search(clean(value))

    if not match:
        return None

    unit_type = match.group(1).lower()
    unit_value = clean(match.group(2)).lower()

    return unit_type, unit_value


def extract_numbers(value: str) -> list[str]:
    text = POSTCODE_RE.sub(" ", clean(value))

    return re.findall(
        r"\b[0-9]+[A-Z]?(?:\.[0-9]+)?\b",
        text,
        flags=re.IGNORECASE,
    )


def normalize_text(value: str) -> str:
    text = clean(value).lower()

    replacements = {
        " road": " rd",
        " street": " st",
        " avenue": " ave",
        " lane": " ln",
        " drive": " dr",
        " close": " cl",
        " court": " ct",
        " place": " pl",
        " gardens": " gdns",
        " crescent": " cres",
        " terrace": " ter",
    }

    for original, abbreviated in replacements.items():
        text = text.replace(original, abbreviated)

    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def review_row(row: pd.Series) -> tuple[str, str, str]:
    query = clean(row.get("query"))
    expected = clean(row.get("expected_address"))
    expected_postcode = normalize_postcode(
        clean(row.get("expected_postcode"))
    )

    query_postcode = extract_postcode(query)

    if query_postcode and expected_postcode:
        if query_postcode != expected_postcode:
            return (
                "rejected",
                "wrong_postcode",
                (
                    f"Query postcode {query_postcode} contradicts "
                    f"the authoritative postcode {expected_postcode}."
                ),
            )

    expected_unit = extract_unit(expected)
    query_unit = extract_unit(query)

    if expected_unit and not query_unit:
        return (
            "rejected",
            "wrong_unit",
            (
                "Synthetic query omits the authoritative flat/unit "
                "identifier and therefore may not uniquely identify "
                "the expected UPRN."
            ),
        )

    if expected_unit and query_unit:
        expected_unit_value = expected_unit[1]
        query_unit_value = query_unit[1]

        if (
            expected_unit_value
            and query_unit_value
            and expected_unit_value != query_unit_value
        ):
            return (
                "rejected",
                "wrong_unit",
                (
                    f"Query unit {query_unit_value!r} contradicts "
                    f"authoritative unit {expected_unit_value!r}."
                ),
            )

    expected_numbers = extract_numbers(expected)
    query_numbers = extract_numbers(query)

    if expected_numbers and not query_numbers:
        return (
            "rejected",
            "wrong_number",
            (
                "Synthetic query omits all numeric property "
                "identifiers present in the authoritative address."
            ),
        )

    similarity = SequenceMatcher(
        None,
        normalize_text(query),
        normalize_text(expected),
    ).ratio()

    if similarity < 0.32:
        return (
            "needs_investigation",
            "other",
            (
                "The normalized query-to-address similarity is low "
                f"({similarity:.2f}); verify this label manually."
            ),
        )

    test_group = clean(row.get("test_group"))

    return (
        "approved",
        "correct",
        (
            "Synthetic query preserves the authoritative property "
            f"identity for the {test_group or 'unspecified'} test group."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    # dtype=str is critical: UPRNs must never be converted into
    # floating-point scientific notation.
    frame = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "benchmark_id",
        "query",
        "expected_uprn",
        "expected_postcode",
        "expected_address",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    reviews = frame.apply(
        review_row,
        axis=1,
        result_type="expand",
    )

    reviews.columns = [
        "human_review_status",
        "human_failure_type",
        "human_notes",
    ]

    frame[
        [
            "human_review_status",
            "human_failure_type",
            "human_notes",
        ]
    ] = reviews

    # Leave reviewer identity and timestamp untouched.
    if "reviewed_by" not in frame.columns:
        frame["reviewed_by"] = ""

    if "reviewed_at" not in frame.columns:
        frame["reviewed_at"] = ""

    output_path = (
        Path(args.output)
        if args.output
        else input_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    print("Rows:", len(frame))
    print()
    print("Review statuses:")
    print(
        frame["human_review_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Failure labels:")
    print(
        frame["human_failure_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Non-approved rows:")

    non_approved = frame[
        frame["human_review_status"] != "approved"
    ]

    if non_approved.empty:
        print("None")
    else:
        print(
            non_approved[
                [
                    "benchmark_id",
                    "query",
                    "expected_address",
                    "human_review_status",
                    "human_failure_type",
                    "human_notes",
                ]
            ].to_string(index=False)
        )

    print()
    print("Wrote:", output_path)


if __name__ == "__main__":
    main()
