from __future__ import annotations

import re

import pandas as pd


NUMBER_RE = re.compile(
    r"(?<![A-Z0-9])([0-9]+[A-Z]?)(?![A-Z0-9])",
    re.IGNORECASE,
)

UNIT_RE = re.compile(
    r"\b(?:FLAT|UNIT|APARTMENT|APT|SUITE|ROOM)\s*"
    r"([0-9]+[A-Z]?|[A-Z])\b",
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


def extract_number(value: object) -> str:
    match = NUMBER_RE.search(clean(value))
    return match.group(1).upper() if match else ""


def extract_unit(value: object) -> str:
    match = UNIT_RE.search(clean(value))
    return match.group(1).upper() if match else ""


def classify_failure(row: pd.Series) -> str:
    error = clean(row.get("error"))

    if error:
        return "EVALUATION_ERROR"

    in_corpus = clean(row.get("uprn_in_corpus", "1"))

    if in_corpus not in {"1", "1.0", "True", "true"}:
        return "ABSENT_FROM_CORPUS"

    top1 = clean(row.get("top1_correct", "0"))
    topk = clean(row.get("topk_correct", "0"))

    if top1 in {"1", "1.0", "True", "true"}:
        return "CORRECT_TOP1"

    if topk in {"1", "1.0", "True", "true"}:
        return "EXPECTED_IN_TOP5_WRONG_RANK"

    predicted_uprn = clean(row.get("predicted_uprn"))

    if not predicted_uprn:
        return "NO_CANDIDATE"

    retrieval_mode = clean(row.get("retrieval_mode"))

    if retrieval_mode == "postcode_sector":
        return "EXACT_POSTCODE_RETRIEVAL_FAILURE"

    if retrieval_mode == "postcode_district":
        return "POSTCODE_SECTOR_RETRIEVAL_FAILURE"

    if retrieval_mode == "global":
        return "MISSING_OR_UNPARSED_POSTCODE"

    expected_address = clean(row.get("expected_address"))
    predicted_address = clean(
        row.get("predicted_address")
        or row.get("matched_address")
    )

    expected_unit = extract_unit(expected_address)
    predicted_unit = extract_unit(predicted_address)

    if (
        expected_unit
        and predicted_unit
        and expected_unit != predicted_unit
    ):
        return "WRONG_UNIT"

    expected_number = extract_number(expected_address)
    predicted_number = extract_number(predicted_address)

    if (
        expected_number
        and predicted_number
        and expected_number != predicted_number
    ):
        return "WRONG_NUMBER"

    expected_postcode = clean(
        row.get("expected_postcode")
    ).upper()

    predicted_postcode = clean(
        row.get("predicted_postcode")
    ).upper()

    if (
        expected_postcode
        and predicted_postcode
        and expected_postcode != predicted_postcode
    ):
        return "WRONG_POSTCODE"

    return "EXPECTED_NOT_IN_TOP5"


def add_failure_types(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["failure_type"] = result.apply(
        classify_failure,
        axis=1,
    )
    return result
