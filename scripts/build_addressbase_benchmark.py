#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

import pandas as pd


ABBREVIATIONS = {
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


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def stable_rng(value: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def remove_punctuation(value: str) -> str:
    value = re.sub(r"[,.;:'\"]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_postcode(value: str, postcode: str) -> str:
    return value.replace(postcode, postcode.replace(" ", "")) if postcode else value


def abbreviate_street(value: str) -> str:
    result = f" {value.lower()} "
    for long_form, short_form in ABBREVIATIONS.items():
        result = result.replace(long_form + " ", short_form + " ")
    return re.sub(r"\s+", " ", result).strip()


def one_typo(value: str, protected: set[str], rng: random.Random) -> str:
    words = value.split()
    candidates = [
        i for i, word in enumerate(words)
        if len(word) >= 6 and word.upper() not in protected and not any(ch.isdigit() for ch in word)
    ]
    if not candidates:
        return value
    i = rng.choice(candidates)
    word = words[i]
    j = rng.randrange(1, len(word) - 1)
    words[i] = word[:j] + word[j + 1:]
    return " ".join(words)


def build_variant(row: pd.Series, category: str, seed: int) -> str:
    raw = clean(row["raw_address"])
    postcode = clean(row["postcode"])
    number = clean(row["number"])
    unit = clean(row["unit"])
    building = clean(row["building"])
    street = clean(row["street"])
    locality = clean(row["locality"])
    town = clean(row["town"])
    rng = stable_rng(clean(row["uprn"]), seed)

    if category == "exact":
        return raw
    if category == "lower_no_punctuation":
        return remove_punctuation(raw).lower()
    if category == "compact_postcode":
        return compact_postcode(remove_punctuation(raw).lower(), postcode.lower())
    if category == "abbreviated_street":
        return abbreviate_street(remove_punctuation(raw))
    if category == "street_typo":
        return one_typo(remove_punctuation(raw).lower(), {postcode.upper()}, rng)
    if category == "missing_postcode":
        return remove_punctuation(raw.replace(postcode, "")).lower()
    if category == "minimal_number_street_postcode":
        return clean(f"{number} {street} {postcode}").lower()
    if category == "reordered":
        return clean(f"{postcode} {unit} {number} {building} {street} {town}").lower()
    if category == "unit_variant":
        if unit:
            simplified = re.sub(
                r"^(flat|apartment|apt|unit|suite|room)\s*",
                "",
                unit,
                flags=re.I,
            )
            return clean(f"flat {simplified} {number} {building} {street} {postcode}").lower()
        return clean(f"{number} {building} {street} {postcode}").lower()
    raise ValueError(category)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation/addressbase_gold_bootstrap.csv"))
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.corpus.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.corpus)
    else:
        df = pd.read_csv(args.corpus, dtype=str, keep_default_na=False)
    required = {
        "uprn","raw_address","postcode","number","unit","building",
        "street","locality","town","source","classification_codes",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    # Keep rows that can support meaningful identity evaluation.
    eligible = df[
        df["uprn"].astype(str).ne("")
        & df["raw_address"].astype(str).ne("")
        & df["postcode"].astype(str).ne("")
    ].copy()

    categories = [
        "exact",
        "lower_no_punctuation",
        "compact_postcode",
        "abbreviated_street",
        "street_typo",
        "missing_postcode",
        "minimal_number_street_postcode",
        "reordered",
        "unit_variant",
    ]

    # Deterministic sample, with extra weight for units and numbered properties.
    eligible["sample_weight"] = 1.0
    eligible.loc[eligible["unit"].astype(str).ne(""), "sample_weight"] = 3.0
    eligible.loc[eligible["number"].astype(str).ne(""), "sample_weight"] += 1.0
    n = min(args.rows, len(eligible))
    sample = eligible.sample(
        n=n,
        weights="sample_weight",
        random_state=args.seed,
        replace=False,
    ).reset_index(drop=True)

    records = []
    for i, row in sample.iterrows():
        category = categories[i % len(categories)]
        query = build_variant(row, category, args.seed)
        records.append(
            {
                "benchmark_id": f"abp-{i+1:07d}",
                "query": query,
                "expected_uprn": clean(row["uprn"]),
                "expected_postcode": clean(row["postcode"]),
                "expected_address": clean(row["raw_address"]),
                "expected_status": "MATCH_CONFIDENT",
                "test_group": category,
                "source": clean(row["source"]),
                "classification_codes": clean(row["classification_codes"]),
                "label_provenance": "addressbase_premium_authoritative_uprn_synthetic_query",
                "human_review_status": "unreviewed",
            }
        )

    output = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    summary = (
        output.groupby("test_group")
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("test_group")
    )
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {len(output):,} benchmark rows to {args.output}")
    print(summary.to_string(index=False))
    print(
        "\nNote: labels are authoritative by UPRN, but query messiness is synthetic. "
        "Human-review a stratified subset before calling it a final gold benchmark."
    )


if __name__ == "__main__":
    main()
