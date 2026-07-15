from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "uprn",
    "os_address_id",
    "udprn",
    "organisation",
    "department",
    "po_box",
    "sub_building",
    "building_name",
    "building_number",
    "dependent_thoroughfare",
    "thoroughfare",
    "post_town",
    "double_dependent_locality",
    "dependent_locality",
    "postcode",
    "postcode_type",
    "easting",
    "northing",
    "latitude",
    "longitude",
    "logical_status",
    "blpu_state",
    "address_status",
    "start_date",
    "entry_date",
    "last_update_date",
    "classification_code",
]


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def join_parts(*values: str) -> str:
    return ", ".join(
        part
        for part in (clean(value) for value in values)
        if part
    )


def build_number(row) -> str:
    return clean(row["building_number"])


def build_street(row) -> str:
    return join_parts(
        row["dependent_thoroughfare"],
        row["thoroughfare"],
    )


def build_raw_address(row) -> str:
    return join_parts(
        row["organisation"],
        row["department"],
        row["sub_building"],
        row["building_name"],
        row["building_number"],
        row["dependent_thoroughfare"],
        row["thoroughfare"],
        row["double_dependent_locality"],
        row["dependent_locality"],
        row["post_town"],
        row["postcode"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    args = parser.parse_args()

    df = pd.read_csv(
        args.source,
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    out = pd.DataFrame()

    out["property_id"] = "ab_" + df["uprn"].astype(str)
    out["uprn"] = df["uprn"].astype(str)
    out["udprn"] = df["udprn"].astype(str)

    out["raw_address"] = df.apply(build_raw_address, axis=1)

    out["unit"] = df["sub_building"].map(clean)
    out["building"] = df["building_name"].map(clean)
    out["number"] = df.apply(build_number, axis=1)
    out["street"] = df.apply(build_street, axis=1)

    out["locality"] = df["dependent_locality"].map(clean)
    out["town"] = df["post_town"].map(clean)
    out["postcode"] = (
        df["postcode"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    out["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )
    out["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    out["easting"] = pd.to_numeric(
        df["easting"],
        errors="coerce",
    )
    out["northing"] = pd.to_numeric(
        df["northing"],
        errors="coerce",
    )

    out["classification_codes"] = df[
        "classification_code"
    ].map(clean)

    out["source"] = "addressbase_flat"
    out["address_status"] = df["address_status"].map(clean)

    out["canonical_text"] = out.apply(
        lambda row: (
            f"{row['raw_address']} | "
            f"postcode={row['postcode']} | "
            f"uprn={row['uprn']}"
        ),
        axis=1,
    )

    out = out[
        out["uprn"].astype(str).str.strip().ne("")
        & out["raw_address"].astype(str).str.strip().ne("")
    ].drop_duplicates(subset=["uprn"], keep="first")

    out = out.sort_values(
        ["postcode", "raw_address", "uprn"],
        na_position="last",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    if args.csv_output:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(csv_output, index=False)

    print("Rows:", len(out))
    print("Unique UPRNs:", out["uprn"].nunique())
    print("Wrote:", output)

    if args.csv_output:
        print("Wrote:", args.csv_output)


if __name__ == "__main__":
    main()
