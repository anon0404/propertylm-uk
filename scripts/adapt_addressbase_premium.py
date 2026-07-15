#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


DPA_FIELDS = [
    "record_identifier","change_type","pro_order","uprn","udprn",
    "organisation_name","department_name","sub_building_name","building_name",
    "building_number","dependent_thoroughfare","thoroughfare",
    "double_dependent_locality","dependent_locality","post_town","postcode",
    "postcode_type","delivery_point_suffix","welsh_dependent_thoroughfare",
    "welsh_thoroughfare","welsh_double_dependent_locality",
    "welsh_dependent_locality","welsh_post_town","po_box_number",
    "process_date","start_date","end_date","last_update_date","entry_date",
]
BLPU_FIELDS = [
    "record_identifier","change_type","pro_order","uprn","logical_status",
    "blpu_state","blpu_state_date","parent_uprn","x","y","latitude","longitude",
    "rpc","local_custodian_code","country","start_date","end_date",
    "last_update_date","entry_date","addressbase_postal","postcode_locator",
    "multi_occ_count",
]
LPI_FIELDS = [
    "record_identifier","change_type","pro_order","uprn","lpi_key","language",
    "logical_status","start_date","end_date","last_update_date","entry_date",
    "sao_start_number","sao_start_suffix","sao_end_number","sao_end_suffix",
    "sao_text","pao_start_number","pao_start_suffix","pao_end_number",
    "pao_end_suffix","pao_text","usrn","usrn_match_indicator","area_name",
    "level","official_flag",
]
STREET_DESC_FIELDS = [
    "record_identifier","change_type","pro_order","usrn","street_description",
    "locality","town_name","administrative_area","language","start_date",
    "end_date","last_update_date","entry_date",
]
ORG_FIELDS = [
    "record_identifier","change_type","pro_order","uprn","org_key",
    "organisation","legal_name","start_date","end_date","last_update_date",
    "entry_date",
]
CLASS_FIELDS = [
    "record_identifier","change_type","pro_order","uprn","class_key",
    "classification_code","class_scheme","scheme_version","start_date",
    "end_date","last_update_date","entry_date",
]


def clean(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_postcode(value: object) -> str:
    compact = re.sub(r"\s+", "", clean(value).upper())
    return f"{compact[:-3]} {compact[-3:]}" if len(compact) >= 5 else compact


def join_nonempty(parts: Iterable[object], sep: str = ", ") -> str:
    values = [clean(x) for x in parts if clean(x)]
    deduped: list[str] = []
    for value in values:
        if not deduped or value.casefold() != deduped[-1].casefold():
            deduped.append(value)
    return sep.join(deduped)


def number_range(start: str, start_suffix: str, end: str, end_suffix: str) -> str:
    first = f"{clean(start)}{clean(start_suffix)}"
    last = f"{clean(end)}{clean(end_suffix)}"
    if first and last:
        return f"{first}-{last}"
    return first or last


def lpi_address(lpi: dict[str, str], street: dict[str, str] | None, postcode: str) -> dict[str, str]:
    sao_number = number_range(
        lpi["sao_start_number"], lpi["sao_start_suffix"],
        lpi["sao_end_number"], lpi["sao_end_suffix"],
    )
    pao_number = number_range(
        lpi["pao_start_number"], lpi["pao_start_suffix"],
        lpi["pao_end_number"], lpi["pao_end_suffix"],
    )
    unit = join_nonempty([lpi["sao_text"], sao_number], " ")
    building = clean(lpi["pao_text"])
    street_name = clean(street["street_description"]) if street else ""
    locality = join_nonempty(
        [street.get("locality", "") if street else "", lpi["area_name"]]
    )
    town = clean(street["town_name"]) if street else ""
    raw = join_nonempty([unit, building, pao_number, street_name, locality, town, postcode])
    return {
        "raw_address": raw,
        "unit": unit,
        "building": building,
        "number": pao_number,
        "street": street_name,
        "locality": locality,
        "town": town,
        "postcode": postcode,
    }


def dpa_address(dpa: dict[str, str]) -> dict[str, str]:
    postcode = normalise_postcode(dpa["postcode"])
    unit = join_nonempty([dpa["department_name"], dpa["sub_building_name"]])
    building = clean(dpa["building_name"])
    number = clean(dpa["building_number"])
    street = join_nonempty([dpa["dependent_thoroughfare"], dpa["thoroughfare"]])
    locality = join_nonempty(
        [dpa["double_dependent_locality"], dpa["dependent_locality"]]
    )
    town = clean(dpa["post_town"])
    raw = join_nonempty(
        [
            dpa["organisation_name"], unit, building, number, street,
            locality, town, postcode,
        ]
    )
    return {
        "raw_address": raw,
        "unit": unit,
        "building": building,
        "number": number,
        "street": street,
        "locality": locality,
        "town": town,
        "postcode": postcode,
    }


def parse_mixed_csv(path: Path):
    blpu: dict[str, dict[str, str]] = {}
    dpas: list[dict[str, str]] = []
    lpis: list[dict[str, str]] = []
    streets: dict[tuple[str, str], dict[str, str]] = {}
    orgs: dict[str, list[str]] = defaultdict(list)
    classes: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            record_type = row[0]
            counts[record_type] += 1
            if record_type == "21" and len(row) == len(BLPU_FIELDS):
                item = dict(zip(BLPU_FIELDS, row))
                blpu[item["uprn"]] = item
            elif record_type == "28" and len(row) == len(DPA_FIELDS):
                dpas.append(dict(zip(DPA_FIELDS, row)))
            elif record_type == "24" and len(row) == len(LPI_FIELDS):
                lpis.append(dict(zip(LPI_FIELDS, row)))
            elif record_type == "15" and len(row) == len(STREET_DESC_FIELDS):
                item = dict(zip(STREET_DESC_FIELDS, row))
                streets[(item["usrn"], item["language"])] = item
            elif record_type == "31" and len(row) == len(ORG_FIELDS):
                item = dict(zip(ORG_FIELDS, row))
                if clean(item["organisation"]):
                    orgs[item["uprn"]].append(clean(item["organisation"]))
            elif record_type == "32" and len(row) == len(CLASS_FIELDS):
                item = dict(zip(CLASS_FIELDS, row))
                if clean(item["classification_code"]):
                    classes[item["uprn"]].append(clean(item["classification_code"]))

    return blpu, dpas, lpis, streets, orgs, classes, dict(counts)


def canonical_text(row: dict[str, object]) -> str:
    fields = [
        ("unit", row.get("unit")),
        ("building", row.get("building")),
        ("number", row.get("number")),
        ("street", row.get("street")),
        ("locality", row.get("locality")),
        ("town", row.get("town")),
        ("postcode", row.get("postcode")),
        ("uprn", row.get("uprn")),
    ]
    return " | ".join(f"{key}={clean(value)}" for key, value in fields if clean(value))


def build_rows(path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    blpu, dpas, lpis, streets, orgs, classes, counts = parse_mixed_csv(path)
    output: list[dict[str, object]] = []
    dpa_uprns: set[str] = set()

    for dpa in dpas:
        uprn = clean(dpa["uprn"])
        dpa_uprns.add(uprn)
        address = dpa_address(dpa)
        b = blpu.get(uprn, {})
        row: dict[str, object] = {
            "property_id": f"abp_{uprn}",
            "uprn": uprn,
            "udprn": clean(dpa["udprn"]),
            "source": "addressbase_premium_dpa",
            "address_status": "postal",
            "logical_status": clean(b.get("logical_status")),
            "classification_codes": "|".join(sorted(set(classes.get(uprn, [])))),
            "organisation": join_nonempty(orgs.get(uprn, []), " | "),
            "latitude": clean(b.get("latitude")),
            "longitude": clean(b.get("longitude")),
            "x": clean(b.get("x")),
            "y": clean(b.get("y")),
            **address,
        }
        row["canonical_text"] = canonical_text(row)
        output.append(row)

    # Add approved geographic addresses only where there is no DPA.
    for lpi in lpis:
        uprn = clean(lpi["uprn"])
        if uprn in dpa_uprns or clean(lpi["logical_status"]) != "1":
            continue
        b = blpu.get(uprn, {})
        postcode = normalise_postcode(b.get("postcode_locator", ""))
        street = (
            streets.get((clean(lpi["usrn"]), clean(lpi["language"])))
            or streets.get((clean(lpi["usrn"]), "ENG"))
            or next(
                (v for (usrn, _), v in streets.items() if usrn == clean(lpi["usrn"])),
                None,
            )
        )
        address = lpi_address(lpi, street, postcode)
        if not address["raw_address"]:
            continue
        row = {
            "property_id": f"abp_{uprn}",
            "uprn": uprn,
            "udprn": "",
            "source": "addressbase_premium_lpi",
            "address_status": "approved_geographic",
            "logical_status": clean(b.get("logical_status")),
            "classification_codes": "|".join(sorted(set(classes.get(uprn, [])))),
            "organisation": join_nonempty(orgs.get(uprn, []), " | "),
            "latitude": clean(b.get("latitude")),
            "longitude": clean(b.get("longitude")),
            "x": clean(b.get("x")),
            "y": clean(b.get("y")),
            **address,
        }
        row["canonical_text"] = canonical_text(row)
        output.append(row)

    # One preferred record per UPRN: DPA wins over LPI.
    by_uprn: dict[str, dict[str, object]] = {}
    for row in output:
        uprn = str(row["uprn"])
        if uprn not in by_uprn or row["source"] == "addressbase_premium_dpa":
            by_uprn[uprn] = row
    return list(by_uprn.values()), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/addressbase_properties.parquet"))
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    rows, counts = build_rows(args.source)
    frame = pd.DataFrame(rows).sort_values(["postcode", "raw_address", "uprn"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv_output, index=False)
    try:
        frame.to_parquet(args.output, index=False)
    except ImportError as exc:
        if not args.csv_output:
            fallback = args.output.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            print(f"Parquet unavailable; wrote CSV fallback to {fallback}")
        print(f"Warning: could not write Parquet: {exc}")
    manifest = {
        "source": str(args.source),
        "records_by_type": counts,
        "output_rows": int(len(frame)),
        "unique_uprns": int(frame["uprn"].nunique()),
        "postal_rows": int((frame["source"] == "addressbase_premium_dpa").sum()),
        "geographic_rows": int((frame["source"] == "addressbase_premium_lpi").sum()),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
