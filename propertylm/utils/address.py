from __future__ import annotations

import re
from dataclasses import dataclass


POSTCODE_RE = re.compile(
    r"\b("
    r"GIR\s?0AA|"
    r"(?:[A-PR-UWYZ][0-9][0-9A-HJKSTUW]?|"
    r"[A-PR-UWYZ][A-HK-Y][0-9][0-9ABEHMNPRV-Y]?)"
    r"\s?[0-9][ABD-HJLNP-UW-Z]{2}"
    r")\b",
    re.IGNORECASE,
)

UNIT_RE = re.compile(
    r"\b(?:flat|apartment|apt|unit|suite|room|floor)\s*"
    r"([0-9]+[a-z]?|[a-z])\b",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(
    r"(?<![a-z0-9])"
    r"([0-9]+(?:\s*[-/]\s*[0-9]+)?[a-z]?)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AddressParts:
    normalized: str
    postcode: str
    house_number: str
    unit: str


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_postcode(value: str) -> str:
    compact = re.sub(r"\s+", "", (value or "").upper())

    if len(compact) < 5:
        return compact

    return f"{compact[:-3]} {compact[-3:]}"


def normalize_identifier(value: str) -> str:
    return re.sub(
        r"\s+",
        "",
        normalize_space(value).lower(),
    )


def extract_postcode(value: str) -> str:
    match = POSTCODE_RE.search(value or "")
    return normalize_postcode(match.group(1)) if match else ""


def remove_postcode(value: str) -> str:
    return POSTCODE_RE.sub(" ", value or "")


def extract_unit(value: str) -> str:
    match = UNIT_RE.search(value or "")
    return normalize_identifier(match.group(1)) if match else ""


def remove_unit_expression(value: str) -> str:
    return UNIT_RE.sub(" ", value or "")


def extract_house_number(value: str) -> str:
    text = remove_postcode(value)
    text = remove_unit_expression(text)

    match = NUMBER_RE.search(text)
    return normalize_identifier(match.group(1)) if match else ""


def parse_address(value: str) -> AddressParts:
    return AddressParts(
        normalized=normalize_space(value).lower(),
        postcode=extract_postcode(value),
        house_number=extract_house_number(value),
        unit=extract_unit(value),
    )


def identifiers_equal(left: str, right: str) -> bool:
    return bool(
        left
        and right
        and normalize_identifier(left) == normalize_identifier(right)
    )


def identifiers_contradict(left: str, right: str) -> bool:
    return bool(
        left
        and right
        and normalize_identifier(left) != normalize_identifier(right)
    )


def postcode_sector(postcode: str) -> str:
    postcode = normalize_postcode(postcode)

    if " " not in postcode:
        return ""

    outward, inward = postcode.split(" ", 1)
    return f"{outward} {inward[:1]}" if inward else outward


def postcode_district(postcode: str) -> str:
    postcode = normalize_postcode(postcode)
    return postcode.split(" ", 1)[0] if postcode else ""


def retrieval_keys(value: str) -> dict[str, str]:
    postcode = extract_postcode(value)

    return {
        "postcode": postcode,
        "sector": postcode_sector(postcode),
        "district": postcode_district(postcode),
    }


@dataclass(frozen=True)
class AddressKeys:
    postcode: str
    sector: str
    district: str
    number: str
    unit: str


def normalize_text(value: str) -> str:
    value = normalize_space(value).lower()
    value = re.sub(r"[^\w\s/-]", " ", value)
    return normalize_space(value)


def address_keys(value: str) -> AddressKeys:
    parts = parse_address(value)

    return AddressKeys(
        postcode=parts.postcode,
        sector=postcode_sector(parts.postcode),
        district=postcode_district(parts.postcode),
        number=parts.house_number,
        unit=parts.unit,
    )
