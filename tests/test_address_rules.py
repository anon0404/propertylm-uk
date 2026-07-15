from propertylm.utils.address import (
    address_keys,
    identifiers_contradict,
    identifiers_equal,
    parse_address,
)


def test_standard_address():
    parts = parse_address("16 Alesmore WS13 8FD")

    assert parts.postcode == "WS13 8FD"
    assert parts.house_number == "16"
    assert parts.unit == ""


def test_compact_postcode():
    parts = parse_address("16 Alesmore WS138FD")

    assert parts.postcode == "WS13 8FD"
    assert parts.house_number == "16"


def test_flat_and_house_number():
    parts = parse_address(
        "Flat 2, 16 Alesmore Meadow, WS13 8FD"
    )

    assert parts.postcode == "WS13 8FD"
    assert parts.house_number == "16"
    assert parts.unit == "2"


def test_house_suffix():
    parts = parse_address("16A High Street BS1 2AB")

    assert parts.house_number == "16a"


def test_number_equality():
    assert identifiers_equal("16A", "16a")
    assert not identifiers_equal("16", "17")


def test_number_contradiction():
    assert identifiers_contradict("16", "17")
    assert identifiers_contradict("16", "16a")
    assert not identifiers_contradict("16", "")


def test_retrieval_keys():
    keys = address_keys("Flat 2, 16 Alesmore WS13 8FD")

    assert keys.postcode == "WS13 8FD"
    assert keys.sector == "WS13 8"
    assert keys.district == "WS13"
    assert keys.number == "16"
    assert keys.unit == "2"
