from propertylm.api.app import matcher


def main():
    result = matcher.match(
        "16 Alesmore WS13 8FD",
        5,
    )

    assert result["retrieval_mode"] == "exact_postcode"
    assert result["best"] is not None

    best = result["best"]
    assert best["property"]["uprn"] == "10009682874"
    assert best["status"] in {
        "MATCH_CONFIDENT",
        "MATCH_PROBABLE",
    }
    assert "house_number_exact" in best["reasons"]
    assert "postcode_exact" in best["reasons"]

    for candidate in result["matches"][1:]:
        candidate_number = candidate.get("candidate_number")

        if candidate_number and candidate_number != "16":
            assert candidate["status"] == "NO_MATCH"
            assert "house_number_contradiction" in candidate["reasons"]

    print("Phase 1 deterministic checks passed")


if __name__ == "__main__":
    main()
