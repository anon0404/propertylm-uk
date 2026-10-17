from __future__ import annotations

import math
from typing import Any

from propertylm.config import settings
from propertylm.utils.address import (
    address_keys,
    identifiers_contradict,
    identifiers_equal,
    normalize_postcode,
    normalize_text,
    parse_address,
)


def sigmoid(value: float) -> float:
    """Convert an unbounded reranker logit to a bounded signal."""

    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)

    z = math.exp(value)
    return z / (1.0 + z)


class Matcher:
    def __init__(self, encoders, store):
        self.encoders = encoders
        self.store = store

    def _retrieve(self, vector, keys, top_k):
        hierarchy = [
            ("postcode", keys.postcode, "exact_postcode"),
            ("sector", keys.sector, "postcode_sector"),
            ("district", keys.district, "postcode_district"),
            (None, None, "global"),
        ]

        retrieve_count = max(top_k * 10, 50)

        for field, value, mode in hierarchy:
            if field and not value:
                continue

            points = self.store.search(
                vector,
                field,
                value,
                retrieve_count,
            )

            if points:
                return points, mode

        return [], "none"

    @staticmethod
    def _candidate_text(payload: dict[str, Any]) -> str:
        return str(
            payload.get("raw_address")
            or payload.get("canonical_text")
            or ""
        )

    @staticmethod
    def _candidate_number(payload: dict[str, Any]) -> str:
        # Prefer a structured payload number when available.
        payload_number = str(payload.get("number") or "").strip()
        if payload_number:
            return payload_number

        return parse_address(
            Matcher._candidate_text(payload)
        ).house_number

    @staticmethod
    def _candidate_unit(payload: dict[str, Any]) -> str:
        payload_unit = str(
            payload.get("unit")
            or payload.get("flat")
            or ""
        ).strip()

        if payload_unit:
            return payload_unit

        return parse_address(
            Matcher._candidate_text(payload)
        ).unit

    @staticmethod
    def _candidate_postcode(payload: dict[str, Any]) -> str:
        payload_postcode = str(
            payload.get("postcode") or ""
        ).strip()

        if payload_postcode:
            return normalize_postcode(payload_postcode)

        return parse_address(
            Matcher._candidate_text(payload)
        ).postcode

    def _rerank(self, query: str, candidates: list[dict]) -> None:
        if not self.encoders.reranker or not candidates:
            return

        pairs = [
            (
                query,
                candidate["property"].get(
                    "canonical_text",
                    self._candidate_text(candidate["property"]),
                ),
            )
            for candidate in candidates
        ]

        scores = self.encoders.reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
            candidate["rerank_probability"] = sigmoid(float(score))

    def _apply_rules(
        self,
        candidate: dict,
        query_keys,
        retrieval_mode: str,
    ) -> None:
        payload = candidate["property"]

        candidate_postcode = self._candidate_postcode(payload)
        candidate_number = self._candidate_number(payload)
        candidate_unit = self._candidate_unit(payload)

        reasons = [retrieval_mode]
        contradiction = False
        rule_adjustment = 0.0

        if settings.disable_rules:
            reasons.append("rules_disabled")

        if not settings.disable_rules:
            if query_keys.postcode and candidate_postcode:
                if query_keys.postcode == candidate_postcode:
                    rule_adjustment += 0.04
                    reasons.append("postcode_exact")
                else:
                    rule_adjustment -= 0.80
                    contradiction = True
                    reasons.append("postcode_contradiction")

            if identifiers_equal(
                query_keys.number,
                candidate_number,
            ):
                rule_adjustment += 0.12
                reasons.append("house_number_exact")
            elif identifiers_contradict(
                query_keys.number,
                candidate_number,
            ):
                rule_adjustment -= 0.85
                contradiction = True
                reasons.append("house_number_contradiction")
            elif query_keys.number and not candidate_number:
                rule_adjustment -= 0.15
                reasons.append("candidate_number_missing")

            if identifiers_equal(
                query_keys.unit,
                candidate_unit,
            ):
                rule_adjustment += 0.10
                reasons.append("unit_exact")
            elif identifiers_contradict(
                query_keys.unit,
                candidate_unit,
            ):
                rule_adjustment -= 0.90
                contradiction = True
                reasons.append("unit_contradiction")
            elif query_keys.unit and not candidate_unit:
                rule_adjustment -= 0.25
                reasons.append("candidate_unit_missing")

        retrieval_probability = (
            float(candidate["retrieval_score"]) + 1.0
        ) / 2.0
        retrieval_probability = max(
            0.0,
            min(1.0, retrieval_probability),
        )

        rerank_probability = candidate.get(
            "rerank_probability",
            retrieval_probability,
        )

        # The cross-encoder remains the main semantic signal.
        base_probability = (
            0.25 * retrieval_probability
            + 0.75 * rerank_probability
        )

        confidence = max(
            0.0,
            min(1.0, base_probability + rule_adjustment),
        )

        candidate.update(
            {
                "candidate_postcode": candidate_postcode,
                "candidate_number": candidate_number,
                "candidate_unit": candidate_unit,
                "rule_adjustment": round(rule_adjustment, 4),
                "contradiction": contradiction,
                "reasons": reasons,
                "confidence": confidence,
            }
        )

    @staticmethod
    def _assign_statuses(candidates: list[dict]) -> None:
        if not candidates:
            return

        best_confidence = candidates[0]["confidence"]
        second_confidence = (
            candidates[1]["confidence"]
            if len(candidates) > 1
            else 0.0
        )
        best_margin = best_confidence - second_confidence

        for index, candidate in enumerate(candidates):
            confidence = candidate["confidence"]
            contradiction = candidate["contradiction"]

            if contradiction:
                status = "NO_MATCH"
            elif (
                index == 0
                and confidence >= 0.92
                and best_margin >= 0.08
            ):
                status = "MATCH_CONFIDENT"
            elif (
                index == 0
                and confidence >= 0.80
                and best_margin >= 0.03
            ):
                status = "MATCH_PROBABLE"
            elif confidence >= 0.65:
                status = "AMBIGUOUS_REVIEW"
            else:
                status = "NO_MATCH"

            candidate["confidence"] = round(confidence, 4)
            candidate["status"] = status
            candidate["margin_from_best"] = round(
                best_confidence - confidence,
                4,
            )

    def match(self, query: str, top_k: int = 5):
        keys = address_keys(query)

        vector = self.encoders.embedder.encode(
            normalize_text(query),
            normalize_embeddings=True,
        ).tolist()

        points, mode = self._retrieve(
            vector,
            keys,
            top_k,
        )

        candidates = [
            {
                "id": point.id,
                "retrieval_score": float(point.score),
                "property": point.payload or {},
            }
            for point in points
        ]

        self._rerank(query, candidates)

        for candidate in candidates:
            self._apply_rules(
                candidate,
                keys,
                mode,
            )

        # Non-contradictory candidates rank before contradictory ones.
        candidates.sort(
            key=lambda candidate: (
                candidate["contradiction"],
                -candidate["confidence"],
                -candidate.get("rerank_probability", 0.0),
            )
        )

        candidates = candidates[:top_k]
        self._assign_statuses(candidates)

        return {
            "query": query,
            "query_parts": {
                "postcode": keys.postcode,
                "sector": keys.sector,
                "district": keys.district,
                "number": keys.number,
                "unit": keys.unit,
            },
            "retrieval_mode": mode,
            "best": candidates[0] if candidates else None,
            "matches": candidates,
        }
