"""Offline re-implementation of the PropertyLM-UK decision stage.

Consumes the output of scripts/dump_candidates.py and reproduces
propertylm.retrieval.matcher.Matcher (adjudication, fusion, ordering, status
assignment) with every constant exposed as a parameter, plus lexical and
neural-only baselines scored over the same cascade blocks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from propertylm.utils.address import identifiers_contradict, identifiers_equal


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class Params:
    alpha: float = 0.25  # weight on retrieval similarity (1 - alpha on reranker)
    use_reranker: bool = True
    use_rules: bool = True
    credit_scale: float = 1.0  # multiplies the agreement credits
    penalty_scale: float = 1.0  # multiplies the absence and contradiction penalties
    tau_conf: float = 0.92
    m_conf: float = 0.08
    tau_prob: float = 0.80
    m_prob: float = 0.03
    tau_rev: float = 0.65
    top_k: int = 5


def adjudicate(keys: dict, cand: dict, p: Params) -> tuple[float, bool]:
    if not p.use_rules:
        return 0.0, False
    adj, flag = 0.0, False
    if keys["postcode"] and cand["postcode"]:
        if keys["postcode"] == cand["postcode"]:
            adj += 0.04 * p.credit_scale
        else:
            adj -= 0.80 * p.penalty_scale
            flag = True
    if identifiers_equal(keys["number"], cand["number"]):
        adj += 0.12 * p.credit_scale
    elif identifiers_contradict(keys["number"], cand["number"]):
        adj -= 0.85 * p.penalty_scale
        flag = True
    elif keys["number"] and not cand["number"]:
        adj -= 0.15 * p.penalty_scale
    if identifiers_equal(keys["unit"], cand["unit"]):
        adj += 0.10 * p.credit_scale
    elif identifiers_contradict(keys["unit"], cand["unit"]):
        adj -= 0.90 * p.penalty_scale
        flag = True
    elif keys["unit"] and not cand["unit"]:
        adj -= 0.25 * p.penalty_scale
    return adj, flag


def contradicts(keys: dict, cand: dict) -> bool:
    """Parsed identity contradiction, independent of whether rules are on."""
    pc = bool(keys["postcode"] and cand["postcode"] and keys["postcode"] != cand["postcode"])
    return (
        pc
        or identifiers_contradict(keys["number"], cand["number"])
        or identifiers_contradict(keys["unit"], cand["unit"])
    )


def statuses(ranked: list[dict], p: Params) -> None:
    if not ranked:
        return
    best = ranked[0]["confidence"]
    second = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
    margin = best - second
    for i, c in enumerate(ranked):
        conf = c["confidence"]
        if c["flag"]:
            s = "NO_MATCH"
        elif i == 0 and conf >= p.tau_conf and margin >= p.m_conf:
            s = "MATCH_CONFIDENT"
        elif i == 0 and conf >= p.tau_prob and margin >= p.m_prob:
            s = "MATCH_PROBABLE"
        elif conf >= p.tau_rev:
            s = "AMBIGUOUS_REVIEW"
        else:
            s = "NO_MATCH"
        c["status"] = s


def decide(record: dict, p: Params) -> list[dict]:
    """Return the ranked, status-labelled top-k exactly as Matcher.match would."""
    out = []
    for c in record["candidates"]:
        r = max(0.0, min(1.0, (c["retrieval_score"] + 1.0) / 2.0))
        x = sigmoid(c["rerank_logit"]) if p.use_reranker else r
        base = p.alpha * r + (1 - p.alpha) * x
        adj, flag = adjudicate(record["keys"], c, p)
        out.append(
            {
                "uprn": c["uprn"],
                "confidence": max(0.0, min(1.0, base + adj)),
                "flag": flag,
                "rerank_p": x,
                "contradicts": contradicts(record["keys"], c),
            }
        )
    out.sort(key=lambda c: (c["flag"], -c["confidence"], -c["rerank_p"]))
    out = out[: p.top_k]
    statuses(out, p)
    return out


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]
