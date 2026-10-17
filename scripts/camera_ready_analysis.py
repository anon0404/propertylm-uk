"""Camera-ready analyses: external baselines, sensitivity sweep, parse strata.

Inputs are the candidate/block dumps from scripts/dump_candidates.py plus the
coverage audit (present/absent regime labels). Every number is an offline
computation over one logged retrieval pass; scripts/offline_rescore.py
reproduces the deployed matcher's decisions exactly.

Usage (cwd = repo root):
    python scripts/camera_ready_analysis.py DUMP_DIR [DUMP_DIR ...] --out OUT_PREFIX [--subset CSV]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from offline_rescore import Params, contradicts, decide, load_jsonl
from propertylm.utils.address import normalize_text

COMMIT = {"MATCH_CONFIDENT", "MATCH_PROBABLE"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def lexical_text(rec: dict) -> str:
    return f"{rec['text']} {rec['postcode']}"


class BM25:
    """Okapi BM25 with document statistics from the tile gazetteer."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.avgdl = sum(map(len, docs)) / max(self.n, 1)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.default_idf = math.log(1 + (self.n + 0.5) / 0.5)

    def score(self, query: list[str], doc: list[str]) -> float:
        tf = Counter(doc)
        dl = len(doc)
        s = 0.0
        for t in set(query):
            if t not in tf:
                continue
            f = tf[t]
            idf = self.idf.get(t, self.default_idf)
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def load_dumps(dirs: list[str]) -> tuple[list[dict], dict[str, list[dict]]]:
    recs, blocks = {}, {}
    for d in dirs:
        for r in load_jsonl(Path(d) / "candidates.jsonl"):
            recs[r["benchmark_id"]] = r
        for b in load_jsonl(Path(d) / "blocks.jsonl"):
            blocks[b["block"]] = b["records"]
    return list(recs.values()), blocks


# ---------------------------------------------------------------------------
# Ranking functions: each returns a ranked list of
#   {"uprn", "score", "contradicts"}  (score in [0, 1] for thresholding)
# ---------------------------------------------------------------------------


def rank_pipeline(rec: dict, p: Params) -> list[dict]:
    return [
        {"uprn": c["uprn"], "score": c["confidence"], "contradicts": c["contradicts"], "status": c["status"]}
        for c in decide(rec, p)
    ]


def rank_dense(rec: dict, veto: bool = False) -> list[dict]:
    out = [
        {
            "uprn": c["uprn"],
            "score": max(0.0, min(1.0, (c["retrieval_score"] + 1) / 2)),
            "contradicts": contradicts(rec["keys"], c),
        }
        for c in rec["candidates"]
    ]
    out.sort(key=lambda c: -c["score"])
    if veto:
        out = [c for c in out if not c["contradicts"]]
    return out


def rank_lexical(rec: dict, block: list[dict], kind: str, bm25: BM25 | None, veto: bool = False) -> list[dict]:
    q = normalize_text(rec["query"])
    qt = tokens(rec["query"])
    out = []
    for b in block:
        text = lexical_text(b)
        if kind == "fuzzy":
            s = fuzz.token_set_ratio(q, normalize_text(text)) / 100.0
        else:
            s = bm25.score(qt, tokens(text))
        out.append({"uprn": b["uprn"], "score": s, "contradicts": contradicts(rec["keys"], b)})
    out.sort(key=lambda c: -c["score"])
    if kind == "bm25" and out:
        # Map BM25 to [0, 1] by the query's self-score so a single threshold is meaningful.
        self_score = bm25.score(qt, qt) or 1.0
        for c in out:
            c["score"] = min(1.0, c["score"] / self_score)
    if veto:
        out = [c for c in out if not c["contradicts"]]
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def threshold_free(rows: list[dict]) -> dict:
    pres = [r for r in rows if r["present"]]
    top1 = sum(r["top1"] for r in pres) / len(pres)
    top5 = sum(r["top5"] for r in pres) / len(pres)
    # Wrong top-1 answers on present properties that cross a parsed identity key.
    contra = sum(r["top1_contradicts"] and not r["top1"] for r in pres) / len(pres)
    return {"top1": top1, "top5": top5, "top1_contradiction_rate": contra}


def at_matched_precision(rows: list[dict], target: float) -> dict:
    """Lowest score threshold whose present-regime commit precision >= target.

    Tuned on the test set itself, so it is an optimistic (oracle) operating
    point for the baseline.
    """
    pres = sorted((r for r in rows if r["present"] and r["score"] is not None), key=lambda r: -r["score"])
    best = None
    correct = 0
    max_precision, max_precision_cov = 0.0, 0.0
    for i, r in enumerate(pres, 1):
        correct += r["top1"]
        if i == len(pres) or pres[i]["score"] < r["score"]:
            if correct / i > max_precision:
                max_precision, max_precision_cov = correct / i, i / len(pres)
            if correct / i >= target:
                best = r["score"]
    if best is None:
        return {
            "threshold": None,
            "coverage": 0.0,
            "absent_false_accept": 0.0,
            "contradicting_commits": 0,
            "max_precision": max_precision,
            "max_precision_coverage": max_precision_cov,
        }
    n_pres = sum(1 for r in rows if r["present"])
    commit_p = [r for r in rows if r["present"] and r["score"] is not None and r["score"] >= best]
    abs_rows = [r for r in rows if not r["present"]]
    commit_a = [r for r in abs_rows if r["score"] is not None and r["score"] >= best]
    commits = commit_p + commit_a
    return {
        "threshold": best,
        "coverage": len(commit_p) / n_pres,
        "absent_false_accept": len(commit_a) / len(abs_rows),
        "contradicting_commits": sum(r["top1_contradicts"] for r in commits),
        "n_commits": len(commits),
        "max_precision": max_precision,
        "max_precision_coverage": max_precision_cov,
    }


def pipeline_rows(recs, present, p: Params) -> list[dict]:
    rows = []
    for rec in recs:
        ranked = rank_pipeline(rec, p)
        best = ranked[0] if ranked else None
        rows.append(
            {
                "id": rec["benchmark_id"],
                "present": present[rec["benchmark_id"]],
                "top1": int(bool(best) and best["uprn"] == rec["expected_uprn"]),
                "top5": int(any(c["uprn"] == rec["expected_uprn"] for c in ranked[:5])),
                "top1_contradicts": int(bool(best) and best["contradicts"]),
                "status": best["status"] if best else "NO_MATCH",
                "score": best["score"] if best else None,
                "group": rec["test_group"],
                "keys": rec["keys"],
            }
        )
    return rows


def generic_rows(recs, present, ranker) -> list[dict]:
    rows = []
    for rec in recs:
        ranked = ranker(rec)
        best = ranked[0] if ranked else None
        rows.append(
            {
                "id": rec["benchmark_id"],
                "present": present[rec["benchmark_id"]],
                "top1": int(bool(best) and best["uprn"] == rec["expected_uprn"]),
                "top5": int(any(c["uprn"] == rec["expected_uprn"] for c in ranked[:5])),
                "top1_contradicts": int(bool(best) and best["contradicts"]),
                "score": best["score"] if best else None,
            }
        )
    return rows


def status_metrics(rows: list[dict]) -> dict:
    pres = [r for r in rows if r["present"]]
    absn = [r for r in rows if not r["present"]]
    conf = [r for r in pres if r["status"] == "MATCH_CONFIDENT"]
    commits = [r for r in rows if r["status"] in COMMIT]
    return {
        "top1": sum(r["top1"] for r in pres) / len(pres),
        "top5": sum(r["top5"] for r in pres) / len(pres),
        "conf_precision": (sum(r["top1"] for r in conf) / len(conf)) if conf else float("nan"),
        "conf_rate": len(conf) / len(pres),
        "commit_rate_present": sum(r["status"] in COMMIT for r in pres) / len(pres),
        "review_present": sum(r["status"] == "AMBIGUOUS_REVIEW" for r in pres) / len(pres),
        "absent_fa_conf": sum(r["status"] == "MATCH_CONFIDENT" for r in absn) / len(absn),
        "absent_match_claims": sum(r["status"] in COMMIT for r in absn) / len(absn),
        "contradicting_commits": sum(r["top1_contradicts"] for r in commits),
        "top1_contradiction_rate": sum(r["top1_contradicts"] and not r["top1"] for r in pres) / len(pres),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset")
    ap.add_argument("--coverage", default="evaluation/SX9090_coverage_current.csv")
    ap.add_argument("--skip-lexical", action="store_true")
    args = ap.parse_args()

    recs, blocks = load_dumps(args.dumps)
    if args.subset:
        keep = set(pd.read_csv(args.subset, dtype=str).benchmark_id)
        recs = [r for r in recs if r["benchmark_id"] in keep]
    cov = pd.read_csv(args.coverage, dtype=str)
    present = dict(zip(cov.benchmark_id, cov.uprn_in_corpus == "1"))
    recs = [r for r in recs if r["benchmark_id"] in present]
    print(f"{len(recs)} queries, {sum(present[r['benchmark_id']] for r in recs)} present")

    results: dict = {"n": len(recs), "n_present": sum(present[r["benchmark_id"]] for r in recs)}

    # --- Pipeline variants (status-based) ---------------------------------
    variants = {
        "full": Params(),
        "no_rules": Params(use_rules=False),
        "no_reranker": Params(use_reranker=False),
        "dense_only": Params(use_rules=False, use_reranker=False),
    }
    vrows = {name: pipeline_rows(recs, present, p) for name, p in variants.items()}
    results["variants"] = {name: status_metrics(rows) for name, rows in vrows.items()}
    target = results["variants"]["full"]["conf_precision"]

    # --- Baselines at an oracle-matched precision operating point ----------
    baselines = {
        "dense_only": generic_rows(recs, present, rank_dense),
        "dense+veto": generic_rows(recs, present, lambda r: rank_dense(r, veto=True)),
        "neural_no_rules": [dict(r) for r in vrows["no_rules"]],
    }
    if not args.skip_lexical:
        gaz = blocks.get("tile_districts") or []
        bm25 = BM25([tokens(lexical_text(b)) for b in gaz])
        for kind in ("bm25", "fuzzy"):
            baselines[kind] = generic_rows(recs, present, lambda r, k=kind: rank_lexical(r, blocks[r["block"]], k, bm25))
            baselines[kind + "+veto"] = generic_rows(
                recs, present, lambda r, k=kind: rank_lexical(r, blocks[r["block"]], k, bm25, veto=True)
            )
    results["baselines"] = {}
    for name, rows in baselines.items():
        results["baselines"][name] = {**threshold_free(rows), **at_matched_precision(rows, target)}
    full_rows = vrows["full"]
    results["baselines"]["full_system"] = {
        **threshold_free(full_rows),
        "coverage": results["variants"]["full"]["conf_rate"],
        "absent_false_accept": results["variants"]["full"]["absent_fa_conf"],
        "contradicting_commits": sum(
            r["top1_contradicts"] for r in full_rows if r["status"] == "MATCH_CONFIDENT"
        ),
    }

    # --- Sensitivity sweeps (full system) ---------------------------------
    sweeps = []
    for a in [round(0.1 * i, 1) for i in range(11)]:
        sweeps.append(("alpha", a, Params(alpha=a)))
    for s in [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]:
        sweeps.append(("credit_scale", s, Params(credit_scale=s)))
    for s in [0.25, 0.5, 1.0, 2.0]:
        sweeps.append(("penalty_scale", s, Params(penalty_scale=s)))
    for t, m in itertools.product([0.86, 0.88, 0.90, 0.92, 0.94, 0.96], [0.04, 0.06, 0.08, 0.10, 0.12]):
        sweeps.append(("tau_margin", f"{t}/{m}", Params(tau_conf=t, m_conf=m)))
    results["sensitivity"] = [
        {"param": name, "value": v, **status_metrics(pipeline_rows(recs, present, p))} for name, v, p in sweeps
    ]

    # --- Parse strata (which identity keys parsed on the query side) ------
    strata = defaultdict(list)
    for r in full_rows:
        k = r["keys"]
        strata[("P" if k["postcode"] else "-") + ("N" if k["number"] else "-") + ("U" if k["unit"] else "-")].append(r)
    results["strata"] = {}
    for key, rows in sorted(strata.items()):
        if any(r["present"] for r in rows) and any(not r["present"] for r in rows):
            results["strata"][key] = {"n": len(rows), **status_metrics(rows)}

    Path(args.out + ".json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: results[k] for k in ("variants", "baselines")}, indent=1, default=float))


if __name__ == "__main__":
    main()
