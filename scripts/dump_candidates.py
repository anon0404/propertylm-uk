"""Log every candidate's component scores for offline re-scoring.

Runs the PropertyLM-UK pipeline in-process (same encoders, same Qdrant
collection, same postcode cascade as propertylm.retrieval.matcher) but,
instead of returning only the decided top-k, writes for each query the full
over-retrieved candidate pool with its raw retrieval similarity, raw
cross-encoder logit, and candidate-side identity keys. It also writes the
complete lexical block at the same cascade level, so lexical baselines can be
scored over the identical search space.

This makes fusion-weight / threshold / credit sensitivity analyses and
external baselines exact offline computations over one retrieval pass.

Usage (Qdrant running, cwd = repo root):
    python scripts/dump_candidates.py BENCHMARK.csv OUT_DIR
Resumable: already-dumped benchmark_ids are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from qdrant_client import models

from propertylm.config import settings
from propertylm.models.encoders import Encoders
from propertylm.retrieval.matcher import Matcher
from propertylm.retrieval.qdrant_store import QdrantStore
from propertylm.utils.address import address_keys, normalize_text, postcode_district


def candidate_record(payload: dict) -> dict:
    return {
        "uprn": str(payload.get("uprn") or ""),
        "text": Matcher._candidate_text(payload),
        "canonical": payload.get("canonical_text", Matcher._candidate_text(payload)),
        "postcode": Matcher._candidate_postcode(payload),
        "number": Matcher._candidate_number(payload),
        "unit": Matcher._candidate_unit(payload),
    }


def scroll_block(store: QdrantStore, conditions: list) -> list[dict]:
    records, offset = [], None
    flt = models.Filter(should=conditions) if len(conditions) > 1 else models.Filter(must=conditions)
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection,
            scroll_filter=flt,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(candidate_record(p.payload or {}) for p in points)
        if offset is None:
            return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cand_path = out / "candidates.jsonl"
    block_path = out / "blocks.jsonl"

    bench = pd.read_csv(args.benchmark, dtype=str).fillna("")
    done = set()
    if cand_path.exists():
        with cand_path.open() as fh:
            done = {json.loads(line)["benchmark_id"] for line in fh}
    seen_blocks = set()
    if block_path.exists():
        with block_path.open() as fh:
            seen_blocks = {json.loads(line)["block"] for line in fh}

    encoders = Encoders(settings.embed_model, settings.rerank_model, "cpu")
    store = QdrantStore(settings.qdrant_url, settings.collection)
    matcher = Matcher(encoders, store)

    # Lexical stand-in for an unfiltered global search: every indexed property
    # in a postcode district that occurs in the benchmark tile.
    tile_districts = sorted({postcode_district(p) for p in bench["expected_postcode"] if p})

    with cand_path.open("a") as cf, block_path.open("a") as bf:
        for i, row in enumerate(bench.itertuples(index=False)):
            if row.benchmark_id in done:
                continue
            t0 = time.perf_counter()
            keys = address_keys(row.query)
            vector = encoders.embedder.encode(normalize_text(row.query), normalize_embeddings=True).tolist()
            points, mode = matcher._retrieve(vector, keys, top_k=5)
            pairs = [
                (row.query, (p.payload or {}).get("canonical_text", Matcher._candidate_text(p.payload or {})))
                for p in points
            ]
            logits = (
                encoders.reranker.predict(pairs, batch_size=8, show_progress_bar=False).tolist()
                if pairs
                else []
            )
            cands = []
            for p, logit in zip(points, logits):
                rec = candidate_record(p.payload or {})
                rec.update({"retrieval_score": float(p.score), "rerank_logit": float(logit)})
                cands.append(rec)

            field_value = {
                "exact_postcode": ("postcode", keys.postcode),
                "postcode_sector": ("sector", keys.sector),
                "postcode_district": ("district", keys.district),
            }.get(mode)
            block = f"{field_value[0]}={field_value[1]}" if field_value else "tile_districts"
            if block not in seen_blocks:
                if field_value:
                    conds = [models.FieldCondition(key=field_value[0], match=models.MatchValue(value=field_value[1]))]
                else:
                    conds = [
                        models.FieldCondition(key="district", match=models.MatchValue(value=d))
                        for d in tile_districts
                    ]
                bf.write(json.dumps({"block": block, "records": scroll_block(store, conds)}) + "\n")
                bf.flush()
                seen_blocks.add(block)

            cf.write(
                json.dumps(
                    {
                        "benchmark_id": row.benchmark_id,
                        "query": row.query,
                        "expected_uprn": row.expected_uprn,
                        "test_group": row.test_group,
                        "keys": {
                            "postcode": keys.postcode,
                            "number": keys.number,
                            "unit": keys.unit,
                        },
                        "retrieval_mode": mode,
                        "block": block,
                        "candidates": cands,
                        "seconds": round(time.perf_counter() - t0, 3),
                    }
                )
                + "\n"
            )
            cf.flush()
            if i % 100 == 0:
                print(f"{i}/{len(bench)} {row.benchmark_id} {mode} {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
