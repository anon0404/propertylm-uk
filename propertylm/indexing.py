from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
from qdrant_client import models


PAYLOAD_COLUMNS = [
    "property_id",
    "epc_id",
    "uprn",
    "raw_address",
    "canonical_text",
    "postcode",
    "number",
    "street",
    "town",
    "inspection_date",
    "lat",
    "lon",
]


def _postcode_sector(postcode: str) -> str:
    postcode = (postcode or "").strip().upper()
    if " " not in postcode:
        return ""
    outward, inward = postcode.split(" ", 1)
    return f"{outward} {inward[:1]}" if inward else outward


def _postcode_district(postcode: str) -> str:
    postcode = (postcode or "").strip().upper()
    return postcode.split(" ", 1)[0] if postcode else ""


def _make_points(
    rows: list[dict[str, Any]],
    vectors,
    start_id: int,
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []

    for local_i, (row, vector) in enumerate(zip(rows, vectors)):
        postcode = str(row.get("postcode") or "").strip().upper()

        payload = {
            key: row.get(key)
            for key in PAYLOAD_COLUMNS
            if key in row
        }

        payload["postcode"] = postcode
        payload["sector"] = _postcode_sector(postcode)
        payload["district"] = _postcode_district(postcode)

        points.append(
            models.PointStruct(
                id=start_id + local_i,
                vector=vector,
                payload=payload,
            )
        )

    return points


def _upload(
    store,
    points: list[models.PointStruct],
    upload_batch_size: int,
    upload_parallel: int,
) -> None:
    store.client.upload_points(
        collection_name=store.collection,
        points=points,
        batch_size=upload_batch_size,
        parallel=upload_parallel,
        wait=True,
    )


def index_corpus(
    corpus: Path,
    encoders,
    store,
    batch_size: int = 4096,
    recreate: bool = False,
    encode_batch_size: int = 512,
    upload_batch_size: int = 1024,
    upload_parallel: int = 4,
) -> None:
    store.create(encoders.dimension, recreate=recreate)

    lazy = pl.scan_parquet(corpus)
    total = lazy.select(pl.len()).collect().item()

    available_columns = set(lazy.collect_schema().names())
    selected_columns = [
        column
        for column in PAYLOAD_COLUMNS
        if column in available_columns
    ]

    if "canonical_text" not in selected_columns:
        raise ValueError("Corpus has no canonical_text column")

    # One background upload allows CPU encoding and Qdrant uploading
    # to happen at the same time.
    executor = ThreadPoolExecutor(max_workers=1)
    pending_upload = None

    for offset in range(0, total, batch_size):
        chunk = (
            lazy
            .slice(offset, batch_size)
            .select(selected_columns)
            .collect()
        )

        texts = chunk["canonical_text"].fill_null("").to_list()

        vectors = encoders.embedder.encode(
            texts,
            batch_size=encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Qdrant accepts NumPy vectors. Do not call vector.tolist()
        # for every record.
        rows = chunk.to_dicts()
        points = _make_points(rows, vectors, offset)

        # Ensure the previous upload finished before submitting another.
        if pending_upload is not None:
            pending_upload.result()

        pending_upload = executor.submit(
            _upload,
            store,
            points,
            upload_batch_size,
            upload_parallel,
        )

        completed = min(offset + chunk.height, total)
        print(f"Encoded/submitted {completed:,}/{total:,}")

    if pending_upload is not None:
        pending_upload.result()

    executor.shutdown(wait=True)
    print(f"Indexed {total:,} records")
