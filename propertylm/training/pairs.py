from __future__ import annotations
import random
import polars as pl
from propertylm.utils.address import normalize_text


def corrupt(text: str) -> str:
    words = normalize_text(text).split()
    if len(words) > 3 and random.random() < .6:
        words.pop(random.randrange(len(words)))
    replacements = {"road":"rd", "street":"st", "avenue":"ave", "close":"cl", "flat":"apt"}
    words = [replacements.get(w, w) for w in words]
    if len(words) > 4 and random.random() < .4:
        i = random.randrange(len(words)-1); words[i], words[i+1] = words[i+1], words[i]
    return " ".join(words)


def generate(corpus: str, output: str, max_rows: int | None = None):
    df = pl.read_parquet(corpus)
    if max_rows: df = df.head(max_rows)
    rows = []
    for row in df.iter_rows(named=True):
        positive = row["canonical_text"]
        rows.append({"anchor": corrupt(row["raw_address"] + " " + row["postcode"]), "positive": positive})
    pl.DataFrame(rows).write_parquet(output)
    print(f"Wrote {len(rows):,} training pairs to {output}")
