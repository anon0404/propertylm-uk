from pathlib import Path
import json
import typer
from rich import print
from propertylm.config import settings
from propertylm.data.prepare import unpack
from propertylm.data.corpus import build_corpus
from propertylm.training.pairs import generate
from propertylm.models.encoders import Encoders
from propertylm.retrieval.qdrant_store import QdrantStore
from propertylm.retrieval.matcher import Matcher
from propertylm.indexing import index_corpus

app = typer.Typer(no_args_is_help=True)

@app.command("prepare-epc")
def prepare_epc(source: Path, clear: bool=False): unpack(source, settings.data_dir/"raw/epc", clear)

@app.command("prepare-openuprn")
def prepare_openuprn(source: Path, clear: bool=False): unpack(source, settings.data_dir/"raw/openuprn", clear)

@app.command("build-corpus")
def corpus(): build_corpus(settings.data_dir/"raw/epc", settings.data_dir/"raw/openuprn", settings.data_dir/"processed/property_corpus.parquet")

@app.command("generate-pairs")
def pairs(max_rows: int|None=None):
    out=settings.data_dir/"training/pairs.parquet"; out.parent.mkdir(parents=True, exist_ok=True)
    generate(str(settings.data_dir/"processed/property_corpus.parquet"), str(out), max_rows)

@app.command("index")
def index(
    batch_size: int = 4096,
    encode_batch_size: int = 512,
    upload_batch_size: int = 1024,
    upload_parallel: int = 4,
    recreate: bool = False,
    device: str | None = None,
):
    enc = Encoders(settings.embed_model, None, device)

    store = QdrantStore(
        settings.qdrant_url,
        settings.collection,
    )

    index_corpus(
        settings.data_dir / "processed/property_corpus.parquet",
        enc,
        store,
        batch_size=batch_size,
        recreate=recreate,
        encode_batch_size=encode_batch_size,
        upload_batch_size=upload_batch_size,
        upload_parallel=upload_parallel,
    )


@app.command("match")
def match(address: str, top_k: int=5, device: str|None=None):
    result=Matcher(Encoders(settings.embed_model, settings.rerank_model, device), QdrantStore(settings.qdrant_url, settings.collection)).match(address, top_k)
    print(json.dumps(result, indent=2, default=str))
