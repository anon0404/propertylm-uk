from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from propertylm.config import settings
from propertylm.models.encoders import Encoders
from propertylm.retrieval.matcher import Matcher
from propertylm.retrieval.qdrant_store import QdrantStore

app = FastAPI(title="PropertyLM UK")


class MatchRequest(BaseModel):
    address: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


encoders = Encoders(
    settings.embed_model,
    settings.rerank_model,
    "cpu",
)

store = QdrantStore(
    settings.qdrant_url,
    settings.collection,
)

matcher = Matcher(encoders, store)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": settings.collection,
        "qdrant": settings.qdrant_url,
    }


@app.post("/match")
async def match(req: MatchRequest):
    try:
        return matcher.match(req.address, req.top_k)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )
