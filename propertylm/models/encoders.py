from sentence_transformers import SentenceTransformer, CrossEncoder

class Encoders:
    def __init__(self, embed_model: str, rerank_model: str | None = None, device: str | None = None):
        self.embedder = SentenceTransformer(embed_model, device=device)
        self.reranker = CrossEncoder(rerank_model, device=device) if rerank_model else None

    @property
    def dimension(self) -> int:
        return self.embedder.get_sentence_embedding_dimension()
