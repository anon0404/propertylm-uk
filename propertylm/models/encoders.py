from sentence_transformers import SentenceTransformer, CrossEncoder

class Encoders:
    def __init__(self, embed_model: str, rerank_model: str | None = None, device: str | None = None):
        self.embedder = SentenceTransformer(embed_model, device=device)
        # "none"/"off"/empty disables the reranker (ablation support;
        # docker-compose ${VAR:-default} cannot pass an empty string).
        use_reranker = bool(rerank_model) and rerank_model.strip().lower() not in {"none", "off"}
        self.reranker = CrossEncoder(rerank_model, device=device) if use_reranker else None

    @property
    def dimension(self) -> int:
        return self.embedder.get_sentence_embedding_dimension()
