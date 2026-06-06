"""Local embeddings via sentence-transformers.

Key fix: the service now EXPOSES the model's max sequence length and a token
counter, so the chunker can clamp chunk size to what the model can actually
embed (instead of silently truncating 750-token chunks at 512)."""
from __future__ import annotations


# bge models expect this prefix on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # The number everything else must respect (512 for bge-small-en-v1.5).
        self.max_seq_length: int = int(self._model.max_seq_length)
        self._tokenizer = self._model.tokenizer

    @property
    def model_version(self) -> str:
        # Stable signature independent of load path: the repo id "BAAI/bge-small-en-v1.5"
        # and a local folder ".../bge-small-en-v1.5" both reduce to the same basename,
        # so loading offline by path does not trigger a spurious full re-index.
        from pathlib import Path
        dim = self._model.get_embedding_dimension()
        return f"{Path(self.model_name).name}@dim{dim}"

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        is_bge = "bge" in self.model_name.lower()
        text = (BGE_QUERY_PREFIX + query) if is_bge else query
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()
