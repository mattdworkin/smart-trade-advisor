import hashlib
from typing import List

import numpy as np

from smart_trade_agent.config import Settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore[assignment]


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        if settings.openai_api_key and OpenAI is not None:
            self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._client:
            kwargs = {
                "model": self.settings.embedding_model,
                "input": texts,
            }
            if "text-embedding-3" in self.settings.embedding_model:
                kwargs["dimensions"] = self.settings.embedding_dimensions
            response = self._client.embeddings.create(**kwargs)
            return [list(item.embedding) for item in response.data]
        return [self._deterministic_embedding(text) for text in texts]

    def embed_one(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def _deterministic_embedding(self, text: str) -> List[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.settings.embedding_dimensions)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.tolist()
        return (vector / norm).tolist()

