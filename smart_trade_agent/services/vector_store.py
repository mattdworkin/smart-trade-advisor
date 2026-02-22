import math
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from psycopg.types.json import Json

from smart_trade_agent.config import Settings
from smart_trade_agent.db.neon import NeonStore
from smart_trade_agent.models import (
    KnowledgeDocumentInput,
    KnowledgeSearchResult,
)
from smart_trade_agent.services.embedding_service import EmbeddingService


class VectorStore:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.neon = NeonStore(settings.neon_database_url)
        self._memory: List[Tuple[KnowledgeDocumentInput, List[float]]] = []

    def initialize(self) -> None:
        if not self.neon.enabled:
            return
        schema_path = Path(__file__).resolve().parents[2] / "infrastructure/sql/neon_pgvector.sql"
        if schema_path.exists():
            self.neon.run_sql_file(schema_path)

    def upsert_documents(self, documents: List[KnowledgeDocumentInput]) -> int:
        if not documents:
            return 0
        embeddings = self.embedding_service.embed_many(
            [f"{doc.title}\n{doc.content}" for doc in documents]
        )
        count = 0

        if self.neon.enabled:
            sql = """
                INSERT INTO knowledge_documents (id, source, title, content, metadata, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (id) DO UPDATE
                SET source = EXCLUDED.source,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    created_at = NOW();
            """
            params = []
            for doc, embedding in zip(documents, embeddings):
                doc_id = doc.id or str(uuid.uuid4())
                params.append(
                    (
                        doc_id,
                        doc.source,
                        doc.title,
                        doc.content,
                        Json(doc.metadata),
                        embedding,
                    )
                )
                count += 1
            self.neon.executemany(sql, params)
            return count

        for doc, embedding in zip(documents, embeddings):
            normalized = doc.model_copy(update={"id": doc.id or str(uuid.uuid4())})
            self._memory.append((normalized, embedding))
            count += 1
        return count

    def search(self, query: str, limit: int = 5) -> List[KnowledgeSearchResult]:
        query_embedding = self.embedding_service.embed_one(query)
        if self.neon.enabled:
            sql = """
                SELECT id::text AS id, source, title, content, metadata,
                       (1 - (embedding <=> %s)) AS similarity
                FROM knowledge_documents
                ORDER BY embedding <=> %s
                LIMIT %s;
            """
            rows = self.neon.fetch_all(sql, (query_embedding, query_embedding, limit))
            return [
                KnowledgeSearchResult(
                    id=row["id"],
                    source=row["source"],
                    title=row["title"],
                    content=row["content"],
                    metadata=row.get("metadata") or {},
                    similarity=float(row["similarity"] or 0.0),
                )
                for row in rows
            ]

        scored: List[KnowledgeSearchResult] = []
        for doc, embedding in self._memory:
            similarity = self._cosine_similarity(query_embedding, embedding)
            scored.append(
                KnowledgeSearchResult(
                    id=doc.id or str(uuid.uuid4()),
                    source=doc.source,
                    title=doc.title,
                    content=doc.content,
                    metadata=doc.metadata,
                    similarity=similarity,
                )
            )
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:limit]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

