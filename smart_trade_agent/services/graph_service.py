import re
from collections import defaultdict
from typing import Dict, List, Optional

from smart_trade_agent.config import Settings
from smart_trade_agent.models import KnowledgeDocumentInput

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional at runtime
    GraphDatabase = None  # type: ignore[assignment]

try:
    import graphiti  # noqa: F401

    GRAPHITI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional at runtime
    GRAPHITI_AVAILABLE = False


class GraphService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._driver = None
        if (
            GraphDatabase is not None
            and settings.neo4j_uri
            and settings.neo4j_username
            and settings.neo4j_password
        ):
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
        self._memory_edges: List[Dict[str, object]] = []

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def ingest_documents(self, documents: List[KnowledgeDocumentInput]) -> int:
        edge_count = 0
        for doc in documents:
            symbols = self._extract_symbols(doc)
            if len(symbols) < 2:
                continue
            symbols_sorted = sorted(symbols)
            for i, left in enumerate(symbols_sorted):
                for right in symbols_sorted[i + 1 :]:
                    self.upsert_relationship(
                        left,
                        right,
                        relationship_kind="co_mentioned",
                        source=doc.source,
                        confidence=0.6,
                    )
                    edge_count += 1
        return edge_count

    def upsert_relationship(
        self,
        left_symbol: str,
        right_symbol: str,
        relationship_kind: str,
        source: str,
        confidence: float = 0.5,
    ) -> None:
        left = left_symbol.upper()
        right = right_symbol.upper()
        if self._driver is not None:
            try:
                query = """
                    MERGE (a:Entity {symbol: $left})
                    MERGE (b:Entity {symbol: $right})
                    MERGE (a)-[r:RELATED_TO {kind: $kind}]->(b)
                    SET r.source = $source,
                        r.confidence = $confidence,
                        r.updated_at = datetime()
                """
                with self._driver.session() as session:
                    session.run(
                        query,
                        left=left,
                        right=right,
                        kind=relationship_kind,
                        source=source,
                        confidence=confidence,
                    )
                return
            except Exception:
                self.close()
                self._driver = None

        edge = {
            "left": left,
            "right": right,
            "kind": relationship_kind,
            "source": source,
            "confidence": confidence,
        }
        if edge not in self._memory_edges:
            self._memory_edges.append(edge)

    def query_relationships(self, symbol: str, limit: int = 20) -> List[Dict[str, object]]:
        needle = symbol.upper()
        if self._driver is not None:
            try:
                query = """
                    MATCH (a:Entity)-[r:RELATED_TO]-(b:Entity)
                    WHERE a.symbol = $needle
                    RETURN a.symbol AS left, b.symbol AS right, r.kind AS kind, r.source AS source,
                           r.confidence AS confidence
                    LIMIT $limit
                """
                with self._driver.session() as session:
                    records = session.run(query, needle=needle, limit=limit)
                    return [record.data() for record in records]
            except Exception:
                self.close()
                self._driver = None

        results = [
            edge
            for edge in self._memory_edges
            if edge["left"] == needle or edge["right"] == needle
        ]
        return results[:limit]

    def get_influence_scores(self, symbols: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        upper_symbols = [symbol.upper() for symbol in symbols]

        if self._driver is not None:
            try:
                query = """
                    MATCH (a:Entity)-[r:RELATED_TO]-()
                    WHERE a.symbol = $symbol
                    RETURN count(r) AS edge_count
                """
                with self._driver.session() as session:
                    for symbol in upper_symbols:
                        record = session.run(query, symbol=symbol).single()
                        edge_count = float(record["edge_count"]) if record else 0.0
                        scores[symbol] = edge_count
                return self._normalize_scores(scores, upper_symbols)
            except Exception:
                self.close()
                self._driver = None

        for symbol in upper_symbols:
            edge_count = sum(
                1
                for edge in self._memory_edges
                if edge["left"] == symbol or edge["right"] == symbol
            )
            scores[symbol] = float(edge_count)
        return self._normalize_scores(scores, upper_symbols)

    def _normalize_scores(self, raw: Dict[str, float], symbols: List[str]) -> Dict[str, float]:
        max_score = max(raw.values()) if raw else 0.0
        if max_score == 0.0:
            return {symbol: 0.0 for symbol in symbols}
        return {symbol: raw.get(symbol, 0.0) / max_score for symbol in symbols}

    def _extract_symbols(self, doc: KnowledgeDocumentInput) -> List[str]:
        metadata_symbols = doc.metadata.get("symbols", [])
        symbol_text = f"{doc.title} {doc.content}"
        regex_symbols = re.findall(r"\b[A-Z]{1,5}\b", symbol_text)
        symbols = {
            value.upper()
            for value in list(metadata_symbols) + regex_symbols
            if isinstance(value, str)
        }
        # Common words that frequently appear in headlines but are not tickers.
        filtered = {token for token in symbols if token not in {"THE", "AND", "FOR", "WITH", "FROM"}}
        return sorted(filtered)
