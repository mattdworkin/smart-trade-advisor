from typing import List

from smart_trade_agent.models import SourceDecision, SourceType


class SourceRouter:
    _graph_terms = (
        "relationship",
        "exposure",
        "supply chain",
        "linked",
        "connected",
        "subsidiary",
        "counterparty",
    )
    _vector_terms = ("document", "policy", "strategy", "knowledge base", "playbook")
    _news_terms = ("news", "headline", "nyt", "new york times", "story", "today")
    _market_terms = (
        "prediction",
        "winner",
        "loser",
        "premarket",
        "pre-market",
        "price",
        "market",
        "long term",
    )

    def route(self, question: str) -> List[SourceDecision]:
        lowered = question.lower()
        decisions: List[SourceDecision] = []

        def add(source: SourceType, reason: str) -> None:
            if source not in {item.source for item in decisions}:
                decisions.append(SourceDecision(source=source, reason=reason))

        if any(term in lowered for term in self._market_terms):
            add(SourceType.MARKET, "Question asks for market movement or prediction.")
        if any(term in lowered for term in self._news_terms):
            add(SourceType.NEWS, "Question references recent headlines or NYT coverage.")
        if any(term in lowered for term in self._graph_terms):
            add(SourceType.GRAPH, "Question requires relationship reasoning across companies.")
            add(SourceType.MARKET, "Relationship questions are grounded in current market context.")
        if any(term in lowered for term in self._vector_terms):
            add(SourceType.VECTOR, "Question asks for knowledge-base retrieval.")

        if "why" in lowered or "because" in lowered:
            add(SourceType.NEWS, "Causal explanation benefits from headline context.")
            add(SourceType.GRAPH, "Causal explanation benefits from relationship graph context.")

        if not decisions:
            add(SourceType.MARKET, "Defaulting to market state for trading questions.")
            add(SourceType.NEWS, "Defaulting to news context for market interpretation.")

        return decisions
