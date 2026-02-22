import json
import re
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, List, Optional

from smart_trade_agent.agent.source_router import SourceRouter
from smart_trade_agent.config import Settings
from smart_trade_agent.models import (
    AgentQueryResponse,
    DashboardPayload,
    KnowledgeDocumentInput,
    SourceDecision,
    SourceType,
    UserContext,
    UserRole,
)
from smart_trade_agent.services.embedding_service import EmbeddingService
from smart_trade_agent.services.graph_service import GraphService
from smart_trade_agent.services.implementation_layer import ImplementationLayer
from smart_trade_agent.services.market_data_service import MarketDataService
from smart_trade_agent.services.news_service import NewsService
from smart_trade_agent.services.prediction_service import PredictionService
from smart_trade_agent.services.profile_service import ProfileService
from smart_trade_agent.services.vector_store import VectorStore

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional at runtime
    OpenAI = None  # type: ignore[assignment]


class AgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_service = EmbeddingService(settings)
        self.vector_store = VectorStore(settings, self.embedding_service)
        self.graph_service = GraphService(settings)
        self.market_data_service = MarketDataService()
        self.news_service = NewsService(settings)
        self.prediction_service = PredictionService(self.market_data_service)
        self.profile_service = ProfileService()
        self.source_router = SourceRouter()
        self.implementation_layer = ImplementationLayer()
        self.dashboard_cache = DashboardPayload(generated_at=datetime.now(timezone.utc))
        self.last_refresh_at: Optional[datetime] = None
        self._refresh_lock = Lock()
        self._llm = None
        if settings.openai_api_key and OpenAI is not None:
            self._llm = OpenAI(api_key=settings.openai_api_key)

    def initialize(self) -> None:
        self.vector_store.initialize()
        self.ingest_documents(self._bootstrap_documents())

    def close(self) -> None:
        self.graph_service.close()

    def resolve_user_context(
        self,
        user_id: str,
        explicit_role: Optional[UserRole] = None,
        question: str = "",
    ) -> UserContext:
        return self.profile_service.resolve(user_id=user_id, explicit_role=explicit_role, question=question)

    def ingest_documents(self, documents: List[KnowledgeDocumentInput]) -> Dict[str, int]:
        vector_count = self.vector_store.upsert_documents(documents)
        graph_edges = self.graph_service.ingest_documents(documents)
        return {"vector_documents": vector_count, "graph_edges": graph_edges}

    def refresh_market_intelligence(self) -> DashboardPayload:
        with self._refresh_lock:
            news_items = self.news_service.fetch_daily_nyt(limit=20)
            documents = [
                KnowledgeDocumentInput(
                    id=item.id,
                    source=item.source,
                    title=item.title,
                    content=item.summary,
                    metadata={
                        "url": item.url,
                        "published_at": item.published_at.isoformat(),
                        "symbols": item.symbols,
                    },
                )
                for item in news_items
            ]
            self.ingest_documents(documents)

            snapshots = self.market_data_service.get_snapshot(self.settings.market_universe)
            graph_scores = self.graph_service.get_influence_scores(list(snapshots.keys()))
            ranked = self.prediction_service.generate(snapshots, news_items, graph_scores)

            dashboard = DashboardPayload(
                generated_at=datetime.now(timezone.utc),
                winners=ranked["winners"],
                losers=ranked["losers"],
                long_term=ranked["long_term"],
                watchlist=ranked["watchlist"],
                nyt_briefing=news_items[:8],
            )
            self.dashboard_cache = dashboard
            self.last_refresh_at = dashboard.generated_at
            return dashboard

    def get_dashboard(self, user_context: UserContext) -> DashboardPayload:
        now = datetime.now(timezone.utc)
        stale = self.last_refresh_at is None or (
            now - self.last_refresh_at > timedelta(seconds=self.settings.refresh_interval_seconds)
        )
        if stale:
            self.refresh_market_intelligence()

        base = self.dashboard_cache
        if user_context.role == UserRole.RETAIL:
            return DashboardPayload(
                generated_at=base.generated_at,
                winners=base.winners[:4],
                losers=base.losers[:4],
                long_term=base.long_term[:4],
                watchlist=base.watchlist[:5],
                nyt_briefing=base.nyt_briefing[:5],
            )
        if user_context.role == UserRole.EXECUTIVE:
            return DashboardPayload(
                generated_at=base.generated_at,
                winners=base.winners[:3],
                losers=base.losers[:3],
                long_term=base.long_term[:6],
                watchlist=base.watchlist[:6],
                nyt_briefing=base.nyt_briefing[:6],
            )
        return base

    def answer_question(
        self,
        question: str,
        user_context: UserContext,
    ) -> AgentQueryResponse:
        dashboard = self.get_dashboard(user_context)
        decisions = self.source_router.route(question)
        evidence = self._collect_evidence(question, decisions, dashboard)
        llm_answer = self._ask_llm(question, user_context, evidence)
        answer = llm_answer or self._compose_fallback_answer(user_context, question, evidence, dashboard)
        actions = self.implementation_layer.build_actions(
            user_context=user_context,
            dashboard=dashboard,
            question=question,
            sources_used=decisions,
        )
        return AgentQueryResponse(
            answer=answer,
            user_context=user_context,
            sources_used=decisions,
            actions=actions,
        )

    def get_relationships(self, symbol: str, limit: int = 20) -> List[Dict[str, object]]:
        return self.graph_service.query_relationships(symbol=symbol, limit=limit)

    def _collect_evidence(
        self,
        question: str,
        decisions: List[SourceDecision],
        dashboard: DashboardPayload,
    ) -> Dict[str, object]:
        evidence: Dict[str, object] = {}
        decision_types = {decision.source for decision in decisions}

        if SourceType.MARKET in decision_types:
            evidence["market"] = {
                "generated_at": dashboard.generated_at.isoformat(),
                "winners": [card.model_dump() for card in dashboard.winners[:4]],
                "losers": [card.model_dump() for card in dashboard.losers[:4]],
                "long_term": [card.model_dump() for card in dashboard.long_term[:4]],
            }
        if SourceType.NEWS in decision_types:
            evidence["news"] = [item.model_dump(mode="json") for item in dashboard.nyt_briefing[:5]]
        if SourceType.VECTOR in decision_types:
            evidence["knowledge"] = [
                item.model_dump(mode="json")
                for item in self.vector_store.search(question, limit=5)
            ]
        if SourceType.GRAPH in decision_types:
            symbols = self._extract_symbols(question)
            if not symbols and dashboard.winners:
                symbols = [dashboard.winners[0].symbol]
            graph_data = {}
            for symbol in symbols[:3]:
                graph_data[symbol] = self.graph_service.query_relationships(symbol, limit=12)
            evidence["graph"] = graph_data

        return evidence

    def _compose_fallback_answer(
        self,
        user_context: UserContext,
        question: str,
        evidence: Dict[str, object],
        dashboard: DashboardPayload,
    ) -> str:
        top_winner = dashboard.winners[0] if dashboard.winners else None
        top_loser = dashboard.losers[0] if dashboard.losers else None
        long_term = dashboard.long_term[0] if dashboard.long_term else None

        lines = []
        lines.append(f"Question: {question}")
        if top_winner and top_loser:
            lines.append(
                f"Current model view: {top_winner.symbol} leads short-term upside "
                f"({top_winner.expected_return_1d:+.2f}%), while {top_loser.symbol} screens as weakest "
                f"({top_loser.expected_return_1d:+.2f}%)."
            )
        if long_term:
            lines.append(
                f"Best long-term candidate right now: {long_term.symbol} "
                f"({long_term.expected_return_30d:+.2f}% expected 30-day return)."
            )
        if "news" in evidence and dashboard.nyt_briefing:
            headlines = ", ".join(item.title for item in dashboard.nyt_briefing[:3])
            lines.append(f"NYT drivers: {headlines}.")
        if "graph" in evidence:
            symbols = ", ".join(list(evidence["graph"].keys()))  # type: ignore[union-attr]
            lines.append(f"Relationship graph was consulted for: {symbols}.")

        if user_context.role == UserRole.RETAIL:
            lines.append("Interpretation: prioritize risk control and avoid chasing open volatility.")
        elif user_context.role == UserRole.PRO_TRADER:
            lines.append("Execution framing: focus on opening range break and relative-volume confirmation.")
        elif user_context.role == UserRole.ADVISOR:
            lines.append("Client framing: pair upside opportunities with downside hedge notes.")
        else:
            lines.append("Allocation framing: treat this as a tactical overlay, not a strategic policy change.")

        return " ".join(lines)

    def _ask_llm(
        self,
        question: str,
        user_context: UserContext,
        evidence: Dict[str, object],
    ) -> Optional[str]:
        if not self._llm:
            return None
        try:
            response = self._llm.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a market research AI agent. Use provided evidence only. "
                            "Be explicit about uncertainty, do not fabricate sources, and tailor "
                            f"wording for role={user_context.role.value} with style={user_context.response_style}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Objectives: {', '.join(user_context.objectives)}\n"
                            f"Evidence JSON: {json.dumps(evidence, default=str)}"
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content if response.choices else None
            return content.strip() if content else None
        except Exception:
            return None

    def _extract_symbols(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[A-Z]{1,5}\b", text.upper())
        return list(dict.fromkeys(tokens))

    def _bootstrap_documents(self) -> List[KnowledgeDocumentInput]:
        return [
            KnowledgeDocumentInput(
                id="boot-risk-playbook",
                source="system",
                title="Risk Guardrails",
                content=(
                    "Open with reduced sizing on high-volatility sessions, enforce stop discipline, "
                    "and downgrade conviction when macro headlines conflict with market breadth."
                ),
                metadata={"symbols": ["SPY", "QQQ"]},
            ),
            KnowledgeDocumentInput(
                id="boot-relationship-playbook",
                source="system",
                title="Cross-Asset Relationship Heuristics",
                content=(
                    "Track semiconductor leaders and cloud hyperscalers together, map upstream suppliers, "
                    "and stress test names with high index concentration."
                ),
                metadata={"symbols": ["NVDA", "MSFT", "AMZN", "AAPL", "GOOGL"]},
            ),
        ]

