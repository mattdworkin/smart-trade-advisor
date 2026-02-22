from collections import defaultdict
from typing import Dict, List

from smart_trade_agent.models import MarketSnapshot, NewsItem, PredictionCard
from smart_trade_agent.services.market_data_service import MarketDataService


class PredictionService:
    def __init__(self, market_data_service: MarketDataService) -> None:
        self.market_data_service = market_data_service

    def generate(
        self,
        snapshots: Dict[str, MarketSnapshot],
        news_items: List[NewsItem],
        graph_scores: Dict[str, float],
    ) -> Dict[str, List[PredictionCard]]:
        sentiment_by_symbol = defaultdict(float)
        catalysts_by_symbol: Dict[str, List[str]] = defaultdict(list)

        for item in news_items:
            for symbol in item.symbols:
                if symbol in snapshots:
                    sentiment_by_symbol[symbol] += item.sentiment
                    if len(catalysts_by_symbol[symbol]) < 4:
                        catalysts_by_symbol[symbol].append(item.title)

        cards: List[PredictionCard] = []
        for symbol, snapshot in snapshots.items():
            sentiment = sentiment_by_symbol[symbol]
            relation_score = graph_scores.get(symbol, 0.0)
            fundamentals = self.market_data_service.get_fundamentals(symbol)
            fundamental_score = self._fundamental_score(fundamentals)

            tactical_score = (
                (0.60 * snapshot.change_pct)
                + (1.80 * sentiment)
                + (2.20 * relation_score)
            )
            strategic_score = (
                (0.35 * snapshot.change_pct)
                + (1.20 * sentiment)
                + (3.00 * fundamental_score)
                + (1.60 * relation_score)
            )
            expected_return_1d = self._clamp(tactical_score, -8.0, 8.0)
            expected_return_30d = self._clamp(strategic_score * 2.2, -25.0, 35.0)
            confidence = self._clamp(
                0.45
                + min(0.30, abs(tactical_score) / 15.0)
                + min(0.18, len(catalysts_by_symbol[symbol]) * 0.04),
                0.30,
                0.96,
            )

            rationale = [
                f"Recent momentum {snapshot.change_pct:+.2f}%.",
                f"News sentiment signal {sentiment:+.2f}.",
                f"Relationship graph influence {relation_score:.2f}.",
            ]
            if fundamentals.get("market_cap", 0.0) > 0:
                rationale.append(
                    f"Fundamental score {fundamental_score:+.2f} from valuation and size."
                )

            cards.append(
                PredictionCard(
                    symbol=symbol,
                    expected_return_1d=round(expected_return_1d, 2),
                    expected_return_30d=round(expected_return_30d, 2),
                    confidence=round(confidence, 3),
                    rationale=rationale,
                    catalysts=catalysts_by_symbol[symbol],
                )
            )

        winners = sorted(cards, key=lambda card: card.expected_return_1d, reverse=True)[:6]
        losers = sorted(cards, key=lambda card: card.expected_return_1d)[:6]
        long_term = sorted(cards, key=lambda card: card.expected_return_30d, reverse=True)[:6]
        watchlist = sorted(
            cards,
            key=lambda card: abs(card.expected_return_1d) + abs(card.expected_return_30d) / 8.0,
            reverse=True,
        )[:8]
        return {
            "winners": winners,
            "losers": losers,
            "long_term": long_term,
            "watchlist": watchlist,
        }

    def _fundamental_score(self, fundamentals: Dict[str, float]) -> float:
        market_cap = fundamentals.get("market_cap", 0.0)
        pe_ratio = fundamentals.get("pe_ratio", 0.0)
        size_signal = 0.0
        valuation_signal = 0.0

        if market_cap > 0:
            if market_cap > 500_000_000_000:
                size_signal = 0.9
            elif market_cap > 100_000_000_000:
                size_signal = 0.6
            elif market_cap > 20_000_000_000:
                size_signal = 0.3
            else:
                size_signal = -0.1

        if pe_ratio > 0:
            if pe_ratio < 12:
                valuation_signal = 0.8
            elif pe_ratio < 25:
                valuation_signal = 0.3
            elif pe_ratio < 35:
                valuation_signal = 0.0
            else:
                valuation_signal = -0.4

        return self._clamp((0.55 * size_signal) + (0.45 * valuation_signal), -1.0, 1.0)

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

