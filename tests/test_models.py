from smart_trade_agent.agent.source_router import SourceRouter
from smart_trade_agent.models import SourceType


def test_source_router_uses_graph_for_relationship_questions():
    router = SourceRouter()
    decisions = router.route("Show relationship exposure between NVDA and cloud providers.")
    decision_types = {decision.source for decision in decisions}
    assert SourceType.GRAPH in decision_types
    assert SourceType.MARKET in decision_types or SourceType.NEWS in decision_types


def test_source_router_defaults_to_market_and_news():
    router = SourceRouter()
    decisions = router.route("What should I focus on before the open?")
    decision_types = {decision.source for decision in decisions}
    assert SourceType.MARKET in decision_types
    assert SourceType.NEWS in decision_types
