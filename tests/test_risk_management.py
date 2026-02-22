from datetime import datetime, timezone

from smart_trade_agent.models import DashboardPayload, PredictionCard, UserRole
from smart_trade_agent.services.implementation_layer import ImplementationLayer
from smart_trade_agent.services.profile_service import ProfileService


def test_profile_service_infers_pro_trader_role():
    service = ProfileService()
    context = service.resolve(
        user_id="u-1",
        explicit_role=None,
        question="I need intraday scalp setups before the open.",
    )
    assert context.role == UserRole.PRO_TRADER


def test_implementation_layer_generates_role_specific_action():
    implementation = ImplementationLayer()
    dashboard = DashboardPayload(
        generated_at=datetime.now(timezone.utc),
        winners=[
            PredictionCard(
                symbol="NVDA",
                expected_return_1d=2.4,
                expected_return_30d=7.1,
                confidence=0.82,
            )
        ],
        losers=[
            PredictionCard(
                symbol="XOM",
                expected_return_1d=-1.4,
                expected_return_30d=-3.2,
                confidence=0.66,
            )
        ],
    )
    context = ProfileService().resolve("user-pro", explicit_role=UserRole.PRO_TRADER)
    actions = implementation.build_actions(
        user_context=context,
        dashboard=dashboard,
        question="Execute implementation plan.",
        sources_used=[],
    )
    action_types = {action.action_type for action in actions}
    assert "set_intraday_alerts" in action_types
    assert "create_paper_trade_plan" in action_types
