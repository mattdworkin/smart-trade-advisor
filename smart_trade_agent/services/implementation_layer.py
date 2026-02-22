from typing import List

from smart_trade_agent.models import (
    DashboardPayload,
    ImplementationAction,
    SourceDecision,
    SourceType,
    UserContext,
    UserRole,
)


class ImplementationLayer:
    def build_actions(
        self,
        user_context: UserContext,
        dashboard: DashboardPayload,
        question: str,
        sources_used: List[SourceDecision],
    ) -> List[ImplementationAction]:
        actions: List[ImplementationAction] = []
        top_winners = [card.symbol for card in dashboard.winners[:3]]
        top_losers = [card.symbol for card in dashboard.losers[:3]]

        if user_context.role == UserRole.PRO_TRADER:
            actions.append(
                ImplementationAction(
                    action_type="set_intraday_alerts",
                    reason="Pro-trader profile favors event-driven execution.",
                    payload={"symbols": top_winners + top_losers, "window": "open_to_lunch"},
                )
            )
        elif user_context.role == UserRole.ADVISOR:
            actions.append(
                ImplementationAction(
                    action_type="prepare_client_digest",
                    reason="Advisor profile favors explainable portfolio updates.",
                    payload={
                        "long_term_focus": [card.symbol for card in dashboard.long_term[:5]],
                        "risk_notes": [card.symbol for card in dashboard.losers[:2]],
                    },
                )
            )
        elif user_context.role == UserRole.EXECUTIVE:
            actions.append(
                ImplementationAction(
                    action_type="generate_macro_risk_memo",
                    reason="Executive profile needs capital-allocation-level summary.",
                    payload={"watchlist": [card.symbol for card in dashboard.watchlist[:6]]},
                )
            )
        else:
            actions.append(
                ImplementationAction(
                    action_type="education_brief",
                    reason="Retail profile benefits from interpretable guardrails.",
                    payload={
                        "focus": [card.symbol for card in dashboard.long_term[:3]],
                        "avoid": [card.symbol for card in dashboard.losers[:3]],
                    },
                )
            )

        if any(item.source == SourceType.GRAPH for item in sources_used):
            actions.append(
                ImplementationAction(
                    action_type="run_relationship_scan",
                    reason="Question depends on company/stock relationship graph reasoning.",
                    payload={"depth": 2, "max_edges": 30},
                )
            )
        if "implement" in question.lower() or "execute" in question.lower():
            actions.append(
                ImplementationAction(
                    action_type="create_paper_trade_plan",
                    reason="User asked for implementation-oriented follow-through.",
                    payload={"symbols": top_winners, "risk_template": user_context.risk_tolerance},
                )
            )

        return actions

