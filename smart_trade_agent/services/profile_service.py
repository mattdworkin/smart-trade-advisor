from typing import Dict, List, Optional, Union

from smart_trade_agent.models import UserContext, UserRole


ROLE_DEFAULTS: Dict[UserRole, Dict[str, Union[str, List[str]]]] = {
    UserRole.RETAIL: {
        "risk_tolerance": "medium",
        "response_style": "plain-language",
        "objectives": ["capital preservation", "steady growth", "daily clarity"],
    },
    UserRole.PRO_TRADER: {
        "risk_tolerance": "high",
        "response_style": "compact-and-data-first",
        "objectives": ["intraday edge", "momentum capture", "execution timing"],
    },
    UserRole.ADVISOR: {
        "risk_tolerance": "medium",
        "response_style": "client-ready-brief",
        "objectives": ["portfolio suitability", "clear narratives", "risk framing"],
    },
    UserRole.EXECUTIVE: {
        "risk_tolerance": "low",
        "response_style": "decision-memo",
        "objectives": ["macro risk scan", "capital allocation", "board-level summary"],
    },
}


class ProfileService:
    def __init__(self) -> None:
        self._cache: Dict[str, UserContext] = {}

    def resolve(
        self,
        user_id: str,
        explicit_role: Optional[Union[str, UserRole]] = None,
        question: str = "",
    ) -> UserContext:
        role = self._normalize_role(explicit_role)
        if role is None and user_id in self._cache:
            role = self._cache[user_id].role
        if role is None:
            role = self._infer_role(question)

        defaults = ROLE_DEFAULTS[role]
        objectives = self._infer_objectives(question, defaults["objectives"])  # type: ignore[arg-type]
        context = UserContext(
            user_id=user_id,
            role=role,
            objectives=objectives,
            risk_tolerance=str(defaults["risk_tolerance"]),
            response_style=str(defaults["response_style"]),
        )
        self._cache[user_id] = context
        return context

    def _normalize_role(self, role: Optional[Union[str, UserRole]]) -> Optional[UserRole]:
        if role is None:
            return None
        if isinstance(role, UserRole):
            return role
        try:
            return UserRole(role)
        except ValueError:
            return None

    def _infer_role(self, question: str) -> UserRole:
        lowered = question.lower()
        if any(token in lowered for token in ("advisor", "client", "portfolio review")):
            return UserRole.ADVISOR
        if any(token in lowered for token in ("desk", "scalp", "intraday", "order flow")):
            return UserRole.PRO_TRADER
        if any(token in lowered for token in ("board", "committee", "allocation memo", "exec")):
            return UserRole.EXECUTIVE
        return UserRole.RETAIL

    def _infer_objectives(self, question: str, defaults: List[str]) -> List[str]:
        lowered = question.lower()
        objectives = list(defaults)
        if "long term" in lowered or "retirement" in lowered:
            objectives.append("long-term compounding")
        if "pre-market" in lowered or "premarket" in lowered:
            objectives.append("pre-market setup")
        if "risk" in lowered or "drawdown" in lowered:
            objectives.append("drawdown control")
        return list(dict.fromkeys(objectives))

