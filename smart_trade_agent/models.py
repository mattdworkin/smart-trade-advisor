from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    RETAIL = "retail"
    PRO_TRADER = "pro_trader"
    ADVISOR = "advisor"
    EXECUTIVE = "executive"


class UserContext(BaseModel):
    user_id: str
    role: UserRole = UserRole.RETAIL
    objectives: List[str] = Field(default_factory=list)
    risk_tolerance: str = "medium"
    response_style: str = "concise"


class KnowledgeDocumentInput(BaseModel):
    id: Optional[str] = None
    source: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchResult(BaseModel):
    id: str
    source: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity: float


class NewsItem(BaseModel):
    id: str
    source: str
    title: str
    url: str
    published_at: datetime
    summary: str
    sentiment: float = 0.0
    symbols: List[str] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    symbol: str
    price: float
    change_pct: float
    volume: int
    as_of: datetime


class PredictionCard(BaseModel):
    symbol: str
    expected_return_1d: float
    expected_return_30d: float
    confidence: float
    rationale: List[str] = Field(default_factory=list)
    catalysts: List[str] = Field(default_factory=list)


class DashboardPayload(BaseModel):
    generated_at: datetime
    winners: List[PredictionCard] = Field(default_factory=list)
    losers: List[PredictionCard] = Field(default_factory=list)
    long_term: List[PredictionCard] = Field(default_factory=list)
    watchlist: List[PredictionCard] = Field(default_factory=list)
    nyt_briefing: List[NewsItem] = Field(default_factory=list)


class SourceType(str, Enum):
    MARKET = "market"
    NEWS = "news"
    VECTOR = "vector"
    GRAPH = "graph"


class SourceDecision(BaseModel):
    source: SourceType
    reason: str


class ImplementationAction(BaseModel):
    action_type: str
    reason: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class AgentQueryRequest(BaseModel):
    question: str
    role: Optional[UserRole] = None


class AgentQueryResponse(BaseModel):
    answer: str
    user_context: UserContext
    sources_used: List[SourceDecision] = Field(default_factory=list)
    actions: List[ImplementationAction] = Field(default_factory=list)


class IngestRequest(BaseModel):
    documents: List[KnowledgeDocumentInput]

