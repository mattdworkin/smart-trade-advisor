import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from smart_trade_agent.agent.orchestrator import AgentOrchestrator
from smart_trade_agent.config import get_settings
from smart_trade_agent.models import (
    AgentQueryRequest,
    AgentQueryResponse,
    DashboardPayload,
    IngestRequest,
    UserRole,
)
from smart_trade_agent.services.graph_service import GRAPHITI_AVAILABLE

logger = logging.getLogger("smart_trade_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

settings = get_settings()
orchestrator = AgentOrchestrator(settings)
scheduler: Optional[BackgroundScheduler] = None


def _parse_role(value: Optional[str]) -> Optional[UserRole]:
    if not value:
        return None
    try:
        return UserRole(value)
    except ValueError:
        return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global scheduler
    orchestrator.initialize()
    try:
        orchestrator.refresh_market_intelligence()
    except Exception as exc:  # pragma: no cover - network/storage availability varies
        logger.exception("Initial refresh failed: %s", exc)

    if settings.enable_background_refresh:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            orchestrator.refresh_market_intelligence,
            trigger="interval",
            seconds=settings.refresh_interval_seconds,
            coalesce=True,
            max_instances=1,
        )
        scheduler.start()
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        orchestrator.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
base_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(base_dir / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "refresh_interval_seconds": settings.refresh_interval_seconds,
        },
    )


@app.get("/api/health")
def healthcheck():
    return {
        "status": "ok",
        "background_refresh": settings.enable_background_refresh,
        "last_refresh_at": orchestrator.last_refresh_at.isoformat() if orchestrator.last_refresh_at else None,
        "graphiti_available": GRAPHITI_AVAILABLE,
    }


@app.get("/api/dashboard", response_model=DashboardPayload)
def dashboard(request: Request, role: Optional[UserRole] = None):
    user_id = request.headers.get("x-user-id", "guest")
    header_role = _parse_role(request.headers.get("x-user-role"))
    context = orchestrator.resolve_user_context(
        user_id=user_id,
        explicit_role=role or header_role,
    )
    return orchestrator.get_dashboard(context)


@app.post("/api/query", response_model=AgentQueryResponse)
def query_agent(payload: AgentQueryRequest, request: Request):
    user_id = request.headers.get("x-user-id", "guest")
    header_role = _parse_role(request.headers.get("x-user-role"))
    context = orchestrator.resolve_user_context(
        user_id=user_id,
        explicit_role=payload.role or header_role,
        question=payload.question,
    )
    return orchestrator.answer_question(payload.question, context)


@app.post("/api/knowledge/documents")
def ingest_documents(payload: IngestRequest):
    result = orchestrator.ingest_documents(payload.documents)
    return {"status": "ok", **result}


@app.get("/api/graph/{symbol}")
def graph_relationships(symbol: str, limit: int = 20):
    return {"symbol": symbol.upper(), "relationships": orchestrator.get_relationships(symbol, limit)}


@app.post("/api/refresh", response_model=DashboardPayload)
def refresh_now():
    return orchestrator.refresh_market_intelligence()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=settings.debug)
