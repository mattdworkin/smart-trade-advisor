# Smart Trade Advisor: Agentic Market Research Platform

This project is now a role-aware AI market research agent designed for the morning trading workflow.

It routes each question to the most useful sources (market data, NYT headlines, vector knowledge, and company relationship graph), then produces a personalized answer and an automated implementation action plan.

## What It Does

- Uses a source-routing agent to decide where to fetch evidence.
- Stores and searches knowledge docs in Neon Postgres with `pgvector`.
- Builds/query company-stock relationships in Neo4j (with Graphiti-compatible architecture).
- Pulls daily New York Times business coverage and generates brief summaries.
- Produces continuously refreshed predictions:
  - biggest winners predicted
  - biggest losers predicted
  - best long-term setups
  - high-attention watchlist
- Adapts output and action planning by user role (`retail`, `pro_trader`, `advisor`, `executive`).

## Architecture

- `app.py`: FastAPI app + scheduler + API routes + UI serving
- `smart_trade_agent/agent/orchestrator.py`: central reasoning/orchestration
- `smart_trade_agent/agent/source_router.py`: source selection logic
- `smart_trade_agent/services/vector_store.py`: Neon pgvector ingest/search
- `smart_trade_agent/services/graph_service.py`: Neo4j relationship ingest/query
- `smart_trade_agent/services/news_service.py`: NYT API/RSS ingest + summary
- `smart_trade_agent/services/prediction_service.py`: ranking engine for winners/losers/long-term/watchlist
- `smart_trade_agent/services/profile_service.py`: role and intent adaptation
- `smart_trade_agent/services/implementation_layer.py`: action generation for implementation workflows
- `infrastructure/sql/neon_pgvector.sql`: Neon schema setup (extensions + tables + indexes)

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
```

3. Fill `.env` with:
- `NEON_DATABASE_URL` for pgvector
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- `NYT_API_KEY`
- `OPENAI_API_KEY` (optional but recommended)

4. Run:

```bash
python app.py
```

Open `http://localhost:8000`.

## API Endpoints

- `GET /api/dashboard`
- `POST /api/query`
- `POST /api/knowledge/documents`
- `GET /api/graph/{symbol}`
- `POST /api/refresh`
- `GET /api/health`

## Notes

- If OpenAI is not configured, embeddings and answer synthesis use deterministic fallback logic.
- If Neo4j is not configured, graph operations run in memory.
- If NYT API key is not configured, NYT RSS is used.
- Legacy modules from the previous architecture were archived under `legacy_archive/` to keep the active stack focused.
