from datetime import datetime, timezone

from smart_trade_agent.config import Settings
from smart_trade_agent.models import NewsItem
from smart_trade_agent.services.news_service import NewsService


def _sample_news_item(source: str) -> NewsItem:
    return NewsItem(
        id=f"id-{source}",
        source=source,
        title="Sample title",
        url="https://example.com/article",
        published_at=datetime.now(timezone.utc),
        summary="Sample summary",
        sentiment=0.1,
        symbols=["AAPL"],
    )


def test_newsapi_ai_is_used_when_provider_is_newsapi_ai():
    settings = Settings(
        news_provider="newsapi_ai",
        newsapi_ai_key="demo-key",
        nyt_api_key="nyt-key",
    )
    service = NewsService(settings)
    service._fetch_from_newsapi_ai = lambda limit: [_sample_news_item("newsapi_ai")]  # type: ignore[method-assign]
    service._fetch_from_nyt_api = lambda limit: [_sample_news_item("new_york_times")]  # type: ignore[method-assign]
    service._fetch_from_nyt_rss = lambda limit: [_sample_news_item("new_york_times_rss")]  # type: ignore[method-assign]

    items = service.fetch_daily_market_news(limit=5)
    assert items
    assert items[0].source == "newsapi_ai"


def test_auto_mode_falls_back_to_nyt_if_newsapi_ai_fails():
    settings = Settings(
        news_provider="auto",
        newsapi_ai_key="demo-key",
        nyt_api_key="nyt-key",
    )
    service = NewsService(settings)

    def _raise(_: int):
        raise RuntimeError("provider failure")

    service._fetch_from_newsapi_ai = _raise  # type: ignore[method-assign]
    service._fetch_from_nyt_api = lambda limit: [_sample_news_item("new_york_times")]  # type: ignore[method-assign]
    service._fetch_from_nyt_rss = lambda limit: [_sample_news_item("new_york_times_rss")]  # type: ignore[method-assign]

    items = service.fetch_daily_market_news(limit=5)
    assert items
    assert items[0].source == "new_york_times"

