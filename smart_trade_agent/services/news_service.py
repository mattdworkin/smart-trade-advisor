import hashlib
import re
from datetime import datetime, timezone
from typing import List

import feedparser
import requests
from bs4 import BeautifulSoup

from smart_trade_agent.config import Settings
from smart_trade_agent.models import NewsItem


class NewsService:
    POSITIVE_TERMS = {"beat", "growth", "rally", "surge", "upgrade", "bullish", "expands"}
    NEGATIVE_TERMS = {"miss", "drop", "fall", "downgrade", "bearish", "slump", "cuts"}
    SYMBOL_ALIASES = {
        "AAPL": ("apple",),
        "MSFT": ("microsoft",),
        "NVDA": ("nvidia",),
        "AMZN": ("amazon",),
        "GOOGL": ("google", "alphabet"),
        "META": ("meta", "facebook"),
        "TSLA": ("tesla",),
        "JPM": ("jpmorgan", "jp morgan"),
        "XOM": ("exxon", "exxonmobil"),
        "LLY": ("eli lilly", "lilly"),
        "SPY": ("s&p 500", "sp500"),
        "QQQ": ("nasdaq 100",),
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch_daily_nyt(self, limit: int = 10) -> List[NewsItem]:
        items = self._fetch_from_nyt_api(limit=limit) if self.settings.nyt_api_key else []
        if items:
            return items
        return self._fetch_from_nyt_rss(limit=limit)

    def _fetch_from_nyt_api(self, limit: int) -> List[NewsItem]:
        section = self.settings.nyt_section.lower().strip() or "business"
        url = f"https://api.nytimes.com/svc/topstories/v2/{section}.json"
        response = requests.get(
            url,
            params={"api-key": self.settings.nyt_api_key},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        items: List[NewsItem] = []
        for result in payload.get("results", [])[:limit]:
            title = result.get("title", "Untitled")
            raw_abstract = result.get("abstract", "")
            summary = self._short_summary(raw_abstract or result.get("title", ""))
            published_at = self._parse_datetime(result.get("published_date"))
            content = f"{title}. {raw_abstract}"
            symbols = self._extract_symbols(content)
            items.append(
                NewsItem(
                    id=self._stable_id(result.get("url", title)),
                    source="new_york_times",
                    title=title,
                    url=result.get("url", ""),
                    published_at=published_at,
                    summary=summary,
                    sentiment=self._sentiment_score(content),
                    symbols=symbols,
                )
            )
        return items

    def _fetch_from_nyt_rss(self, limit: int) -> List[NewsItem]:
        feed = feedparser.parse("https://rss.nytimes.com/services/xml/rss/nyt/Business.xml")
        items: List[NewsItem] = []
        for entry in feed.entries[:limit]:
            title = str(getattr(entry, "title", "Untitled"))
            link = str(getattr(entry, "link", ""))
            description = str(getattr(entry, "summary", ""))
            cleaned_description = self._strip_html(description)
            summary = self._short_summary(cleaned_description or title)
            published_at = self._parse_feed_date(entry)
            content = f"{title}. {cleaned_description}"
            symbols = self._extract_symbols(content)
            items.append(
                NewsItem(
                    id=self._stable_id(link or title),
                    source="new_york_times_rss",
                    title=title,
                    url=link,
                    published_at=published_at,
                    summary=summary,
                    sentiment=self._sentiment_score(content),
                    symbols=symbols,
                )
            )
        return items

    def _short_summary(self, text: str, max_sentences: int = 2) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        selected = " ".join(sentences[:max_sentences]).strip()
        return selected if selected else text[:240]

    def _sentiment_score(self, text: str) -> float:
        lowered = text.lower()
        positive = sum(1 for term in self.POSITIVE_TERMS if term in lowered)
        negative = sum(1 for term in self.NEGATIVE_TERMS if term in lowered)
        raw = positive - negative
        return max(-1.0, min(1.0, raw / 4.0))

    def _extract_symbols(self, text: str) -> List[str]:
        candidates = re.findall(r"\b[A-Z]{1,5}\b", text)
        filtered = {
            symbol
            for symbol in candidates
            if symbol not in {"NYT", "CEO", "U.S", "THE", "AND", "FOR", "FROM", "WITH"}
        }
        lowered = text.lower()
        for symbol, aliases in self.SYMBOL_ALIASES.items():
            if any(alias in lowered for alias in aliases):
                filtered.add(symbol)
        return sorted(filtered)

    def _parse_datetime(self, value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

    def _parse_feed_date(self, entry: object) -> datetime:
        parsed = getattr(entry, "published_parsed", None)
        if not parsed:
            return datetime.now(timezone.utc)
        return datetime(
            year=parsed.tm_year,
            month=parsed.tm_mon,
            day=parsed.tm_mday,
            hour=parsed.tm_hour,
            minute=parsed.tm_min,
            second=parsed.tm_sec,
            tzinfo=timezone.utc,
        )

    def _stable_id(self, value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()

    def _strip_html(self, html: str) -> str:
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
