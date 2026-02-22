from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Smart Trade Advisor Agent"
    environment: str = "development"
    debug: bool = False

    neon_database_url: Optional[str] = None
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    nyt_api_key: Optional[str] = None
    nyt_section: str = "business"

    refresh_interval_seconds: int = 900
    enable_background_refresh: bool = True
    market_universe: List[str] = Field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "JPM",
            "XOM",
            "LLY",
            "SPY",
            "QQQ",
        ]
    )

    @field_validator("market_universe", mode="before")
    @classmethod
    def parse_market_universe(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

