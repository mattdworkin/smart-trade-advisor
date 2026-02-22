import hashlib
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from smart_trade_agent.models import MarketSnapshot

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional at runtime
    yf = None  # type: ignore[assignment]


class MarketDataService:
    def get_snapshot(self, symbols: List[str]) -> Dict[str, MarketSnapshot]:
        snapshots: Dict[str, MarketSnapshot] = {}
        for symbol in symbols:
            snapshots[symbol] = self._fetch_symbol_snapshot(symbol)
        return snapshots

    def get_fundamentals(self, symbol: str) -> Dict[str, float]:
        if yf is not None:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                market_cap = float(
                    info.get("marketCap")
                    or info.get("market_cap")
                    or info.get("market_capitalization")
                    or 0.0
                )
                pe_ratio = float(info.get("trailingPE") or info.get("trailing_pe") or 0.0)
                return {"market_cap": market_cap, "pe_ratio": pe_ratio}
            except Exception:
                pass
        return self._fallback_fundamentals(symbol)

    def _fetch_symbol_snapshot(self, symbol: str) -> MarketSnapshot:
        if yf is not None:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
                if not hist.empty:
                    last_close = float(hist["Close"].iloc[-1])
                    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
                    volume = int(hist["Volume"].iloc[-1]) if "Volume" in hist else 0
                    change_pct = ((last_close - prev_close) / prev_close * 100.0) if prev_close else 0.0
                    return MarketSnapshot(
                        symbol=symbol,
                        price=round(last_close, 2),
                        change_pct=round(change_pct, 3),
                        volume=volume,
                        as_of=datetime.now(timezone.utc),
                    )
            except Exception:
                pass

        return self._fallback_snapshot(symbol)

    def _fallback_snapshot(self, symbol: str) -> MarketSnapshot:
        seed = int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed + datetime.now(timezone.utc).day)
        price = float(rng.uniform(20.0, 850.0))
        change_pct = float(rng.normal(0.0, 2.0))
        volume = int(rng.uniform(500_000, 90_000_000))
        return MarketSnapshot(
            symbol=symbol,
            price=round(price, 2),
            change_pct=round(change_pct, 3),
            volume=volume,
            as_of=datetime.now(timezone.utc),
        )

    def _fallback_fundamentals(self, symbol: str) -> Dict[str, float]:
        seed = int(hashlib.sha256(f"fund-{symbol}".encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        return {
            "market_cap": float(rng.uniform(3_000_000_000, 2_200_000_000_000)),
            "pe_ratio": float(rng.uniform(8.0, 45.0)),
        }

