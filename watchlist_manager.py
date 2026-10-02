import json
import os
from typing import List

WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")

DEFAULT_WATCHLIST = [
    "NVDA",
    "TSLA",
    "AAPL",
    "MSTR",
    "COIN",
    "RIOT",
    "PLTR",
    "CELH",
    "MARA",
    "AMD"
]

def load_watchlist() -> List[str]:
    """Loads watchlist from file or returns defaults."""
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return [ticker.upper().strip() for ticker in data if isinstance(ticker, str)]
        except Exception:
            pass
    return DEFAULT_WATCHLIST.copy()

def save_watchlist(tickers: List[str]) -> bool:
    """Saves watchlist to file."""
    try:
        clean_tickers = list(dict.fromkeys([t.upper().strip() for t in tickers if t.strip()]))
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_tickers, f, indent=2)
        return True
    except Exception:
        return False

def add_to_watchlist(ticker: str) -> bool:
    """Adds a ticker to watchlist."""
    ticker = ticker.upper().strip()
    if not ticker:
        return False
    tickers = load_watchlist()
    if ticker not in tickers:
        tickers.append(ticker)
        return save_watchlist(tickers)
    return True

def remove_from_watchlist(ticker: str) -> bool:
    """Removes a ticker from watchlist."""
    ticker = ticker.upper().strip()
    tickers = load_watchlist()
    if ticker in tickers:
        tickers.remove(ticker)
        return save_watchlist(tickers)
    return False
