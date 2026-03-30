"""
ingestion/news_fetcher.py — Fetches financial news and macroeconomic data.
 
Sources:
  1. NewsAPI.org  — financial headlines (free tier: 100 req/day)
  2. FRED API     — macro time series (free, requires key)
"""
 
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
 
import pandas as pd
 
from src.utils.logger import get_logger
 
log = get_logger(__name__)
 
ASSET_KEYWORDS = {
    "EURUSD=X":  ["euro dollar exchange rate", "ECB monetary policy"],
    "KES=X":     ["Kenya shilling dollar", "Kenya central bank rate"],
    "BTC-USD":   ["Bitcoin price", "crypto market rally"],
    "ETH-USD":   ["Ethereum price", "DeFi market"],
    "GC=F":      ["gold price market", "gold futures rally"],
}
 
# Verified FRED series IDs (confirmed working as of 2025)
# Format: series_id -> human label
MACRO_SERIES = {
    "CPIAUCSL":   "US CPI (inflation)",          # Monthly
    "DFF":        "Fed Funds Rate (daily)",       # Daily — replaces FEDFUNDS
    "T10Y2Y":     "10Y-2Y Treasury spread",       # Daily
    "DEXUSEU":    "USD/EUR Exchange Rate",        # Daily
    "DEXCHUS":    "USD/CNY Exchange Rate",        # Daily — useful macro signal
    "GOLDAMGBD228NLBM": "Gold Price London Fix",  # Daily
}
 
 
def fetch_news(config: dict) -> pd.DataFrame:
    """Fetches financial news from NewsAPI and saves to CSV."""
    api_key  = config["api_keys"].get("news_api", "")
    news_dir = Path(config["data"]["raw_dir"]) / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
 
    if not api_key or api_key == "YOUR_NEWSAPI_KEY":
        log.warning("No NewsAPI key — skipping news fetch")
        log.info("Get free key at https://newsapi.org and add to configs/config.local.yaml")
        return pd.DataFrame()
 
    all_articles = []
    for symbol, keywords in ASSET_KEYWORDS.items():
        log.info(f"Fetching news for {symbol}...")
        for query in keywords[:2]:
            articles = _fetch_newsapi(query, api_key)
            for a in articles:
                a["symbol"] = symbol
                a["query"]  = query
            all_articles.extend(articles)
            time.sleep(0.5)
 
    if not all_articles:
        log.warning("No news articles retrieved")
        return pd.DataFrame()
 
    df = _clean_news(pd.DataFrame(all_articles))
    path = news_dir / "headlines.csv"
    df.to_csv(path, index=False)
    log.info(f"Saved {len(df):,} news articles → {path}")
    return df
 
 
def fetch_macro_data(config: dict) -> pd.DataFrame:
    """Fetches macroeconomic time series from FRED API."""
    api_key = config["api_keys"].get("fred_api", "")
    raw_dir = Path(config["data"]["raw_dir"])
    start   = config["data"]["start_date"]
    end     = config["data"]["end_date"]
 
    if not api_key or api_key == "YOUR_FRED_API_KEY":
        log.warning("No FRED API key — skipping macro fetch")
        log.info("Get free key at https://fred.stlouisfed.org and add to configs/config.local.yaml")
        return pd.DataFrame()
 
    frames = {}
    for series_id, description in MACRO_SERIES.items():
        log.info(f"Fetching FRED: {series_id} ({description})")
        df = _fetch_fred_series(series_id, api_key, start, end)
        if df is not None and not df.empty:
            frames[series_id] = df["value"]
            log.info(f"  ✓ {len(df):,} observations")
        else:
            log.warning(f"  ✗ No data for {series_id} — skipping")
        time.sleep(0.3)
 
    if not frames:
        log.warning("No FRED series fetched — check API key and connection")
        return pd.DataFrame()
 
    macro_df = pd.DataFrame(frames)
    macro_df.index.name = "date"
    macro_df = macro_df.sort_index().ffill()
 
    path = raw_dir / "macro_fred.parquet"
    macro_df.to_parquet(path)
    log.info(f"Saved macro data {macro_df.shape} → {path.name}")
    return macro_df
 
 
def load_news(raw_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    path = Path(raw_dir) / "news" / "headlines.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["published_at"])
 
 
def load_macro(raw_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    path = Path(raw_dir) / "macro_fred.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
 
 
# ── Internal helpers ──────────────────────────────────────────
 
def _fetch_newsapi(query: str, api_key: str, days_back: int = 30) -> list[dict]:
    """Calls NewsAPI with retry on timeout."""
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url    = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey":   api_key,
    }
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [
                {
                    "published_at": a.get("publishedAt"),
                    "title":        a.get("title", ""),
                    "description":  a.get("description", ""),
                    "source":       a.get("source", {}).get("name", ""),
                    "url":          a.get("url", ""),
                }
                for a in articles
            ]
        except requests.exceptions.Timeout:
            if attempt == 0:
                log.warning(f"NewsAPI timeout for '{query}' — retrying...")
                time.sleep(3)
            else:
                log.warning(f"NewsAPI timeout for '{query}' after retry — skipping")
        except Exception as e:
            log.warning(f"NewsAPI error for '{query}': {e}")
            break
    return []
 
 
def _fetch_fred_series(series_id: str, api_key: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetches a single FRED series with retry on 500 errors."""
    url    = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":         series_id,
        "api_key":           api_key,
        "file_type":         "json",
        "observation_start": start,
        "observation_end":   end,
    }
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=15)
 
            # FRED occasionally returns 500 transiently — retry once
            if resp.status_code == 500 and attempt == 0:
                log.warning(f"  FRED 500 for {series_id} — retrying in 3s...")
                time.sleep(3)
                continue
 
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
 
            if not observations:
                return None
 
            df = pd.DataFrame(observations)
            df["date"]  = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.set_index("date")[["value"]].dropna()
            return df
 
        except requests.exceptions.HTTPError as e:
            log.warning(f"  FRED HTTP error for {series_id}: {e}")
            return None
        except Exception as e:
            log.warning(f"  FRED error for {series_id}: {e}")
            return None
 
    return None
 
 
def _clean_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["published_at"] = df["published_at"].dt.tz_localize(None)
    df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    df = df.dropna(subset=["published_at"])
    df = df.drop_duplicates(subset=["title"])
    df = df.sort_values("published_at").reset_index(drop=True)
    return df
 