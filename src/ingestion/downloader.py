"""
ingestion/downloader.py — Downloads OHLCV price data for all CARMS assets.
"""
 
import time
from pathlib import Path
from typing import Optional
 
import pandas as pd
import yfinance as yf
 
from src.utils.logger import get_logger
 
log = get_logger(__name__)
 
 
def download_all_assets(config: dict) -> dict[str, pd.DataFrame]:
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
 
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )
 
    results = {}
    for asset in all_assets:
        symbol = asset["symbol"]
        name   = asset["name"]
        cat    = asset["category"]
        log.info(f"Downloading {name} ({symbol})  [{cat}]")
 
        df_daily  = _download_with_retry(symbol, config, "daily",  retries=3)
        df_hourly = _download_with_retry(symbol, config, "hourly", retries=3)
 
        if df_daily is not None and not df_daily.empty:
            _save(df_daily,  raw_dir, symbol, "daily")
            _save(df_hourly, raw_dir, symbol, "hourly")
            results[symbol] = df_daily
            hourly_rows = len(df_hourly) if df_hourly is not None else 0
            log.info(f"  ✓ {name}: {len(df_daily):,} daily | {hourly_rows:,} hourly rows")
        else:
            # Last resort: try loading from cache if already downloaded before
            cached = _load_cached(raw_dir, symbol, "daily")
            if cached is not None:
                log.warning(f"  ↩ {name}: live download failed — using cached data ({len(cached):,} rows)")
                results[symbol] = cached
            else:
                log.warning(f"  ✗ No data for {symbol}")
 
        time.sleep(0.5)
 
    log.info(f"Download complete — {len(results)}/{len(all_assets)} assets")
    return results
 
 
def load_asset(symbol: str, interval: str = "daily", raw_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    path = _parquet_path(Path(raw_dir), symbol, interval)
    if not path.exists():
        log.warning(f"No cached data for {symbol} ({interval})")
        return None
    df = pd.read_parquet(path)
    log.debug(f"Loaded {symbol} ({interval}): {len(df):,} rows")
    return df
 
 
def load_all_assets(config: dict, interval: str = "daily") -> dict[str, pd.DataFrame]:
    raw_dir = Path(config["data"]["raw_dir"])
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )
    results = {}
    for asset in all_assets:
        df = load_asset(asset["symbol"], interval, str(raw_dir))
        if df is not None:
            results[asset["symbol"]] = df
    return results
 
 
# ── Internal ──────────────────────────────────────────────────
 
def _download_with_retry(symbol: str, config: dict, interval: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """Tries yf.download first, falls back to ticker.history on failure."""
    for attempt in range(retries):
        try:
            df = _download_via_download(symbol, config, interval)
            if df is not None and not df.empty:
                return df
            # Empty result — try ticker.history fallback
            df = _download_via_ticker(symbol, config, interval)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            wait = 2 * (attempt + 1)
            if attempt < retries - 1:
                log.warning(f"  Attempt {attempt+1} failed for {symbol} ({interval}): {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"  All {retries} attempts failed for {symbol} ({interval}): {e}")
    return None
 
 
def _download_via_download(symbol: str, config: dict, interval: str) -> Optional[pd.DataFrame]:
    """Primary method: yf.download()"""
    data_cfg    = config["data"]
    yf_interval = data_cfg["intervals"][interval]
 
    kwargs = dict(
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
 
    if interval == "hourly":
        kwargs["period"]   = "729d"
        kwargs["interval"] = yf_interval
    else:
        kwargs["start"]    = data_cfg["start_date"]
        kwargs["end"]      = data_cfg["end_date"]
        kwargs["interval"] = yf_interval
 
    df = yf.download(symbol, **kwargs)
    if df is None or df.empty:
        return None
    return _clean_ohlcv(df, symbol)
 
 
def _download_via_ticker(symbol: str, config: dict, interval: str) -> Optional[pd.DataFrame]:
    """Fallback method: yf.Ticker.history() — different code path in yfinance."""
    data_cfg    = config["data"]
    yf_interval = data_cfg["intervals"][interval]
 
    ticker = yf.Ticker(symbol)
 
    if interval == "hourly":
        df = ticker.history(period="729d", interval=yf_interval, auto_adjust=True)
    else:
        df = ticker.history(
            start=data_cfg["start_date"],
            end=data_cfg["end_date"],
            interval=yf_interval,
            auto_adjust=True,
        )
 
    if df is None or df.empty:
        return None
    return _clean_ohlcv(df, symbol)
 
 
def _load_cached(raw_dir: Path, symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Loads previously saved parquet if it exists."""
    path = _parquet_path(raw_dir, symbol, interval)
    if path.exists():
        return pd.read_parquet(path)
    return None
 
 
def _clean_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
 
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
 
    df.columns = [str(c).lower().strip() for c in df.columns]
 
    # Forex pairs have no volume — fill with 0
    if "volume" not in df.columns:
        df["volume"] = 0.0
 
    required = ["open", "high", "low", "close", "volume"]
    present  = [c for c in required if c in df.columns]
    df = df[present]
 
    if "close" not in df.columns or df.empty:
        return pd.DataFrame()
 
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
 
    df.index.name = "date"
    df = df.sort_index().dropna(subset=["close"])
    df["symbol"] = symbol
    return df
 
 
def _parquet_path(raw_dir: Path, symbol: str, interval: str) -> Path:
    safe = symbol.replace("=", "_").replace("-", "_").replace("/", "_")
    return raw_dir / f"{safe}_{interval}.parquet"
 
 
def _save(df: Optional[pd.DataFrame], raw_dir: Path, symbol: str, interval: str):
    if df is None or df.empty:
        return
    path = _parquet_path(raw_dir, symbol, interval)
    df.to_parquet(path, index=True)
    log.debug(f"  Saved → {path.name}  ({len(df):,} rows)")