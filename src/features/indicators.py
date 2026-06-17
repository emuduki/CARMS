"""
features/indicators.py — Computes technical indicators for all CARMS assets.
 
Handles both assets WITH volume (Crypto, Gold) and WITHOUT (Forex pairs).
"""
 
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
 
try:
    import pandas_ta_classic as ta
    HAS_PANDAS_TA = True
except ImportError:
    try:
        import pandas_ta as ta
        HAS_PANDAS_TA = True
    except ImportError:
        HAS_PANDAS_TA = False
 
from src.utils.logger import get_logger
 
log = get_logger(__name__)
 
 
def compute_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Computes all technical indicators for a single asset DataFrame.
    Safely handles Forex pairs that have zero/missing volume.
    """
    df = df.copy()
    has_volume = "volume" in df.columns and df["volume"].sum() > 0
    log.info(f"Computing features for {symbol} ({len(df):,} rows)  [volume={'yes' if has_volume else 'no — forex'}]")
 
    # ── Returns & volatility ──────────────────────────────────
    df["return_1d"]     = df["close"].pct_change()
    df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
    df["volatility_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)
    df["range_hl"]      = (df["high"] - df["low"]) / df["close"]
 
    if HAS_PANDAS_TA:
        df = _compute_pandas_ta(df, has_volume)
    else:
        log.warning("pandas-ta not installed — using manual fallbacks")
        df = _compute_manual(df, has_volume)
 
    # ── Price normalisation ───────────────────────────────────
    df["close_norm"] = df["close"] / df["close"].rolling(50).mean()
 
    # ── Drop NaN rows (indicator warm-up) ────────────────────
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    log.info(f"  ✓ {len(df):,} clean rows ({dropped} dropped for indicator warm-up)")
 
    return df
 
 
def compute_all_features(
    asset_data: dict[str, pd.DataFrame],
    processed_dir: str = "data/processed",
) -> dict[str, pd.DataFrame]:
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    results = {}
    for symbol, df in asset_data.items():
        try:
            feat_df = compute_features(df, symbol)
            if feat_df.empty:
                log.warning(f"  No features produced for {symbol} — skipping")
                continue
            path = out_dir / f"{_safe(symbol)}_features.parquet"
            feat_df.to_parquet(path)
            log.info(f"  Saved → {path.name}  ({feat_df.shape[1]} features)")
            results[symbol] = feat_df
        except Exception as e:
            log.error(f"Feature computation failed for {symbol}: {e}")
 
    log.info(f"Feature engineering complete — {len(results)}/{len(asset_data)} assets")
    return results
 
 
def load_features(symbol: str, processed_dir: str = "data/processed") -> Optional[pd.DataFrame]:
    path = Path(processed_dir) / f"{_safe(symbol)}_features.parquet"
    if not path.exists():
        log.warning(f"No features found for {symbol} — run Phase 1 first")
        return None
    return pd.read_parquet(path)
 
 
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    raw_cols = {"open", "high", "low", "close", "volume", "symbol"}
    return [c for c in df.columns if c not in raw_cols]
 
 
# ── Indicator computation ─────────────────────────────────────
 
def _compute_pandas_ta(df: pd.DataFrame, has_volume: bool) -> pd.DataFrame:
    df["rsi_14"] = ta.rsi(df["close"], length=14)
 
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd"]        = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"]   = macd.iloc[:, 2]
 
    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df["bb_upper"] = bbands.iloc[:, 0]
        df["bb_mid"]   = bbands.iloc[:, 1]
        df["bb_lower"] = bbands.iloc[:, 2]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
 
    df["atr_14"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ema_20"]    = ta.ema(df["close"], length=20)
    df["ema_50"]    = ta.ema(df["close"], length=50)
    df["ema_cross"] = (df["ema_20"] > df["ema_50"]).astype(int)
 
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    if stoch is not None and not stoch.empty:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
 
    # Volume-based indicators — only computed when volume is meaningful
    if has_volume and "volume" in df.columns:
        df["obv"]          = ta.obv(df["close"], df["volume"])
        df["volume_sma"]   = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
    else:
        # Forex: fill volume indicators with neutral values so shape is consistent
        df["obv"]          = 0.0
        df["volume_sma"]   = 0.0
        df["volume_ratio"] = 1.0
 
    return df
 
 
def _compute_manual(df: pd.DataFrame, has_volume: bool) -> pd.DataFrame:
    # RSI
    delta  = df["close"].diff()
    gain   = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
    loss   = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
 
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
 
    # Bollinger Bands
    sma20  = df["close"].rolling(20).mean()
    std20  = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_mid"]   = sma20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
 
    # ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"]    = tr.rolling(14).mean()
    df["ema_20"]    = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"]    = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_cross"] = (df["ema_20"] > df["ema_50"]).astype(int)
 
    if has_volume and "volume" in df.columns:
        df["volume_sma"]   = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
        df["obv"]          = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    else:
        df["obv"]          = 0.0
        df["volume_sma"]   = 0.0
        df["volume_ratio"] = 1.0
 
    return df
 
 
def _safe(symbol: str) -> str:
    return symbol.replace("=", "_").replace("-", "_").replace("/", "_")