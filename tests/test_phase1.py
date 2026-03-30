"""
tests/test_phase1.py — Unit tests for Phase 1 pipeline.
 
Run with: python -m pytest tests/test_phase1.py -v
"""
 
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
 
# ── Fixtures ──────────────────────────────────────────────────
 
@pytest.fixture
def sample_ohlcv():
    """Creates a minimal OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    close = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
    df = pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.002),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close":  close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        "symbol": "TEST-USD",
    }, index=dates)
    df.index.name = "date"
    return df
 
 
@pytest.fixture
def sample_config():
    """Minimal config dict for testing."""
    return {
        "data": {
            "raw_dir": "/tmp/carms_test/raw",
            "processed_dir": "/tmp/carms_test/processed",
            "charts_dir": "/tmp/carms_test/charts",
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "intervals": {"daily": "1d", "hourly": "1h", "hourly_lookback_days": 730},
        },
        "features": {
            "chart_image": {"window_size": 20, "image_size": 64, "style": "charles"},
        },
        "assets": {
            "forex": [{"symbol": "EURUSD=X", "name": "EUR/USD", "category": "forex"}],
            "crypto": [{"symbol": "BTC-USD", "name": "Bitcoin", "category": "crypto"}],
            "commodities": [{"symbol": "GC=F", "name": "Gold", "category": "gold"}],
        },
        "api_keys": {"news_api": "", "fred_api": ""},
    }
 
 
# ── OHLCV cleaning tests ───────────────────────────────────────
 
class TestDownloader:
    def test_clean_ohlcv_columns(self, sample_ohlcv):
        from src.ingestion.downloader import _clean_ohlcv
        df = sample_ohlcv.drop(columns=["symbol"])
        cleaned = _clean_ohlcv(df, "TEST-USD")
        assert "close" in cleaned.columns
        assert "symbol" in cleaned.columns
        assert cleaned["symbol"].iloc[0] == "TEST-USD"
 
    def test_clean_ohlcv_no_nans_in_close(self, sample_ohlcv):
        from src.ingestion.downloader import _clean_ohlcv
        df = sample_ohlcv.drop(columns=["symbol"])
        df.loc[df.index[5], "close"] = np.nan
        cleaned = _clean_ohlcv(df, "TEST-USD")
        assert cleaned["close"].isna().sum() == 0
 
    def test_clean_ohlcv_sorted(self, sample_ohlcv):
        from src.ingestion.downloader import _clean_ohlcv
        df = sample_ohlcv.drop(columns=["symbol"]).sample(frac=1)  # Shuffle
        cleaned = _clean_ohlcv(df, "TEST-USD")
        assert cleaned.index.is_monotonic_increasing
 
 
# ── Feature engineering tests ─────────────────────────────────
 
class TestIndicators:
    def test_compute_features_returns_more_columns(self, sample_ohlcv):
        from src.features.indicators import compute_features
        feat = compute_features(sample_ohlcv, "TEST-USD")
        assert len(feat.columns) > len(sample_ohlcv.columns)
 
    def test_rsi_in_valid_range(self, sample_ohlcv):
        from src.features.indicators import compute_features
        feat = compute_features(sample_ohlcv, "TEST-USD")
        assert "rsi_14" in feat.columns
        assert feat["rsi_14"].between(0, 100).all()
 
    def test_no_nans_after_warmup_drop(self, sample_ohlcv):
        from src.features.indicators import compute_features
        feat = compute_features(sample_ohlcv, "TEST-USD")
        assert feat.isnull().sum().sum() == 0
 
    def test_log_return_computed(self, sample_ohlcv):
        from src.features.indicators import compute_features
        feat = compute_features(sample_ohlcv, "TEST-USD")
        assert "log_return" in feat.columns
        assert "volatility_20" in feat.columns
 
    def test_macd_histogram_present(self, sample_ohlcv):
        from src.features.indicators import compute_features
        feat = compute_features(sample_ohlcv, "TEST-USD")
        assert "macd_hist" in feat.columns
 
    def test_get_feature_columns_excludes_raw(self, sample_ohlcv):
        from src.features.indicators import compute_features, get_feature_columns
        feat = compute_features(sample_ohlcv, "TEST-USD")
        feat_cols = get_feature_columns(feat)
        assert "open" not in feat_cols
        assert "close" not in feat_cols
        assert "rsi_14" in feat_cols
 
    def test_save_and_load_features(self, sample_ohlcv, sample_config, tmp_path):
        from src.features.indicators import compute_all_features, load_features
        sample_config["data"]["processed_dir"] = str(tmp_path)
        result = compute_all_features({"TEST-USD": sample_ohlcv}, str(tmp_path))
        assert "TEST-USD" in result
        loaded = load_features("TEST-USD", str(tmp_path))
        assert loaded is not None
        assert len(loaded) > 0
 
 
# ── News fetcher tests ────────────────────────────────────────
 
class TestNewsFetcher:
    def test_fetch_news_no_key_returns_empty(self, sample_config):
        from src.ingestion.news_fetcher import fetch_news
        df = fetch_news(sample_config)
        assert isinstance(df, pd.DataFrame)
 
    def test_fetch_macro_no_key_returns_empty(self, sample_config):
        from src.ingestion.news_fetcher import fetch_macro_data
        df = fetch_macro_data(sample_config)
        assert isinstance(df, pd.DataFrame)
 
 
# ── Logger tests ──────────────────────────────────────────────
 
class TestLogger:
    def test_get_logger_returns_logger(self, tmp_path):
        from src.utils.logger import get_logger
        log = get_logger("test", log_dir=str(tmp_path))
        assert log is not None
 
    def test_load_config_returns_dict(self):
        from src.utils.logger import load_config
        cfg = load_config("configs/config.yaml")
        assert isinstance(cfg, dict)
        assert "assets" in cfg
        assert "data" in cfg
 