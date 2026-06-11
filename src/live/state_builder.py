"""
live/state_builder.py — Builds live 138-d observation vectors in real time.

Fetches live prices via yfinance (Forex/Gold) and Binance Testnet (Crypto),
applies the trained Phase 2 encoders, and runs the Phase 3 HMM to produce
a regime-labelled state vector ready for the specialist RL agents.

Update cycle: every 60 seconds (daily candle closes at market end).
For intraday paper trading we use the latest available daily close.
"""

from pathlib import Path
from typing import Optional
import time
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

POLL_INTERVAL  = 60      # Seconds between live price updates
WINDOW_SIZE    = 60      # Days of history needed for TFT encoder
ASSETS = {
    "forex":  ["EURUSD=X", "KES=X"],
    "crypto": ["BTC-USD",  "ETH-USD"],
    "gold":   ["GC=F"],
}


class LiveStateBuilder:
    """
    Assembles real-time 138-d observation vectors for all assets.

    Pipeline per tick:
      1. Fetch latest OHLCV candles (yfinance or Binance)
      2. Compute technical indicators (rolling window)
      3. Run TFT encoder → 64-d price embedding
      4. Run HMM → regime label + 4-d regime probabilities
      5. Assemble portfolio state (6-d)
      6. Concatenate → 138-d observation

    Args:
        config:    Parsed config dict.
        save_dir:  Where trained model checkpoints are saved.
        device:    'cpu' or 'cuda'.
    """

    def __init__(self, config: dict, save_dir: str = "models", device: str = "cpu"):
        self.config   = config
        self.save_dir = Path(save_dir)
        self.device   = device

        # Price history buffers (rolling WINDOW_SIZE days)
        self.price_history: dict[str, pd.DataFrame] = {}
        self.latest_prices:  dict[str, float]       = {}
        self.last_update:    dict[str, pd.Timestamp] = {}

        # Loaded models (lazy-loaded on first use)
        self._tft_models    = {}
        self._hmm_model     = None
        self._hmm_pca       = None
        self._hmm_scaler    = None
        self._regime_map    = {}

        log.info("LiveStateBuilder initialised")

    # ── Public API ────────────────────────────────────────────

    def get_observation(self, symbol: str, portfolio_state: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns the current 138-d observation vector for one symbol.

        Args:
            symbol:          Asset symbol e.g. 'BTC-USD'.
            portfolio_state: 6-d vector: [position, pnl, drawdown,
                             holding_days, trade_freq, is_long].

        Returns:
            np.ndarray of shape (138,) or None if data unavailable.
        """
        # Step 1: ensure price history is fresh
        self._refresh_prices(symbol)

        if symbol not in self.price_history or len(self.price_history[symbol]) < WINDOW_SIZE:
            log.warning(f"Insufficient price history for {symbol} "
                        f"({len(self.price_history.get(symbol, []))} rows, need {WINDOW_SIZE})")
            return None

        # Step 2: compute features
        feat_df = self._compute_features(symbol)
        if feat_df is None or feat_df.empty:
            return None

        # Step 3: TFT embedding (64-d)
        tft_embed = self._get_tft_embedding(symbol, feat_df)

        # Step 4: regime probabilities (4-d)
        regime_probs = self._get_regime_probs(tft_embed)

        # Step 5: assemble observation
        obs = np.concatenate([
            tft_embed.astype(np.float32),
            regime_probs.astype(np.float32),
            portfolio_state.astype(np.float32),
        ])

        return obs

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Returns the latest available price for a symbol."""
        self._refresh_prices(symbol)
        return self.latest_prices.get(symbol)

    def get_current_regime(self) -> dict:
        """Returns the current market regime for all assets combined."""
        from src.regime.hmm_detector import load_hmm, REGIME_NAMES
        if self._hmm_model is None:
            self._hmm_model, self._hmm_pca, self._hmm_scaler, self._regime_map = \
                load_hmm(str(self.save_dir))

        if self._hmm_model is None:
            return {"regime": 2, "name": "ranging", "confidence": 0.5}

        # Build a combined feature vector from all available assets
        all_embeds = []
        for sym_list in ASSETS.values():
            for sym in sym_list:
                self._refresh_prices(sym)
                feat_df = self._compute_features(sym)
                if feat_df is not None and not feat_df.empty:
                    embed = self._get_tft_embedding(sym, feat_df)
                    all_embeds.append(embed)

        if not all_embeds:
            return {"regime": 2, "name": "ranging", "confidence": 0.5}

        combined = np.concatenate(all_embeds).reshape(1, -1)
        try:
            scaled = self._hmm_scaler.transform(combined)
            reduced = self._hmm_pca.transform(scaled)
            raw_state  = self._hmm_model.predict(reduced)[0]
            probs      = self._hmm_model.predict_proba(reduced)[0]
            regime_id  = self._regime_map.get(int(raw_state), int(raw_state))
            confidence = float(probs[raw_state])
            return {
                "regime":        regime_id,
                "name":          REGIME_NAMES.get(regime_id, "ranging"),
                "probabilities": probs.tolist(),
                "confidence":    confidence,
            }
        except Exception as e:
            log.warning(f"Regime detection failed: {e}")
            return {"regime": 2, "name": "ranging", "confidence": 0.5}

    # ── Price fetching ────────────────────────────────────────

    def _refresh_prices(self, symbol: str, force: bool = False):
        """Fetches latest price data if cache is stale (>60s old)."""
        now = pd.Timestamp.now()
        last = self.last_update.get(symbol)

        if not force and last is not None:
            if (now - last).total_seconds() < POLL_INTERVAL:
                return   # Cache still fresh

        try:
            if symbol in ["BTC-USD", "ETH-USD"]:
                df = self._fetch_binance_testnet(symbol)
            else:
                df = self._fetch_yfinance(symbol)

            if df is not None and not df.empty:
                self.price_history[symbol]  = df
                self.latest_prices[symbol]  = float(df["close"].iloc[-1])
                self.last_update[symbol]    = now
                log.debug(f"  {symbol}: ${self.latest_prices[symbol]:,.4f}  "
                          f"({len(df)} candles)")
        except Exception as e:
            log.warning(f"Price fetch failed for {symbol}: {e}")

    def _fetch_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches daily OHLCV via yfinance (Forex, Gold)."""
        import yfinance as yf
        try:
            df = yf.download(
                symbol, period="120d", interval="1d",
                auto_adjust=True, progress=False, multi_level_index=False,
            )
            if df is None or df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            if "volume" not in df.columns:
                df["volume"] = 0.0
            if hasattr(df.index, "tz") and df.index.tz:
                df.index = df.index.tz_localize(None)
            df.index.name = "date"
            df["symbol"] = symbol
            return df.dropna(subset=["close"]).tail(WINDOW_SIZE + 30)
        except Exception as e:
            log.warning(f"yfinance error for {symbol}: {e}")
            return None

    def _fetch_binance_testnet(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetches daily OHLCV from Binance Testnet.
        Falls back to yfinance if testnet is unreachable.
        """
        binance_symbol = symbol.replace("-", "").replace("USD", "USDT")
        try:
            import requests
            url = "https://testnet.binance.vision/api/v3/klines"
            params = {
                "symbol":   binance_symbol,
                "interval": "1d",
                "limit":    WINDOW_SIZE + 30,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qav","n_trades","tbbav","tbqav","ignore"
            ])
            df["date"]   = pd.to_datetime(df["open_time"], unit="ms")
            df           = df.set_index("date")
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["symbol"] = symbol
            return df[["open","high","low","close","volume","symbol"]].dropna()
        except Exception as e:
            log.warning(f"Binance testnet failed for {symbol}: {e} — falling back to yfinance")
            return self._fetch_yfinance(symbol)

    # ── Feature computation ───────────────────────────────────

    def _compute_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Computes technical indicators on the cached price history."""
        from src.features.indicators import compute_features
        df = self.price_history.get(symbol)
        if df is None or len(df) < 55:
            return None
        try:
            feat_df = compute_features(df, symbol)
            return feat_df if not feat_df.empty else None
        except Exception as e:
            log.warning(f"Feature computation failed for {symbol}: {e}")
            return None

    # ── Encoder inference ─────────────────────────────────────

    def _get_tft_embedding(self, symbol: str, feat_df: pd.DataFrame) -> np.ndarray:
        """Runs the TFT encoder to get a 64-d price embedding."""
        from src.encoders.tft_encoder import load_tft_encoder, extract_tft_embeddings
        import torch

        if symbol not in self._tft_models:
            model = load_tft_encoder(symbol, str(self.save_dir), self.device)
            if model is None:
                return np.zeros(64, dtype=np.float32)
            self._tft_models[symbol] = model

        model = self._tft_models[symbol]
        try:
            embeds = extract_tft_embeddings(model, feat_df, self.device)
            if embeds.empty:
                return np.zeros(64, dtype=np.float32)
            return embeds.iloc[-1].values.astype(np.float32)
        except Exception as e:
            log.warning(f"TFT embedding failed for {symbol}: {e}")
            return np.zeros(64, dtype=np.float32)

    def _get_regime_probs(self, tft_embed: np.ndarray) -> np.ndarray:
        """Returns 4-d regime probability vector."""
        from src.regime.hmm_detector import load_hmm
        if self._hmm_model is None:
            self._hmm_model, self._hmm_pca, self._hmm_scaler, self._regime_map = \
                load_hmm(str(self.save_dir))

        if self._hmm_model is None:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        try:
            # Pad to expected dimension if needed
            expected = self._hmm_pca.n_features_in_
            x = tft_embed[:expected] if len(tft_embed) >= expected \
                else np.pad(tft_embed, (0, expected - len(tft_embed)))
            x_scaled  = self._hmm_scaler.transform(x.reshape(1, -1))
            x_reduced = self._hmm_pca.transform(x_scaled)
            probs = self._hmm_model.predict_proba(x_reduced)[0]
            # Reorder to match semantic regime labels
            reordered = np.zeros(4, dtype=np.float32)
            for raw, final in self._regime_map.items():
                if raw < len(probs) and final < 4:
                    reordered[final] = probs[raw]
            return reordered
        except Exception as e:
            log.warning(f"Regime prob failed: {e}")
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)