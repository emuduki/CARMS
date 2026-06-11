"""
encoders/tft_encoder.py — Temporal Fusion Transformer encoder for price features.

Reads the 29-feature parquets from Phase 1 and trains a TFT model to:
  1. Predict next-day return (supervised pre-training)
  2. Extract a 64-dimensional embedding per time step

The 64-d embedding captures:
  - Short-term momentum (RSI, MACD)
  - Volatility regime (BB width, ATR)
  - Trend strength (EMA crossover, close_norm)

Architecture (lightweight for MVP — no pytorch-forecasting dependency):
  Input (29 features, window=60) → LSTM(128) → Linear(64) → embedding
  Training target: next-day log return (regression)

GPU: Runs fine on CPU. Use Google Colab T4 for faster training.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.logger import get_logger

log = get_logger(__name__)

EMBED_DIM   = 64
HIDDEN_DIM  = 128
WINDOW_SIZE = 60      # Days of history per sample
BATCH_SIZE  = 64
MAX_EPOCHS  = 50
LR          = 1e-3
PATIENCE    = 8       # Early stopping patience


# ── Dataset ───────────────────────────────────────────────────

class PriceDataset(Dataset):
    """
    Sliding-window dataset over feature-engineered price data.

    Each sample:
      X : (WINDOW_SIZE, n_features)  — normalised feature matrix
      y : scalar                     — next-day log return (target)
    """

    def __init__(self, df: pd.DataFrame, window: int = WINDOW_SIZE):
        from src.features.indicators import get_feature_columns
        feat_cols = get_feature_columns(df)

        # Drop any remaining NaNs and normalise features (z-score per column)
        data = df[feat_cols].copy().dropna()
        self.mean = data.mean()
        self.std  = data.std().replace(0, 1)
        data_norm = (data - self.mean) / self.std

        self.X = data_norm.values.astype(np.float32)
        self.y = df["log_return"].reindex(data.index).values.astype(np.float32)
        self.window = window

    def __len__(self):
        return max(0, len(self.X) - self.window - 1)

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.window]
        y = self.y[idx + self.window]
        return torch.tensor(x), torch.tensor(y)

    @property
    def n_features(self):
        return self.X.shape[1]


# ── Model ─────────────────────────────────────────────────────

class TFTEncoder(nn.Module):
    """
    Lightweight LSTM-based encoder inspired by TFT.

    For the MVP we use a stacked LSTM rather than full TFT attention
    (which requires pytorch-forecasting). Full TFT can replace this in v2.

    Architecture:
      Input  : (batch, seq_len, n_features)
      LSTM   : 2 layers, hidden=128, dropout=0.2
      Linear : 128 → 64  (embedding)
      Head   : 64  → 1   (return prediction, only used during training)
    """

    def __init__(self, n_features: int, hidden: int = HIDDEN_DIM, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.pred_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x               : (batch, seq_len, n_features)
            return_embedding: if True return 64-d embed, else return scalar prediction

        Returns:
            embedding (batch, 64) or prediction (batch, 1)
        """
        out, _ = self.lstm(x)
        last    = out[:, -1, :]         # Take the last timestep
        embed   = self.embed_proj(last)
        if return_embedding:
            return embed
        return self.pred_head(embed)


# ── Training ──────────────────────────────────────────────────

def train_tft_encoder(
    config: dict,
    symbol: str = "BTC-USD",
    save_dir: str = "models",
    device: str = "cpu",
) -> TFTEncoder:
    """
    Trains the TFT encoder on one asset's feature data.

    Args:
        config:   Parsed config dict.
        symbol:   Asset to train on (default BTC-USD — most data).
        save_dir: Where to save the trained checkpoint.
        device:   "cpu" or "cuda".

    Returns:
        Trained TFTEncoder model.
    """
    from src.features.indicators import load_features

    log.info(f"Training TFT encoder on {symbol}...")
    df = load_features(symbol, config["data"]["processed_dir"])
    if df is None or df.empty:
        raise ValueError(f"No feature data for {symbol} — run Phase 1 first")

    # Train / val split (last 20% = validation)
    split = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:split], df.iloc[split:]

    ds_train = PriceDataset(df_train)
    ds_val   = PriceDataset(df_val)

    if len(ds_train) == 0:
        raise ValueError(f"Not enough data to train ({len(df_train)} rows after split)")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model     = TFTEncoder(n_features=ds_train.n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr  = 0
    save_path     = Path(save_dir) / f"tft_encoder_{_safe(symbol)}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"  Train samples: {len(ds_train):,}  |  Val samples: {len(ds_val):,}")
    log.info(f"  Features: {ds_train.n_features}  |  Device: {device}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X, y in dl_train:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(dl_train)

        # ── Validate ──────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in dl_val:
                X, y = X.to(device), y.to(device)
                pred = model(X).squeeze(-1)
                val_loss += criterion(pred, y).item()
        val_loss /= max(len(dl_val), 1)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:>3}/{MAX_EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}")

        # ── Early stopping ────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save({
                "epoch":       epoch,
                "state_dict":  model.state_dict(),
                "val_loss":    val_loss,
                "n_features":  ds_train.n_features,
                "mean":        ds_train.mean.to_dict(),
                "std":         ds_train.std.to_dict(),
                "symbol":      symbol,
            }, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.6f})")
                break

    log.info(f"  ✓ TFT encoder saved → {save_path}  (best val loss: {best_val_loss:.6f})")
    return model


def train_all_tft_encoders(config: dict, device: str = "cpu") -> dict[str, TFTEncoder]:
    """Trains one TFT encoder per asset. Returns dict of symbol → model."""
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )
    models = {}
    for asset in all_assets:
        symbol = asset["symbol"]
        try:
            model = train_tft_encoder(config, symbol=symbol, device=device)
            models[symbol] = model
        except Exception as e:
            log.error(f"TFT training failed for {symbol}: {e}")
    return models


# ── Inference ─────────────────────────────────────────────────

def load_tft_encoder(symbol: str, save_dir: str = "models", device: str = "cpu") -> Optional[TFTEncoder]:
    """Loads a saved TFT encoder checkpoint."""
    path = Path(save_dir) / f"tft_encoder_{_safe(symbol)}.pt"
    if not path.exists():
        log.warning(f"No TFT checkpoint for {symbol} — train first")
        return None
    ckpt  = torch.load(path, map_location=device)
    model = TFTEncoder(n_features=ckpt["n_features"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info(f"Loaded TFT encoder for {symbol} (val_loss={ckpt['val_loss']:.6f})")
    return model


def extract_tft_embeddings(
    model: TFTEncoder,
    df: pd.DataFrame,
    device: str = "cpu",
    window: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Runs the trained encoder over a feature DataFrame and returns
    a DataFrame of 64-d embeddings aligned to the original dates.

    Returns:
        DataFrame with columns [tft_0, tft_1, ..., tft_63] indexed by date.
    """
    ds     = PriceDataset(df, window=window)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()

    embeds = []
    with torch.no_grad():
        for X, _ in loader:
            e = model(X.to(device), return_embedding=True)
            embeds.append(e.cpu().numpy())

    if not embeds:
        return pd.DataFrame()

    embed_arr = np.vstack(embeds)

    from src.features.indicators import get_feature_columns
    feat_cols = get_feature_columns(df)
    data      = df[feat_cols].dropna()
    dates     = data.index[window: window + len(embed_arr)]

    cols = [f"tft_{i}" for i in range(embed_arr.shape[1])]
    return pd.DataFrame(embed_arr, index=dates, columns=cols)


def _safe(symbol: str) -> str:
    return symbol.replace("=", "_").replace("-", "_").replace("/", "_")