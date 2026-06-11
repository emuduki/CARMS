"""
encoders/fusion.py — Multi-modal fusion layer for CARMS.

Combines the three encoder embeddings into a single unified state vector:
  TFT embedding   (64-d)  — price & technical indicators
  CNN embedding   (64-d)  — candlestick visual patterns
  Sentiment embed (32-d)  — news sentiment per asset

Total input: 160-d → Fusion MLP → 128-d unified state vector

The fusion layer uses:
  1. Concatenation of all three embeddings
  2. Cross-modal attention gating — learns which modality to trust more
     in each market regime (e.g. news matters more around FOMC events)
  3. Layer normalisation + residual connection
  4. Final projection to 128-d state vector

The 128-d state vector is the input to:
  - Phase 3: HMM regime detector
  - Phase 4: RL specialist agents
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils.logger import get_logger

log = get_logger(__name__)

TFT_DIM    = 64
CNN_DIM    = 64
SENT_DIM   = 32
TOTAL_DIM  = TFT_DIM + CNN_DIM + SENT_DIM   # 160
STATE_DIM  = 128     # Output unified state vector size
BATCH_SIZE = 64
MAX_EPOCHS = 30
LR         = 5e-4


# ── Model ─────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft attention gate that learns which modality to emphasise.

    For each input vector, computes a scalar weight in [0,1] and
    scales the embedding. During training, the gate learns that:
      - In trending markets: TFT (price) weight increases
      - Around news events: sentiment weight increases
      - In pattern-heavy markets: CNN weight increases
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.gate(x)
        return x * weight


class FusionLayer(nn.Module):
    """
    Cross-modal attention fusion combining TFT + CNN + Sentiment embeddings.

    Architecture:
      3 inputs (64, 64, 32) → attention gates → concat (160-d)
      → LayerNorm → MLP (160 → 256 → 128) → output (128-d state)

    The 128-d output is the unified market state vector passed to
    the regime detector (Phase 3) and RL agents (Phase 4).
    """

    def __init__(
        self,
        tft_dim:   int = TFT_DIM,
        cnn_dim:   int = CNN_DIM,
        sent_dim:  int = SENT_DIM,
        state_dim: int = STATE_DIM,
    ):
        super().__init__()
        total = tft_dim + cnn_dim + sent_dim

        # Per-modality attention gates
        self.gate_tft  = AttentionGate(tft_dim)
        self.gate_cnn  = AttentionGate(cnn_dim)
        self.gate_sent = AttentionGate(sent_dim)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, state_dim),
            nn.LayerNorm(state_dim),
        )

        # Residual projection for skip connection
        self.residual_proj = nn.Linear(total, state_dim)

    def forward(
        self,
        tft_embed:  torch.Tensor,
        cnn_embed:  torch.Tensor,
        sent_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tft_embed  : (batch, 64)
            cnn_embed  : (batch, 64)
            sent_embed : (batch, 32)

        Returns:
            state_vector : (batch, 128)
        """
        # Apply attention gates
        tft_gated  = self.gate_tft(tft_embed)
        cnn_gated  = self.gate_cnn(cnn_embed)
        sent_gated = self.gate_sent(sent_embed)

        # Concatenate all modalities
        combined = torch.cat([tft_gated, cnn_gated, sent_gated], dim=-1)

        # MLP + residual
        state = self.fusion_mlp(combined) + self.residual_proj(combined)
        return state


# ── Dataset for fusion training ───────────────────────────────

class FusionDataset(torch.utils.data.Dataset):
    """
    Dataset that aligns TFT, CNN, and sentiment embeddings by date.

    For dates where sentiment is missing (no news), fills with zeros.
    Label: next-day return direction (binary).
    """

    def __init__(
        self,
        tft_embeds:  pd.DataFrame,   # (dates, 64) indexed by date
        cnn_embeds:  pd.DataFrame,   # (dates, 64) indexed by date
        sent_embeds: pd.DataFrame,   # (symbol×date, 32+) MultiIndex
        symbol:      str,
        price_df:    pd.DataFrame,   # feature df for labels
    ):
        # Find common dates across all modalities
        tft_dates  = set(tft_embeds.index)
        cnn_dates  = set(cnn_embeds.index)
        common     = sorted(tft_dates & cnn_dates)

        self.tft_arr  = tft_embeds.loc[common].values.astype(np.float32)
        self.cnn_arr  = cnn_embeds.loc[common].values.astype(np.float32)
        self.dates    = pd.DatetimeIndex(common)

        # Sentiment — fill missing dates with zeros
        sent_cols = [c for c in sent_embeds.columns if c.startswith("sent_")]
        if symbol in sent_embeds.index.get_level_values(0):
            sym_sent = sent_embeds.loc[symbol][sent_cols]
            self.sent_arr = np.array([
                sym_sent.loc[d].values if d in sym_sent.index else np.zeros(SENT_DIM)
                for d in self.dates
            ], dtype=np.float32)
        else:
            self.sent_arr = np.zeros((len(self.dates), SENT_DIM), dtype=np.float32)

        # Labels: next-day direction from price data
        ret = price_df["log_return"].reindex(self.dates)
        self.labels = (ret.values > 0).astype(np.float32)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tft_arr[idx]),
            torch.tensor(self.cnn_arr[idx]),
            torch.tensor(self.sent_arr[idx]),
            torch.tensor(self.labels[idx]),
        )


# ── Training ──────────────────────────────────────────────────

def train_fusion_layer(
    config:      dict,
    symbol:      str,
    tft_embeds:  pd.DataFrame,
    cnn_embeds:  pd.DataFrame,
    sent_embeds: pd.DataFrame,
    save_dir:    str = "models",
    device:      str = "cpu",
) -> Optional[FusionLayer]:
    """
    Trains the fusion layer to combine all three embedding streams.

    The fusion layer is trained end-to-end on the direction prediction
    task — it learns to weight modalities appropriately.

    Args:
        config:      Parsed config dict.
        symbol:      Asset symbol.
        tft_embeds:  DataFrame of TFT embeddings (dates × 64).
        cnn_embeds:  DataFrame of CNN embeddings (dates × 64).
        sent_embeds: DataFrame of sentiment embeddings (symbol×date × 32).
        save_dir:    Model save directory.
        device:      "cpu" or "cuda".

    Returns:
        Trained FusionLayer.
    """
    from src.features.indicators import load_features

    price_df = load_features(symbol, config["data"]["processed_dir"])
    if price_df is None:
        log.warning(f"No price features for {symbol}")
        return None

    dataset = FusionDataset(tft_embeds, cnn_embeds, sent_embeds, symbol, price_df)
    if len(dataset) < 50:
        log.warning(f"Too few fusion samples for {symbol} ({len(dataset)})")
        return None

    split    = int(len(dataset) * 0.8)
    dl_train = DataLoader(
        torch.utils.data.Subset(dataset, range(split)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    dl_val = DataLoader(
        torch.utils.data.Subset(dataset, range(split, len(dataset))),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model     = FusionLayer().to(device)
    # Add a classification head for training only
    head      = nn.Linear(STATE_DIM, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=LR
    )

    log.info(f"Training fusion layer for {symbol} ({len(dataset):,} samples)...")

    best_acc  = 0.0
    save_path = Path(save_dir) / f"fusion_{_safe(symbol)}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    patience_ctr = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train(); head.train()
        for tft, cnn, sent, lbl in dl_train:
            tft, cnn, sent, lbl = [t.to(device) for t in (tft, cnn, sent, lbl)]
            optimizer.zero_grad()
            state  = model(tft, cnn, sent)
            logits = head(state).squeeze(-1)
            loss   = criterion(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

        # Validate
        model.eval(); head.eval()
        correct = total = 0
        with torch.no_grad():
            for tft, cnn, sent, lbl in dl_val:
                tft, cnn, sent, lbl = [t.to(device) for t in (tft, cnn, sent, lbl)]
                state  = model(tft, cnn, sent)
                preds  = (torch.sigmoid(head(state).squeeze(-1)) > 0.5).float()
                correct += (preds == lbl).sum().item()
                total   += len(lbl)

        acc = correct / max(total, 1)
        if epoch % 5 == 0:
            log.info(f"  Epoch {epoch}/{MAX_EPOCHS}  val_acc={acc:.1%}")

        if acc > best_acc:
            best_acc = acc
            patience_ctr = 0
            torch.save({"state_dict": model.state_dict(), "val_acc": acc}, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= 8:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    log.info(f"  ✓ Fusion layer saved → {save_path}  (best val acc: {best_acc:.1%})")
    return model


# ── Inference ─────────────────────────────────────────────────

def load_fusion_layer(symbol: str, save_dir: str = "models", device: str = "cpu") -> Optional[FusionLayer]:
    """Loads a saved fusion layer checkpoint."""
    path = Path(save_dir) / f"fusion_{_safe(symbol)}.pt"
    if not path.exists():
        log.warning(f"No fusion checkpoint for {symbol}")
        return None
    ckpt  = torch.load(path, map_location=device)
    model = FusionLayer().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info(f"Loaded fusion layer for {symbol} (val_acc={ckpt.get('val_acc', 0):.1%})")
    return model


def build_state_vectors(
    fusion_model: FusionLayer,
    tft_embeds:   pd.DataFrame,
    cnn_embeds:   pd.DataFrame,
    sent_embeds:  pd.DataFrame,
    symbol:       str,
    device:       str = "cpu",
) -> pd.DataFrame:
    """
    Generates the final 128-d unified state vectors for all dates.

    This is the direct input to the Phase 3 regime detector and
    Phase 4 RL agents.

    Returns:
        DataFrame with columns [state_0, ..., state_127] indexed by date.
    """
    tft_dates = set(tft_embeds.index)
    cnn_dates = set(cnn_embeds.index)
    common    = sorted(tft_dates & cnn_dates)

    tft_arr  = tft_embeds.loc[common].values.astype(np.float32)
    cnn_arr  = cnn_embeds.loc[common].values.astype(np.float32)

    sent_cols = [c for c in sent_embeds.columns if c.startswith("sent_")]
    if symbol in sent_embeds.index.get_level_values(0):
        sym_sent  = sent_embeds.loc[symbol][sent_cols]
        sent_arr  = np.array([
            sym_sent.loc[d].values if d in sym_sent.index else np.zeros(SENT_DIM)
            for d in pd.DatetimeIndex(common)
        ], dtype=np.float32)
    else:
        sent_arr = np.zeros((len(common), SENT_DIM), dtype=np.float32)

    fusion_model.eval()
    states = []
    with torch.no_grad():
        for i in range(0, len(common), 256):
            tft_b  = torch.tensor(tft_arr[i:i+256]).to(device)
            cnn_b  = torch.tensor(cnn_arr[i:i+256]).to(device)
            sent_b = torch.tensor(sent_arr[i:i+256]).to(device)
            s      = fusion_model(tft_b, cnn_b, sent_b)
            states.append(s.cpu().numpy())

    state_arr = np.vstack(states)
    cols = [f"state_{i}" for i in range(state_arr.shape[1])]
    df   = pd.DataFrame(state_arr, index=pd.DatetimeIndex(common), columns=cols)
    log.info(f"Built {len(df):,} state vectors for {symbol} ({STATE_DIM}-d)")
    return df


def _safe(symbol: str) -> str:
    return symbol.replace("=", "_").replace("-", "_").replace("/", "_")