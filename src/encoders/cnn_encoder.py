"""
encoders/cnn_encoder.py — ResNet-18 CNN encoder for candlestick chart images.

Reads the 64×64 grayscale PNG chart images from Phase 1 and trains a
ResNet-18 to:
  1. Classify next-day direction (up/down) — supervised pre-training
  2. Extract a 64-dimensional visual embedding per image

The 64-d embedding captures spatial visual patterns:
  - Candlestick shape and body size distribution
  - Wick length ratios (indecision vs conviction)
  - Trend structure in the 20-candle window
  - Volume profile shape (where present)

Training:
  - Uses transfer learning from torchvision's pretrained ResNet-18
  - Adapts first conv layer for single-channel (grayscale) input
  - Trains classification head first, then fine-tunes full network
  - Binary cross-entropy on next-day direction (up=1, down=0)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from src.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_SIZE   = 64
EMBED_DIM    = 64
BATCH_SIZE   = 64
MAX_EPOCHS   = 30
LR           = 1e-3
PATIENCE     = 6


# ── Dataset ───────────────────────────────────────────────────

class ChartImageDataset(Dataset):
    """
    Dataset of candlestick chart PNG images with binary direction labels.

    Images are loaded lazily from disk on __getitem__ to avoid
    loading 8,000+ images into RAM at once.
    """

    def __init__(self, metadata: pd.DataFrame, image_size: int = IMAGE_SIZE):
        # Filter to rows where the image file actually exists
        self.meta       = metadata[metadata["image_path"].apply(
            lambda p: Path(p).exists()
        )].reset_index(drop=True)
        self.image_size = image_size

        if len(self.meta) < len(metadata):
            missing = len(metadata) - len(self.meta)
            log.warning(f"  {missing} image files missing — skipped")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row   = self.meta.iloc[idx]
        img   = self._load_image(row["image_path"])
        label = float(row["label_1d"])
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def _load_image(self, path: str) -> np.ndarray:
        """Loads a grayscale PNG and returns a (1, H, W) float32 array."""
        try:
            from PIL import Image
            img = Image.open(path).convert("L")
            img = img.resize((self.image_size, self.image_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr[np.newaxis, :, :]   # (1, H, W)
        except Exception:
            return np.zeros((1, self.image_size, self.image_size), dtype=np.float32)


# ── Model ─────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    ResNet-18 adapted for grayscale candlestick chart classification.

    Modifications from standard ResNet-18:
      - First conv: 3-channel → 1-channel (grayscale input)
      - Final FC:   512 → 64 embedding + binary classification head

    Architecture:
      Input (1, 64, 64) → ResNet-18 backbone → avg pool (512-d)
      → embed_proj (512 → 64)
      → pred_head  (64  → 1, sigmoid for binary classification)
    """

    def __init__(self, embed_dim: int = EMBED_DIM, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt first conv layer for grayscale (1 channel instead of 3)
        # Average the 3-channel pretrained weights into 1 channel
        old_conv   = backbone.conv1
        new_conv   = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Remove the original classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to avg pool

        self.embed_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pred_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x               : (batch, 1, H, W) grayscale image tensor
            return_embedding: True → return 64-d embed, False → return logit

        Returns:
            embedding (batch, 64) or logit (batch, 1)
        """
        features = self.backbone(x)
        embed    = self.embed_proj(features)
        if return_embedding:
            return embed
        return self.pred_head(embed)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ── Training ──────────────────────────────────────────────────

def train_cnn_encoder(
    config: dict,
    symbol: str = "BTC-USD",
    save_dir: str = "models",
    device: str = "cpu",
) -> Optional[CNNEncoder]:
    """
    Trains the CNN encoder on candlestick chart images for one asset.

    Two-stage training:
      Stage 1 (10 epochs): freeze ResNet backbone, train embed+head only
      Stage 2 (20 epochs): unfreeze full network, fine-tune end-to-end

    Args:
        config:   Parsed config dict.
        symbol:   Asset to train on.
        save_dir: Where to save checkpoint.
        device:   "cpu" or "cuda".

    Returns:
        Trained CNNEncoder, or None if no image data available.
    """
    from src.features.chart_generator import load_chart_metadata

    meta = load_chart_metadata(symbol, config["data"]["charts_dir"])
    if meta is None or meta.empty:
        log.warning(f"No chart metadata for {symbol} — skipping CNN training")
        return None

    # Train / val split
    split    = int(len(meta) * 0.8)
    ds_train = ChartImageDataset(meta.iloc[:split])
    ds_val   = ChartImageDataset(meta.iloc[split:])

    if len(ds_train) == 0:
        log.warning(f"No usable chart images for {symbol}")
        return None

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    log.info(f"Training CNN encoder on {symbol}...")
    log.info(f"  Train: {len(ds_train):,}  |  Val: {len(ds_val):,}  |  Device: {device}")

    model     = CNNEncoder(embed_dim=EMBED_DIM, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    save_path = Path(save_dir) / f"cnn_encoder_{_safe(symbol)}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc    = 0.0
    patience_ctr = 0

    # ── Stage 1: backbone frozen ──────────────────────────────
    log.info("  Stage 1: training head (backbone frozen)...")
    model.freeze_backbone()
    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    for epoch in range(1, 11):
        _train_epoch(model, dl_train, opt1, criterion, device)
        acc = _eval_epoch(model, dl_val, device)
        if epoch % 2 == 0:
            log.info(f"    Epoch {epoch}/10  val_acc={acc:.1%}")
        if acc > best_acc:
            best_acc = acc

    # ── Stage 2: full fine-tune ───────────────────────────────
    log.info("  Stage 2: fine-tuning full network...")
    model.unfreeze_backbone()
    opt2 = torch.optim.Adam(model.parameters(), lr=LR / 5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=MAX_EPOCHS)

    for epoch in range(1, MAX_EPOCHS + 1):
        _train_epoch(model, dl_train, opt2, criterion, device)
        acc = _eval_epoch(model, dl_val, device)
        scheduler.step()

        if epoch % 5 == 0:
            log.info(f"    Epoch {epoch}/{MAX_EPOCHS}  val_acc={acc:.1%}")

        if acc > best_acc:
            best_acc     = acc
            patience_ctr = 0
            torch.save({"state_dict": model.state_dict(), "val_acc": acc, "symbol": symbol}, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    log.info(f"  ✓ CNN encoder saved → {save_path}  (best val acc: {best_acc:.1%})")
    return model


def train_all_cnn_encoders(config: dict, device: str = "cpu") -> dict[str, CNNEncoder]:
    """Trains one CNN encoder per asset. Returns dict of symbol → model."""
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )
    models = {}
    for asset in all_assets:
        symbol = asset["symbol"]
        try:
            model = train_cnn_encoder(config, symbol=symbol, device=device)
            if model is not None:
                models[symbol] = model
        except Exception as e:
            log.error(f"CNN training failed for {symbol}: {e}")
    return models


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(-1)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _eval_epoch(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = (torch.sigmoid(model(imgs).squeeze(-1)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += len(labels)
    return correct / max(total, 1)


# ── Inference ─────────────────────────────────────────────────

def load_cnn_encoder(symbol: str, save_dir: str = "models", device: str = "cpu") -> Optional[CNNEncoder]:
    """Loads a saved CNN encoder checkpoint."""
    path = Path(save_dir) / f"cnn_encoder_{_safe(symbol)}.pt"
    if not path.exists():
        log.warning(f"No CNN checkpoint for {symbol} — run Phase 2 first")
        return None
    ckpt  = torch.load(path, map_location=device)
    model = CNNEncoder(embed_dim=EMBED_DIM, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info(f"Loaded CNN encoder for {symbol} (val_acc={ckpt.get('val_acc', 0):.1%})")
    return model


def extract_cnn_embeddings(
    model: CNNEncoder,
    symbol: str,
    config: dict,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Runs the CNN encoder over all chart images for a symbol.

    Returns:
        DataFrame with columns [cnn_0, ..., cnn_63] indexed by date.
    """
    from src.features.chart_generator import load_chart_metadata

    meta = load_chart_metadata(symbol, config["data"]["charts_dir"])
    if meta is None or meta.empty:
        return pd.DataFrame()

    ds     = ChartImageDataset(meta)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    model.eval()

    embeds = []
    with torch.no_grad():
        for imgs, _ in loader:
            e = model(imgs.to(device), return_embedding=True)
            embeds.append(e.cpu().numpy())

    if not embeds:
        return pd.DataFrame()

    embed_arr = np.vstack(embeds)
    dates     = ds.meta["date"].values[:len(embed_arr)]

    cols = [f"cnn_{i}" for i in range(embed_arr.shape[1])]
    return pd.DataFrame(embed_arr, index=pd.DatetimeIndex(dates), columns=cols)


def _safe(symbol: str) -> str:
    return symbol.replace("=", "_").replace("-", "_").replace("/", "_")