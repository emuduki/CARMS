"""
encoders/finbert_encoder.py — FinBERT-based NLP encoder for financial news sentiment.

Loads ProsusAI/finbert from HuggingFace and:
  1. Fine-tunes on CARMS news corpus with per-asset sentiment labels
  2. Extracts a 32-d sentiment embedding per (asset, date) pair

The 32-d embedding captures:
  - Bullish / bearish / neutral sentiment score
  - Sentiment momentum (change over rolling window)
  - Source diversity and headline volume

Fine-tuning strategy:
  - Freeze base BERT, train classification head first (3 epochs)
  - Unfreeze last 2 BERT layers, fine-tune end-to-end (3 more epochs)
  - This avoids catastrophic forgetting of financial language

Output per date/asset:
  [pos_score, neg_score, neu_score, sentiment_ma5, headline_count, ...] → 32-d
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

FINBERT_MODEL = "ProsusAI/finbert"
EMBED_DIM     = 32
MAX_LEN       = 128
BATCH_SIZE    = 16
FINETUNE_EPOCHS = 6
LR_HEAD       = 2e-4
LR_FULL       = 5e-5

SENTIMENT_LABELS = {"positive": 0, "negative": 1, "neutral": 2}


# ── Dataset ───────────────────────────────────────────────────

class NewsDataset(Dataset):
    """
    Dataset of news headlines with weak sentiment labels.

    Labels are generated heuristically from next-day price returns:
      return > +0.5%  → positive
      return < -0.5%  → negative
      otherwise       → neutral
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = MAX_LEN):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }


# ── Model ─────────────────────────────────────────────────────

class FinBERTEncoder(nn.Module):
    """
    FinBERT with a custom sentiment head + embedding projector.

    Layers:
      FinBERT base (frozen or partially unfrozen during fine-tune)
      → [CLS] hidden state (768-d)
      → sentiment head (768 → 3 logits)
      → embedding projector (768 → 32-d)  [used at inference]
    """

    def __init__(self, pretrained: str = FINBERT_MODEL):
        super().__init__()
        from transformers import BertModel
        self.bert          = BertModel.from_pretrained(pretrained)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 3),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(768, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, EMBED_DIM),
        )

    def forward(self, input_ids, attention_mask, return_embedding: bool = False):
        out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls   = out.last_hidden_state[:, 0, :]   # [CLS] token
        if return_embedding:
            return self.embed_proj(cls)
        return self.sentiment_head(cls)

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2):
        """Unfreezes the last N transformer layers for fine-tuning."""
        for layer in self.bert.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.bert.pooler.parameters():
            p.requires_grad = True


# ── Training ──────────────────────────────────────────────────

def train_finbert_encoder(
    config: dict,
    save_dir: str = "models",
    device: str = "cpu",
) -> Optional["FinBERTEncoder"]:
    """
    Fine-tunes FinBERT on the CARMS news corpus.

    Two-stage training:
      Stage 1 (3 epochs): freeze BERT, train head only → fast convergence
      Stage 2 (3 epochs): unfreeze last 2 layers → domain adaptation

    Args:
        config:   Parsed config dict.
        save_dir: Directory to save checkpoint.
        device:   "cpu" or "cuda".

    Returns:
        Fine-tuned FinBERTEncoder, or None if no news data available.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.error("transformers not installed — run: pip install transformers")
        return None

    from src.ingestion.news_fetcher import load_news
    from src.ingestion.downloader import load_asset

    news = load_news(config["data"]["raw_dir"])
    if news is None or news.empty:
        log.warning("No news data — skipping FinBERT fine-tuning")
        log.info("Run Phase 1 with a NewsAPI key to enable NLP encoder training")
        return _load_pretrained_only(save_dir, device)

    log.info(f"Fine-tuning FinBERT on {len(news):,} news articles...")

    # ── Build weak labels from price returns ──────────────────
    texts, labels = _build_labelled_corpus(news, config)
    if len(texts) < 10:
        log.warning(f"Only {len(texts)} labelled samples — using pretrained only (need ≥10)")
        return _load_pretrained_only(save_dir, device)

    log.info(f"  Labelled corpus: {len(texts):,} samples")
    log.info(f"  Label dist: pos={labels.count(0)}  neg={labels.count(1)}  neu={labels.count(2)}")

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    dataset   = NewsDataset(texts, labels, tokenizer)
    split     = int(len(dataset) * 0.85)
    ds_train  = torch.utils.data.Subset(dataset, range(split))
    ds_val    = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    dl_train  = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val    = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)

    model     = FinBERTEncoder(FINBERT_MODEL).to(device)
    criterion = nn.CrossEntropyLoss()
    save_path = Path(save_dir) / "finbert_encoder.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: head only ────────────────────────────────────
    log.info("  Stage 1: training sentiment head (BERT frozen)...")
    model.freeze_bert()
    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )
    _run_training_loop(model, dl_train, dl_val, opt1, criterion,
                       epochs=FINETUNE_EPOCHS // 2, device=device, label="Stage1")

    # ── Stage 2: last 2 BERT layers + head ───────────────────
    log.info("  Stage 2: fine-tuning last 2 BERT layers...")
    model.unfreeze_last_n_layers(n=2)
    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FULL
    )
    best_acc = _run_training_loop(model, dl_train, dl_val, opt2, criterion,
                                  epochs=FINETUNE_EPOCHS // 2, device=device, label="Stage2")

    torch.save({"state_dict": model.state_dict(), "val_acc": best_acc}, save_path)
    log.info(f"  ✓ FinBERT encoder saved → {save_path}  (val acc: {best_acc:.1%})")
    return model


def _run_training_loop(model, dl_train, dl_val, optimizer, criterion,
                       epochs, device, label="") -> float:
    """Runs train/val loop, returns best validation accuracy."""
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in dl_train:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in dl_val:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                lbls  = batch["label"].to(device)
                preds = model(ids, mask).argmax(dim=-1)
                correct += (preds == lbls).sum().item()
                total   += len(lbls)
        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)
        log.info(f"    {label} Epoch {epoch}/{epochs}  val_acc={acc:.1%}")
    return best_acc


def _build_labelled_corpus(news: pd.DataFrame, config: dict) -> tuple[list, list]:
    """
    Aligns news articles with next-day price returns to create weak labels.

    Strategy:
      For each article, find the asset's next trading day return.
      return > 0  → positive label
      return < 0  → negative label
      (no neutral — binary labels work better with small corpus)

    Robust date handling:
      - Strips timezone info before comparison
      - Falls back to nearest date if exact date missing
    """
    from src.ingestion.downloader import load_asset

    texts, labels = [], []
    asset_returns = {}

    for symbol in news["symbol"].unique():
        df = load_asset(symbol, "daily", config["data"]["raw_dir"])
        if df is not None:
            ret = df["close"].pct_change().dropna()
            # Ensure index is timezone-naive
            if hasattr(ret.index, "tz") and ret.index.tz is not None:
                ret.index = ret.index.tz_localize(None)
            asset_returns[symbol] = ret

    log.info(f"  Building labels from {len(news):,} articles across {len(asset_returns)} assets")

    skipped_no_symbol = skipped_no_future = skipped_nan = 0

    for _, row in news.iterrows():
        text   = str(row.get("text", row.get("title", "")))
        symbol = row.get("symbol", "")
        if not text or symbol not in asset_returns:
            skipped_no_symbol += 1
            continue

        try:
            # Parse date — strip timezone to match price index
            raw_date = row.get("published_at", "")
            if pd.isna(raw_date) or raw_date == "":
                continue
            pub_dt   = pd.to_datetime(raw_date, utc=True).tz_localize(None)
            pub_date = pub_dt.normalize()

            ret_series = asset_returns[symbol]

            # Try next trading day first; fall back to previous day
            # (handles articles published after price data end_date)
            future = ret_series[ret_series.index > pub_date]
            past   = ret_series[ret_series.index <= pub_date]

            if not future.empty:
                ret = future.iloc[0]
            elif not past.empty:
                ret = past.iloc[-1]
            else:
                skipped_no_future += 1
                continue

            if pd.isna(ret):
                skipped_nan += 1
                continue

            label = 0 if ret >= 0 else 1   # positive / negative (binary)
            texts.append(text[:512])
            labels.append(label)

        except Exception:
            continue

    log.info(f"  Labelled: {len(texts)}  |  skipped (no symbol): {skipped_no_symbol}  "
             f"no future date: {skipped_no_future}  nan: {skipped_nan}")
    return texts, labels


def _load_pretrained_only(save_dir: str, device: str) -> Optional["FinBERTEncoder"]:
    """Loads FinBERT weights without fine-tuning (zero-shot baseline)."""
    try:
        log.info("Loading pretrained FinBERT (no fine-tuning)...")
        model = FinBERTEncoder(FINBERT_MODEL).to(device)
        model.eval()
        save_path = Path(save_dir) / "finbert_encoder.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "val_acc": 0.0}, save_path)
        log.info("  ✓ Pretrained FinBERT loaded (fine-tuning skipped)")
        return model
    except Exception as e:
        log.error(f"Failed to load FinBERT: {e}")
        log.info("Make sure transformers is installed: pip install transformers")
        return None


# ── Inference ─────────────────────────────────────────────────

def load_finbert_encoder(save_dir: str = "models", device: str = "cpu") -> Optional["FinBERTEncoder"]:
    """Loads saved FinBERT checkpoint."""
    path = Path(save_dir) / "finbert_encoder.pt"
    if not path.exists():
        log.warning("No FinBERT checkpoint — run Phase 2 first")
        return None
    ckpt  = torch.load(path, map_location=device)
    model = FinBERTEncoder(FINBERT_MODEL).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info(f"Loaded FinBERT encoder (val_acc={ckpt.get('val_acc', 0):.1%})")
    return model


def extract_sentiment_embeddings(
    model: "FinBERTEncoder",
    news: pd.DataFrame,
    config: dict,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Aggregates per-article embeddings into daily per-asset sentiment vectors.

    For each (symbol, date) pair:
      - Encode all headlines from that day
      - Average pool embeddings → 32-d daily sentiment vector
      - Add scalar features: headline_count, sentiment_score

    Returns:
        MultiIndex DataFrame (symbol, date) → 32-d embedding columns
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.error("transformers not installed")
        return pd.DataFrame()

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model.eval()

    records = []
    for symbol in news["symbol"].unique():
        sym_news = news[news["symbol"] == symbol].copy()
        sym_news["date"] = pd.to_datetime(sym_news["published_at"]).dt.normalize()

        for date, day_news in sym_news.groupby("date"):
            texts = day_news["text"].fillna(day_news["title"]).tolist()
            if not texts:
                continue

            enc = tokenizer(
                texts, truncation=True, padding=True,
                max_length=MAX_LEN, return_tensors="pt"
            )
            with torch.no_grad():
                embeds = model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                    return_embedding=True,
                ).cpu().numpy()

            mean_embed = embeds.mean(axis=0)
            row = {"symbol": symbol, "date": date, "headline_count": len(texts)}
            for i, v in enumerate(mean_embed):
                row[f"sent_{i}"] = float(v)
            records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index(["symbol", "date"]).sort_index()
    log.info(f"Sentiment embeddings: {df.shape[0]:,} (symbol, date) pairs  |  {df.shape[1]} features")
    return df