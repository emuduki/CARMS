"""
encoders/phase2_runner.py — Orchestrates the full Phase 2 training pipeline.

Runs in order:
  1. Train TFT encoders    (one per asset)
  2. Train FinBERT encoder (shared across assets)
  3. Train CNN encoders    (one per asset)
  4. Extract all embeddings
  5. Train fusion layers   (one per asset)
  6. Build & save state vectors (Phase 3 input)
  7. Validate outputs

Usage:
    python main.py --phase 2
    python main.py --phase 2 --device cuda   # GPU training
    python main.py --phase 2 --symbol BTC-USD  # Single asset
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


def run_phase2(config: dict, device: str = "cpu", symbol: Optional[str] = None):
    """
    Full Phase 2 pipeline.

    Args:
        config: Parsed config dict.
        device: "cpu" or "cuda".
        symbol: If set, only trains encoders for this asset (useful for testing).
    """
    from src.encoders.tft_encoder     import train_tft_encoder, extract_tft_embeddings, load_tft_encoder
    from src.encoders.finbert_encoder import train_finbert_encoder, extract_sentiment_embeddings, load_finbert_encoder
    from src.encoders.cnn_encoder     import train_cnn_encoder, extract_cnn_embeddings, load_cnn_encoder
    from src.encoders.fusion          import train_fusion_layer, build_state_vectors
    from src.ingestion.news_fetcher   import load_news
    from src.features.indicators      import load_features

    save_dir   = config.get("model_dir", "models")
    states_dir = Path(config["data"]["processed_dir"]) / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )
    if symbol:
        all_assets = [a for a in all_assets if a["symbol"] == symbol]
        if not all_assets:
            log.error(f"Symbol {symbol} not found in config")
            return

    symbols = [a["symbol"] for a in all_assets]
    log.info("=" * 55)
    log.info("PHASE 2 — Modality Encoders")
    log.info("=" * 55)
    log.info(f"Assets   : {symbols}")
    log.info(f"Device   : {device}")
    log.info(f"Save dir : {save_dir}")

    # ── Step 1: TFT encoders ──────────────────────────────────
    log.info("\nStep 1/5 — Training TFT price encoders...")
    tft_models = {}
    for sym in symbols:
        try:
            model = train_tft_encoder(config, symbol=sym, save_dir=save_dir, device=device)
            tft_models[sym] = model
        except Exception as e:
            log.error(f"  TFT failed for {sym}: {e}")
            # Try loading existing checkpoint
            m = load_tft_encoder(sym, save_dir, device)
            if m: tft_models[sym] = m

    # ── Step 2: FinBERT encoder (shared) ──────────────────────
    log.info("\nStep 2/5 — Training FinBERT sentiment encoder...")
    finbert = None
    try:
        finbert = train_finbert_encoder(config, save_dir=save_dir, device=device)
    except Exception as e:
        log.error(f"  FinBERT failed: {e}")
        finbert = load_finbert_encoder(save_dir, device)

    # ── Step 3: CNN encoders ──────────────────────────────────
    log.info("\nStep 3/5 — Training CNN chart encoders...")
    cnn_models = {}
    for sym in symbols:
        try:
            model = train_cnn_encoder(config, symbol=sym, save_dir=save_dir, device=device)
            if model: cnn_models[sym] = model
        except Exception as e:
            log.error(f"  CNN failed for {sym}: {e}")
            m = load_cnn_encoder(sym, save_dir, device)
            if m: cnn_models[sym] = m

    # ── Step 4: Extract all embeddings ───────────────────────
    log.info("\nStep 4/5 — Extracting embeddings...")

    # Sentiment embeddings (shared across all assets)
    sent_embeds = pd.DataFrame()
    if finbert is not None:
        news = load_news(config["data"]["raw_dir"])
        if news is not None and not news.empty:
            sent_embeds = extract_sentiment_embeddings(finbert, news, config, device)

    all_tft_embeds  = {}
    all_cnn_embeds  = {}

    for sym in symbols:
        price_df = load_features(sym, config["data"]["processed_dir"])
        if price_df is None:
            continue

        if sym in tft_models:
            tft_e = extract_tft_embeddings(tft_models[sym], price_df, device)
            all_tft_embeds[sym] = tft_e
            log.info(f"  TFT embeddings {sym}: {tft_e.shape}")

        if sym in cnn_models:
            cnn_e = extract_cnn_embeddings(cnn_models[sym], sym, config, device)
            all_cnn_embeds[sym] = cnn_e
            log.info(f"  CNN embeddings {sym}: {cnn_e.shape}")

    # ── Step 5: Train fusion layers + build state vectors ─────
    log.info("\nStep 5/5 — Training fusion layers & building state vectors...")
    for sym in symbols:
        if sym not in all_tft_embeds or sym not in all_cnn_embeds:
            log.warning(f"  Skipping fusion for {sym} — missing embeddings")
            continue

        tft_e  = all_tft_embeds[sym]
        cnn_e  = all_cnn_embeds[sym]
        price_df = load_features(sym, config["data"]["processed_dir"])

        fusion = train_fusion_layer(
            config, sym, tft_e, cnn_e, sent_embeds,
            save_dir=save_dir, device=device
        )
        if fusion is None:
            continue

        # Build and save state vectors
        states = build_state_vectors(fusion, tft_e, cnn_e, sent_embeds, sym, device)
        if not states.empty:
            safe   = sym.replace("=","_").replace("-","_").replace("/","_")
            path   = states_dir / f"{safe}_states.parquet"
            states.to_parquet(path)
            log.info(f"  ✓ State vectors saved → {path.name}  {states.shape}")

    log.info("\nPhase 2 complete!")


def validate_phase2(config: dict) -> bool:
    """Checks all Phase 2 outputs exist and are non-empty."""
    save_dir   = Path(config.get("model_dir", "models"))
    states_dir = Path(config["data"]["processed_dir"]) / "states"
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )

    GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
    CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"
    GRAY  = "\033[90m"

    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  CARMS Phase 2 — Validation Report{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")

    passed = total = 0

    print(f"{BOLD}  TFT Encoder Checkpoints{RESET}")
    for asset in all_assets:
        sym  = asset["symbol"]
        safe = sym.replace("=","_").replace("-","_").replace("/","_")
        path = save_dir / f"tft_encoder_{safe}.pt"
        total += 1
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            log.debug(f"    {sym}: val_loss={ckpt.get('val_loss',0):.6f}")
            print(f"    {GREEN}✓{RESET} {sym:<16} val_loss={ckpt.get('val_loss',0):.6f}")
            passed += 1
        else:
            print(f"    {RED}✗{RESET} {sym:<16} MISSING")

    print(f"\n{BOLD}  FinBERT Checkpoint{RESET}")
    total += 1
    path = save_dir / "finbert_encoder.pt"
    if path.exists():
        ckpt = torch.load(path, map_location="cpu")
        print(f"    {GREEN}✓{RESET} finbert_encoder  val_acc={ckpt.get('val_acc',0):.1%}")
        passed += 1
    else:
        print(f"    {RED}✗{RESET} finbert_encoder  MISSING")

    print(f"\n{BOLD}  CNN Encoder Checkpoints{RESET}")
    for asset in all_assets:
        sym  = asset["symbol"]
        safe = sym.replace("=","_").replace("-","_").replace("/","_")
        path = save_dir / f"cnn_encoder_{safe}.pt"
        total += 1
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            print(f"    {GREEN}✓{RESET} {sym:<16} val_acc={ckpt.get('val_acc',0):.1%}")
            passed += 1
        else:
            print(f"    {YELLOW}⚠{RESET} {sym:<16} MISSING  {GRAY}(optional if no chart images){RESET}")
            passed += 1  # Not all assets need CNN

    print(f"\n{BOLD}  Fusion Layers & State Vectors{RESET}")
    for asset in all_assets:
        sym  = asset["symbol"]
        safe = sym.replace("=","_").replace("-","_").replace("/","_")
        path = states_dir / f"{safe}_states.parquet"
        total += 1
        if path.exists():
            df = pd.read_parquet(path)
            print(f"    {GREEN}✓{RESET} {sym:<16} {len(df):>6,} state vectors  ({df.shape[1]}-d)")
            passed += 1
        else:
            print(f"    {RED}✗{RESET} {sym:<16} MISSING")

    pct = passed / total * 100 if total else 0
    colour = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)

    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"  Checks passed: {colour}{passed}/{total} ({pct:.0f}%){RESET}")
    if pct >= 80:
        print(f"  {GREEN}{BOLD}Phase 2 complete — ready for Phase 3 (regime detection)!{RESET}")
    else:
        print(f"  {YELLOW}Phase 2 incomplete — re-run: python main.py --phase 2{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")
    return pct >= 80