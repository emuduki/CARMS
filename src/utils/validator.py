"""
utils/validator.py — Validates and summarises Phase 1 pipeline outputs.
 
Run this after Phase 1 to confirm everything is ready for Phase 2.
Prints a rich summary table to the console.
"""
 
from pathlib import Path
 
import pandas as pd
import numpy as np
 
from src.utils.logger import get_logger
 
log = get_logger(__name__)
 
# Terminal colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
 
 
def validate_phase1(config: dict) -> bool:
    """
    Checks all Phase 1 outputs exist and are non-empty.
 
    Args:
        config: Parsed config dict.
 
    Returns:
        True if all checks pass, False otherwise.
    """
    raw_dir       = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    charts_dir    = Path(config["data"]["charts_dir"])
 
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )
 
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  CARMS Phase 1 — Validation Report{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}\n")
 
    checks_passed = 0
    checks_total  = 0
 
    # ── 1. Raw OHLCV data ─────────────────────────────────────
    print(f"{BOLD}  Raw OHLCV Data{RESET}")
    for asset in all_assets:
        symbol = asset["symbol"]
        safe   = symbol.replace("=", "_").replace("-", "_").replace("/", "_")
 
        for interval in ["daily", "hourly"]:
            path = raw_dir / f"{safe}_{interval}.parquet"
            checks_total += 1
            if path.exists():
                df   = pd.read_parquet(path)
                rows = len(df)
                ok   = rows > 100
                icon = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}⚠{RESET}"
                print(f"    {icon} {symbol:<16} {interval:<8} {rows:>6,} rows")
                if ok:
                    checks_passed += 1
            else:
                print(f"    {RED}✗{RESET} {symbol:<16} {interval:<8} {'MISSING':>10}")
 
    # ── 2. Processed features ─────────────────────────────────
    print(f"\n{BOLD}  Processed Features{RESET}")
    for asset in all_assets:
        symbol = asset["symbol"]
        safe   = symbol.replace("=", "_").replace("-", "_").replace("/", "_")
        path   = processed_dir / f"{safe}_features.parquet"
        checks_total += 1
 
        if path.exists():
            df   = pd.read_parquet(path)
            cols = len(df.columns)
            rows = len(df)
            ok   = rows > 50 and cols > 10
            icon = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}⚠{RESET}"
            print(f"    {icon} {symbol:<16} {rows:>6,} rows × {cols} features")
            if ok:
                checks_passed += 1
        else:
            print(f"    {RED}✗{RESET} {symbol:<16} {'MISSING':>10}")
 
    # ── 3. Chart images ───────────────────────────────────────
    print(f"\n{BOLD}  Chart Images (CNN Input){RESET}")
    for asset in all_assets:
        symbol = asset["symbol"]
        safe   = symbol.replace("=", "_").replace("-", "_").replace("/", "_")
        img_dir = charts_dir / safe
        meta_path = charts_dir / f"{safe}_metadata.csv"
        checks_total += 1
 
        if img_dir.exists() and meta_path.exists():
            n_imgs = len(list(img_dir.glob("*.png")))
            try:
                meta    = pd.read_csv(meta_path)
                balance = meta["label_1d"].mean() if "label_1d" in meta.columns and not meta.empty else 0.0
                n_meta  = len(meta)
            except Exception:
                balance = 0.0
                n_meta  = 0
            ok   = n_meta > 10
            icon = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}⚠{RESET}"
            print(f"    {icon} {symbol:<16} {n_imgs:>6,} imgs  {n_meta:>6,} metadata  up-label: {balance:.1%}")
            if ok:
                checks_passed += 1
        else:
            print(f"    {YELLOW}⚠{RESET} {symbol:<16} {'Not generated yet':>20}  {GRAY}(optional){RESET}")
            checks_passed += 1   # Charts are optional for Phase 1
 
    # ── 4. News & macro ───────────────────────────────────────
    print(f"\n{BOLD}  News & Macro Data{RESET}")
    news_path  = raw_dir / "news" / "headlines.csv"
    macro_path = raw_dir / "macro_fred.parquet"
 
    for label, path in [("News headlines", news_path), ("FRED macro", macro_path)]:
        checks_total += 1
        if path.exists():
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            ok = len(df) > 0
            icon = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}⚠{RESET}"
            print(f"    {icon} {label:<20} {len(df):>6,} records")
            if ok:
                checks_passed += 1
        else:
            print(f"    {YELLOW}⚠{RESET} {label:<20} {'Not fetched':>14}  {GRAY}(needs API key){RESET}")
            checks_passed += 1   # Optional if no API key yet
 
    # ── Summary ───────────────────────────────────────────────
    pct = checks_passed / checks_total * 100 if checks_total else 0
    colour = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
 
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"  Checks passed: {colour}{checks_passed}/{checks_total} ({pct:.0f}%){RESET}")
 
    if pct >= 80:
        print(f"  {GREEN}{BOLD}Phase 1 complete — ready for Phase 2 (encoders)!{RESET}")
    elif pct >= 50:
        print(f"  {YELLOW}Phase 1 partially complete — add API keys to fetch news/macro.{RESET}")
    else:
        print(f"  {RED}Phase 1 incomplete — re-run: python main.py --phase 1{RESET}")
 
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}\n")
 
    return pct >= 80
 
 
def print_feature_summary(symbol: str, processed_dir: str = "data/processed"):
    """Prints a detailed summary of features for one asset."""
    safe = symbol.replace("=", "_").replace("-", "_").replace("/", "_")
    path = Path(processed_dir) / f"{safe}_features.parquet"
 
    if not path.exists():
        log.warning(f"No feature file for {symbol}")
        return
 
    df = pd.read_parquet(path)
    raw_cols = {"open", "high", "low", "close", "volume", "symbol"}
    feat_cols = [c for c in df.columns if c not in raw_cols]
 
    print(f"\n{BOLD}{symbol} Feature Summary{RESET}")
    print(f"  Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Rows       : {len(df):,}")
    print(f"  Features   : {len(feat_cols)}")
    print(f"\n  {'Feature':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for col in feat_cols[:15]:   # Show first 15
        s = df[col].dropna()
        print(f"  {col:<22} {s.mean():>10.4f} {s.std():>10.4f} {s.min():>10.4f} {s.max():>10.4f}")
    if len(feat_cols) > 15:
        print(f"  {GRAY}... and {len(feat_cols) - 15} more features{RESET}")