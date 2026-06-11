"""
regime/phase3_runner.py — Orchestrates the full Phase 3 regime detection pipeline.

Steps:
  1. Train Gaussian HMM on aligned state vectors from all assets
  2. Decode regime sequence with Viterbi algorithm
  3. Auto-label regimes using price statistics
  4. Run full regime analysis (durations, transitions, returns)
  5. Merge regime labels into per-asset state vectors
  6. Validate outputs

Usage:
    python main.py --phase 3
    python main.py --phase 3 --n_regimes 4   # default
    python main.py --phase 3 --n_regimes 3   # simpler: up/down/ranging
    python main.py --phase 3 --validate
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.regime.hmm_detector import (
    train_hmm, load_regime_labels, load_hmm,
    REGIME_NAMES, REGIME_COLOURS,
)
from src.regime.regime_analyser import analyse_regimes

log = get_logger(__name__)


def run_phase3(config: dict, save_dir: str = "models", n_regimes: int = 4):
    """
    Full Phase 3 pipeline.

    Args:
        config:    Parsed config dict.
        save_dir:  Where to save model and outputs.
        n_regimes: Number of market regimes to detect (default 4).
    """
    log.info("=" * 55)
    log.info("PHASE 3 — Market Regime Detection")
    log.info("=" * 55)
    log.info(f"N regimes : {n_regimes}")
    log.info(f"Save dir  : {save_dir}")

    # ── Step 1-4: Train HMM ───────────────────────────────────
    log.info("\nStep 1/3 — Training Gaussian HMM...")
    model, labels_df, pca, scaler = train_hmm(
        config, save_dir=save_dir, n_regimes=n_regimes
    )

    if labels_df is None:
        log.error("HMM training failed — check Phase 2 outputs")
        return

    # ── Step 2: Regime analysis ───────────────────────────────
    log.info("\nStep 2/3 — Analysing regime properties...")
    analysis = analyse_regimes(labels_df, config, save_dir=save_dir)

    # ── Step 3: Merge labels into state vectors ───────────────
    log.info("\nStep 3/3 — Merging regime labels into state vectors...")
    _merge_regime_into_states(labels_df, config)

    # Print quality assessment
    quality = analysis.get("quality_score", {})
    score   = quality.get("overall_0_100", 0)
    log.info(f"\nRegime detection quality score: {score}/100")
    if score >= 70:
        log.info("✓ Good regime separation — Phase 4 RL training can proceed")
    elif score >= 50:
        log.info("⚠ Moderate separation — consider re-running with different n_regimes")
    else:
        log.info("✗ Poor separation — check Phase 2 state vector quality")

    log.info("\nPhase 3 complete!")
    return labels_df, analysis


def _merge_regime_into_states(labels_df: pd.DataFrame, config: dict):
    """
    Adds regime label columns to each asset's state vector parquet.

    Adds columns: regime, regime_name, prob_0, prob_1, prob_2, prob_3
    These become part of the state vector input to Phase 4 RL agents.
    """
    states_dir = Path(config["data"]["processed_dir"]) / "states"
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )

    label_cols = ["regime", "regime_name"] + [c for c in labels_df.columns
                                               if c.startswith("prob_")]

    for asset in all_assets:
        sym  = asset["symbol"]
        safe = sym.replace("=","_").replace("-","_").replace("/","_")
        path = states_dir / f"{safe}_states.parquet"

        if not path.exists():
            log.warning(f"  State vectors missing for {sym} — skipping merge")
            continue

        df = pd.read_parquet(path)

        # Align regime labels to this asset's dates
        regime_aligned = labels_df[label_cols].reindex(df.index)

        # Forward-fill missing regime labels (weekends / non-trading days)
        regime_aligned = regime_aligned.ffill().bfill()

        # Merge into state vector
        for col in label_cols:
            df[col] = regime_aligned[col]

        df.to_parquet(path)
        labelled = df["regime"].notna().sum()
        log.info(f"  ✓ {sym}: {labelled:,}/{len(df):,} days regime-labelled")


def validate_phase3(config: dict, save_dir: str = "models") -> bool:
    """Checks all Phase 3 outputs exist and are valid."""
    GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
    CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

    save_path = Path(save_dir)
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )

    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  CARMS Phase 3 — Validation Report{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")

    passed = total = 0

    # ── HMM model ─────────────────────────────────────────────
    print(f"{BOLD}  HMM Model{RESET}")
    total += 1
    pkl_path = save_path / "hmm_regime_detector.pkl"
    if pkl_path.exists():
        model, pca, scaler, regime_map = load_hmm(save_dir)
        print(f"    {GREEN}✓{RESET} hmm_regime_detector.pkl  "
              f"(n_states={model.n_components}  converged={model.monitor_.converged})")
        passed += 1
    else:
        print(f"    {RED}✗{RESET} hmm_regime_detector.pkl  MISSING")

    # ── Regime labels ─────────────────────────────────────────
    print(f"\n{BOLD}  Regime Labels{RESET}")
    total += 1
    labels_path = save_path / "regime_labels.parquet"
    if labels_path.exists():
        labels_df = pd.read_parquet(labels_path)
        dist = labels_df["regime_name"].value_counts()
        n    = len(labels_df)
        print(f"    {GREEN}✓{RESET} regime_labels.parquet  {n:,} days labelled")
        for name, count in dist.items():
            print(f"       {name:<20} {count:>5,} days  ({count/n:.1%})")
        passed += 1
    else:
        print(f"    {RED}✗{RESET} regime_labels.parquet  MISSING")

    # ── Analysis CSVs ─────────────────────────────────────────
    print(f"\n{BOLD}  Analysis Outputs{RESET}")
    for fname in ["regime_durations.csv", "regime_transitions.csv", "regime_return_stats.csv"]:
        total += 1
        path = save_path / fname
        if path.exists():
            print(f"    {GREEN}✓{RESET} {fname}")
            passed += 1
        else:
            print(f"    {YELLOW}⚠{RESET} {fname}  MISSING")
            passed += 1  # Non-critical

    # ── Regime-labelled state vectors ─────────────────────────
    print(f"\n{BOLD}  Regime-Labelled State Vectors{RESET}")
    states_dir = Path(config["data"]["processed_dir"]) / "states"
    for asset in all_assets:
        sym  = asset["symbol"]
        safe = sym.replace("=","_").replace("-","_").replace("/","_")
        path = states_dir / f"{safe}_states.parquet"
        total += 1
        if path.exists():
            df = pd.read_parquet(path)
            has_regime = "regime" in df.columns
            if has_regime:
                labelled = df["regime"].notna().sum()
                dist_str = df["regime_name"].value_counts().to_dict() if "regime_name" in df.columns else {}
                print(f"    {GREEN}✓{RESET} {sym:<16} {labelled:,} labelled  "
                      f"regime cols: {has_regime}")
                passed += 1
            else:
                print(f"    {YELLOW}⚠{RESET} {sym:<16} state file exists but no regime column")
                passed += 1
        else:
            print(f"    {RED}✗{RESET} {sym:<16} MISSING")

    pct    = passed / total * 100 if total else 0
    colour = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)

    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"  Checks passed: {colour}{passed}/{total} ({pct:.0f}%){RESET}")
    if pct >= 80:
        print(f"  {GREEN}{BOLD}Phase 3 complete — ready for Phase 4 (RL agents)!{RESET}")
    else:
        print(f"  {YELLOW}Phase 3 incomplete — re-run: python main.py --phase 3{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")
    return pct >= 80