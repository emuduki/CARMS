"""
agents/phase4_runner.py — Orchestrates full Phase 4 RL training pipeline.

Steps:
  1. Verify Phase 3 outputs exist (state vectors + regime labels)
  2. Train Forex agent (PPO)
  3. Train Crypto agent (SAC)
  4. Train Gold agent (PPO)
  5. Run backtest on held-out test period
  6. Compare against baselines
  7. Validate outputs

Usage:
    python main.py --phase 4
    python main.py --phase 4 --device cuda    # GPU training
    python main.py --phase 4 --agent crypto   # Train single agent
    python main.py --phase 4 --validate       # Validate only
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.agents.trainer import train_all_agents, evaluate_agent, AGENT_CONFIGS, AGENT_ASSET_MAP
from src.agents.backtester import run_backtest, compare_to_baselines

log = get_logger(__name__)


def run_phase4(
    config:    dict,
    save_dir:  str = "models",
    device:    str = "cpu",
    agent:     Optional[str] = None,
) -> dict:
    """
    Full Phase 4 pipeline.

    Args:
        config:   Parsed config dict.
        save_dir: Model save directory.
        device:   'cpu' or 'cuda'.
        agent:    If set, only trains this agent ('forex', 'crypto', 'gold').
    """
    log.info("=" * 55)
    log.info("PHASE 4 — Specialist RL Agent Training")
    log.info("=" * 55)

    # ── Verify Phase 3 prerequisites ──────────────────────────
    states_dir = Path(config["data"]["processed_dir"]) / "states"
    labels_path = Path(save_dir) / "regime_labels.parquet"

    if not labels_path.exists():
        log.error("Regime labels not found — run Phase 3 first:")
        log.error("  python main.py --phase 3")
        return {}

    state_files = list(states_dir.glob("*_states.parquet"))
    if not state_files:
        log.error("No state vectors found — run Phase 2 & 3 first")
        return {}

    log.info(f"  Phase 3 outputs verified ({len(state_files)} state files)")

    # Check regime columns are present
    sample = pd.read_parquet(state_files[0])
    if "regime" not in sample.columns:
        log.error("State vectors missing regime labels — re-run Phase 3:")
        log.error("  python main.py --phase 3")
        return {}

    log.info(f"  Regime columns present: {[c for c in sample.columns if 'regime' in c]}")

    # ── Train agents ──────────────────────────────────────────
    if agent:
        # Single agent mode
        from src.agents.trainer import train_ppo_agent, train_sac_agent, AGENT_ASSET_MAP
        cfg = AGENT_CONFIGS.get(agent)
        if not cfg:
            log.error(f"Unknown agent '{agent}'. Choose: forex, crypto, gold")
            return {}

        if cfg["type"] == "ppo":
            results = {agent: train_ppo_agent(
                config, agent, cfg["symbols"], save_dir, device
            )}
        else:
            results = {agent: train_sac_agent(
                config, agent, cfg["symbols"], save_dir, device
            )}
    else:
        results = train_all_agents(config, save_dir, device)

    # ── Backtest ──────────────────────────────────────────────
    log.info("\nRunning backtest on test period...")
    backtest_df = run_backtest(config, save_dir, device)

    # ── Baseline comparison ───────────────────────────────────
    log.info("\nComputing baseline comparisons...")
    baseline_df = compare_to_baselines(config, save_dir)

    return {
        "training": results,
        "backtest": backtest_df,
        "baselines": baseline_df,
    }


def validate_phase4(config: dict, save_dir: str = "models") -> bool:
    """Checks all Phase 4 outputs exist and are valid."""
    GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
    CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

    save_path = Path(save_dir)

    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  CARMS Phase 4 — Validation Report{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")

    passed = total = 0

    # ── Agent checkpoints ─────────────────────────────────────
    print(f"{BOLD}  Agent Checkpoints{RESET}")
    agent_files = {
        "Forex  (PPO)":  save_path / "ppo_forex.pt",
        "Crypto (SAC)":  save_path / "sac_crypto.pt",
        "Gold   (PPO)":  save_path / "ppo_gold.pt",
    }
    for name, path in agent_files.items():
        total += 1
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"    {GREEN}✓{RESET} {name:<16} {path.name}  ({size_mb:.1f} MB)")
            passed += 1
        else:
            print(f"    {RED}✗{RESET} {name:<16} MISSING")

    # ── Training metrics ──────────────────────────────────────
    print(f"\n{BOLD}  Training Metrics{RESET}")
    for agent_name in ["forex", "crypto", "gold"]:
        agent_type = AGENT_CONFIGS.get(agent_name, {}).get("type", "ppo")
        metrics_path = save_path / f"{agent_type}_{agent_name}_metrics.csv"
        total += 1
        if metrics_path.exists():
            df    = pd.read_csv(metrics_path)
            last  = df.tail(50)
            sharpe = last["sharpe"].mean()
            ret    = last["total_return"].mean()
            col    = GREEN if sharpe > 0 else YELLOW
            print(f"    {GREEN}✓{RESET} {agent_name:<8} metrics  "
                  f"last50ep: sharpe={col}{sharpe:.2f}{RESET}  ret={ret:+.1%}")
            passed += 1
        else:
            print(f"    {RED}✗{RESET} {agent_name:<8} metrics  MISSING")

    # ── Backtest results ──────────────────────────────────────
    print(f"\n{BOLD}  Backtest Results{RESET}")
    total += 1
    bt_path = save_path / "backtest_results.csv"
    if bt_path.exists():
        df = pd.read_csv(bt_path)
        print(f"    {GREEN}✓{RESET} backtest_results.csv  ({len(df)} agent-symbol pairs)")
        for _, row in df.iterrows():
            col = GREEN if row.get("sharpe", 0) > 0 else YELLOW
            print(f"       {row['agent']:<10} {row['symbol']:<14} "
                  f"return={row.get('total_return_%',0):+.1f}%  "
                  f"sharpe={col}{row.get('sharpe',0):.2f}{RESET}")
        passed += 1
    else:
        print(f"    {YELLOW}⚠{RESET} backtest_results.csv  MISSING (run full phase 4)")
        passed += 1

    # ── Paper trading readiness ───────────────────────────────
    print(f"\n{BOLD}  Paper Trading Readiness{RESET}")
    all_agents_ready = all(p.exists() for p in agent_files.values())
    total += 1
    if all_agents_ready:
        print(f"    {GREEN}✓{RESET} All agents trained — ready for paper trading!")
        print(f"    Connect to Binance Testnet (crypto) or OANDA demo (forex/gold)")
        print(f"    Run: python main.py --phase 5")
        passed += 1
    else:
        missing = [n for n, p in agent_files.items() if not p.exists()]
        print(f"    {YELLOW}⚠{RESET} Missing: {missing}")

    pct    = passed / total * 100 if total else 0
    colour = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"  Checks passed: {colour}{passed}/{total} ({pct:.0f}%){RESET}")
    if pct >= 80:
        print(f"  {GREEN}{BOLD}Phase 4 complete — ready for Phase 5 (meta-controller)!{RESET}")
    else:
        print(f"  {YELLOW}Phase 4 incomplete — re-run: python main.py --phase 4{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")
    return pct >= 80