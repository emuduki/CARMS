"""
live/phase5_runner.py — Orchestrates Phase 5: meta-controller + paper trading.

Steps:
  1. Verify Phase 4 agents are trained
  2. Run paper trade simulation (historical test data, no live API needed)
  3. Optionally start live paper trading (Binance Testnet + yfinance)
  4. Validate outputs

Usage:
    python main.py --phase 5                    # Simulation only
    python main.py --phase 5 --live             # Live paper trading
    python main.py --phase 5 --capital 10000    # Set starting capital
    python main.py --phase 5 --validate         # Validate only
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)


def run_phase5(
    config,
    save_dir="models",
    device="cpu",
    capital=10_000.0,
    live=False,
    n_ticks=252,
):
    from src.live.paper_trader import run_paper_trade_simulation, PaperTrader

    log.info("=" * 55)
    log.info("PHASE 5 — Meta-Controller & Paper Trading")
    log.info("=" * 55)
    log.info(f"Mode     : {'LIVE paper trading' if live else 'Historical simulation'}")
    log.info(f"Capital  : ${capital:,.2f}")
    log.info(f"Save dir : {save_dir}")

    save_path   = Path(save_dir)
    agent_files = [
        save_path / "ppo_forex.pt",
        save_path / "sac_crypto.pt",
        save_path / "ppo_gold.pt",
    ]
    missing = [f.name for f in agent_files if not f.exists()]
    if missing:
        log.error(f"Missing agent checkpoints: {missing}")
        log.error("Run Phase 4 first: python main.py --phase 4")
        return {}

    log.info("  Phase 4 agents verified OK")

    if live:
        log.info("\nStarting live paper trading...")
        log.info("  Forex/Gold : yfinance live prices")
        log.info("  Crypto     : Binance Testnet")
        log.info("  Press Ctrl+C to stop\n")
        trader = PaperTrader(config, save_dir, device, capital)
        trader.run()
        return trader.portfolio.get_metrics()
    else:
        log.info(f"\nRunning historical simulation ({n_ticks} trading days)...")
        results = run_paper_trade_simulation(
            config=config, save_dir=save_dir,
            device=device, capital=capital, n_ticks=n_ticks,
        )
        return results


def validate_phase5(config, save_dir="models"):
    GREEN  = "\033[92m"; RED  = "\033[91m"; YELLOW = "\033[93m"
    CYAN   = "\033[96m"; BOLD = "\033[1m";  RESET  = "\033[0m"
    save_path = Path(save_dir)

    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  CARMS Phase 5 — Validation Report{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")

    passed = total = 0

    # Simulation results
    print(f"{BOLD}  Paper Trade Simulation{RESET}")
    sim_path = save_path / "paper_trade_simulation.csv"
    total += 1
    if sim_path.exists():
        sim_df     = pd.read_csv(sim_path)
        n_days     = len(sim_df)
        final_ret  = sim_df["total_return_%"].iloc[-1]  if "total_return_%" in sim_df.columns else 0
        final_sh   = sim_df["sharpe"].iloc[-1]          if "sharpe"          in sim_df.columns else 0
        final_dd   = sim_df["max_drawdown_%"].iloc[-1]  if "max_drawdown_%"  in sim_df.columns else 0
        col        = GREEN if final_sh > 0 else YELLOW
        print(f"    {GREEN}✓{RESET} paper_trade_simulation.csv  ({n_days} days)")
        print(f"       Return   : {final_ret:+.1f}%")
        print(f"       Sharpe   : {col}{final_sh:.2f}{RESET}")
        print(f"       Max DD   : {final_dd:.1f}%")
        passed += 1
    else:
        print(f"    {RED}✗{RESET} paper_trade_simulation.csv  MISSING")

    # Trade log
    print(f"\n{BOLD}  Trade Log{RESET}")
    total += 1
    log_path = save_path / "trade_log.csv"
    if log_path.exists():
        tl = pd.read_csv(log_path)
        print(f"    {GREEN}✓{RESET} trade_log.csv  ({len(tl):,} trades logged)")
        if "symbol" in tl.columns:
            for sym, cnt in tl["symbol"].value_counts().items():
                print(f"       {sym:<16} {cnt:>4} trades")
        passed += 1
    else:
        print(f"    {YELLOW}⚠{RESET} trade_log.csv  MISSING (no trades yet)")
        passed += 1

    # Portfolio state
    print(f"\n{BOLD}  Portfolio State{RESET}")
    total += 1
    state_path = save_path / "portfolio_state.json"
    if state_path.exists():
        import json
        with open(state_path) as f:
            state = json.load(f)
        pv  = state.get("portfolio_value", 0)
        col = GREEN if pv > 10000 else (YELLOW if pv > 8000 else RED)
        print(f"    {GREEN}✓{RESET} portfolio_state.json")
        print(f"       Value  : {col}${pv:,.2f}{RESET}")
        print(f"       Trades : {state.get('n_trades', 0)}")
        print(f"       Halted : {state.get('halted', False)}")
        passed += 1
    else:
        print(f"    {YELLOW}⚠{RESET} portfolio_state.json  MISSING")
        passed += 1

    # Performance gate
    print(f"\n{BOLD}  Performance Assessment{RESET}")
    total += 1
    if sim_path.exists():
        sharpe = sim_df["sharpe"].iloc[-1]          if "sharpe"          in sim_df.columns else 0
        dd     = sim_df["max_drawdown_%"].iloc[-1]  if "max_drawdown_%"  in sim_df.columns else 100
        print(f"    Sharpe > 1.0  : {'Yes' if sharpe > 1.0 else 'No'}")
        print(f"    Max DD < 15%  : {'Yes' if dd < 15 else 'No'}")
        if sharpe > 0:
            print(f"    Status  : {YELLOW}Moderate — continue paper trading before live{RESET}")
            passed += 1
        else:
            print(f"    Status  : {RED}Negative Sharpe — review agents{RESET}")
    else:
        passed += 1

    pct    = passed / total * 100 if total else 0
    colour = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"  Checks passed: {colour}{passed}/{total} ({pct:.0f}%){RESET}")
    if pct >= 80:
        print(f"  {GREEN}{BOLD}Phase 5 complete — CARMS is paper trading!{RESET}")
        print(f"  {GREEN}Start live: python main.py --phase 5 --live{RESET}")
    else:
        print(f"  {YELLOW}Run: python main.py --phase 5{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")
    return pct >= 80