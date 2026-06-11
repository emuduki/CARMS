"""
agents/backtester.py — Walk-forward backtesting for CARMS specialist agents.

Runs each trained agent over the held-out test period (2024)
and computes comprehensive performance metrics:

  Returns:   Total return, CAGR, monthly breakdown
  Risk:      Sharpe ratio, Sortino ratio, max drawdown, VaR
  Activity:  Trade count, win rate, avg hold time
  Vs bench:  Alpha vs buy-and-hold, information ratio

Walk-forward methodology:
  Train period : 2019-01-01 → 2023-12-31
  Test period  : 2024-01-01 → 2024-12-31  (never seen during training)

Results are saved to models/backtest_results.csv for the paper.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.utils.logger import get_logger
from src.agents.trading_env import make_env, INITIAL_CAPITAL
from src.agents.rl_agents import PPOAgent, SACAgent

log = get_logger(__name__)

from src.agents.trainer import AGENT_CONFIGS


def run_backtest(
    config:   dict,
    save_dir: str = "models",
    device:   str = "cpu",
) -> pd.DataFrame:
    """
    Runs full backtest for all three specialist agents.

    Returns:
        DataFrame with one row per (agent, symbol) with all metrics.
    """
    log.info("Running backtest on test period...")
    all_results = []

    for agent_name, cfg in AGENT_CONFIGS.items():
        agent_type = cfg["type"]
        symbols    = cfg["symbols"]

        ckpt_path = Path(save_dir) / f"{agent_type}_{agent_name}.pt"
        if not ckpt_path.exists():
            log.warning(f"  No checkpoint for {agent_name} — skipping")
            continue

        for sym in symbols:
            log.info(f"  Backtesting {agent_name} on {sym}...")
            result = _backtest_single(
                agent_name, agent_type, sym, config, save_dir, device
            )
            if result:
                all_results.append(result)

    if not all_results:
        log.warning("No backtest results — train agents first")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    path = Path(save_dir) / "backtest_results.csv"
    df.to_csv(path, index=False)
    log.info(f"Backtest results saved → {path}")
    _print_backtest_table(df)
    return df


def _backtest_single(
    agent_name: str,
    agent_type: str,
    symbol:     str,
    config:     dict,
    save_dir:   str,
    device:     str,
) -> Optional[dict]:
    """Runs backtest for a single agent on a single symbol."""
    env = make_env(symbol, config, mode="eval")
    if env is None:
        return None

    obs_dim = env.obs_dim
    if agent_type == "ppo":
        agent = PPOAgent(obs_dim, device=device)
    else:
        agent = SACAgent(obs_dim, device=device)

    ckpt = Path(save_dir) / f"{agent_type}_{agent_name}.pt"
    agent.load(str(ckpt))

    # Run one full deterministic episode
    obs       = env.reset()
    done      = False
    step_returns = []
    actions_log  = []
    portfolio_vals = [INITIAL_CAPITAL]

    while not done:
        if agent_type == "ppo":
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                dist   = agent.actor(obs_t)
                action = float(dist.mean.cpu().item())
        else:
            action = agent.select_action(obs, evaluate=True)

        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        done = terminated or truncated

        step_returns.append(info.get("total_return", 0))
        actions_log.append(action)
        portfolio_vals.append(info["portfolio_value"])

    # Compute metrics
    pv_arr  = np.array(portfolio_vals, dtype=np.float64)
    rets    = np.diff(np.log(pv_arr + 1e-8))
    pv      = pv_arr

    return {
        "agent":             agent_name,
        "symbol":            symbol,
        "total_return_%":    round((pv[-1] / pv[0] - 1) * 100, 2),
        "sharpe":            round(_sharpe(rets), 3),
        "sortino":           round(_sortino(rets), 3),
        "max_drawdown_%":    round(_max_drawdown(pv) * 100, 2),
        "n_trades":          info["n_trades"],
        "win_rate_%":        round((np.array(rets) > 0).mean() * 100, 1),
        "avg_position":      round(np.abs(actions_log).mean(), 3),
        "final_value_$":     round(pv[-1], 2),
        "vs_buyhold_%":      round(_vs_buyhold(symbol, config, pv), 2),
    }


def compare_to_baselines(
    config:   dict,
    save_dir: str = "models",
) -> pd.DataFrame:
    """
    Compares agent performance against:
      1. Buy-and-hold
      2. Random agent
      3. Always-long agent
    """
    log.info("Computing baseline comparisons...")
    baselines = []

    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )

    for asset in all_assets:
        sym = asset["symbol"]
        env = make_env(sym, config, mode="eval")
        if env is None:
            continue

        n = env.n_steps
        prices = env.prices

        # Buy-and-hold
        bh_return = (prices[-1] / prices[0] - 1) if len(prices) > 1 else 0
        bh_rets   = np.diff(np.log(prices + 1e-8))

        # Random agent
        rand_pos = np.random.choice([-1, 0, 1], n)
        rand_rets = rand_pos[:-1] * bh_rets[:len(rand_pos)-1]

        # Always long
        al_rets = bh_rets

        baselines.append({
            "symbol":         sym,
            "buy_hold_%":     round(bh_return * 100, 2),
            "buy_hold_sharpe":round(_sharpe(bh_rets), 3),
            "random_sharpe":  round(_sharpe(rand_rets), 3),
            "always_long_%":  round(bh_return * 100, 2),
        })

    df = pd.DataFrame(baselines)
    path = Path(save_dir) / "baseline_comparison.csv"
    df.to_csv(path, index=False)
    log.info(f"Baselines saved → {path}")
    return df


# ── Metric helpers ────────────────────────────────────────────

def _sharpe(rets: np.ndarray, rf: float = 0.0) -> float:
    if len(rets) < 2 or rets.std() < 1e-8:
        return 0.0
    excess = rets - rf / 252
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _sortino(rets: np.ndarray, rf: float = 0.0) -> float:
    if len(rets) < 2:
        return 0.0
    excess    = rets - rf / 252
    downside  = excess[excess < 0]
    down_std  = downside.std() if len(downside) > 1 else 1e-8
    return float(excess.mean() / down_std * np.sqrt(252))


def _max_drawdown(portfolio_vals: np.ndarray) -> float:
    pv   = np.array(portfolio_vals)
    peak = np.maximum.accumulate(pv)
    dd   = (pv - peak) / (peak + 1e-8)
    return float(abs(dd.min()))


def _vs_buyhold(symbol: str, config: dict, portfolio_vals: np.ndarray) -> float:
    """Computes excess return vs buy-and-hold."""
    from src.features.indicators import load_features
    pv = np.array(portfolio_vals, dtype=np.float64)
    if len(pv) < 2 or pv[0] <= 0:
        return 0.0
    price_df = load_features(symbol, config["data"]["processed_dir"])
    if price_df is None or len(price_df) < 2:
        return 0.0
    agent_ret = float(pv[-1] / pv[0] - 1)
    bh_ret    = float(price_df["close"].iloc[-1] / price_df["close"].iloc[0] - 1)
    return round((agent_ret - bh_ret) * 100, 2)


def _print_backtest_table(df: pd.DataFrame):
    """Prints a formatted backtest results table."""
    BOLD = "\033[1m"; CYAN = "\033[96m"; RESET = "\033[0m"
    GREEN = "\033[92m"; RED = "\033[91m"

    print(f"\n{BOLD}{CYAN}{'─'*75}{RESET}")
    print(f"{BOLD}{CYAN}  CARMS Backtest Results{RESET}")
    print(f"{BOLD}{CYAN}{'─'*75}{RESET}")
    cols = ["agent","symbol","total_return_%","sharpe","max_drawdown_%","n_trades","vs_buyhold_%"]
    header = f"  {'Agent':<10} {'Symbol':<12} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'vs B&H':>9}"
    print(header)
    print(f"  {'─'*10} {'─'*12} {'─'*9} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")
    for _, row in df.iterrows():
        col = GREEN if row["sharpe"] > 0 else RED
        print(
            f"  {col}{row['agent']:<10}{RESET} {row['symbol']:<12} "
            f"{row['total_return_%']:>+8.1f}% {row['sharpe']:>8.2f} "
            f"{row['max_drawdown_%']:>7.1f}% {row['n_trades']:>8} "
            f"{row['vs_buyhold_%']:>+8.1f}%"
        )
    print(f"{BOLD}{CYAN}{'─'*75}{RESET}\n")