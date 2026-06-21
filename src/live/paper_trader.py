"""
live/paper_trader.py — Main paper trading loop for CARMS.

Runs a continuous loop that:
  1. Fetches live prices (yfinance + Binance Testnet)
  2. Builds observation vectors for each asset
  3. Gets regime from HMM
  4. Queries each specialist agent for an action
  5. Meta-controller allocates capital weights
  6. Portfolio manager executes paper trades with risk controls
  7. Logs everything and updates dashboard

Usage:
    python main.py --phase 5 --paper_trade
    python main.py --phase 5 --paper_trade --capital 10000

Press Ctrl+C to stop gracefully.
"""

import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.utils.logger import get_logger
from src.live.state_builder import LiveStateBuilder
from src.live.meta_controller import load_meta_controller
from src.live.portfolio_manager import PortfolioManager
from src.agents.rl_agents import PPOAgent, SACAgent
from src.agents.trainer import AGENT_CONFIGS

log = get_logger(__name__)

TICK_INTERVAL = 300    # Seconds between trading decisions (5 min)
ALL_SYMBOLS   = ["EURUSD=X", "KES=X", "BTC-USD", "ETH-USD", "GC=F"]
SYMBOL_AGENT  = {
    "EURUSD=X": "forex",
    "KES=X":    "forex",
    "BTC-USD":  "crypto",
    "ETH-USD":  "crypto",
    "GC=F":     "gold",
}


class PaperTrader:
    """
    Main CARMS paper trading engine.

    Args:
        config:   Parsed config dict.
        save_dir: Model and log directory.
        device:   'cpu' or 'cuda'.
        capital:  Starting paper capital in USD.
    """

    def __init__(
        self,
        config:   dict,
        save_dir: str   = "models",
        device:   str   = "cpu",
        capital:  float = 10_000.0,
    ):
        self.config   = config
        self.save_dir = Path(save_dir)
        self.device   = device
        self.running  = False

        log.info("=" * 55)
        log.info("CARMS Paper Trader")
        log.info("=" * 55)
        log.info(f"Capital  : ${capital:,.2f}")
        log.info(f"Symbols  : {ALL_SYMBOLS}")
        log.info(f"Interval : {TICK_INTERVAL}s")

        # Initialise components
        self.state_builder  = LiveStateBuilder(config, save_dir, device)
        self.meta_controller = load_meta_controller(save_dir)
        self.portfolio       = PortfolioManager(save_dir, capital)
        self.agents          = self._load_agents()

        log.info(f"Loaded {len(self.agents)} specialist agents")

    def _load_agents(self) -> dict:
        """Loads all trained specialist agents."""
        agents = {}
        obs_dim = 137   # Fixed observation dimension

        for agent_name, cfg in AGENT_CONFIGS.items():
            agent_type = cfg["type"]
            ckpt_path  = self.save_dir / f"{agent_type}_{agent_name}.pt"

            if not ckpt_path.exists():
                log.warning(f"No checkpoint for {agent_name} at {ckpt_path}")
                continue

            if agent_type == "ppo":
                agent = PPOAgent(obs_dim, device=self.device)
            else:
                agent = SACAgent(obs_dim, device=self.device)

            agent.load(str(ckpt_path))
            agents[agent_name] = agent
            log.info(f"  Loaded {agent_name} ({agent_type.upper()})")

        return agents

    def run(self, max_ticks: Optional[int] = None):
        """
        Starts the paper trading loop.

        Args:
            max_ticks: Maximum number of ticks (None = run forever).
        """
        self.running = True
        tick = 0

        # Graceful shutdown on Ctrl+C
        def _shutdown(sig, frame):
            log.info("\nShutting down gracefully...")
            self.running = False
        signal.signal(signal.SIGINT, _shutdown)

        log.info("\nPaper trading started. Press Ctrl+C to stop.\n")

        while self.running:
            if max_ticks and tick >= max_ticks:
                break

            tick += 1
            tick_start = time.time()

            log.info(f"\n{'─'*50}")
            log.info(f"Tick {tick}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log.info(f"{'─'*50}")

            try:
                self._run_tick()
            except Exception as e:
                log.error(f"Tick error: {e}")
                import traceback
                log.debug(traceback.format_exc())

            # Wait for next tick
            elapsed = time.time() - tick_start
            wait    = max(0, TICK_INTERVAL - elapsed)
            if wait > 0 and self.running:
                log.info(f"Next tick in {wait:.0f}s...")
                time.sleep(wait)

        # Final summary
        self._print_session_summary()

    def _run_tick(self):
        """Executes one complete trading decision cycle."""

        # ── Step 1: Get current regime ────────────────────────
        regime_info = self.state_builder.get_current_regime()
        log.info(f"Regime: {regime_info['name']}  "
                 f"(confidence={regime_info['confidence']:.0%})")

        # ── Step 2: Get portfolio state ───────────────────────
        portfolio_vec = self.portfolio.get_portfolio_state_vector()

        # ── Step 3: Get agent signals ─────────────────────────
        agent_signals   = {}
        agent_obs       = {}

        for symbol in ALL_SYMBOLS:
            obs = self.state_builder.get_observation(symbol, portfolio_vec)
            if obs is None:
                log.warning(f"  No observation for {symbol} — skipping")
                continue

            agent_name = SYMBOL_AGENT[symbol]
            agent      = self.agents.get(agent_name)
            if agent is None:
                continue

            # Get deterministic action from agent
            if isinstance(agent, PPOAgent):
                with torch.no_grad():
                    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    dist   = agent.actor(obs_t)
                    action = float(dist.mean.cpu().item())
            else:
                action = agent.select_action(obs, evaluate=True)

            agent_obs[symbol]    = obs
            agent_signals[agent_name] = action
            log.info(f"  {symbol:<14} signal={action:+.3f}")

        if not agent_signals:
            log.warning("No agent signals this tick")
            return

        # ── Step 4: Meta-controller allocation ────────────────
        weights = self.meta_controller.get_weights(
            regime_info     = regime_info,
            agent_signals   = agent_signals,
            portfolio_state = self.portfolio.get_metrics(),
        )
        log.info(f"  Weights: forex={weights['forex']:.2f}  "
                 f"crypto={weights['crypto']:.2f}  "
                 f"gold={weights['gold']:.2f}")

        # ── Step 5: Update prices and execute trades ──────────
        current_prices = {}
        for symbol in ALL_SYMBOLS:
            price = self.state_builder.get_current_price(symbol)
            if price:
                current_prices[symbol] = price

        self.portfolio.update_prices(current_prices)

        if self.portfolio.halted:
            log.warning(f"Trading halted: {self.portfolio.halt_reason}")
            return

        # Execute per-symbol trades
        for symbol, obs in agent_obs.items():
            agent_name    = SYMBOL_AGENT[symbol]
            agent         = self.agents.get(agent_name)
            current_price = current_prices.get(symbol)

            if agent is None or current_price is None:
                continue

            if isinstance(agent, PPOAgent):
                with torch.no_grad():
                    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    dist   = agent.actor(obs_t)
                    action = float(dist.mean.cpu().item())
            else:
                action = agent.select_action(obs, evaluate=True)

            weight = weights.get(agent_name, 1/3)
            self.portfolio.execute_trade(symbol, action, weight, current_price)

        # ── Step 6: Print metrics ─────────────────────────────
        metrics = self.portfolio.get_metrics()
        log.info(f"\n  Portfolio: ${metrics.get('portfolio_value',0):,.2f}  "
                 f"return={metrics.get('total_return_%',0):+.2f}%  "
                 f"sharpe={metrics.get('sharpe',0):.2f}  "
                 f"dd={metrics.get('current_dd_%',0):.1f}%  "
                 f"trades={metrics.get('n_trades',0)}")

    def _print_session_summary(self):
        """Prints end-of-session performance summary."""
        metrics = self.portfolio.get_metrics()
        log.info("\n" + "="*55)
        log.info("CARMS Paper Trading Session Summary")
        log.info("="*55)
        for k, v in metrics.items():
            log.info(f"  {k:<25} {v}")

        trade_log_path = self.save_dir / "trade_log.csv"
        if trade_log_path.exists():
            log.info(f"\n  Trade log saved → {trade_log_path}")
        log.info("="*55)


def run_paper_trade_simulation(
    config:    dict,
    save_dir:  str   = "models",
    device:    str   = "cpu",
    capital:   float = 10_000.0,
    n_ticks:   int   = 100,
) -> dict:
    """
    Runs a paper trade simulation using historical data (no live API needed).

    Instead of waiting for real-time prices, steps through the held-out
    test period day by day — equivalent to a live paper trading session
    but runs in minutes instead of months.

    Args:
        config:   Parsed config dict.
        save_dir: Model directory.
        device:   'cpu' or 'cuda'.
        capital:  Starting capital.
        n_ticks:  Number of trading days to simulate.

    Returns:
        Dict of performance metrics.
    """
    from src.features.indicators import load_features
    from src.regime.hmm_detector import load_regime_labels

    log.info("Running paper trade simulation on historical test data...")
    log.info(f"  Simulating {n_ticks} trading days")

    portfolio = PortfolioManager(save_dir, capital)
    agents    = {}
    obs_dim   = 137

    for agent_name, cfg in AGENT_CONFIGS.items():
        agent_type = cfg["type"]
        ckpt_path  = Path(save_dir) / f"{agent_type}_{agent_name}.pt"
        if not ckpt_path.exists():
            continue
        if agent_type == "ppo":
            agent = PPOAgent(obs_dim, device=device)
        else:
            agent = SACAgent(obs_dim, device=device)
        agent.load(str(ckpt_path))
        agents[agent_name] = agent

    meta    = load_meta_controller(save_dir)
    labels  = load_regime_labels(save_dir)

    # Load test period state vectors
    states_dir = Path(config["data"]["processed_dir"]) / "states"
    all_states = {}
    for symbol in ALL_SYMBOLS:
        safe = symbol.replace("=","_").replace("-","_").replace("/","_")
        path = states_dir / f"{safe}_states.parquet"
        if path.exists():
            all_states[symbol] = pd.read_parquet(path)

    if not all_states:
        log.error("No state vectors found — run Phase 2 & 3 first")
        return {}

    # Use the last n_ticks dates as simulation period
    all_dates = sorted(set.intersection(*[set(df.index) for df in all_states.values()]))
    sim_dates = all_dates[-n_ticks:]

    log.info(f"  Simulation period: {sim_dates[0].date()} → {sim_dates[-1].date()}")

    tick_results = []

    for t, sim_date in enumerate(sim_dates):
        portfolio_vec = portfolio.get_portfolio_state_vector()
        regime_info   = {"name": "ranging", "confidence": 0.5}

        if labels is not None and sim_date in labels.index:
            row = labels.loc[sim_date]
            prob_cols = [c for c in labels.columns if c.startswith("prob_")]
            regime_info = {
                "name":          row.get("regime_name", "ranging"),
                "confidence":    float(row[prob_cols].max()) if prob_cols else 0.5,
                "probabilities": [float(row[c]) for c in prob_cols] if prob_cols else [0.25]*4,
            }

        agent_signals = {}
        sym_actions   = {}

        for symbol, states_df in all_states.items():
            if sim_date not in states_df.index:
                continue

            state_cols = [c for c in states_df.columns if c.startswith("state_")]
            prob_cols  = [c for c in states_df.columns if c.startswith("prob_")]
            row        = states_df.loc[sim_date]

            state_vec = row[state_cols].values.astype(np.float32)
            prob_vec  = row[prob_cols].values.astype(np.float32) if prob_cols else np.zeros(4, dtype=np.float32)
            obs       = np.concatenate([state_vec, prob_vec, portfolio_vec])

            agent_name = SYMBOL_AGENT.get(symbol)
            agent      = agents.get(agent_name)
            if agent is None:
                continue

            if isinstance(agent, PPOAgent):
                with torch.no_grad():
                    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = float(agent.actor(obs_t).mean.cpu().item())
            else:
                action = agent.select_action(obs, evaluate=True)

            agent_signals[agent_name] = action
            sym_actions[symbol]       = action

        weights = meta.get_weights(
            regime_info     = regime_info,
            agent_signals   = agent_signals,
            portfolio_state = portfolio.get_metrics(),
        )

        # Simulate prices from feature data
        prices = {}
        for symbol in ALL_SYMBOLS:
            feat = load_features(symbol, config["data"]["processed_dir"])
            if feat is not None and sim_date in feat.index:
                prices[symbol] = float(feat.loc[sim_date, "close"])

        portfolio.update_prices(prices)

        for symbol, action in sym_actions.items():
            price = prices.get(symbol)
            if price:
                agent_name = SYMBOL_AGENT.get(symbol)
                weight     = weights.get(agent_name, 1/3)
                portfolio.execute_trade(symbol, action, weight, price)

        metrics = portfolio.get_metrics()
        tick_results.append({
            "date":   sim_date,
            "regime": regime_info["name"],
            **metrics,
        })

        if (t + 1) % 20 == 0:
            log.info(f"  Day {t+1}/{n_ticks}  "
                     f"val=${metrics.get('portfolio_value',0):,.2f}  "
                     f"ret={metrics.get('total_return_%',0):+.1f}%  "
                     f"sharpe={metrics.get('sharpe',0):.2f}")

        if portfolio.halted:
            log.warning(f"  Circuit breaker triggered at day {t+1}")
            break

    # Save simulation results
    sim_df   = pd.DataFrame(tick_results)
    sim_path = Path(save_dir) / "paper_trade_simulation.csv"
    sim_df.to_csv(sim_path, index=False)

    final = portfolio.get_metrics()
    log.info(f"\nSimulation complete!")
    log.info(f"  Final portfolio : ${final.get('portfolio_value',0):,.2f}")
    log.info(f"  Total return    : {final.get('total_return_%',0):+.1f}%")
    log.info(f"  Sharpe ratio    : {final.get('sharpe',0):.2f}")
    log.info(f"  Max drawdown    : {final.get('max_drawdown_%',0):.1f}%")
    log.info(f"  Total trades    : {final.get('n_trades',0)}")
    log.info(f"  Results saved   → {sim_path}")

    return final