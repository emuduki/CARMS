"""
live/portfolio_manager.py — Portfolio risk manager for CARMS paper trading.

Responsibilities:
  1. Position sizing (Kelly criterion variant)
  2. Drawdown circuit breaker (halt if DD > 15%)
  3. Per-asset position limits (max 40% in any one asset)
  4. Daily loss limit (halt if daily loss > 5%)
  5. P&L tracking and trade logging
  6. Performance metrics (Sharpe, drawdown, win rate)

All trades are paper trades — no real money moves.
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

INITIAL_CAPITAL   = 10_000.0    # Paper trading starting capital ($)
MAX_DRAWDOWN      = 0.15        # 15% — circuit breaker threshold
MAX_DAILY_LOSS    = 0.05        # 5% — daily loss limit
MAX_SINGLE_ASSET  = 0.40        # 40% — max allocation to any one asset
TRANSACTION_COST  = 0.0001      # 0.01% per trade


class PortfolioManager:
    """
    Manages portfolio state, risk limits, and trade execution for paper trading.

    Args:
        save_dir:        Where to save trade logs and portfolio state.
        initial_capital: Starting paper capital.
    """

    def __init__(self, save_dir: str = "models", initial_capital: float = INITIAL_CAPITAL):
        self.save_dir        = Path(save_dir)
        self.initial_capital = initial_capital

        # Portfolio state
        self.cash            = initial_capital
        self.positions: dict[str, float] = {}   # symbol → quantity (can be negative = short)
        self.entry_prices: dict[str, float] = {}
        self.portfolio_value = initial_capital
        self.peak_value      = initial_capital

        # Risk state
        self.halted          = False
        self.halt_reason     = ""
        self.daily_start_val = initial_capital
        self.today           = date.today()

        # Metrics tracking
        self.trade_log: list[dict] = []
        self.daily_values: list[dict] = []
        self.n_trades    = 0
        self.n_wins      = 0

        # Load existing state if any
        self._load_state()
        log.info(f"PortfolioManager: ${self.portfolio_value:,.2f}  "
                 f"(initial: ${self.initial_capital:,.2f})")

    # ── Public API ────────────────────────────────────────────

    def update_prices(self, prices: dict[str, float]):
        """
        Updates portfolio value with latest market prices.

        Args:
            prices: Dict of symbol → current price.
        """
        # Reset daily tracker if new day
        if date.today() != self.today:
            self.daily_start_val = self.portfolio_value
            self.today = date.today()

        # Mark-to-market positions
        position_value = 0.0
        for symbol, qty in self.positions.items():
            price = prices.get(symbol, self.entry_prices.get(symbol, 0))
            position_value += qty * price

        self.portfolio_value = self.cash + position_value
        self.peak_value      = max(self.peak_value, self.portfolio_value)

        # Log daily value
        self.daily_values.append({
            "timestamp":       datetime.now().isoformat(),
            "portfolio_value": round(self.portfolio_value, 2),
            "cash":            round(self.cash, 2),
            "drawdown":        round(self.current_drawdown, 4),
        })

        # Check risk limits
        self._check_risk_limits()
        self._save_state()

    def execute_trade(
        self,
        symbol:        str,
        action:        float,   # -1 to +1
        agent_weight:  float,   # 0 to 1 (from meta-controller)
        current_price: float,
    ) -> dict:
        """
        Executes a paper trade.

        Args:
            symbol:        Asset symbol.
            action:        Agent action (-1=full short, 0=hold, +1=full long).
            agent_weight:  Capital allocation from meta-controller.
            current_price: Current market price.

        Returns:
            Trade record dict.
        """
        if self.halted:
            log.warning(f"Trading halted ({self.halt_reason}) — skipping {symbol}")
            return {"status": "halted", "reason": self.halt_reason}

        if current_price <= 0:
            return {"status": "skipped", "reason": "invalid_price"}

        # ── Position sizing ───────────────────────────────────
        # Kelly-inspired: size = action × weight × available_capital
        target_capital = self.portfolio_value * agent_weight * abs(action)
        target_capital = min(target_capital, self.portfolio_value * MAX_SINGLE_ASSET)

        # Target quantity (positive = long, negative = short)
        direction = np.sign(action) if abs(action) > 0.05 else 0
        target_qty = (target_capital / current_price) * direction

        # Current position
        current_qty = self.positions.get(symbol, 0.0)
        trade_qty   = target_qty - current_qty

        if abs(trade_qty) < 1e-6:
            return {"status": "no_trade", "symbol": symbol}

        # ── Transaction cost ──────────────────────────────────
        trade_value = abs(trade_qty) * current_price
        cost        = trade_value * TRANSACTION_COST

        if cost > self.cash * 0.1:   # Don't let costs eat >10% of cash
            log.warning(f"Transaction cost too high for {symbol}: ${cost:.2f}")
            return {"status": "skipped", "reason": "cost_too_high"}

        # ── Execute ───────────────────────────────────────────
        old_qty     = current_qty
        self.positions[symbol] = target_qty

        if direction != 0:
            self.entry_prices[symbol] = current_price

        # Update cash (simplified — assume we can always trade)
        self.cash -= cost
        if trade_qty > 0:
            self.cash -= trade_value    # Buying costs cash
        else:
            self.cash += abs(trade_value)  # Selling returns cash

        self.cash = max(self.cash, 0.0)  # Floor at 0

        # ── Log trade ─────────────────────────────────────────
        self.n_trades += 1
        trade_record = {
            "timestamp":     datetime.now().isoformat(),
            "symbol":        symbol,
            "action":        round(float(action), 4),
            "direction":     "LONG" if direction > 0 else ("SHORT" if direction < 0 else "CLOSE"),
            "quantity":      round(float(trade_qty), 6),
            "price":         round(current_price, 4),
            "trade_value":   round(trade_value, 2),
            "cost":          round(cost, 4),
            "portfolio_val": round(self.portfolio_value, 2),
            "n_trade":       self.n_trades,
        }
        self.trade_log.append(trade_record)
        self._save_state()

        log.info(f"  TRADE {symbol}: {trade_record['direction']}  "
                 f"qty={trade_qty:+.4f}  price=${current_price:.4f}  "
                 f"val=${trade_value:.2f}  cost=${cost:.4f}")

        return trade_record

    def get_portfolio_state_vector(self) -> np.ndarray:
        """Returns 6-d portfolio state vector for RL agent observation."""
        total_pos_val = sum(
            abs(qty) * self.entry_prices.get(sym, 0)
            for sym, qty in self.positions.items()
        )
        return np.array([
            (self.portfolio_value / self.initial_capital) - 1.0,   # Total return
            self.current_drawdown,                                  # Current drawdown
            self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0,
            total_pos_val / max(self.portfolio_value, 1.0),        # Position ratio
            self.n_trades / max(len(self.daily_values), 1),        # Trade frequency
            float(any(v > 0 for v in self.positions.values())),    # Any long?
        ], dtype=np.float32)

    def get_metrics(self) -> dict:
        """Returns current performance metrics."""
        if not self.daily_values:
            return {}

        vals = pd.DataFrame(self.daily_values)["portfolio_value"].values
        rets = np.diff(np.log(vals + 1e-8)) if len(vals) > 1 else np.array([0.0])

        total_return = (self.portfolio_value / self.initial_capital) - 1.0
        sharpe       = float(rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)) if len(rets) > 1 else 0.0
        win_rate     = self.n_wins / max(self.n_trades, 1)

        return {
            "portfolio_value":  round(self.portfolio_value, 2),
            "total_return_%":   round(total_return * 100, 2),
            "sharpe":           round(sharpe, 3),
            "max_drawdown_%":   round(self._all_time_max_drawdown() * 100, 2),
            "current_dd_%":     round(self.current_drawdown * 100, 2),
            "n_trades":         self.n_trades,
            "win_rate_%":       round(win_rate * 100, 1),
            "halted":           self.halted,
            "halt_reason":      self.halt_reason,
        }

    # ── Risk management ───────────────────────────────────────

    @property
    def current_drawdown(self) -> float:
        return (self.peak_value - self.portfolio_value) / max(self.peak_value, 1.0)

    def _check_risk_limits(self):
        """Checks all risk limits and halts trading if breached."""
        if self.halted:
            return

        # Max drawdown circuit breaker
        if self.current_drawdown > MAX_DRAWDOWN:
            self._halt(f"Max drawdown breached: {self.current_drawdown:.1%} > {MAX_DRAWDOWN:.0%}")
            return

        # Daily loss limit
        daily_loss = (self.daily_start_val - self.portfolio_value) / max(self.daily_start_val, 1.0)
        if daily_loss > MAX_DAILY_LOSS:
            self._halt(f"Daily loss limit: {daily_loss:.1%} > {MAX_DAILY_LOSS:.0%}")
            return

    def _halt(self, reason: str):
        self.halted      = True
        self.halt_reason = reason
        log.warning(f"⚠ TRADING HALTED: {reason}")
        log.warning("  Close all positions and review before restarting.")

    def resume(self):
        """Manually resumes trading after reviewing halt."""
        log.info("Trading resumed by operator")
        self.halted      = False
        self.halt_reason = ""
        self.daily_start_val = self.portfolio_value

    # ── State persistence ─────────────────────────────────────

    def _save_state(self):
        """Saves portfolio state and trade log to disk."""
        state_path = self.save_dir / "portfolio_state.json"
        log_path   = self.save_dir / "trade_log.csv"

        state = {
            "cash":            self.cash,
            "portfolio_value": self.portfolio_value,
            "peak_value":      self.peak_value,
            "positions":       self.positions,
            "entry_prices":    self.entry_prices,
            "n_trades":        self.n_trades,
            "halted":          self.halted,
            "halt_reason":     self.halt_reason,
            "last_updated":    datetime.now().isoformat(),
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        if self.trade_log:
            pd.DataFrame(self.trade_log).to_csv(log_path, index=False)

    def _load_state(self):
        """Loads existing portfolio state from disk if available."""
        state_path = self.save_dir / "portfolio_state.json"
        if not state_path.exists():
            return
        try:
            with open(state_path) as f:
                state = json.load(f)
            self.cash            = state.get("cash", self.initial_capital)
            self.portfolio_value = state.get("portfolio_value", self.initial_capital)
            self.peak_value      = state.get("peak_value", self.initial_capital)
            self.positions       = state.get("positions", {})
            self.entry_prices    = state.get("entry_prices", {})
            self.n_trades        = state.get("n_trades", 0)
            self.halted          = state.get("halted", False)
            self.halt_reason     = state.get("halt_reason", "")
            log.info(f"Loaded portfolio state: ${self.portfolio_value:,.2f}")
        except Exception as e:
            log.warning(f"Could not load portfolio state: {e}")

    def _all_time_max_drawdown(self) -> float:
        if len(self.daily_values) < 2:
            return 0.0
        vals = np.array([d["portfolio_value"] for d in self.daily_values])
        peak = np.maximum.accumulate(vals)
        dd   = (vals - peak) / (peak + 1e-8)
        return float(abs(dd.min()))