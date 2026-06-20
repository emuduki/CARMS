"""
agents/trading_env.py — Multi-asset trading gym environment for CARMS.

A custom OpenAI Gymnasium environment that simulates trading one asset
using the regime-labelled state vectors from Phase 3.

State space:
  - 128-d fused state vector (TFT + CNN + FinBERT)
  - 4-d regime probabilities
  - 5-d portfolio state (position, cash, drawdown, holding_days, unrealised_pnl)
  - Total: 137-d observation vector

Action space (continuous):
  - Single float in [-1, 1]
    -1.0 = fully short (or sell everything)
     0.0 = hold current position
    +1.0 = fully long (max position)

Reward:
  Step reward = log return of current position × position_size
              − transaction_cost (if trade occurred)
              − drawdown_penalty (if currently in drawdown)
  Episode reward = Sharpe ratio of full episode returns

Key design decisions:
  - Transaction cost: 0.01% per trade (realistic for crypto/forex)
  - Max drawdown circuit breaker: episode ends if drawdown > 20%
  - Position sizing: continuous (0–100% of portfolio)
  - Short selling: allowed (negative positions)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

from src.utils.logger import get_logger

log = get_logger(__name__)

TRANSACTION_COST = 0.0001   # 0.01% per trade
MAX_DRAWDOWN     = 0.20     # 20% drawdown ends episode
INITIAL_CAPITAL  = 10_000   # Starting portfolio value ($)
STATE_DIM        = 128      # From Phase 2 fusion layer
REGIME_DIM       = 4        # Regime probabilities
PORTFOLIO_DIM    = 6        # Portfolio state features
OBS_DIM          = STATE_DIM + REGIME_DIM + PORTFOLIO_DIM  # 138


class TradingEnv:
    """
    Single-asset trading environment.

    Compatible with both gymnasium and gym APIs.
    Falls back to a pure numpy implementation if neither is installed.

    Args:
        states_df:   DataFrame of state vectors + regime columns, indexed by date.
        prices_df:   DataFrame with 'close' column, indexed by date.
        symbol:      Asset symbol (for logging).
        mode:        'train' or 'eval' — eval mode disables episode randomisation.
    """

    def __init__(
        self,
        states_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        symbol:    str = "BTC-USD",
        mode:      str = "train",
    ):
        self.symbol  = symbol
        self.mode    = mode

        # Align data
        state_cols   = [c for c in states_df.columns if c.startswith("state_")]
        prob_cols    = [c for c in states_df.columns if c.startswith("prob_")]

        common = states_df.index.intersection(prices_df.index)

        # Drop rows where regime probabilities OR state vectors are NaN
        # (NaN state vectors propagate through the MLP and cause nan mean → crash)
        drop_cols = []
        if prob_cols:
            drop_cols += prob_cols
        if state_cols:
            drop_cols += state_cols
        if drop_cols:
            states_df = states_df.loc[common].dropna(subset=drop_cols)
            common = states_df.index

        self.states  = states_df.loc[common, state_cols].values.astype(np.float32)
        self.probs   = states_df.loc[common, prob_cols].values.astype(np.float32) \
                       if prob_cols else np.zeros((len(common), 4), dtype=np.float32)
        self.prices  = prices_df.loc[common, "close"].values.astype(np.float32)
        self.dates   = common

        self.n_steps  = len(self.dates)
        self.obs_dim  = STATE_DIM + self.probs.shape[1] + PORTFOLIO_DIM

        # Gymnasium spaces (if available)
        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        self.reset()
        log.info(f"TradingEnv [{symbol}]: {self.n_steps:,} steps  obs={self.obs_dim}-d")

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Resets environment to start of episode."""
        # In train mode: random start; in eval mode: always start at beginning
        if self.mode == "train" and self.n_steps > 252:
            self.t = np.random.randint(0, self.n_steps - 252)
        else:
            self.t = 0

        self.position       = 0.0    # Current position (-1 to 1)
        self.cash           = float(INITIAL_CAPITAL)
        self.portfolio_val  = float(INITIAL_CAPITAL)
        self.peak_val       = float(INITIAL_CAPITAL)
        self.holding_days   = 0
        self.episode_returns = []
        self.trade_count    = 0

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple:
        """
        Takes one trading step.

        Args:
            action: float in [-1, 1] representing target position.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))

        # ── Execute trade ─────────────────────────────────────
        old_position   = self.position
        self.position  = action
        traded         = abs(action - old_position) > 0.01

        # ── Price change ──────────────────────────────────────
        if self.t + 1 >= self.n_steps:
            terminated = True
            return self._get_obs(), 0.0, terminated, False, self._get_info()

        price_now  = self.prices[self.t]
        price_next = self.prices[self.t + 1]
        log_return = float(np.log(price_next / (price_now + 1e-8)))

        # ── Portfolio update ──────────────────────────────────
        position_return  = self.position * log_return
        transaction_cost = TRANSACTION_COST * abs(action - old_position) if traded else 0.0
        step_return      = position_return - transaction_cost

        self.portfolio_val *= np.exp(step_return)
        self.peak_val       = max(self.peak_val, self.portfolio_val)
        drawdown            = (self.peak_val - self.portfolio_val) / self.peak_val
        self.holding_days   = self.holding_days + 1 if not traded else 0
        if traded:
            self.trade_count += 1

        self.episode_returns.append(step_return)
        self.t += 1

        # ── Reward ────────────────────────────────────────────
        reward = self._compute_reward(step_return, drawdown)

        # ── Termination ───────────────────────────────────────
        terminated = drawdown > MAX_DRAWDOWN or self.t >= self.n_steps - 1
        truncated  = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _compute_reward(self, step_return: float, drawdown: float) -> float:
        """
        Reward = scaled step return
               − drawdown penalty (only beyond 15%, linear not quadratic)
               + short-term Sharpe bonus

        Reward scaling: multiply by 100 so gradients are not vanishingly small.
        Financial log returns are tiny (~0.001) — without scaling the agent
        sees near-zero rewards and struggles to learn a signal.
        """
        # Scale return to a learnable magnitude
        scaled_return = step_return * 100.0

        # Drawdown penalty: only kicks in beyond 15%, and linear (not quadratic)
        # Quadratic penalties were too aggressive at early training
        dd_penalty = 0.0
        if drawdown > 0.15:
            dd_penalty = (drawdown - 0.15) * 5.0

        # Short-term Sharpe bonus: reward consistency, not just returns
        consistency = 0.0
        if len(self.episode_returns) >= 10:
            recent = np.array(self.episode_returns[-10:]) * 100.0
            std    = recent.std()
            if std > 1e-8:
                consistency = (recent.mean() / std) * 0.05

        return float(scaled_return - dd_penalty + consistency)

    def _get_obs(self) -> np.ndarray:
        """Builds the observation vector for the current timestep."""
        t   = min(self.t, self.n_steps - 1)
        state_vec = self.states[t]
        prob_vec  = self.probs[t]

        # Normalise portfolio features
        portfolio_vec = np.array([
            self.position,
            (self.portfolio_val / INITIAL_CAPITAL) - 1.0,
            (self.peak_val - self.portfolio_val) / (self.peak_val + 1e-8),
            min(self.holding_days / 20.0, 1.0),
            self.trade_count / max(self.t, 1),
            float(self.position > 0),
        ], dtype=np.float32)

        obs = np.concatenate([state_vec, prob_vec, portfolio_vec])

        # Add slight Gaussian noise during training to prevent sequence memorisation
        if self.mode == "train":
            noise = np.random.normal(0, 0.01, size=obs.shape).astype(np.float32)
            obs += noise

        # Final NaN/Inf guard — should never fire after dropna above, but just in case
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _get_info(self) -> dict:
        """Returns episode statistics."""
        rets = np.array(self.episode_returns) if self.episode_returns else np.array([0.0])
        return {
            "portfolio_value": self.portfolio_val,
            "total_return":    (self.portfolio_val / INITIAL_CAPITAL) - 1.0,
            "sharpe":          rets.mean() / (rets.std() + 1e-8) * np.sqrt(252),
            "max_drawdown":    (self.peak_val - self.portfolio_val) / self.peak_val,
            "n_trades":        self.trade_count,
            "n_steps":         self.t,
        }

    def render(self):
        info = self._get_info()
        print(f"  [{self.symbol}] t={self.t}  "
              f"pos={self.position:+.2f}  "
              f"val=${self.portfolio_val:,.0f}  "
              f"ret={info['total_return']:+.1%}  "
              f"sharpe={info['sharpe']:.2f}")


def make_env(symbol: str, config: dict, mode: str = "train") -> Optional["TradingEnv"]:
    """
    Factory function that loads data and creates a TradingEnv for one asset.

    Args:
        symbol: Asset symbol e.g. 'BTC-USD'
        config: Parsed config dict.
        mode:   'train' or 'eval'

    Returns:
        TradingEnv instance or None if data is missing.
    """
    from src.features.indicators import load_features

    safe       = symbol.replace("=","_").replace("-","_").replace("/","_")
    states_dir = Path(config["data"]["processed_dir"]) / "states"
    states_path = states_dir / f"{safe}_states.parquet"

    if not states_path.exists():
        log.warning(f"No state vectors for {symbol} — run Phase 2 & 3 first")
        return None

    states_df = pd.read_parquet(states_path)
    prices_df = load_features(symbol, config["data"]["processed_dir"])

    if prices_df is None:
        log.warning(f"No price features for {symbol}")
        return None

    return TradingEnv(states_df, prices_df, symbol=symbol, mode=mode)