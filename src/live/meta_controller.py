"""
live/meta_controller.py — RL meta-controller that routes capital between agents.

The meta-controller observes:
  - Current regime (from HMM)
  - Each specialist agent's signal strength
  - Current portfolio allocation
  - Recent performance per asset class

And outputs a weight vector [w_forex, w_crypto, w_gold] that sums to 1,
representing how much capital to allocate to each specialist agent.

In crisis regime: heavily weights Gold (safe-haven).
In trending-up:   weights Crypto more aggressively.
In ranging:       favours Forex (mean-reversion).

For MVP: uses a rule-based meta-controller first, then optionally
trains an RL meta-controller on top if performance is good.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Dirichlet

from src.utils.logger import get_logger
from src.agents.rl_agents import MLP

log = get_logger(__name__)

REGIME_PRIOR_WEIGHTS = {
    # regime_name → [forex_w, crypto_w, gold_w]
    "trending_up":   [0.20, 0.60, 0.20],
    "trending_down": [0.30, 0.20, 0.50],
    "ranging":       [0.50, 0.30, 0.20],
    "crisis":        [0.20, 0.10, 0.70],
    "unknown":       [0.33, 0.34, 0.33],
}

META_OBS_DIM  = 4 + 3 + 3 + 6   # regime_probs + agent_signals + recent_perf + portfolio
META_ACT_DIM  = 3                # weight for each agent class


class RuleBasedMetaController:
    """
    Simple rule-based meta-controller using regime priors.

    For the MVP this is used first — no training required.
    If paper trading results are good, swap for RLMetaController.

    Args:
        blend_factor: How much to blend regime prior with equal weights.
                      0.0 = pure equal weight, 1.0 = pure regime routing.
    """

    def __init__(self, blend_factor: float = 0.7):
        self.blend_factor = blend_factor
        log.info(f"RuleBasedMetaController initialised (blend={blend_factor})")

    def get_weights(
        self,
        regime_info:     dict,
        agent_signals:   dict,
        portfolio_state: dict,
    ) -> dict:
        """
        Returns capital allocation weights.

        Args:
            regime_info:     Dict with 'name', 'confidence', 'probabilities'.
            agent_signals:   Dict with 'forex', 'crypto', 'gold' → signal strength [-1,1].
            portfolio_state: Dict with 'total_value', 'drawdown', 'positions'.

        Returns:
            Dict with 'forex', 'crypto', 'gold' weights summing to 1.0.
        """
        regime_name = regime_info.get("name", "unknown")
        confidence  = regime_info.get("confidence", 0.5)

        # Base weights from regime prior
        prior = REGIME_PRIOR_WEIGHTS.get(regime_name, REGIME_PRIOR_WEIGHTS["unknown"])

        # Equal weight baseline
        equal = [1/3, 1/3, 1/3]

        # Blend: high confidence → follow regime more strongly
        effective_blend = self.blend_factor * confidence
        weights = [
            equal[i] * (1 - effective_blend) + prior[i] * effective_blend
            for i in range(3)
        ]

        # Scale by signal strength: if agent signal is weak, reduce its weight
        signals = [
            abs(agent_signals.get("forex",  0.0)),
            abs(agent_signals.get("crypto", 0.0)),
            abs(agent_signals.get("gold",   0.0)),
        ]
        # Avoid zeroing out if all signals are zero
        if sum(signals) > 0.01:
            signal_boost = [0.5 + 0.5 * s for s in signals]
            weights = [w * b for w, b in zip(weights, signal_boost)]

        # Crisis override: if drawdown > 10% reduce all risk
        drawdown = portfolio_state.get("drawdown", 0.0)
        if drawdown > 0.10:
            risk_scale = max(0.3, 1.0 - drawdown * 2)
            weights = [w * risk_scale for w in weights]

        # Normalise
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1/3, 1/3, 1/3]

        return {"forex": weights[0], "crypto": weights[1], "gold": weights[2]}


class RLMetaController(nn.Module):
    """
    Optional RL-trained meta-controller.

    Uses PPO to learn optimal capital routing based on:
      - Regime probabilities (4-d)
      - Agent action signals (3-d)
      - Recent per-agent performance (3-d)
      - Portfolio state (6-d)

    Output: Dirichlet concentration parameters → weight distribution
    """

    def __init__(self, obs_dim: int = META_OBS_DIM, hidden: int = 128):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden, META_ACT_DIM),
            nn.Softplus(),   # Ensure positive Dirichlet params
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Dirichlet concentration params (positive)."""
        x = self.backbone(obs)
        return self.alpha_head(x) + 0.1   # Add small floor for stability

    def get_weights(self, obs: np.ndarray) -> np.ndarray:
        """Samples allocation weights from Dirichlet distribution."""
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            alphas = self(obs_t)
            dist   = Dirichlet(alphas)
            sample = dist.sample().squeeze(0).numpy()
        return sample

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["state_dict"])
        self.eval()


def load_meta_controller(save_dir: str = "models") -> RuleBasedMetaController:
    """
    Loads meta-controller. Returns RL version if trained, else rule-based.
    """
    rl_path = Path(save_dir) / "meta_controller_rl.pt"
    if rl_path.exists():
        log.info("Loading trained RL meta-controller...")
        mc = RLMetaController()
        mc.load(str(rl_path))
        return mc
    else:
        log.info("Using rule-based meta-controller (no RL training yet)")
        return RuleBasedMetaController(blend_factor=0.7)