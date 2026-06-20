"""
agents/rl_agents.py — PPO and SAC RL agent implementations for CARMS.

Two algorithms used:
  PPO (Proximal Policy Optimisation) — Forex and Gold agents
    - More stable, better for assets with clearer trends
    - Clip ratio prevents destructive policy updates
    - Works well with discrete-ish action distributions

  SAC (Soft Actor-Critic) — Crypto agent
    - Better for continuous actions with high volatility
    - Entropy regularisation encourages exploration
    - More sample-efficient but needs more memory

Both use the same network architecture:
  Input (138-d obs) → MLP(256, 256) → policy/value heads

The agents are regime-aware: the observation vector includes
regime probabilities from Phase 3, so the policy learns to
behave differently in trending vs ranging vs crisis markets.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.utils.logger import get_logger

log = get_logger(__name__)

HIDDEN_DIM   = 256
LR_ACTOR     = 3e-4
LR_CRITIC    = 3e-4
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2    # PPO clip ratio
ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
MAX_GRAD_NORM = 0.5

# SAC specific
ALPHA        = 0.2    # Entropy temperature
TAU          = 0.005  # Soft update coefficient
BUFFER_SIZE  = 100_000


# ── Shared network components ─────────────────────────────────

class MLP(nn.Module):
    """Shared MLP backbone for all agents."""
    def __init__(self, input_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── PPO Agent ─────────────────────────────────────────────────

class PPOActor(nn.Module):
    """PPO policy network outputting mean and log_std of action distribution."""
    def __init__(self, obs_dim: int, action_dim: int = 1, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden)
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        x       = self.backbone(obs)
        # Clamp pre-activation to prevent exploding values before tanh
        raw     = self.mean_head(x).clamp(-10.0, 10.0)
        mean    = torch.tanh(raw)
        # Safety: replace any residual NaN (e.g. from corrupt input) with 0
        mean    = torch.nan_to_num(mean, nan=0.0)
        log_std = self.log_std_head.clamp(-4, 2)
        std     = log_std.exp()
        dist    = Normal(mean, std)
        return dist

    def get_action(self, obs: torch.Tensor):
        dist   = self(obs)
        action = dist.sample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class PPOCritic(nn.Module):
    """PPO value network."""
    def __init__(self, obs_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden)
        self.value    = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(self.backbone(obs))


class PPOAgent:
    """
    Proximal Policy Optimisation agent.
    Used for Forex (EUR/USD, USD/KES) and Gold (GC=F) assets.

    Key hyperparameters:
      clip_eps    = 0.2   (prevents large policy updates)
      gae_lambda  = 0.95  (advantage estimation smoothing)
      n_epochs    = 10    (number of update passes per batch)
    """

    def __init__(self, obs_dim: int, device: str = "cpu", lr: float = LR_ACTOR):
        self.device  = device
        self.actor   = PPOActor(obs_dim).to(device)
        self.critic  = PPOCritic(obs_dim).to(device)
        self.opt_a   = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_c   = torch.optim.Adam(self.critic.parameters(), lr=lr * 2)

        # Rollout buffer
        self._clear_buffer()

    def _clear_buffer(self):
        self.buf_obs      = []
        self.buf_actions  = []
        self.buf_logprobs = []
        self.buf_rewards  = []
        self.buf_dones    = []
        self.buf_values   = []

    def select_action(self, obs: np.ndarray) -> float:
        """Selects action given observation. Returns scalar action."""
        # Sanitise obs before it enters the network
        obs = np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.get_action(obs_t)
            value  = self.critic(obs_t)
        self.buf_obs.append(obs)
        self.buf_actions.append(action.cpu().numpy()[0])
        self.buf_logprobs.append(log_prob.cpu().item())
        self.buf_values.append(value.cpu().item())
        return float(action.cpu().numpy()[0][0])

    def store_reward(self, reward: float, done: bool):
        """Stores step reward and done flag."""
        self.buf_rewards.append(reward)
        self.buf_dones.append(done)

    def update(self, n_epochs: int = 10) -> dict:
        """
        Runs PPO update on collected rollout buffer.
        Returns dict of loss metrics.
        """
        if len(self.buf_rewards) < 2:
            return {}

        # ── Compute GAE advantages ────────────────────────────
        advantages, returns = self._compute_gae()

        obs_t      = torch.FloatTensor(np.array(self.buf_obs)).to(self.device)
        actions_t  = torch.FloatTensor(np.array(self.buf_actions)).unsqueeze(-1).to(self.device)
        old_lp_t   = torch.FloatTensor(self.buf_logprobs).to(self.device)
        adv_t      = torch.FloatTensor(advantages).to(self.device)
        ret_t      = torch.FloatTensor(returns).to(self.device)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        for _ in range(n_epochs):
            # Policy update
            dist     = self.actor(obs_t)
            new_lp   = dist.log_prob(actions_t).sum(-1)
            entropy  = dist.entropy().sum(-1).mean()
            ratio    = (new_lp - old_lp_t).exp()

            surr1    = ratio * adv_t
            surr2    = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            p_loss   = -torch.min(surr1, surr2).mean()
            e_loss   = -ENTROPY_COEF * entropy
            actor_loss = p_loss + e_loss

            self.opt_a.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.opt_a.step()

            # Value update
            values   = self.critic(obs_t).squeeze(-1)
            v_loss   = VALUE_COEF * F.mse_loss(values, ret_t)

            self.opt_c.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.opt_c.step()

            metrics["policy_loss"] += p_loss.item() / n_epochs
            metrics["value_loss"]  += v_loss.item() / n_epochs
            metrics["entropy"]     += entropy.item() / n_epochs

        self._clear_buffer()
        return metrics

    def _compute_gae(self):
        """Computes Generalised Advantage Estimation."""
        rewards  = self.buf_rewards
        dones    = self.buf_dones
        values   = self.buf_values + [0.0]   # Bootstrap with 0

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
            gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor.eval()
        self.critic.eval()


# ── SAC Agent ─────────────────────────────────────────────────

class ReplayBuffer:
    """Experience replay buffer for SAC."""

    def __init__(self, obs_dim: int, capacity: int = BUFFER_SIZE):
        self.capacity = capacity
        self.ptr = self.size = 0
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity, 1),       dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),       dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),       dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu"):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def __len__(self):
        return self.size


class SACActorNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden)
        self.mean_head    = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x       = self.backbone(obs)
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(-4, 2)
        std     = log_std.exp()
        dist    = Normal(mean, std)
        action_raw = dist.rsample()
        action  = torch.tanh(action_raw)
        # Correct log prob for tanh squashing
        log_prob = dist.log_prob(action_raw) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class SACCriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        # Twin Q-networks to reduce overestimation bias
        self.q1 = nn.Sequential(MLP(obs_dim + 1, hidden), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(MLP(obs_dim + 1, hidden), nn.Linear(hidden, 1))

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    """
    Soft Actor-Critic agent.
    Used for the Crypto agent (BTC-USD, ETH-USD).

    More sample-efficient than PPO for high-volatility assets.
    Entropy regularisation prevents premature convergence.
    """

    def __init__(self, obs_dim: int, device: str = "cpu",
                 lr: float = LR_ACTOR, alpha: float = ALPHA):
        self.device = device
        self.alpha  = alpha

        self.actor   = SACActorNet(obs_dim).to(device)
        self.critic  = SACCriticNet(obs_dim).to(device)
        self.critic_target = SACCriticNet(obs_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(obs_dim)

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> float:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if evaluate:
                # Deterministic action for evaluation
                x    = self.actor.backbone(obs_t)
                mean = self.actor.mean_head(x)
                return float(torch.tanh(mean).cpu().item())
            action, _ = self.actor(obs_t)
        return float(action.cpu().item())

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self, batch_size: int = 256) -> dict:
        if len(self.buffer) < batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + GAMMA * (1 - dones) * q_next

        # Critic update
        q1, q2   = self.critic(obs, actions)
        c_loss   = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_c.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.opt_c.step()

        # Actor update
        new_actions, log_pi = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_actions)
        q_pi   = torch.min(q1_pi, q2_pi)
        a_loss = (self.alpha * log_pi - q_pi).mean()
        self.opt_a.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        self.opt_a.step()

        # Soft update target network
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        return {
            "critic_loss": c_loss.item(),
            "actor_loss":  a_loss.item(),
            "entropy":     -log_pi.mean().item(),
        }

    def save(self, path: str):
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor.eval()