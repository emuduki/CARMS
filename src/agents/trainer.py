"""
agents/trainer.py — Trains all three specialist RL agents.

Training loop for each agent:
  1. Forex agent  (PPO) — EUR/USD and USD/KES combined
  2. Crypto agent (SAC) — BTC-USD and ETH-USD combined
  3. Gold agent   (PPO) — GC=F

Each agent is trained for N_EPISODES episodes on its specific
asset class, with regime-conditioned state vectors from Phase 3.

Metrics tracked per episode:
  - Total portfolio return
  - Sharpe ratio
  - Maximum drawdown
  - Number of trades
  - Win rate

Early stopping: if mean Sharpe > 1.5 over last 50 episodes,
training is considered converged.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.agents.trading_env import make_env, TradingEnv
from src.agents.rl_agents import PPOAgent, SACAgent

log = get_logger(__name__)

# Training config
N_EPISODES_PPO  = 600    # Forex and Gold (increased for better convergence)
N_EPISODES_SAC  = 600    # Crypto (reduced — SAC converges faster with tuned reward)
ROLLOUT_STEPS   = 256    # PPO: smaller rollouts = more frequent updates
PPO_EPOCHS      = 8      # PPO: update epochs per rollout
SAC_BATCH       = 128    # SAC: smaller batch = more frequent updates early on
SAC_UPDATE_FREQ = 2      # SAC: 2 updates per step for faster learning
LOG_INTERVAL    = 50     # Episodes between progress logs
EVAL_INTERVAL   = 100    # Episodes between full evaluations

AGENT_ASSET_MAP = {
    "forex":  ["EURUSD=X", "KES=X"],
    "crypto": ["BTC-USD",  "ETH-USD"],
    "gold":   ["GC=F"],
}

# Canonical agent config used by trainer, runner, backtester, and notebook
AGENT_CONFIGS = {
    "forex":  {"type": "ppo", "symbols": ["EURUSD=X", "KES=X"]},
    "crypto": {"type": "sac", "symbols": ["BTC-USD",  "ETH-USD"]},
    "gold":   {"type": "ppo", "symbols": ["GC=F"]},
}


def train_all_agents(
    config:   dict,
    save_dir: str = "models",
    device:   str = "cpu",
) -> dict:
    """
    Trains all three specialist RL agents.

    Args:
        config:   Parsed config dict.
        save_dir: Where to save trained agent checkpoints.
        device:   'cpu' or 'cuda'.

    Returns:
        Dict mapping agent_name → training metrics DataFrame.
    """
    results = {}

    log.info("=" * 55)
    log.info("PHASE 4 — Specialist RL Agent Training")
    log.info("=" * 55)
    log.info(f"Device   : {device}")
    log.info(f"Save dir : {save_dir}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Forex agent (PPO) ─────────────────────────────────────
    log.info("\n[1/3] Training Forex agent (PPO)...")
    forex_metrics = train_ppo_agent(
        config     = config,
        agent_name = "forex",
        symbols    = AGENT_ASSET_MAP["forex"],
        save_dir   = save_dir,
        device     = device,
        n_episodes = N_EPISODES_PPO,
    )
    results["forex"] = forex_metrics

    # ── Crypto agent (SAC) ────────────────────────────────────
    log.info("\n[2/3] Training Crypto agent (SAC)...")
    crypto_metrics = train_sac_agent(
        config     = config,
        agent_name = "crypto",
        symbols    = AGENT_ASSET_MAP["crypto"],
        save_dir   = save_dir,
        device     = device,
        n_episodes = N_EPISODES_SAC,
    )
    results["crypto"] = crypto_metrics

    # ── Gold agent (PPO) ──────────────────────────────────────
    log.info("\n[3/3] Training Gold agent (PPO)...")
    gold_metrics = train_ppo_agent(
        config     = config,
        agent_name = "gold",
        symbols    = AGENT_ASSET_MAP["gold"],
        save_dir   = save_dir,
        device     = device,
        n_episodes = N_EPISODES_PPO,
    )
    results["gold"] = gold_metrics

    log.info("\nAll agents trained!")
    _print_training_summary(results, save_dir)
    return results


def train_ppo_agent(
    config:     dict,
    agent_name: str,
    symbols:    list,
    save_dir:   str,
    device:     str = "cpu",
    n_episodes: int = N_EPISODES_PPO,
) -> pd.DataFrame:
    """
    Trains a PPO agent across one or more asset symbols.

    For multi-symbol agents (e.g. Forex with EUR/USD + USD/KES),
    the agent alternates between environments each episode.
    """
    # Build environments
    envs = {}
    for sym in symbols:
        env = make_env(sym, config, mode="train")
        if env is not None:
            envs[sym] = env

    if not envs:
        log.error(f"No environments for {agent_name} — missing Phase 3 outputs")
        return pd.DataFrame()

    obs_dim = list(envs.values())[0].obs_dim
    agent   = PPOAgent(obs_dim, device=device)
    sym_list = list(envs.keys())

    metrics_rows = []
    best_sharpe  = -np.inf

    log.info(f"  Environments: {sym_list}")
    log.info(f"  Obs dim: {obs_dim}  |  Episodes: {n_episodes}")

    for ep in range(1, n_episodes + 1):
        # Alternate between symbols
        sym = sym_list[(ep - 1) % len(sym_list)]
        env = envs[sym]
        obs = env.reset()

        ep_reward = 0.0
        step      = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(
                np.array([action])
            )
            agent.store_reward(reward, terminated or truncated)
            ep_reward += reward
            obs = next_obs
            step += 1

            # PPO update every ROLLOUT_STEPS
            if step % ROLLOUT_STEPS == 0:
                agent.update(n_epochs=PPO_EPOCHS)

            if terminated or truncated:
                break

        # Final update at episode end
        if len(agent.buf_rewards) > 0:
            agent.update(n_epochs=PPO_EPOCHS)

        row = {
            "episode":     ep,
            "symbol":      sym,
            "total_return": info["total_return"],
            "sharpe":       info["sharpe"],
            "max_drawdown": info["max_drawdown"],
            "n_trades":     info["n_trades"],
            "ep_reward":    ep_reward,
        }
        metrics_rows.append(row)

        if ep % LOG_INTERVAL == 0:
            recent = pd.DataFrame(metrics_rows[-LOG_INTERVAL:])
            log.info(
                f"  Ep {ep:>4}/{n_episodes}  "
                f"ret={recent['total_return'].mean():+.2%}  "
                f"sharpe={recent['sharpe'].mean():.2f}  "
                f"dd={recent['max_drawdown'].mean():.1%}"
            )

        # Save best checkpoint
        if info["sharpe"] > best_sharpe and ep > 10:
            best_sharpe = info["sharpe"]
            agent.save(str(Path(save_dir) / f"ppo_{agent_name}.pt"))

    # Always save final
    agent.save(str(Path(save_dir) / f"ppo_{agent_name}_final.pt"))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(Path(save_dir) / f"ppo_{agent_name}_metrics.csv", index=False)
    log.info(f"  ✓ {agent_name} PPO agent saved  (best Sharpe: {best_sharpe:.2f})")
    return metrics_df


def train_sac_agent(
    config:     dict,
    agent_name: str,
    symbols:    list,
    save_dir:   str,
    device:     str = "cpu",
    n_episodes: int = N_EPISODES_SAC,
) -> pd.DataFrame:
    """Trains a SAC agent across one or more asset symbols."""
    envs = {}
    for sym in symbols:
        env = make_env(sym, config, mode="train")
        if env is not None:
            envs[sym] = env

    if not envs:
        log.error(f"No environments for {agent_name}")
        return pd.DataFrame()

    obs_dim  = list(envs.values())[0].obs_dim
    agent    = SACAgent(obs_dim, device=device)
    sym_list = list(envs.keys())

    metrics_rows = []
    best_sharpe  = -np.inf

    log.info(f"  Environments: {sym_list}")
    log.info(f"  Obs dim: {obs_dim}  |  Episodes: {n_episodes}")

    for ep in range(1, n_episodes + 1):
        sym = sym_list[(ep - 1) % len(sym_list)]
        env = envs[sym]
        obs = env.reset()

        ep_reward = 0.0

        while True:
            action   = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(
                np.array([action])
            )
            agent.store(obs, action, reward, next_obs, float(terminated))
            ep_reward += reward
            obs = next_obs

            # SAC update every step once buffer is large enough
            if len(agent.buffer) > SAC_BATCH:
                for _ in range(SAC_UPDATE_FREQ):
                    agent.update(SAC_BATCH)

            if terminated or truncated:
                break

        row = {
            "episode":      ep,
            "symbol":       sym,
            "total_return": info["total_return"],
            "sharpe":       info["sharpe"],
            "max_drawdown": info["max_drawdown"],
            "n_trades":     info["n_trades"],
            "ep_reward":    ep_reward,
        }
        metrics_rows.append(row)

        if ep % LOG_INTERVAL == 0:
            recent = pd.DataFrame(metrics_rows[-LOG_INTERVAL:])
            log.info(
                f"  Ep {ep:>4}/{n_episodes}  "
                f"ret={recent['total_return'].mean():+.2%}  "
                f"sharpe={recent['sharpe'].mean():.2f}  "
                f"dd={recent['max_drawdown'].mean():.1%}"
            )

        if info["sharpe"] > best_sharpe and ep > 20:
            best_sharpe = info["sharpe"]
            agent.save(str(Path(save_dir) / f"sac_{agent_name}.pt"))

    agent.save(str(Path(save_dir) / f"sac_{agent_name}_final.pt"))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(Path(save_dir) / f"sac_{agent_name}_metrics.csv", index=False)
    log.info(f"  ✓ {agent_name} SAC agent saved  (best Sharpe: {best_sharpe:.2f})")
    return metrics_df


def evaluate_agent(
    agent_name: str,
    agent_type: str,   # 'ppo' or 'sac'
    symbols:    list,
    config:     dict,
    save_dir:   str = "models",
    device:     str = "cpu",
    n_episodes: int = 20,
) -> pd.DataFrame:
    """
    Evaluates a trained agent on held-out data (deterministic policy).

    Uses the last 20% of data (test split) not seen during training.
    """
    from src.features.indicators import load_features

    path = Path(save_dir) / f"{agent_type}_{agent_name}.pt"
    if not path.exists():
        log.warning(f"No checkpoint at {path}")
        return pd.DataFrame()

    rows = []
    for sym in symbols:
        env = make_env(sym, config, mode="eval")
        if env is None:
            continue

        obs_dim = env.obs_dim
        if agent_type == "ppo":
            agent = PPOAgent(obs_dim, device=device)
        else:
            agent = SACAgent(obs_dim, device=device)
        agent.load(str(path))

        for ep in range(n_episodes):
            obs  = env.reset()
            done = False
            while not done:
                if agent_type == "ppo":
                    with torch.no_grad():
                        import torch
                        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        dist   = agent.actor(obs_t)
                        action = float(dist.mean.cpu().item())
                else:
                    action = agent.select_action(obs, evaluate=True)
                obs, _, terminated, truncated, info = env.step(np.array([action]))
                done = terminated or truncated

            rows.append({
                "episode":      ep,
                "symbol":       sym,
                "total_return": info["total_return"],
                "sharpe":       info["sharpe"],
                "max_drawdown": info["max_drawdown"],
                "n_trades":     info["n_trades"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info(f"\n{agent_name} Evaluation ({n_episodes} episodes per symbol):")
        log.info(f"  Mean return : {df['total_return'].mean():+.2%}")
        log.info(f"  Mean Sharpe : {df['sharpe'].mean():.2f}")
        log.info(f"  Mean DD     : {df['max_drawdown'].mean():.1%}")
    return df


def _print_training_summary(results: dict, save_dir: str):
    """Prints a summary table of all agent training results."""
    BOLD = "\033[1m"; CYAN = "\033[96m"; RESET = "\033[0m"
    GREEN = "\033[92m"; YELLOW = "\033[93m"

    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Phase 4 Training Summary{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"  {'Agent':<12} {'Episodes':<10} {'Mean Sharpe':>12} {'Mean Return':>12} {'Mean DD':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*12} {'─'*12} {'─'*10}")

    for name, df in results.items():
        if df.empty:
            continue
        last_100 = df.tail(100)
        sharpe   = last_100["sharpe"].mean()
        ret      = last_100["total_return"].mean()
        dd       = last_100["max_drawdown"].mean()
        col      = GREEN if sharpe > 0.5 else YELLOW
        print(f"  {col}{name:<12}{RESET} {len(df):<10,} {sharpe:>12.2f} {ret:>+11.1%} {dd:>10.1%}")

    print(f"{BOLD}{CYAN}{'─'*60}{RESET}\n")
    checkpoints = list(Path(save_dir).glob("ppo_*.pt")) + list(Path(save_dir).glob("sac_*.pt"))
    log.info(f"Saved {len(checkpoints)} checkpoint files to {save_dir}/")