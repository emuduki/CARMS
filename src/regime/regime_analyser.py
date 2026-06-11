"""
regime/regime_analyser.py — Analyses and validates detected market regimes.

Computes:
  - Regime duration statistics (how long each regime lasts)
  - Transition probability matrix
  - Per-regime return and risk statistics
  - Regime persistence score
  - Asset-specific regime behaviour

Used in Phase 4 to validate that the HMM is producing
meaningful regime distinctions before training RL agents on them.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.regime.hmm_detector import REGIME_NAMES, REGIME_COLOURS, load_regime_labels

log = get_logger(__name__)


def analyse_regimes(
    labels_df: pd.DataFrame,
    config: dict,
    save_dir: str = "models",
) -> dict:
    """
    Full regime analysis.

    Args:
        labels_df: DataFrame with columns [regime, regime_name] indexed by date.
        config:    Parsed config dict.
        save_dir:  Where to save analysis outputs.

    Returns:
        Dict of analysis results including statistics, transitions, and per-asset metrics.
    """
    results = {}

    log.info("Running regime analysis...")

    # ── Duration statistics ───────────────────────────────────
    log.info("  Computing regime durations...")
    results["durations"]   = _compute_durations(labels_df)

    # ── Transition matrix ─────────────────────────────────────
    log.info("  Computing transition probabilities...")
    results["transitions"] = _compute_transitions(labels_df)

    # ── Return statistics per regime ──────────────────────────
    log.info("  Computing per-regime return statistics...")
    results["return_stats"] = _compute_return_stats(labels_df, config)

    # ── Regime quality score ──────────────────────────────────
    results["quality_score"] = _compute_quality_score(results)

    # ── Per-asset analysis ────────────────────────────────────
    log.info("  Computing per-asset regime behaviour...")
    results["asset_stats"] = _compute_asset_stats(labels_df, config)

    # Save results
    _save_analysis(results, save_dir)
    _print_analysis(results)

    return results


def _compute_durations(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes run-length statistics for each regime.

    For each contiguous sequence of the same regime, records its length.
    Returns mean, median, max, min duration per regime.
    """
    regimes = labels_df["regime"].values
    rows    = []
    i = 0
    while i < len(regimes):
        j = i
        while j < len(regimes) and regimes[j] == regimes[i]:
            j += 1
        rows.append({
            "regime":      regimes[i],
            "regime_name": REGIME_NAMES.get(int(regimes[i]), "?"),
            "duration":    j - i,
            "start":       labels_df.index[i],
            "end":         labels_df.index[j-1],
        })
        i = j

    runs_df = pd.DataFrame(rows)
    stats   = runs_df.groupby("regime_name")["duration"].agg(
        mean_days="mean", median_days="median",
        max_days="max", min_days="min", n_episodes="count"
    ).round(1)
    return stats


def _compute_transitions(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the empirical regime transition probability matrix.

    P[i,j] = probability of transitioning from regime i to regime j.
    Diagonal = probability of staying in the same regime (persistence).
    """
    regimes = labels_df["regime"].values
    n       = 4
    counts  = np.zeros((n, n), dtype=int)

    for t in range(len(regimes) - 1):
        r_now  = int(regimes[t])
        r_next = int(regimes[t+1])
        if 0 <= r_now < n and 0 <= r_next < n:
            counts[r_now, r_next] += 1

    # Normalise rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    probs    = np.where(row_sums > 0, counts / row_sums, 0)

    names = [REGIME_NAMES.get(i, f"r{i}") for i in range(n)]
    return pd.DataFrame(probs, index=names, columns=names).round(3)


def _compute_return_stats(labels_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Computes return and risk statistics for each regime using BTC as reference.
    """
    from src.features.indicators import load_features

    price_df = load_features("BTC-USD", config["data"]["processed_dir"])
    if price_df is None:
        return pd.DataFrame()

    ret = price_df["return_1d"].reindex(labels_df.index)
    vol = price_df["volatility_20"].reindex(labels_df.index)
    rsi = price_df["rsi_14"].reindex(labels_df.index)

    rows = []
    for regime_id, name in REGIME_NAMES.items():
        mask = labels_df["regime"] == regime_id
        if mask.sum() == 0:
            continue
        r = ret[mask].dropna()
        v = vol[mask].dropna()
        rows.append({
            "regime":        name,
            "n_days":        int(mask.sum()),
            "mean_ret_%":    round(r.mean() * 100, 3),
            "std_ret_%":     round(r.std() * 100, 3),
            "sharpe":        round(r.mean() / (r.std() + 1e-8) * np.sqrt(252), 2),
            "win_rate_%":    round((r > 0).mean() * 100, 1),
            "mean_vol":      round(v.mean(), 4),
            "max_drawdown_%":round(_max_drawdown(r) * 100, 2),
        })

    return pd.DataFrame(rows).set_index("regime")


def _compute_quality_score(results: dict) -> dict:
    """
    Computes a quality score for the regime detection:

      1. Persistence score: % of time staying in same regime (high = good)
      2. Separation score: how different regime returns are (high = good)
      3. Balance score:    no regime < 5% of time (penalises degenerate states)
    """
    scores = {}

    # Persistence: mean of diagonal of transition matrix
    trans = results.get("transitions")
    if trans is not None:
        diag = np.diag(trans.values)
        scores["persistence"] = float(diag.mean())

    # Return separation
    ret_stats = results.get("return_stats")
    if ret_stats is not None and "mean_ret_%" in ret_stats.columns:
        rets  = ret_stats["mean_ret_%"].values
        spread = float(rets.max() - rets.min())
        scores["return_separation_%"] = round(spread, 3)

    # Balance: min regime frequency
    durations = results.get("durations")
    if durations is not None and "n_episodes" in durations.columns:
        total    = durations["n_episodes"].sum()
        min_freq = durations["n_episodes"].min() / total if total > 0 else 0
        scores["min_regime_freq"] = round(float(min_freq), 3)

    # Overall quality 0-100
    p = scores.get("persistence", 0)
    s = min(scores.get("return_separation_%", 0) / 0.5, 1.0)  # normalise
    b = min(scores.get("min_regime_freq", 0) / 0.1, 1.0)      # need >10%
    scores["overall_0_100"] = round((p * 0.4 + s * 0.4 + b * 0.2) * 100, 1)

    return scores


def _compute_asset_stats(labels_df: pd.DataFrame, config: dict) -> dict:
    """Computes per-asset return statistics within each regime."""
    from src.features.indicators import load_features

    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )

    asset_stats = {}
    for asset in all_assets:
        sym      = asset["symbol"]
        price_df = load_features(sym, config["data"]["processed_dir"])
        if price_df is None:
            continue

        ret = price_df["return_1d"].reindex(labels_df.index)
        rows = []
        for regime_id, name in REGIME_NAMES.items():
            mask = labels_df["regime"] == regime_id
            r    = ret[mask].dropna()
            if len(r) == 0:
                continue
            rows.append({
                "regime":     name,
                "mean_ret_%": round(r.mean() * 100, 3),
                "sharpe":     round(r.mean() / (r.std() + 1e-8) * np.sqrt(252), 2),
                "win_rate_%": round((r > 0).mean() * 100, 1),
            })
        if rows:
            asset_stats[sym] = pd.DataFrame(rows).set_index("regime")

    return asset_stats


def _max_drawdown(returns: pd.Series) -> float:
    """Computes maximum drawdown from a return series."""
    wealth = (1 + returns).cumprod()
    peak   = wealth.expanding().max()
    dd     = (wealth - peak) / peak
    return float(dd.min())


def _save_analysis(results: dict, save_dir: str):
    """Saves analysis DataFrames to CSV for use in notebooks."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(results.get("durations"), pd.DataFrame):
        results["durations"].to_csv(path / "regime_durations.csv")
    if isinstance(results.get("transitions"), pd.DataFrame):
        results["transitions"].to_csv(path / "regime_transitions.csv")
    if isinstance(results.get("return_stats"), pd.DataFrame):
        results["return_stats"].to_csv(path / "regime_return_stats.csv")

    log.info(f"  Analysis CSVs saved → {path}/regime_*.csv")


def _print_analysis(results: dict):
    """Prints a formatted analysis summary to console."""
    BOLD = "\033[1m"; RESET = "\033[0m"; CYAN = "\033[96m"

    print(f"\n{BOLD}{CYAN}  Regime Duration Statistics{RESET}")
    dur = results.get("durations")
    if dur is not None:
        print(dur.to_string())

    print(f"\n{BOLD}{CYAN}  Transition Probability Matrix{RESET}")
    trans = results.get("transitions")
    if trans is not None:
        print(trans.to_string())
        diag = np.diag(trans.values)
        print(f"  Mean persistence (diagonal): {diag.mean():.1%}")

    print(f"\n{BOLD}{CYAN}  Return Statistics per Regime (BTC reference){RESET}")
    ret_stats = results.get("return_stats")
    if ret_stats is not None:
        print(ret_stats.to_string())

    quality = results.get("quality_score", {})
    print(f"\n{BOLD}{CYAN}  Quality Scores{RESET}")
    for k, v in quality.items():
        print(f"    {k:<30} {v}")