"""
regime/hmm_detector.py — Hidden Markov Model for market regime detection.

Reads the 128-d unified state vectors from Phase 2 and trains a
Gaussian HMM to label every trading day as one of 4 regimes:

  Regime 0 — Trending up    : sustained positive momentum, expanding price
  Regime 1 — Trending down  : sustained negative momentum, declining price
  Regime 2 — Ranging        : mean-reverting, low directionality
  Regime 3 — Crisis         : high volatility spike, correlation surge

Pipeline:
  1. Load all state vectors (all 5 assets)
  2. Cross-asset concatenation → richer feature matrix
  3. PCA reduction (128→16 per asset) to remove noise
  4. Gaussian HMM training with 4 hidden states
  5. Viterbi decoding → regime label per day
  6. Regime auto-labelling using price statistics
  7. Save model + labelled dataset

Why HMM:
  Markets are non-stationary — they switch between regimes with
  memory (today's regime likely = yesterday's). HMMs model this
  explicitly through transition probabilities, unlike clustering.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

log = get_logger(__name__)

N_REGIMES   = 3
PCA_DIMS    = 16     # Reduce 128-d state to 16-d before HMM
N_ITER      = 200    # HMM EM iterations
RANDOM_SEED = 42

REGIME_NAMES = {
    0: "trending_up",
    1: "trending_down",
    2: "ranging",
}

REGIME_COLOURS = {
    0: "#1D9E75",   # green
    1: "#E24B4A",   # red
    2: "#888780",   # gray
}


# ── Public API ────────────────────────────────────────────────

def train_hmm(
    config: dict,
    save_dir: str = "models",
    n_regimes: int = N_REGIMES,
) -> tuple:
    """
    Full HMM training pipeline.

    Steps:
      1. Load state vectors for all assets
      2. PCA reduction per asset
      3. Concatenate across assets → cross-asset feature matrix
      4. Train Gaussian HMM
      5. Viterbi decode → regime labels
      6. Auto-label regimes using price statistics
      7. Save model, labels, and diagnostics

    Args:
        config:    Parsed config dict.
        save_dir:  Where to save HMM model and outputs.
        n_regimes: Number of hidden states (default 4).

    Returns:
        (model, labels_df, pca, scaler) tuple
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        log.error("hmmlearn not installed — run: pip install hmmlearn")
        return None, None, None, None

    states_dir = Path(config["data"]["processed_dir"]) / "states"
    save_path  = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load state vectors ────────────────────────────
    log.info("Loading state vectors from Phase 2...")
    all_states, common_dates = _load_and_align_states(states_dir, config)

    if all_states is None or len(all_states) == 0:
        log.error("No state vectors found — run Phase 2 first")
        return None, None, None, None

    if len(common_dates) < 100:
        log.warning("=" * 65)
        log.warning(f"CRITICAL WARNING: Only {len(common_dates)} common dates available for regime detection.")
        log.warning("This is usually caused by running Phase 1/2 in '--quick' mode or data constraints.")
        log.warning("Training HMM on so few dates will result in a degenerate solution and severe RL overfitting.")
        log.warning("Please make sure to run the full pipeline without '--quick':")
        log.warning("  python main.py --phase 1")
        log.warning("  python main.py --phase 2")
        log.warning("=" * 65)

    log.info(f"  Loaded {all_states.shape[0]:,} observations × {all_states.shape[1]} features")

    # ── Step 2: Scale + PCA reduction ────────────────────────
    target_dims = PCA_DIMS * _n_assets(config)
    log.info(f"Reducing dimensions: {all_states.shape[1]} → {target_dims}...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_states)

    max_components = min(target_dims, X_scaled.shape[1], X_scaled.shape[0])
    if max_components < target_dims:
        log.warning(
            "Only %d samples available for PCA; reducing n_components from %d to %d",
            X_scaled.shape[0], target_dims, max_components,
        )

    pca = PCA(n_components=max_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    variance_explained = pca.explained_variance_ratio_.cumsum()[-1]
    log.info(f"  PCA variance explained: {variance_explained:.1%}")
    log.info(f"  Feature matrix: {X_pca.shape}")

    # ── Step 3: Train HMM ─────────────────────────────────────
    log.info(f"Training Gaussian HMM ({n_regimes} states, {N_ITER} iterations)...")
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    model.fit(X_pca)
    log.info(f"  HMM converged: {model.monitor_.converged}")
    log.info(f"  Log-likelihood: {model.score(X_pca):.2f}")

    # ── Step 4: Viterbi decode ────────────────────────────────
    log.info("Decoding regime sequence (Viterbi)...")
    raw_labels = model.predict(X_pca)

    # ── Step 5: Auto-label regimes ────────────────────────────
    log.info("Auto-labelling regimes using price statistics...")
    regime_map  = _auto_label_regimes(raw_labels, common_dates, config)
    final_labels = np.array([regime_map.get(r, r) for r in raw_labels])

    # ── Step 6: Build labelled DataFrame ─────────────────────
    labels_df = pd.DataFrame({
        "date":        common_dates,
        "regime":      final_labels,
        "regime_name": [REGIME_NAMES.get(r, f"regime_{r}") for r in final_labels],
        "raw_label":   raw_labels,
    }).set_index("date")

    # Add regime probabilities (soft assignments)
    probs = model.predict_proba(X_pca)
    for i in range(n_regimes):
        labels_df[f"prob_{i}"] = probs[:, regime_map.get(i, i)]

    # ── Step 7: Save everything ───────────────────────────────
    _save_model(model, pca, scaler, regime_map, save_path)
    labels_path = save_path / "regime_labels.parquet"
    labels_df.to_parquet(labels_path)

    # Print summary
    _print_regime_summary(labels_df, common_dates, config)

    log.info(f"✓ Regime labels saved → {labels_path}")
    return model, labels_df, pca, scaler


def predict_regime(
    state_vector: np.ndarray,
    save_dir: str = "models",
) -> dict:
    """
    Real-time regime prediction for a single new state vector.

    Args:
        state_vector: 1-d array of shape (128,) — current market state.
        save_dir:     Where the trained model is saved.

    Returns:
        Dict with keys: regime (int), name (str), probabilities (array),
                        confidence (float)
    """
    model, pca, scaler, regime_map = load_hmm(save_dir)
    if model is None:
        return {"regime": -1, "name": "unknown", "confidence": 0.0}

    x = state_vector.reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_pca    = pca.transform(x_scaled)

    raw       = model.predict(x_pca)[0]
    regime    = regime_map.get(int(raw), int(raw))
    probs     = model.predict_proba(x_pca)[0]
    confidence = float(probs[raw])

    return {
        "regime":        regime,
        "name":          REGIME_NAMES.get(regime, f"regime_{regime}"),
        "probabilities": probs.tolist(),
        "confidence":    confidence,
    }


def load_hmm(save_dir: str = "models"):
    """Loads trained HMM, PCA, scaler and regime map from disk."""
    import pickle
    path = Path(save_dir) / "hmm_regime_detector.pkl"
    if not path.exists():
        log.warning(f"No HMM model at {path} — run Phase 3 first")
        return None, None, None, {}
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    log.info(f"Loaded HMM from {path}")
    return bundle["model"], bundle["pca"], bundle["scaler"], bundle["regime_map"]


def load_regime_labels(save_dir: str = "models") -> Optional[pd.DataFrame]:
    """Loads the regime label DataFrame saved during training."""
    path = Path(save_dir) / "regime_labels.parquet"
    if not path.exists():
        log.warning("No regime labels found — run Phase 3 first")
        return None
    return pd.read_parquet(path)


# ── Internal helpers ──────────────────────────────────────────

def _load_and_align_states(states_dir: Path, config: dict):
    """
    Loads state vectors for all assets and aligns them to common dates.
    Returns a combined feature matrix and the common date index.
    """
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )

    frames = {}
    for asset in all_assets:
        sym  = asset["symbol"]
        safe = sym.replace("=","_").replace("-","_").replace("/","_")
        path = states_dir / f"{safe}_states.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            state_cols = [c for c in df.columns if c.startswith("state_")]
            df = df[state_cols]
            frames[sym] = df
            log.info(f"  Loaded {sym}: {df.shape}")
        else:
            log.warning(f"  Missing state vectors for {sym}")

    if not frames:
        return None, None

    # Find common dates across all available assets
    common_dates = None
    for df in frames.values():
        dates = set(df.index)
        common_dates = dates if common_dates is None else common_dates & dates

    common_dates = sorted(common_dates)
    log.info(f"  Common dates: {len(common_dates):,} ({common_dates[0]} → {common_dates[-1]})")

    # Build concatenated feature matrix: one block per asset
    blocks = []
    for sym, df in frames.items():
        block = df.reindex(common_dates).fillna(0).values
        blocks.append(block)

    combined = np.hstack(blocks)
    return combined, pd.DatetimeIndex(common_dates)


def _auto_label_regimes(
    raw_labels: np.ndarray,
    dates: pd.DatetimeIndex,
    config: dict,
) -> dict:
    """
    Assigns meaningful regime names to HMM states using price statistics.

    For each HMM state we compute:
      - Mean 20-day return     → positive = trending up, negative = down
      - Mean 20-day volatility → high = crisis, low = ranging or trending

    Returns a mapping {raw_label → final_label} where final labels are:
      0 = trending_up, 1 = trending_down, 2 = ranging, 3 = crisis
    """
    from src.features.indicators import load_features

    # Use BTC as the reference asset for labelling (most data, most volatile)
    ref_symbol = "BTC-USD"
    price_df   = load_features(ref_symbol, config["data"]["processed_dir"])

    if price_df is None:
        # Fallback: use raw labels as-is
        log.warning("  Cannot load BTC features for auto-labelling — using raw labels")
        return {i: i for i in range(N_REGIMES)}

    ret = price_df["return_1d"].reindex(dates).fillna(0)
    vol = price_df["volatility_20"].reindex(dates).fillna(price_df["volatility_20"].median())

    # Compute per-regime statistics
    n_states = len(np.unique(raw_labels))
    stats = {}
    for state in range(n_states):
        mask          = raw_labels == state
        stats[state]  = {
            "mean_ret":   ret.values[mask].mean(),
            "mean_vol":   vol.values[mask].mean(),
            "count":      mask.sum(),
        }
        log.info(f"  Raw state {state}: n={mask.sum():,}  "
                 f"ret={stats[state]['mean_ret']:+.4f}  "
                 f"vol={stats[state]['mean_vol']:.4f}")

    # Assign semantic labels for 3 regimes:
    # Sort all states by mean return
    sorted_by_ret = sorted(stats.keys(), key=lambda s: stats[s]["mean_ret"])
    
    regime_map = {}
    if len(sorted_by_ret) >= 3:
        regime_map[sorted_by_ret[-1]] = 0   # highest return = trending up
        regime_map[sorted_by_ret[0]]  = 1   # lowest return  = trending down
        regime_map[sorted_by_ret[1]]  = 2   # middle = ranging
    else:
        # Fallback if fewer than 3 states are predicted
        for i, s in enumerate(sorted_by_ret):
            regime_map[s] = i

    log.info(f"  Regime mapping: {regime_map}")
    return regime_map


def _save_model(model, pca, scaler, regime_map, save_path: Path):
    """Saves HMM model bundle as pickle."""
    import pickle
    bundle = {
        "model":      model,
        "pca":        pca,
        "scaler":     scaler,
        "regime_map": regime_map,
        "n_regimes":  N_REGIMES,
        "pca_dims":   PCA_DIMS,
    }
    path = save_path / "hmm_regime_detector.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"  HMM model saved → {path}")


def _print_regime_summary(labels_df: pd.DataFrame, dates, config: dict):
    """Prints a summary table of regime assignments."""
    GREEN  = "\033[92m"; RED = "\033[91m"; GRAY = "\033[90m"
    ORANGE = "\033[93m"; BOLD = "\033[1m"; RESET = "\033[0m"
    CYAN   = "\033[96m"

    REGIME_COLOUR_CODES = {
        "trending_up":   GREEN,
        "trending_down": RED,
        "ranging":       GRAY,
        "crisis":        ORANGE,
    }

    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  CARMS Regime Detection Summary{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"  Date range : {dates[0].date()} → {dates[-1].date()}")
    print(f"  Total days : {len(dates):,}")
    print()
    print(f"  {'Regime':<20} {'Days':>6} {'%':>6} {'Avg return':>12} {'Avg vol':>10}")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*12} {'─'*10}")

    from src.features.indicators import load_features
    price_df = load_features("BTC-USD", config["data"]["processed_dir"])

    for name in ["trending_up", "trending_down", "ranging"]:
        mask = labels_df["regime_name"] == name
        n    = mask.sum()
        pct  = n / len(labels_df) * 100
        col  = REGIME_COLOUR_CODES.get(name, RESET)

        ret_str = vol_str = "N/A"
        if price_df is not None and n > 0:
            ret = price_df["return_1d"].reindex(labels_df.index[mask])
            vol = price_df["volatility_20"].reindex(labels_df.index[mask])
            ret_str = f"{ret.mean()*100:+.2f}%"
            vol_str = f"{vol.mean():.3f}"

        print(f"  {col}{name:<20}{RESET} {n:>6,} {pct:>5.1f}%  {ret_str:>12} {vol_str:>10}")

    print(f"{BOLD}{CYAN}{'─'*55}{RESET}\n")


def _n_assets(config: dict) -> int:
    return (len(config["assets"]["forex"])
            + len(config["assets"]["crypto"])
            + len(config["assets"]["commodities"]))