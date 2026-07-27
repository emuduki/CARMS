from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp

from src.utils.logger import get_logger
from src.utils.data_splits import get_test_start
from src.regime.constants import REGIME_NAMES, REGIME_COLOURS
from src.regime.regime_analyser import analyse_regimes

log = get_logger(__name__)

N_REGIMES   = 3
PCA_DIMS    = 16
N_ITER      = 200
RANDOM_SEED = 42


def train_hmm(
    config: dict,
    save_dir: str = "models",
    n_regimes: int = N_REGIMES,
) -> tuple:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        log.error("hmmlearn not installed — run: pip install hmmlearn")
        return None, None, None, None

    states_dir = Path(config["data"]["processed_dir"]) / "states"
    save_path  = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading state vectors from Phase 2...")
    all_states, common_dates = _load_and_align_states(states_dir, config)

    if all_states is None or len(all_states) == 0:
        log.error("No state vectors found — run Phase 2 first")
        return None, None, None, None

    if len(common_dates) < 100:
        log.warning("=" * 65)
        log.warning("CRITICAL WARNING: Only %d common dates for regime detection.", len(common_dates))
        log.warning("Training HMM on so few dates will result in a degenerate solution.")
        log.warning("Please run full pipeline without '--quick':")
        log.warning("  python main.py --phase 1")
        log.warning("  python main.py --phase 2")
        log.warning("=" * 65)
        raise RuntimeError(f"Insufficient data for HMM: {len(common_dates)} common dates")

    log.info("Loaded %,d observations × %d features", all_states.shape[0], all_states.shape[1])

    target_dims = PCA_DIMS * _n_assets(config)
    log.info("Reducing dimensions: %d → %d...", all_states.shape[1], target_dims)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_states)

    max_components = min(target_dims, X_scaled.shape[1], X_scaled.shape[0])
    if max_components < target_dims:
        log.warning(
            "Only %d samples for PCA; reducing n_components from %d to %d",
            X_scaled.shape[0], target_dims, max_components,
        )

    pca = PCA(n_components=max_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    variance_explained = pca.explained_variance_ratio_.cumsum()[-1]
    log.info("PCA variance explained: %.1f%%", variance_explained * 100)
    log.info("Feature matrix: %s", X_pca.shape)

    test_start = get_test_start(config)
    train_mask = common_dates < test_start
    n_train = int(train_mask.sum())
    n_test = int((~train_mask).sum())
    log.info("Train period: %,d days (< %s)", n_train, test_start.date())
    log.info("Test period : %,d days (>= %s)", n_test, test_start.date())

    X_fit = X_pca if n_train < 100 else X_pca[train_mask]

    log.info("Training Gaussian HMM (%d states, %d iterations)...", n_regimes, N_ITER)
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    model.fit(X_fit)
    log.info("HMM converged: %s", model.monitor_.converged)
    log.info("Log-likelihood: %.2f", model.score(X_fit))

    log.info("Decoding regime sequence (causal forward filter)...")
    raw_labels, filtered_probs = _forward_filter_regimes(model, X_pca)

    log.info("Auto-labeling regimes using price statistics...")
    regime_map = _auto_label_regimes(raw_labels, common_dates, config)
    final_labels = np.array([regime_map.get(r, r) for r in raw_labels])

    labels_df = pd.DataFrame({
        "date": common_dates,
        "regime": final_labels,
        "regime_name": [REGIME_NAMES.get(r, f"regime_{r}") for r in final_labels],
        "raw_label": raw_labels,
    }).set_index("date")

    for i in range(n_regimes):
        mapped = regime_map.get(i, i)
        labels_df[f"prob_{mapped}"] = filtered_probs[:, i]

    _save_model(model, pca, scaler, regime_map, save_path, n_regimes)
    labels_path = save_path / "regime_labels.parquet"
    labels_df.to_parquet(labels_path)

    _print_regime_summary(labels_df, common_dates, config)
    log.info("Regime labels saved → %s", labels_path)
    return model, labels_df, pca, scaler


def predict_regime(
    state_vector: np.ndarray,
    save_dir: str = "models",
) -> dict:
    model, pca, scaler, regime_map = load_hmm(save_dir)
    if model is None:
        return {"regime": -1, "name": "unknown", "confidence": 0.0}

    x = state_vector.reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    raw = model.predict(x_pca)[0]
    regime = regime_map.get(int(raw), int(raw))
    probs = model.predict_proba(x_pca)[0]
    confidence = float(probs[raw])

    return {
        "regime": regime,
        "name": REGIME_NAMES.get(regime, f"regime_{regime}"),
        "probabilities": probs.tolist(),
        "confidence": confidence,
    }


def load_hmm(save_dir: str = "models"):
    import pickle
    path = Path(save_dir) / "hmm_regime_detector.pkl"
    if not path.exists():
        log.warning("No HMM model at %s — run Phase 3 first", path)
        return None, None, None, {}
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    log.info("Loaded HMM from %s", path)
    return bundle["model"], bundle["pca"], bundle["scaler"], bundle["regime_map"]


def load_regime_labels(save_dir: str = "models") -> Optional[pd.DataFrame]:
    path = Path(save_dir) / "regime_labels.parquet"
    if not path.exists():
        log.warning("No regime labels found — run Phase 3 first")
        return None
    return pd.read_parquet(path)


def _forward_filter_regimes(model, X_pca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_prob = model._compute_log_likelihood(X_pca)
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)

    n_samples, n_components = log_prob.shape
    filtered_probs = np.zeros((n_samples, n_components))
    labels = np.zeros(n_samples, dtype=int)

    log_alpha = log_start + log_prob[0]
    log_alpha -= logsumexp(log_alpha)
    filtered_probs[0] = np.exp(log_alpha)
    labels[0] = int(np.argmax(log_alpha))

    for t in range(1, n_samples):
        log_alpha = logsumexp(log_alpha[:, np.newaxis] + log_trans, axis=0) + log_prob[t]
        log_alpha -= logsumexp(log_alpha)
        filtered_probs[t] = np.exp(log_alpha)
        labels[t] = int(np.argmax(log_alpha))

    return labels, filtered_probs


def _load_and_align_states(states_dir: Path, config: dict):
    all_assets = (
        config["assets"]["forex"]
        + config["assets"]["crypto"]
        + config["assets"]["commodities"]
    )

    frames = {}
    for asset in all_assets:
        sym = asset["symbol"]
        safe = sym.replace("=", "_").replace("-", "_").replace("/", "_")
        path = states_dir / f"{safe}_states.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            state_cols = [c for c in df.columns if c.startswith("state_")]
            df = df[state_cols]
            frames[sym] = df
            log.info("Loaded %s: %s", sym, df.shape)
        else:
            log.warning("Missing state vectors for %s", sym)

    if not frames:
        return None, None

    common_dates = None
    for df in frames.values():
        dates = set(df.index)
        common_dates = dates if common_dates is None else common_dates & dates

    common_dates = sorted(common_dates)
    log.info("Common dates: %,d (%s → %s)", len(common_dates), common_dates[0], common_dates[-1])

    blocks = []
    for sym, df in frames.items():
        block = df.reindex(common_dates).fillna(0).values
        blocks.append(block)

    combined = np.hstack(blocks)
    return combined, pd.DatetimeIndex(common_dates)


def _auto_label_regimes(raw_labels, dates, config):
    from src.features.indicators import load_features

    ref_symbol = "BTC-USD"
    price_df = load_features(ref_symbol, config["data"]["processed_dir"])

    if price_df is None:
        log.warning("Cannot load BTC features for auto-labelling — using raw labels")
        return {i: i for i in range(N_REGIMES)}

    ret = price_df["return_1d"].reindex(dates).fillna(0)
    vol = price_df["volatility_20"].reindex(dates).fillna(price_df["volatility_20"].median())

    n_states = len(np.unique(raw_labels))
    stats = {}
    for state in range(n_states):
        mask = raw_labels == state
        stats[state] = {
            "mean_ret": ret.values[mask].mean(),
            "mean_vol": vol.values[mask].mean(),
            "count": mask.sum(),
        }
        log.info("Raw state %d: n=%,d  ret=%.6f  vol=%.4f", state, mask.sum(), stats[state]["mean_ret"], stats[state]["mean_vol"])

    sorted_by_ret = sorted(stats.keys(), key=lambda s: stats[s]["mean_ret"])

    regime_map = {}
    if len(sorted_by_ret) >= 3:
        regime_map[sorted_by_ret[-1]] = 0
        regime_map[sorted_by_ret[0]] = 1
        regime_map[sorted_by_ret[1]] = 2
    else:
        for i, s in enumerate(sorted_by_ret):
            regime_map[s] = i

    log.info("Regime mapping: %s", regime_map)
    return regime_map


def _save_model(model, pca, scaler, regime_map, save_path: Path, n_regimes: int = N_REGIMES):
    import pickle
    bundle = {
        "model": model,
        "pca": pca,
        "scaler": scaler,
        "regime_map": regime_map,
        "n_regimes": n_regimes,
        "pca_dims": PCA_DIMS,
    }
    path = save_path / "hmm_regime_detector.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("HMM model saved → %s", path)


def _print_regime_summary(labels_df: pd.DataFrame, dates, config: dict):
    BOLD = "\033[1m"; RESET = "\033[0m"; CYAN = "\033[96m"
    ORANGE = "\033[93m"; GREEN = "\033[92m"; RED = "\033[91m"; GRAY = "\033[90m"

    REGIME_COLOUR_CODES = {
        "trending_up": GREEN, "trending_down": RED, "ranging": GRAY, "crisis": ORANGE,
    }

    print(f"\n{BOLD}{CYAN}{'─' * 55}{RESET}")
    print(f"{BOLD}{CYAN}  CARMS Regime Detection Summary{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 55}{RESET}")
    print(f"  Date range : {dates[0].date()} → {dates[-1].date()}")
    print(f"  Total days : {len(dates):,}")
    print()
    print(f"  {'Regime':<20} {'Days':>6} {'%':>6} {'Avg return':>12} {'Avg vol':>10}")
    print(f"  {'─' * 20} {'─' * 6} {'─' * 6} {'─' * 12} {'─' * 10}")

    from src.features.indicators import load_features

    price_df = load_features("BTC-USD", config["data"]["processed_dir"])

    for name in ["trending_up", "trending_down", "ranging"]:
        mask = labels_df["regime_name"] == name
        n = mask.sum()
        pct = n / len(labels_df) * 100
        col = REGIME_COLOUR_CODES.get(name, RESET)

        ret_str = vol_str = "N/A"
        if price_df is not None and n > 0:
            ret = price_df["return_1d"].reindex(labels_df.index[mask])
            vol = price_df["volatility_20"].reindex(labels_df.index[mask])
            ret_str = f"{ret.mean() * 100:+.2f}%"
            vol_str = f"{vol.mean():.3f}"

        print(f"  {col}{name:<20}{RESET} {n:>6,} {pct:>5.1f}%  {ret_str:>12} {vol_str:>10}")

    print(f"{BOLD}{CYAN}{'─' * 55}{RESET}\n")


def _n_assets(config: dict) -> int:
    return len(config["assets"]["forex"]) + len(config["assets"]["crypto"]) + len(config["assets"]["commodities"])