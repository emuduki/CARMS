"""
features/chart_generator.py — Generates candlestick chart images for CNN encoder.
 
Each image = a 20-candle window saved as 64x64 grayscale PNG.
Falls back to a matplotlib-only renderer if mplfinance fails.
"""
 
import io
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
from src.utils.logger import get_logger
 
log = get_logger(__name__)
 
try:
    import mplfinance as mpf
    HAS_MPF = True
except ImportError:
    HAS_MPF = False
    log.warning("mplfinance not installed — using fallback renderer. Run: pip install mplfinance")
 
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    log.warning("Pillow not installed — chart images disabled. Run: pip install Pillow")
 
 
def generate_chart_dataset(
    df: pd.DataFrame,
    symbol: str,
    charts_dir: str = "data/charts",
    window_size: int = 20,
    image_size: int = 64,
    stride: int = 1,
    max_images: Optional[int] = None,
) -> pd.DataFrame:
    """
    Slides a window over price data generating one chart image per step.
    Returns metadata DataFrame with image paths and direction labels.
    """
    EMPTY_DF = pd.DataFrame(columns=["date", "symbol", "image_path", "label_1d", "label_5d", "return_1d"])
 
    if not HAS_PIL:
        log.error("Pillow required — run: pip install Pillow")
        return EMPTY_DF
 
    out_dir = Path(charts_dir) / _safe(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    df_plot = df[[c for c in ohlcv_cols if c in df.columns]].copy()
    df_plot = df_plot.dropna(subset=["close"])
 
    # Ensure volume column exists (needed for mplfinance even if zero)
    if "volume" not in df_plot.columns:
        df_plot["volume"] = 0.0
 
    n = len(df_plot)
    if n < window_size + 5:
        log.warning(f"  Not enough rows for {symbol} ({n} rows, need {window_size+5}) — skipping charts")
        return EMPTY_DF
 
    indices = list(range(window_size, n - 5, stride))
    if max_images:
        indices = indices[:max_images]
 
    log.info(f"Generating {len(indices):,} chart images for {symbol}...")
 
    records    = []
    saved      = 0
    render_err = 0
 
    for i in indices:
        window = df_plot.iloc[i - window_size: i].copy()
        date   = df_plot.index[i]
 
        ret_1d = (df_plot["close"].iloc[i] - df_plot["close"].iloc[i-1]) / df_plot["close"].iloc[i-1]
        ret_5d = (df_plot["close"].iloc[min(i+4, n-1)] - df_plot["close"].iloc[i]) / df_plot["close"].iloc[i]
 
        fname = f"{date.strftime('%Y%m%d_%H%M') if hasattr(date, 'hour') else date.strftime('%Y%m%d')}.png"
        fpath = out_dir / fname
 
        if not fpath.exists():
            arr = _render_chart(window, image_size)
            if arr is not None:
                _save_image(arr, fpath)
                saved += 1
            else:
                render_err += 1
 
        records.append({
            "date":       date,
            "symbol":     symbol,
            "image_path": str(fpath),
            "label_1d":   int(ret_1d > 0),
            "label_5d":   int(ret_5d > 0),
            "return_1d":  round(float(ret_1d), 6),
        })
 
    if render_err > 0:
        log.warning(f"  {render_err} render failures (check mplfinance install)")
 
    if not records:
        return EMPTY_DF
 
    meta_df   = pd.DataFrame(records)
    meta_path = Path(charts_dir) / f"{_safe(symbol)}_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
 
    balance = meta_df["label_1d"].mean()
    log.info(f"  ✓ {saved:,} images saved → {out_dir}")
    log.info(f"  Metadata → {meta_path.name}  ({len(meta_df):,} records)")
    log.info(f"  Label balance (1d up): {balance:.1%}")
 
    return meta_df
 
 
def generate_all_charts(
    asset_data: dict[str, pd.DataFrame],
    config: dict,
    max_images_per_asset: Optional[int] = 500,
) -> dict[str, pd.DataFrame]:
    charts_dir  = config["data"]["charts_dir"]
    window_size = config["features"]["chart_image"]["window_size"]
    image_size  = config["features"]["chart_image"]["image_size"]
 
    results = {}
    for symbol, df in asset_data.items():
        try:
            meta = generate_chart_dataset(
                df, symbol,
                charts_dir=charts_dir,
                window_size=window_size,
                image_size=image_size,
                max_images=max_images_per_asset,
            )
            if not meta.empty:
                results[symbol] = meta
        except Exception as e:
            log.error(f"Chart generation failed for {symbol}: {e}")
    return results
 
 
def load_chart_metadata(symbol: str, charts_dir: str = "data/charts") -> Optional[pd.DataFrame]:
    path = Path(charts_dir) / f"{_safe(symbol)}_metadata.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty or len(df.columns) < 2:
            return None
        return df
    except Exception:
        return None
 
 
# ── Renderers ─────────────────────────────────────────────────
 
def _render_chart(window: pd.DataFrame, image_size: int) -> Optional[np.ndarray]:
    """Tries mplfinance first, falls back to plain matplotlib."""
    arr = None
    if HAS_MPF:
        arr = _render_mpf(window, image_size)
    if arr is None:
        arr = _render_fallback(window, image_size)
    return arr
 
 
def _render_mpf(window: pd.DataFrame, image_size: int) -> Optional[np.ndarray]:
    """mplfinance candlestick renderer."""
    try:
        has_real_vol = window["volume"].sum() > 0
        fig, _ = mpf.plot(
            window,
            type="candle",
            style="charles",
            volume=has_real_vol,
            returnfig=True,
            figsize=(image_size / 72, image_size / 72),
            axisoff=True,
            tight_layout=True,
        )
        return _fig_to_array(fig, image_size)
    except Exception as e:
        log.debug(f"mplfinance render error: {e}")
        return None
 
 
def _render_fallback(window: pd.DataFrame, image_size: int) -> Optional[np.ndarray]:
    """
    Pure matplotlib candlestick fallback — works without mplfinance.
    Draws simple OHLC bars manually.
    """
    try:
        fig, ax = plt.subplots(figsize=(image_size / 72, image_size / 72))
        ax.set_axis_off()
 
        n      = len(window)
        closes = window["close"].values
        opens  = window["open"].values
        highs  = window["high"].values
        lows   = window["low"].values
        x      = np.arange(n)
        width  = 0.6
 
        for i in range(n):
            color = "#26a69a" if closes[i] >= opens[i] else "#ef5350"
            # Candle body
            body_bottom = min(opens[i], closes[i])
            body_height = abs(closes[i] - opens[i])
            ax.add_patch(mpatches.Rectangle(
                (i - width/2, body_bottom), width, max(body_height, 1e-8),
                color=color, zorder=2
            ))
            # Wicks
            ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5, zorder=1)
 
        ax.set_xlim(-1, n)
        ax.set_ylim(lows.min() * 0.999, highs.max() * 1.001)
        fig.tight_layout(pad=0)
 
        return _fig_to_array(fig, image_size)
    except Exception as e:
        log.debug(f"Fallback render error: {e}")
        return None
 
 
def _fig_to_array(fig, image_size: int) -> Optional[np.ndarray]:
    """Converts a matplotlib figure to a grayscale numpy array."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = PILImage.open(buf).convert("L")
        img = img.resize((image_size, image_size), PILImage.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        plt.close(fig)
        buf.close()
        return arr
    except Exception as e:
        log.debug(f"fig_to_array error: {e}")
        plt.close(fig)
        return None
 
 
def _save_image(arr: np.ndarray, path: Path):
    PILImage.fromarray(arr).save(path, format="PNG")
 
 
def _safe(symbol: str) -> str:
    return symbol.replace("=", "_").replace("-", "_").replace("/", "_")