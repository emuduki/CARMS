"""
main.py — CARMS entry point.

Usage:
    python main.py --phase 1                          # Data pipeline
    python main.py --phase 1 --validate               # Phase 1 + validate
    python main.py --phase 2                          # Train encoders (CPU)
    python main.py --phase 2 --device cuda            # Train encoders (GPU)
    python main.py --phase 2 --symbol BTC-USD         # Single asset only
    python main.py --phase 2 --validate               # Phase 2 + validate
    python main.py --validate                         # Validate current outputs
    python main.py --phase 1 --quick                  # Quick test run
"""

import argparse
import sys
from pathlib import Path

# Configure stdout and stderr to use UTF-8 encoding to avoid Windows cp1252 print crashes
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.utils.logger import get_logger, load_config, print_banner
from src.utils.validator import validate_phase1


def run_phase1(config: dict, quick: bool = False):
    from src.ingestion.downloader import download_all_assets
    from src.ingestion.news_fetcher import fetch_news, fetch_macro_data
    from src.features.indicators import compute_all_features
    from src.features.chart_generator import generate_all_charts

    log = get_logger("phase1")
    max_imgs = 100 if quick else None

    log.info("=" * 55)
    log.info("PHASE 1 — Data Pipeline & Feature Engineering")
    log.info("=" * 55)

    log.info("Step 1/5 — Downloading OHLCV price data...")
    asset_data = download_all_assets(config)
    if not asset_data:
        log.error("No asset data — check internet connection")
        sys.exit(1)

    log.info("Step 2/5 — Fetching news headlines...")
    fetch_news(config)

    log.info("Step 3/5 — Fetching macro indicators from FRED...")
    fetch_macro_data(config)

    log.info("Step 4/5 — Computing technical indicators...")
    processed = compute_all_features(asset_data, processed_dir=config["data"]["processed_dir"])

    log.info("Step 5/5 — Generating candlestick chart images...")
    generate_all_charts(processed, config, max_images_per_asset=max_imgs)

    log.info("Phase 1 complete!")
    return processed


def run_phase2(config: dict, device: str = "cpu", symbol: str = None):
    from src.encoders.phase2_runner import run_phase2 as _run
    _run(config, device=device, symbol=symbol)


def main():
    parser = argparse.ArgumentParser(
        description="CARMS — Cross-Asset Regime-Aware Multi-Agent Trading System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--phase",    type=int, choices=[1,2,3,4,5,6])
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--quick",    action="store_true", help="Limit images to 100 per asset")
    import torch
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device",   default=default_device, choices=["cpu","cuda"], help="Training device")
    parser.add_argument("--symbol",     default=None,  help="Train only this symbol (Phase 2)")
    parser.add_argument("--n_regimes",  default=3, type=int, help="Number of HMM regimes (Phase 3)")
    parser.add_argument("--agent",      default=None,  help="Train single agent: forex, crypto, gold (Phase 4)")
    parser.add_argument("--live",       action="store_true", help="Live paper trading via APIs (Phase 5)")
    parser.add_argument("--capital",    default=10000.0, type=float, help="Starting paper capital (Phase 5)")
    parser.add_argument("--n_ticks",    default=252, type=int, help="Simulation days (Phase 5)")
    parser.add_argument("--dashboard",  action="store_true", help="Launch live dashboard (Phase 5)")
    parser.add_argument("--config",   default="configs/config.yaml")

    args = parser.parse_args()

    print_banner()
    config = load_config(args.config)
    # Inject project root so all phases can resolve relative paths correctly
    config["_base_dir"] = str(Path(args.config).parent.parent.resolve())
    log    = get_logger("main")

    for dir_key in ["raw_dir", "processed_dir", "charts_dir"]:
        Path(config["data"][dir_key]).mkdir(parents=True, exist_ok=True)
    Path(config.get("model_dir", "models")).mkdir(parents=True, exist_ok=True)

    if args.phase == 1:
        run_phase1(config, quick=args.quick)
        if args.validate:
            validate_phase1(config)

    elif args.phase == 2:
        run_phase2(config, device=args.device, symbol=args.symbol)
        if args.validate:
            from src.encoders.phase2_runner import validate_phase2
            validate_phase2(config)

    elif args.phase == 3:
        from src.regime.phase3_runner import run_phase3, validate_phase3
        n_regimes = getattr(args, 'n_regimes', 4)
        run_phase3(config, save_dir=config.get("model_dir","models"), n_regimes=n_regimes)
        if args.validate:
            validate_phase3(config, config.get("model_dir","models"))

    elif args.validate and not args.phase:
        validate_phase1(config)
        try:
            from src.encoders.phase2_runner import validate_phase2
            validate_phase2(config)
        except Exception:
            pass

    elif args.phase is None and not args.validate:
        log.info("No phase specified.")
        log.info("Run: python main.py --phase 1   (data pipeline)")
        log.info("     python main.py --phase 2   (train encoders)")
        parser.print_help()

    elif args.phase == 4:
        from src.agents.phase4_runner import run_phase4, validate_phase4
        agent = getattr(args, 'agent', None)
        run_phase4(config, save_dir=config.get("model_dir","models"),
                   device=args.device, agent=agent)
        if args.validate:
            validate_phase4(config, config.get("model_dir","models"))

    elif args.phase == 5:
        from src.live.phase5_runner import run_phase5, validate_phase5
        live      = getattr(args, 'live',      False)
        capital   = getattr(args, 'capital',   10000.0)
        n_ticks   = getattr(args, 'n_ticks',   252)
        dashboard = getattr(args, 'dashboard', False)
        if dashboard:
            from src.live.dashboard import run_dashboard
            run_dashboard(save_dir=config.get("model_dir", "models"))
        else:
            run_phase5(config,
                       save_dir=config.get("model_dir", "models"),
                       device=args.device,
                       capital=capital,
                       live=live,
                       n_ticks=n_ticks)
            if args.validate:
                validate_phase5(config, config.get("model_dir", "models"))

    else:
        log.warning(f"Phase {args.phase} not yet implemented")
        log.info("Available: --phase 1 to --phase 5")


if __name__ == "__main__":
    main()