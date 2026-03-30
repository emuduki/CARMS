
import argparse
import sys
from pathlib import Path
 
from src.utils.logger import get_logger, load_config, print_banner
from src.utils.validator import validate_phase1
 
 
def run_phase1(config: dict, quick: bool = False):
    """
    Executes the full Phase 1 data pipeline:
      1. Download OHLCV data for all assets
      2. Fetch news headlines (requires NewsAPI key)
      3. Fetch FRED macro data (requires FRED API key)
      4. Compute technical indicators
      5. Generate candlestick chart images
    """
    from src.ingestion.downloader import download_all_assets, load_all_assets
    from src.ingestion.news_fetcher import fetch_news, fetch_macro_data
    from src.features.indicators import compute_all_features
    from src.features.chart_generator import generate_all_charts
 
    log = get_logger("phase1")
    max_imgs = 100 if quick else None    # Limit images in quick mode
 
    log.info("=" * 55)
    log.info("PHASE 1 — Data Pipeline & Feature Engineering")
    log.info("=" * 55)
 
    #  Step 1: Download OHLCV 
    log.info("Step 1/5 — Downloading OHLCV price data...")
    asset_data = download_all_assets(config)
 
    if not asset_data:
        log.error("No asset data downloaded — check your internet connection")
        sys.exit(1)
 
    #  Step 2: News headlines 
    log.info("Step 2/5 — Fetching news headlines...")
    fetch_news(config)
 
    #  Step 3: FRED macro data 
    log.info("Step 3/5 — Fetching macro indicators from FRED...")
    fetch_macro_data(config)
 
    #  Step 4: Technical indicators
    log.info("Step 4/5 — Computing technical indicators...")
    processed = compute_all_features(
        asset_data,
        processed_dir=config["data"]["processed_dir"],
    )
 
    #  Step 5: Chart images
    log.info("Step 5/5 — Generating candlestick chart images...")
    generate_all_charts(processed, config, max_images_per_asset=max_imgs)
 
    log.info("Phase 1 complete!")
    return processed
 
 
def main():
    parser = argparse.ArgumentParser(
        description="CARMS — Cross-Asset Regime-Aware Multi-Agent Trading System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Which phase to run (1 = data pipeline)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate Phase 1 outputs and print summary"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run: limit chart images to 100 per asset (for testing)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)"
    )
 
    args = parser.parse_args()
 
    #  Startup 
    print_banner()
    config = load_config(args.config)
    log    = get_logger("main")
 
    #  Ensure output dirs exist 
    for dir_key in ["raw_dir", "processed_dir", "charts_dir"]:
        Path(config["data"][dir_key]).mkdir(parents=True, exist_ok=True)
 
    #  Route to phase 
    if args.phase == 1:
        run_phase1(config, quick=args.quick)
        if args.validate:
            validate_phase1(config)
 
    elif args.validate and not args.phase:
        validate_phase1(config)
 
    elif args.phase is None and not args.validate:
        log.info("No phase specified. Run: python main.py --phase 1")
        log.info("Or:                       python main.py --validate")
        parser.print_help()
 
    else:
        log.warning(f"Phase {args.phase} not yet implemented — coming soon!")
        log.info("Currently available: --phase 1")
 
 
if __name__ == "__main__":
    main()