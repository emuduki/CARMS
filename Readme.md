CARMS — Cross-Asset Regime-Aware Multi-Agent Trading System
A multi-modal RL trading system that fuses Time Series, NLP, and Computer Vision
to trade Forex, Crypto, and Gold using a regime-aware meta-controller.
Project Structure
carms/
├── configs/            # All configuration (assets, paths, hyperparams)
├── data/
│   ├── raw/            # Raw OHLCV data from APIs
│   ├── processed/      # Feature-engineered datasets
│   └── charts/         # Candlestick chart images (CNN input)
├── src/
│   ├── ingestion/      # Data download & storage (Phase 1)
│   ├── features/       # Technical indicators & feature engineering (Phase 1)
│   ├── encoders/       # TFT, FinBERT, CNN encoders (Phase 2)
│   ├── agents/         # RL specialist agents + meta-controller (Phase 4-5)
│   └── utils/          # Logging, helpers, plotting
├── models/             # Saved model checkpoints
├── notebooks/          # Exploration & visualization notebooks
├── tests/              # Unit tests
├── logs/               # Training & pipeline logs
├── requirements.txt
└── main.py             # Entry point
Phases
PhaseDescriptionStatus1Data pipeline & feature engineering✅ MVP Ready2Modality encoders (TFT, FinBERT, CNN)🔜 Next3Regime detection (HMM)🔜4Specialist RL agents🔜5Meta-controller & portfolio manager🔜6Paper trading & research paper🔜
Quickstart
bash# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your assets and API keys
cp configs/config.yaml configs/config.local.yaml
# Edit configs/config.local.yaml with your API keys

# 3. Run Phase 1 — download data and build features
python main.py --phase 1

# 4. Verify outputs
ls data/raw/        # OHLCV parquet files
ls data/processed/  # Feature matrices
ls data/charts/     # Candlestick images
Paper Trading (Phase 6)

Forex/Gold: OANDA demo account (free at oanda.com)
Crypto: Binance Testnet (free at testnet.binance.vision)


⚠️ This system is for research purposes only. Never trade real money based
solely on an ML model. Always paper trade first.