"""
utils/logger.py — Centralised logging and config loading for CARMS.
"""
 
import yaml
import logging
import sys
from pathlib import Path
from datetime import datetime
 
 
# ── Colour codes for terminal output ─────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GRAY   = "\033[90m"
 
 
def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Returns a logger that writes to both console (coloured) and a daily log file.
 
    Usage:
        from src.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Starting pipeline...")
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
 
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
 
    if logger.handlers:
        return logger  # Avoid adding duplicate handlers on re-import
 
    # ── Console handler (coloured) ────────────────────────────
    class ColouredFormatter(logging.Formatter):
        COLOURS = {
            logging.DEBUG:    GRAY,
            logging.INFO:     CYAN,
            logging.WARNING:  YELLOW,
            logging.ERROR:    RED,
            logging.CRITICAL: BOLD + RED,
        }
 
        def format(self, record):
            colour = self.COLOURS.get(record.levelno, RESET)
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            module = record.name.split(".")[-1]
            msg = record.getMessage()
            return f"{GRAY}{ts}{RESET} {colour}{record.levelname:<8}{RESET} {GRAY}[{module}]{RESET} {msg}"
 
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColouredFormatter())
    logger.addHandler(console)
 
    # ── File handler (plain text) ─────────────────────────────
    log_path = Path(log_dir) / f"carms_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)
 
    return logger
 
 
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Loads YAML config. Tries config.local.yaml first (has real API keys),
    falls back to config.yaml (safe defaults).
 
    Returns:
        dict: Parsed config dictionary.
    """
    local_path = Path(config_path).parent / "config.local.yaml"
    path = local_path if local_path.exists() else Path(config_path)
 
    with open(path, "r") as f:
        config = yaml.safe_load(f)
 
    return config
 
 
def print_banner():
    """Prints a startup banner to the console."""
    banner = f"""
{CYAN}{BOLD}
  ██████╗ █████╗ ██████╗ ███╗   ███╗███████╗
 ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝
 ██║     ███████║██████╔╝██╔████╔██║███████╗
 ██║     ██╔══██║██╔══██╗██║╚██╔╝██║╚════██║
 ╚██████╗██║  ██║██║  ██║██║ ╚═╝ ██║███████║
  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
{RESET}
{GRAY}  Cross-Asset Regime-Aware Multi-Agent Trading System{RESET}
{GRAY}  Phase 1 — Data Pipeline & Feature Engineering{RESET}
"""
    print(banner)