"""
regime/constants.py — Shared constants for regime detection.
"""

REGIME_NAMES = {
    0: "trending_up",
    1: "trending_down",
    2: "ranging",
    3: "crisis",
}

REGIME_COLOURS = {
    0: "#1D9E75",   # green
    1: "#E24B4A",   # red
    2: "#888780",   # gray
    3: "#FF8C00",   # orange
}

REGIME_COLOUR_CODES = {
    "trending_up":   "\033[92m",   # green
    "trending_down": "\033[91m",   # red
    "ranging":       "\033[90m",   # gray
    "crisis":        "\033[93m",   # orange
}