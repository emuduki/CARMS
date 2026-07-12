"""Train / test date splitting utilities for CARMS."""

from __future__ import annotations

from typing import Literal

import pandas as pd

SplitMode = Literal["train", "eval", "all"]


def get_test_start(config: dict) -> pd.Timestamp:
    """Returns the hold-out test period start date from config."""
    return pd.Timestamp(config["data"].get("test_start", "2024-01-01"))


def filter_by_split(
    df: pd.DataFrame,
    config: dict,
    mode: SplitMode = "all",
) -> pd.DataFrame:
    """
    Filters a date-indexed DataFrame to train or eval period.

    train : dates strictly before test_start
    eval  : dates from test_start onward
    all   : no filtering
    """
    if mode == "all" or df.empty:
        return df

    test_start = get_test_start(config)
    if mode == "train":
        return df.loc[df.index < test_start]
    if mode == "eval":
        return df.loc[df.index >= test_start]
    raise ValueError(f"Unknown split mode: {mode!r}")
