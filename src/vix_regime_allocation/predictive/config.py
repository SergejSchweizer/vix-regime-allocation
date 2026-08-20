"""Immutable configuration for the causal predictive extension."""

from __future__ import annotations

import pandas as pd

ASSET_ORDER: tuple[str, str, str] = ("TLT", "GLD", "SPY")
SUPPORTED_STATE_COUNTS: tuple[int, int] = (2, 3)
SWITCH_HURDLES_BPS: tuple[float, float, float, float] = (0.0, 5.0, 10.0, 20.0)
ONE_WAY_COST_BPS: float = 5.0
PROBABILITY_TOL: float = 1e-10
SELECTION_TOL: float = 1e-12

INITIAL_HISTORY_END = pd.Timestamp("2014-12-31")
VALIDATION_START = pd.Timestamp("2015-01-01")
VALIDATION_END = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")

FAMILY_PRIORITY: tuple[str, str] = ("markov", "hmm")
