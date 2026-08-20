"""Chronological split and refit schedule primitives."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import INITIAL_HISTORY_END, TEST_START, VALIDATION_END, VALIDATION_START


@dataclass(frozen=True)
class PredictivePeriods:
    """Resolved observed dates for the three chronological experiment periods."""

    initial_history: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex


def _validate_index(data: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data index must be a DatetimeIndex.")
    index = pd.DatetimeIndex(data.index, name="Date")
    if index.tz is not None:
        raise ValueError("data index must be timezone-naive.")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("data dates must be unique and sorted ascending.")
    if len(index) < 4:
        raise ValueError("data must contain enough observations for all periods.")
    return index


def split_periods(data: pd.DataFrame) -> PredictivePeriods:
    """Resolve the fixed initial-history, validation, and final-test periods."""

    index = _validate_index(data)
    initial = index[index <= INITIAL_HISTORY_END]
    validation = index[(index >= VALIDATION_START) & (index <= VALIDATION_END)]
    test = index[index >= TEST_START]
    if len(initial) < 2:
        raise ValueError("initial estimation history is missing or too short.")
    if len(validation) < 2:
        raise ValueError("validation period is missing or too short.")
    if len(test) < 2:
        raise ValueError("final test period is missing or too short.")
    if not (initial[-1] < validation[0] < test[0]):
        raise ValueError("chronological periods must be strictly ordered and non-overlapping.")
    if validation[-1] >= test[0]:
        raise ValueError("validation and test periods overlap.")
    return PredictivePeriods(initial, validation, test)


def is_monthly_refit(
    previous_date: pd.Timestamp | None, current_date: pd.Timestamp
) -> bool:
    """Return true for the first decision or first observed decision of a new month."""

    current = pd.Timestamp(current_date)
    if previous_date is None:
        return True
    previous = pd.Timestamp(previous_date)
    if current <= previous:
        raise ValueError("current_date must be later than previous_date.")
    return (current.year, current.month) != (previous.year, previous.month)
