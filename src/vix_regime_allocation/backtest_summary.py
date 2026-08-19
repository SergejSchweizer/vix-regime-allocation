"""Step 5 aligned portfolio comparison and shared performance summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .backtest import ROTATION_DETAIL_COLUMNS
from .benchmarks import EQUAL_WEIGHT_NAME, SPY_NAME
from .performance import PERFORMANCE_KEYS, performance_metrics

COMPARISON_COLUMNS: tuple[str, ...] = (
    "regime_rotation",
    EQUAL_WEIGHT_NAME,
    SPY_NAME,
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "portfolio",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "observations",
)


def _validate_rotation(rotation_detail: pd.DataFrame) -> pd.Series:
    if not isinstance(rotation_detail, pd.DataFrame):
        raise TypeError("rotation_detail must be a pandas DataFrame.")
    if tuple(rotation_detail.columns) != ROTATION_DETAIL_COLUMNS:
        raise ValueError("rotation_detail columns must match the canonical Step 5 schema exactly.")
    if not isinstance(rotation_detail.index, pd.DatetimeIndex):
        raise ValueError("rotation_detail index must be a pandas DatetimeIndex.")
    if rotation_detail.index.name != "Date" or rotation_detail.index.tz is not None:
        raise ValueError("rotation_detail index must be timezone-naive and named 'Date'.")
    if len(rotation_detail) == 0:
        raise ValueError("rotation_detail must contain observations.")
    if rotation_detail.index.has_duplicates or not rotation_detail.index.is_monotonic_increasing:
        raise ValueError("rotation_detail dates must be unique and sorted ascending.")
    series = rotation_detail["regime_rotation_return"]
    if not is_numeric_dtype(series.dtype):
        raise ValueError("regime_rotation_return must be numeric.")
    values = series.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("regime_rotation_return must be finite and greater than -1.")
    return series.rename("regime_rotation").astype(float)


def _validate_benchmark(series: pd.Series, expected_name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{expected_name} must be a pandas Series.")
    if series.name != expected_name:
        raise ValueError(f"benchmark Series must be named {expected_name!r}.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{expected_name} index must be a pandas DatetimeIndex.")
    if series.index.name != "Date" or series.index.tz is not None:
        raise ValueError(f"{expected_name} index must be timezone-naive and named 'Date'.")
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError(f"{expected_name} dates must be unique and sorted ascending.")
    if not is_numeric_dtype(series.dtype):
        raise ValueError(f"{expected_name} must be numeric.")
    values = series.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError(f"{expected_name} returns must be finite and greater than -1.")
    return series.astype(float)


def build_comparison(
    rotation_detail: pd.DataFrame, equal_weight: pd.Series, spy: pd.Series
) -> pd.DataFrame:
    """Build the exact three-column comparison after enforcing identical dates."""
    rotation = _validate_rotation(rotation_detail)
    equal_weight_validated = _validate_benchmark(equal_weight, EQUAL_WEIGHT_NAME)
    spy_validated = _validate_benchmark(spy, SPY_NAME)
    if not rotation.index.equals(equal_weight_validated.index) or not rotation.index.equals(
        spy_validated.index
    ):
        raise ValueError("rotation and benchmark series must have exactly identical Date indexes.")
    result = pd.concat([rotation, equal_weight_validated, spy_validated], axis=1)
    result = result.loc[:, list(COMPARISON_COLUMNS)]
    result.index.name = "Date"
    return result


def build_performance_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Build the exact three-row performance summary by delegating all metric math."""
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    if tuple(comparison.columns) != COMPARISON_COLUMNS:
        raise ValueError("comparison columns must match the canonical Step 5 order exactly.")
    if not isinstance(comparison.index, pd.DatetimeIndex):
        raise ValueError("comparison index must be a pandas DatetimeIndex.")
    if comparison.index.name != "Date" or comparison.index.tz is not None:
        raise ValueError("comparison index must be timezone-naive and named 'Date'.")
    if comparison.index.has_duplicates or not comparison.index.is_monotonic_increasing:
        raise ValueError("comparison dates must be unique and sorted ascending.")
    rows: list[dict[str, float | int | str]] = []
    for portfolio in COMPARISON_COLUMNS:
        metrics = performance_metrics(comparison[portfolio])
        if tuple(metrics.keys()) != PERFORMANCE_KEYS:
            raise ValueError("performance_metrics returned an unexpected metric schema.")
        rows.append({"portfolio": portfolio, **metrics})
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
