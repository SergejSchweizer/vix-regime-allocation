"""Step 5 aligned comparison summaries with a strict four-portfolio HMM API."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .backtest import ROTATION_DETAIL_COLUMNS
from .benchmarks import EQUAL_WEIGHT_NAME, SPY_NAME
from .performance import PERFORMANCE_KEYS, performance_metrics
from .strategy_comparison import COMPARISON_COLUMNS as DUAL_COMPARISON_COLUMNS

# Transitional legacy names remain until the full canonical rebuild migrates old callers.
COMPARISON_COLUMNS: tuple[str, ...] = ("regime_rotation", EQUAL_WEIGHT_NAME, SPY_NAME)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "portfolio",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "observations",
)


def _validate_index(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("comparison index must be a pandas DatetimeIndex.")
    if frame.index.name != "Date" or frame.index.tz is not None:
        raise ValueError("comparison index must be timezone-naive and named 'Date'.")
    if len(frame) == 0:
        raise ValueError("comparison must contain observations.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("comparison dates must be unique and sorted ascending.")


def _validate_numeric_returns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if not is_numeric_dtype(frame[column].dtype):
            raise ValueError(f"comparison column {column!r} must be numeric.")
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("comparison simple returns must be finite and greater than -1.")


def build_four_portfolio_performance_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed four-row HMM/benchmark summary using shared metric math only."""
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    if tuple(comparison.columns) != DUAL_COMPARISON_COLUMNS:
        raise ValueError("comparison must contain the exact four HMM/benchmark series in order.")
    _validate_index(comparison)
    _validate_numeric_returns(comparison, DUAL_COMPARISON_COLUMNS)
    rows: list[dict[str, float | int | str]] = []
    for portfolio in DUAL_COMPARISON_COLUMNS:
        metrics = performance_metrics(comparison[portfolio])
        if tuple(metrics.keys()) != PERFORMANCE_KEYS:
            raise ValueError("performance_metrics returned an unexpected metric schema.")
        rows.append({"portfolio": portfolio, **metrics})
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def _validate_rotation(rotation_detail: pd.DataFrame) -> pd.Series:
    if not isinstance(rotation_detail, pd.DataFrame):
        raise TypeError("rotation_detail must be a pandas DataFrame.")
    if tuple(rotation_detail.columns) != ROTATION_DETAIL_COLUMNS:
        raise ValueError("rotation_detail columns must match the transitional Step 5 schema.")
    _validate_index(rotation_detail)
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
    """Temporary legacy three-series adapter kept for pre-rebuild callers."""
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
    """Dispatch to strict four-row summary or the temporary legacy three-row adapter."""
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    if tuple(comparison.columns) == DUAL_COMPARISON_COLUMNS:
        return build_four_portfolio_performance_summary(comparison)
    if tuple(comparison.columns) != COMPARISON_COLUMNS:
        raise ValueError("comparison columns do not match a supported Step 5 schema.")
    _validate_index(comparison)
    _validate_numeric_returns(comparison, COMPARISON_COLUMNS)
    rows: list[dict[str, float | int | str]] = []
    for portfolio in COMPARISON_COLUMNS:
        metrics = performance_metrics(comparison[portfolio])
        if tuple(metrics.keys()) != PERFORMANCE_KEYS:
            raise ValueError("performance_metrics returned an unexpected metric schema.")
        rows.append({"portfolio": portfolio, **metrics})
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
