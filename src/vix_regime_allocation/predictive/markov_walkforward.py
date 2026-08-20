"""Causal monthly-refit Markov walk-forward signal engine."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .config import ASSET_ORDER, SUPPORTED_STATE_COUNTS
from .markov_forecast import MarkovForecastModel, fit_markov_forecaster, forecast_next_regime
from .refit import build_refit_schedule
from .returns import asset_simple_returns
from .state_returns import expected_asset_returns, hard_state_asset_means


def _validate_decisions(
    data: pd.DataFrame, decision_dates: pd.DatetimeIndex, n_states: int
) -> pd.DatetimeIndex:
    if n_states not in SUPPORTED_STATE_COUNTS:
        raise ValueError(f"n_states must be one of {SUPPORTED_STATE_COUNTS}.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data index must be a DatetimeIndex.")
    if not isinstance(decision_dates, pd.DatetimeIndex) or len(decision_dates) == 0:
        raise ValueError("decision_dates must be a non-empty DatetimeIndex.")
    if decision_dates.has_duplicates or not decision_dates.is_monotonic_increasing:
        raise ValueError("decision_dates must be unique and sorted.")
    index = pd.DatetimeIndex(data.index, name="Date")
    positions = index.get_indexer(decision_dates)
    if np.any(positions < 1) or np.any(positions >= len(index) - 1):
        raise ValueError("each decision requires a prior training row and a following return row.")
    if len(positions) > 1 and np.any(np.diff(positions) != 1):
        raise ValueError("decision_dates must be a contiguous observed-date slice.")
    if "VIX_change" not in data.columns:
        raise ValueError("data must contain VIX_change.")
    return index


def build_markov_signals(
    data: pd.DataFrame, decision_dates: pd.DatetimeIndex, n_states: int
) -> pd.DataFrame:
    """Generate causal one-step regime and asset-return forecasts."""

    index = _validate_decisions(data, decision_dates, n_states)
    simple_returns = asset_simple_returns(data)
    schedule = build_refit_schedule(index, decision_dates[0])
    schedule = schedule.loc[schedule["decision_date"].isin(decision_dates)].reset_index(drop=True)
    if len(schedule) != len(decision_dates):
        raise RuntimeError("refit schedule does not cover every requested decision.")

    model: MarkovForecastModel | None = None
    state_means: pd.DataFrame | None = None
    active_training_end: pd.Timestamp | None = None
    rows: list[dict[str, object]] = []

    for schedule_row in schedule.itertuples(index=False):
        decision = cast(pd.Timestamp, schedule_row.decision_date)
        if bool(schedule_row.refit) or model is None or state_means is None:
            active_training_end = cast(pd.Timestamp, schedule_row.training_end)
            training = data.loc[:active_training_end]
            vix = training["VIX_change"].astype(float).rename("VIX_change")
            model = fit_markov_forecaster(vix, n_states)
            state_means = hard_state_asset_means(
                simple_returns.loc[training.index], model.training_states, n_states
            )

        if active_training_end is None or active_training_end >= decision:
            raise RuntimeError("active training window is not causal.")
        current_vix = float(cast(float, data.at[decision, "VIX_change"]))
        next_probabilities = forecast_next_regime(model, current_vix)
        expected = expected_asset_returns(next_probabilities, state_means)
        position = cast(int, index.get_loc(decision))
        return_date = index[position + 1]
        row: dict[str, object] = {
            "decision_date": decision,
            "return_date": return_date,
            "family": "markov",
            "n_states": n_states,
            "training_end": active_training_end,
        }
        for state, probability in enumerate(next_probabilities):
            row[f"p_state_{state}"] = float(probability)
        for asset in ASSET_ORDER:
            row[f"expected_{asset}"] = float(expected[asset])
        rows.append(row)

    result = pd.DataFrame(rows)
    if not (result["training_end"] < result["decision_date"]).all():
        raise RuntimeError("walk-forward output contains look-ahead.")
    return result
