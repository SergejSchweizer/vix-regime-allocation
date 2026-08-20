"""Causal monthly-refit HMM walk-forward signal engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ASSET_ORDER, SUPPORTED_STATE_COUNTS
from .hmm_filter import (
    HMMFilterModel,
    filter_observation,
    filtered_probabilities,
    fit_hmm_filter,
    forecast_next_regime,
)
from .refit import build_refit_schedule
from .returns import asset_simple_returns
from .state_returns import expected_asset_returns, soft_state_asset_means


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


def build_hmm_signals(
    data: pd.DataFrame, decision_dates: pd.DatetimeIndex, n_states: int
) -> pd.DataFrame:
    """Generate strictly one-sided HMM regime and expected-return forecasts."""

    index = _validate_decisions(data, decision_dates, n_states)
    simple_returns = asset_simple_returns(data)
    schedule = build_refit_schedule(index, decision_dates[0])
    schedule = schedule.loc[schedule["decision_date"].isin(decision_dates)].reset_index(drop=True)
    if len(schedule) != len(decision_dates):
        raise RuntimeError("refit schedule does not cover every requested decision.")

    model: HMMFilterModel | None = None
    state_means: pd.DataFrame | None = None
    alpha: np.ndarray | None = None
    active_training_end: pd.Timestamp | None = None
    rows: list[dict[str, object]] = []

    for schedule_row in schedule.itertuples(index=False):
        decision = pd.Timestamp(schedule_row.decision_date)
        if bool(schedule_row.refit) or model is None or state_means is None or alpha is None:
            active_training_end = pd.Timestamp(schedule_row.training_end)
            training = data.loc[:active_training_end]
            vix = training["VIX_change"].astype(float).rename("VIX_change")
            model = fit_hmm_filter(vix, n_states)
            filtered_training = filtered_probabilities(model, vix)
            state_means = soft_state_asset_means(
                simple_returns.loc[training.index], filtered_training
            )
            alpha = filtered_training.iloc[-1].to_numpy(dtype=float)

        if active_training_end is None or active_training_end >= decision:
            raise RuntimeError("active training window is not causal.")
        alpha = filter_observation(model, alpha, float(data.at[decision, "VIX_change"]))
        next_probabilities = forecast_next_regime(model, alpha)
        expected = expected_asset_returns(next_probabilities, state_means)
        position = int(index.get_loc(decision))
        return_date = index[position + 1]
        row: dict[str, object] = {
            "decision_date": decision,
            "return_date": return_date,
            "family": "hmm",
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
