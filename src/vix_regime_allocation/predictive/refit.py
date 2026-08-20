"""Monthly expanding-window refit schedule."""

from __future__ import annotations

from typing import cast

import pandas as pd

from .split import is_monthly_refit

REFIT_COLUMNS: tuple[str, str, str] = ("decision_date", "refit", "training_end")


def build_refit_schedule(
    index: pd.DatetimeIndex, first_decision_date: pd.Timestamp
) -> pd.DataFrame:
    """Build one causal refit flag per eligible decision date."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex.")
    if index.tz is not None or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("index must be timezone-naive, unique, and sorted.")
    first = pd.Timestamp(first_decision_date)
    if first not in index:
        raise ValueError("first_decision_date must be an observed row.")
    first_position = cast(int, index.get_loc(first))
    if first_position == 0:
        raise ValueError("first_decision_date requires at least one prior training row.")
    decisions = index[first_position:-1]
    if len(decisions) == 0:
        raise ValueError("at least one decision with a following return row is required.")
    rows: list[dict[str, object]] = []
    previous: pd.Timestamp | None = None
    active_training_end = index[first_position - 1]
    for decision in decisions:
        refit = is_monthly_refit(previous, decision)
        if refit:
            position = cast(int, index.get_loc(decision))
            active_training_end = index[position - 1]
        if active_training_end >= decision:
            raise RuntimeError("training_end must precede decision_date.")
        rows.append(
            {
                "decision_date": decision,
                "refit": refit,
                "training_end": active_training_end,
            }
        )
        previous = decision
    return pd.DataFrame(rows, columns=list(REFIT_COLUMNS))
