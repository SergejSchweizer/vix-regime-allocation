"""Notebook-facing helper for the Step 5 state-count sensitivity output."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def step_5_state_count_sensitivity() -> Any:  # pragma: no cover
    """Compute, persist, and display K=2 versus K=3 sensitivity results."""
    import json

    import pandas as pd
    from IPython.display import Markdown, display

    from vix_regime_allocation.sensitivity import build_state_count_sensitivity

    repo_root = (
        Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
    )
    selected_model = json.loads(
        (repo_root / "reports/generated/step3_selected_model.json").read_text(encoding="utf-8")
    )
    preferred_family = selected_model["family"]
    data = pd.read_csv(
        repo_root / "data/processed/step1_data.csv",
        parse_dates=["Date"],
    ).set_index("Date")
    data.index = pd.DatetimeIndex(data.index, name="Date")

    states_by_k: dict[int, pd.Series] = {}
    for n_states in (2, 3):
        states_path = repo_root / f"reports/tables/step2_{preferred_family}_{n_states}_states.csv"
        state_frame = pd.read_csv(states_path, parse_dates=["Date"]).set_index("Date")
        state_frame.index = pd.DatetimeIndex(state_frame.index, name="Date")
        states_by_k[n_states] = state_frame["state"].astype(int).rename("state")

    sensitivity = build_state_count_sensitivity(data, preferred_family, states_by_k)
    sensitivity_path = repo_root / "reports/tables/step5_state_count_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    display(Markdown("### K=2 versus K=3 sensitivity within the preferred family"))
    display(sensitivity)
    return sensitivity
