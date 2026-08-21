"""Notebook-facing helper for the HMM K-by-allocation sensitivity output."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def step_5_state_count_sensitivity() -> Any:  # pragma: no cover - notebook presentation
    """Recompute and display HMM K=2/K=3 crossed with both allocation methods."""
    import pandas as pd
    from IPython.display import Markdown, display

    from vix_regime_allocation.sensitivity import build_hmm_state_count_sensitivity

    cwd = Path.cwd().resolve()
    repo_root = cwd.parent if cwd.name == "notebooks" else cwd
    data = pd.read_csv(
        repo_root / "data/processed/step1_data.csv",
        parse_dates=["Date"],
    ).set_index("Date")
    data.index = pd.DatetimeIndex(data.index, name="Date")

    states_by_k: dict[int, pd.Series] = {}
    for n_states in (2, 3):
        state_frame = pd.read_csv(
            repo_root / f"reports/tables/step2_hmm_{n_states}_states.csv",
            parse_dates=["Date"],
        ).set_index("Date")
        state_frame.index = pd.DatetimeIndex(state_frame.index, name="Date")
        states_by_k[n_states] = state_frame["state"].astype("int64").rename("state")

    sensitivity = build_hmm_state_count_sensitivity(data, states_by_k)
    expected = [
        ["hmm", 2, "100_keep"],
        ["hmm", 2, "60_40_spread"],
        ["hmm", 3, "100_keep"],
        ["hmm", 3, "60_40_spread"],
    ]
    if sensitivity[["family", "n_states", "method"]].values.tolist() != expected:
        raise RuntimeError("HMM sensitivity does not match the canonical four-row order.")

    sensitivity_path = repo_root / "reports/tables/step5_state_count_sensitivity.csv"
    if sensitivity_path.is_file():
        persisted = pd.read_csv(sensitivity_path)
        pd.testing.assert_frame_equal(
            sensitivity.reset_index(drop=True),
            persisted.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-9,
            atol=1e-11,
        )
    display(Markdown("### Step 5 — HMM state-count and allocation-method sensitivity"))
    display(sensitivity)
    return sensitivity
