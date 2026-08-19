from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from vix_regime_allocation.backtest_plot import plot_cumulative_performance
from vix_regime_allocation.hmm_state_plot import plot_hmm_vix_states
from vix_regime_allocation.markov_plots import plot_markov_vix_states
from vix_regime_allocation.plots import plot_etf_log_returns, plot_vix_change
from vix_regime_allocation.sensitivity import build_state_count_sensitivity
from vix_regime_allocation.state_statistics_plot import plot_state_asset_statistics

ROOT = Path(__file__).resolve().parents[1]
STEP1 = ROOT / "data/processed/step1_data.csv"
SELECTED = ROOT / "reports/generated/step3_selected_model.json"
STATE_STATS = ROOT / "reports/tables/step3_state_asset_statistics.csv"
DAILY = ROOT / "reports/tables/step5_daily_returns.csv"
OUT_SENSITIVITY = ROOT / "reports/tables/step5_state_count_sensitivity.csv"
OUT_MANIFEST = ROOT / "reports/generated/step5_manifest.json"


def _load_indexed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    return frame


def _write_sensitivity_and_manifest() -> None:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    family = selected["family"]
    if family not in {"markov", "hmm"}:
        raise RuntimeError(f"Unexpected preferred family: {family!r}")

    step1 = _load_indexed_csv(STEP1)
    states_by_k: dict[int, pd.Series] = {}
    for n_states in (2, 3):
        path = ROOT / f"reports/tables/step2_{family}_{n_states}_states.csv"
        frame = _load_indexed_csv(path)
        states_by_k[n_states] = frame["state"].astype(int).rename("state")

    sensitivity = build_state_count_sensitivity(step1, family, states_by_k)
    OUT_SENSITIVITY.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(OUT_SENSITIVITY, index=False)

    tables = [
        "reports/tables/step5_daily_returns.csv",
        "reports/tables/step5_performance_summary.csv",
        "reports/tables/step5_state_count_sensitivity.csv",
    ]
    figures = ["reports/figures/step5_cumulative_performance.png"]
    for relative in tables + figures:
        path = ROOT / relative
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing canonical Step 5 artifact: {relative}")

    manifest = {
        "schema_version": 1,
        "input_data_path": "data/processed/step1_data.csv",
        "input_data_sha256": hashlib.sha256(STEP1.read_bytes()).hexdigest(),
        "selected_model_path": "reports/generated/step3_selected_model.json",
        "selected_states_path": selected["selected_states_path"],
        "allocation_path": "reports/tables/step4_allocation_mapping.csv",
        "notebook_path": "notebooks/gwp2_vix_regime_allocation.ipynb",
        "tables": sorted(tables),
        "figures": sorted(figures),
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _regenerate_figures() -> None:
    step1 = _load_indexed_csv(STEP1)
    plot_etf_log_returns(step1, ROOT / "reports/figures/step1_etf_log_returns.png")
    plot_vix_change(step1, ROOT / "reports/figures/step1_vix_change.png")

    vix = step1["VIX"].rename("VIX")
    markov_2 = (
        _load_indexed_csv(ROOT / "reports/tables/step2_markov_2_states.csv")["state"]
        .astype(int)
        .rename("state")
    )
    markov_3 = (
        _load_indexed_csv(ROOT / "reports/tables/step2_markov_3_states.csv")["state"]
        .astype(int)
        .rename("state")
    )
    plot_markov_vix_states(
        vix,
        markov_2,
        markov_3,
        ROOT / "reports/figures/step2_markov_vix_states.png",
    )

    hmm_2 = _load_indexed_csv(ROOT / "reports/tables/step2_hmm_2_states.csv")["state"]
    hmm_2 = hmm_2.astype(int).rename("state")
    hmm_3 = _load_indexed_csv(ROOT / "reports/tables/step2_hmm_3_states.csv")["state"]
    hmm_3 = hmm_3.astype(int).rename("state")
    plot_hmm_vix_states(
        vix,
        hmm_2,
        hmm_3,
        ROOT / "reports/figures/step2_hmm_vix_states.png",
    )

    statistics = pd.read_csv(STATE_STATS)
    plot_state_asset_statistics(
        statistics,
        ROOT / "reports/figures/step3_state_asset_statistics.png",
    )

    daily = _load_indexed_csv(DAILY)
    plot_cumulative_performance(
        daily,
        ROOT / "reports/figures/step5_cumulative_performance.png",
    )


def main() -> None:
    """Regenerate canonical review artifacts without editing notebook cells."""
    _regenerate_figures()
    _write_sensitivity_and_manifest()
    print("Analysis artifacts rebuilt; notebook presentation is handled separately.")


if __name__ == "__main__":
    main()
