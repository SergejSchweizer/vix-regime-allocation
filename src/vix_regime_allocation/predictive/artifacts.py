"""Canonical predictive-analysis computation and deterministic artifact writing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ONE_WAY_COST_BPS, TEST_START, VALIDATION_END, VALIDATION_START
from .hmm_walkforward import build_hmm_signals
from .holdout import HoldoutResult, run_final_holdout
from .markov_walkforward import build_markov_signals
from .returns import asset_simple_returns
from .selection import build_validation_summary, selected_configuration
from .split import split_periods

TABLE_FILENAMES: tuple[str, str, str, str] = (
    "candidate_validation_summary.csv",
    "selected_test_daily.csv",
    "selected_test_performance.csv",
    "test_asset_dominance.csv",
)
SELECTED_STRATEGY_FILENAME = "selected_strategy.json"
MANIFEST_FILENAME = "predictive_manifest.json"


@dataclass(frozen=True)
class PredictiveAnalysis:
    """All canonical in-memory outputs required by the predictive extension."""

    validation_summary: pd.DataFrame
    selected_test_daily: pd.DataFrame
    selected_test_performance: pd.DataFrame
    test_asset_dominance: pd.DataFrame
    selected_strategy: dict[str, object]
    cagr_dominance_margin: float
    dominates_all_individual_assets: bool


def _signals_for_family(
    data: pd.DataFrame, decision_dates: pd.DatetimeIndex, family: str, n_states: int
) -> pd.DataFrame:
    if family == "markov":
        return build_markov_signals(data, decision_dates, n_states)
    if family == "hmm":
        return build_hmm_signals(data, decision_dates, n_states)
    raise ValueError("family must be markov or hmm.")


def _merge_signal_provenance(signals: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    keys = ["decision_date", "return_date", "family", "n_states"]
    merged = signals.merge(
        daily,
        on=keys,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(signals) or len(merged) != len(daily):
        raise RuntimeError("test signal provenance and realized backtest rows are misaligned.")
    return merged


def compute_predictive_analysis(data: pd.DataFrame) -> PredictiveAnalysis:
    """Run the pre-registered validation selection and one frozen final holdout."""

    periods = split_periods(data)
    validation_decisions = periods.validation[:-1]
    test_decisions = periods.test[:-1]
    if len(validation_decisions) < 2 or len(test_decisions) < 2:
        raise ValueError("validation and test each require at least two predictive decisions.")

    validation_signals: dict[tuple[str, int], pd.DataFrame] = {}
    for family in ("markov", "hmm"):
        for n_states in (2, 3):
            validation_signals[(family, n_states)] = _signals_for_family(
                data, validation_decisions, family, n_states
            )

    asset_returns = asset_simple_returns(data)
    validation_summary = build_validation_summary(validation_signals, asset_returns)
    family, n_states, hurdle = selected_configuration(validation_summary)
    selected_validation = validation_summary.loc[validation_summary["selected"].astype(bool)].iloc[
        0
    ]

    test_signals = _signals_for_family(data, test_decisions, family, n_states)
    holdout: HoldoutResult = run_final_holdout(
        data,
        validation_summary,
        {(family, n_states): test_signals},
    )
    selected_test_daily = _merge_signal_provenance(test_signals, holdout.daily)

    selected_strategy: dict[str, object] = {
        "family": family,
        "n_states": n_states,
        "switch_hurdle_bps": hurdle,
        "transaction_cost_bps": ONE_WAY_COST_BPS,
        "validation_start": VALIDATION_START.date().isoformat(),
        "validation_end": VALIDATION_END.date().isoformat(),
        "test_start": TEST_START.date().isoformat(),
        "training_rule": "expanding_window_training_prefix_only",
        "refit_rule": "first_observed_decision_row_of_each_calendar_month",
        "selection_metric": "validation_mean_log_growth_of_net_returns",
        "validation_mean_log_growth": float(selected_validation["mean_log_growth"]),
        "validation_mean_turnover": float(selected_validation["mean_turnover"]),
    }
    return PredictiveAnalysis(
        validation_summary=validation_summary,
        selected_test_daily=selected_test_daily,
        selected_test_performance=holdout.performance,
        test_asset_dominance=holdout.dominance,
        selected_strategy=selected_strategy,
        cagr_dominance_margin=holdout.cagr_dominance_margin,
        dominates_all_individual_assets=holdout.dominates_all_individual_assets,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_predictive_manifest(
    root: Path,
    *,
    figure_paths: tuple[str, ...] = (),
) -> Path:
    """Write a deterministic hash manifest for the canonical predictive artifacts."""

    root = Path(root)
    generated_dir = root / "reports/predictive/generated"
    table_paths = tuple(f"reports/predictive/tables/{name}" for name in TABLE_FILENAMES)
    selected_path = f"reports/predictive/generated/{SELECTED_STRATEGY_FILENAME}"
    required = (*table_paths, selected_path, *figure_paths)
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"Missing predictive artifacts for manifest: {missing}")
    input_path = root / "data/processed/step1_data.csv"
    if not input_path.is_file():
        raise RuntimeError("Canonical Step 1 dataset is missing.")
    hashes = {relative: _sha256(root / relative) for relative in sorted(required)}
    manifest = {
        "schema_version": 1,
        "input_data_path": "data/processed/step1_data.csv",
        "input_data_sha256": _sha256(input_path),
        "selected_strategy_path": selected_path,
        "tables": sorted(table_paths),
        "figures": sorted(figure_paths),
        "sha256": hashes,
    }
    generated_dir.mkdir(parents=True, exist_ok=True)
    output = generated_dir / MANIFEST_FILENAME
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_predictive_artifacts(root: Path, analysis: PredictiveAnalysis) -> dict[str, Path]:
    """Persist the canonical tables, selected configuration, and base manifest."""

    root = Path(root)
    table_dir = root / "reports/predictive/tables"
    generated_dir = root / "reports/predictive/generated"
    table_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "candidate_validation_summary": analysis.validation_summary,
        "selected_test_daily": analysis.selected_test_daily,
        "selected_test_performance": analysis.selected_test_performance,
        "test_asset_dominance": analysis.test_asset_dominance,
    }
    outputs: dict[str, Path] = {}
    for name, frame in tables.items():
        path = table_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = path

    selected_path = generated_dir / SELECTED_STRATEGY_FILENAME
    selected_path.write_text(
        json.dumps(analysis.selected_strategy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["selected_strategy"] = selected_path
    outputs["manifest"] = write_predictive_manifest(root)
    return outputs


def load_step1_data(path: Path) -> pd.DataFrame:
    """Load the canonical persisted Step 1 dataset with its exact Date index."""

    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    return frame


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data = load_step1_data(root / "data/processed/step1_data.csv")
    analysis = compute_predictive_analysis(data)
    write_predictive_artifacts(root, analysis)
    print(
        "Predictive analysis artifacts written; "
        f"dominates_all_individual_assets={analysis.dominates_all_individual_assets}, "
        f"cagr_dominance_margin={analysis.cagr_dominance_margin:.12f}."
    )


if __name__ == "__main__":
    main()
