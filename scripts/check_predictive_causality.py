from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vix_regime_allocation.predictive.config import TEST_START, VALIDATION_END, VALIDATION_START
from vix_regime_allocation.predictive.selection import candidate_grid

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "reports/predictive/tables/candidate_validation_summary.csv"
DAILY = ROOT / "reports/predictive/tables/selected_test_daily.csv"
SELECTED = ROOT / "reports/predictive/generated/selected_strategy.json"


def validate_predictive_causality(
    daily: pd.DataFrame, validation: pd.DataFrame, selected: dict[str, object]
) -> None:
    """Independently reject time leakage, candidate drift, or post-validation mutation."""

    if len(validation) != 16:
        raise RuntimeError("Predictive validation must contain exactly 16 candidates.")
    required_validation = {"family", "n_states", "switch_hurdle_bps", "selected"}
    if not required_validation.issubset(validation.columns):
        raise RuntimeError("Predictive validation summary is missing required columns.")
    actual_grid = {
        (str(row.family), int(row.n_states), float(row.switch_hurdle_bps))
        for row in validation.itertuples(index=False)
    }
    if actual_grid != set(candidate_grid()):
        raise RuntimeError("Predictive validation candidate grid differs from the pre-registered grid.")
    winners = validation.loc[validation["selected"].astype(bool)]
    if len(winners) != 1:
        raise RuntimeError("Predictive validation must contain exactly one selected candidate.")
    winner = winners.iloc[0]
    expected = (
        str(winner["family"]),
        int(winner["n_states"]),
        float(winner["switch_hurdle_bps"]),
    )
    actual = (
        str(selected.get("family")),
        int(selected.get("n_states", -1)),
        float(selected.get("switch_hurdle_bps", float("nan"))),
    )
    if actual != expected:
        raise RuntimeError("Selected strategy JSON does not match the validation winner.")
    if str(selected.get("validation_start")) != VALIDATION_START.date().isoformat():
        raise RuntimeError("Selected strategy validation_start is not the fixed pre-registered date.")
    if str(selected.get("validation_end")) != VALIDATION_END.date().isoformat():
        raise RuntimeError("Selected strategy validation_end is not the fixed pre-registered date.")
    if str(selected.get("test_start")) != TEST_START.date().isoformat():
        raise RuntimeError("Selected strategy test_start is not the fixed pre-registered date.")

    required_daily = {
        "training_end",
        "decision_date",
        "return_date",
        "family",
        "n_states",
        "switch_hurdle_bps",
    }
    if not required_daily.issubset(daily.columns) or len(daily) < 2:
        raise RuntimeError("Selected test daily artifact is missing causal provenance.")
    training_end = pd.to_datetime(daily["training_end"])
    decision = pd.to_datetime(daily["decision_date"])
    realized = pd.to_datetime(daily["return_date"])
    if not ((training_end < decision) & (decision < realized)).all():
        raise RuntimeError("Predictive daily artifact violates training < decision < return chronology.")
    if (realized < TEST_START).any():
        raise RuntimeError("Predictive final holdout contains pre-test returns.")
    if not (
        (daily["family"].astype(str) == expected[0])
        & (daily["n_states"].astype(int) == expected[1])
        & (daily["switch_hurdle_bps"].astype(float) == expected[2])
    ).all():
        raise RuntimeError("Final holdout configuration changed after validation selection.")


def main() -> None:
    for path in (VALIDATION, DAILY, SELECTED):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing predictive artifact: {path.relative_to(ROOT)}")
    validation = pd.read_csv(VALIDATION)
    daily = pd.read_csv(DAILY)
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    validate_predictive_causality(daily, validation, selected)
    print("Predictive causality and frozen-selection audit passed.")


if __name__ == "__main__":
    main()
