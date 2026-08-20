from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_predictive_causality.py"
SPEC = importlib.util.spec_from_file_location("check_predictive_causality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _validation() -> pd.DataFrame:
    rows = []
    for family in ("markov", "hmm"):
        for k in (2, 3):
            for hurdle in (0.0, 5.0, 10.0, 20.0):
                rows.append(
                    {
                        "family": family,
                        "n_states": k,
                        "switch_hurdle_bps": hurdle,
                        "selected": family == "markov" and k == 2 and hurdle == 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _selected() -> dict[str, object]:
    return {
        "family": "markov",
        "n_states": 2,
        "switch_hurdle_bps": 0.0,
        "validation_start": "2015-01-01",
        "validation_end": "2020-12-31",
        "test_start": "2021-01-01",
    }


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "training_end": pd.to_datetime(["2020-12-31", "2021-01-04"]),
            "decision_date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
            "return_date": pd.to_datetime(["2021-01-05", "2021-01-06"]),
            "family": ["markov", "markov"],
            "n_states": [2, 2],
            "switch_hurdle_bps": [0.0, 0.0],
        }
    )


def test_valid_causal_artifacts_pass() -> None:
    module.validate_predictive_causality(_daily(), _validation(), _selected())


@pytest.mark.parametrize("kind", ["leakage", "grid", "mutation"])
def test_corruption_is_rejected(kind: str) -> None:
    daily = _daily()
    validation = _validation()
    selected = _selected()
    if kind == "leakage":
        daily.loc[0, "training_end"] = daily.loc[0, "decision_date"]
    elif kind == "grid":
        validation = validation.iloc[:-1].copy()
    else:
        selected["n_states"] = 3
    with pytest.raises(RuntimeError):
        module.validate_predictive_causality(daily, validation, selected)
