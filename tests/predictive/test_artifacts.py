from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from vix_regime_allocation.predictive.artifacts import (
    PredictiveAnalysis,
    write_predictive_artifacts,
)


def _analysis() -> PredictiveAnalysis:
    validation = pd.DataFrame(
        [
            {
                "family": "markov",
                "n_states": 2,
                "switch_hurdle_bps": 0.0,
                "mean_log_growth": 0.001,
                "selected": True,
            }
        ]
    )
    daily = pd.DataFrame(
        [
            {
                "training_end": "2020-12-31",
                "decision_date": "2021-01-04",
                "return_date": "2021-01-05",
                "family": "markov",
                "n_states": 2,
                "switch_hurdle_bps": 0.0,
                "net_return": 0.01,
            }
        ]
    )
    performance = pd.DataFrame(
        [{"portfolio": "selected_predictive_net", "annualized_return": 0.1}]
    )
    dominance = pd.DataFrame(
        [
            {
                "benchmark": "SPY",
                "benchmark_cagr": 0.08,
                "strategy_net_cagr": 0.1,
                "cagr_difference": 0.02,
            }
        ]
    )
    selected = {
        "family": "markov",
        "n_states": 2,
        "switch_hurdle_bps": 0.0,
        "transaction_cost_bps": 5.0,
        "validation_start": "2015-01-01",
        "validation_end": "2020-12-31",
        "test_start": "2021-01-01",
        "training_rule": "expanding_window_training_prefix_only",
        "refit_rule": "first_observed_decision_row_of_each_calendar_month",
        "selection_metric": "validation_mean_log_growth_of_net_returns",
        "validation_mean_log_growth": 0.001,
        "validation_mean_turnover": 0.2,
    }
    return PredictiveAnalysis(validation, daily, performance, dominance, selected, 0.02, True)


def test_artifact_writer_and_manifest_are_deterministic(tmp_path: Path) -> None:
    data_dir = tmp_path / "data/processed"
    data_dir.mkdir(parents=True)
    data_path = data_dir / "step1_data.csv"
    data_path.write_text("Date,x\n2020-01-01,1\n", encoding="utf-8")

    outputs = write_predictive_artifacts(tmp_path, _analysis())
    assert set(outputs) == {
        "candidate_validation_summary",
        "selected_test_daily",
        "selected_test_performance",
        "test_asset_dominance",
        "selected_strategy",
        "manifest",
    }
    selected = json.loads(outputs["selected_strategy"].read_text(encoding="utf-8"))
    assert set(selected) == {
        "family",
        "n_states",
        "switch_hurdle_bps",
        "transaction_cost_bps",
        "validation_start",
        "validation_end",
        "test_start",
        "training_rule",
        "refit_rule",
        "selection_metric",
        "validation_mean_log_growth",
        "validation_mean_turnover",
    }
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert "timestamp" not in manifest
    assert manifest["figures"] == []
    assert manifest["input_data_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    for relative, digest in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == digest
