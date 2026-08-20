from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vix_regime_allocation.predictive import notebook_section


def _dominance(differences: tuple[float, float, float]) -> pd.DataFrame:
    strategy = 0.12
    return pd.DataFrame(
        {
            "benchmark": ["TLT", "GLD", "SPY"],
            "benchmark_cagr": [strategy - value for value in differences],
            "strategy_net_cagr": [strategy, strategy, strategy],
            "cagr_difference": list(differences),
        }
    )


def test_predictive_conclusion_reports_outperformance_only_when_all_differences_positive() -> None:
    success = notebook_section.predictive_conclusion(_dominance((0.01, 0.02, 0.03)))
    failure = notebook_section.predictive_conclusion(_dominance((0.01, 0.00, -0.02)))

    assert "beat TLT, GLD, and SPY" in success
    assert "not a guarantee of future outperformance" in success
    assert "did not beat every individual asset" in failure
    assert "not retuned" in failure


def test_predictive_conclusion_rejects_wrong_benchmark_set() -> None:
    frame = _dominance((0.01, 0.02, 0.03)).copy()
    frame.loc[2, "benchmark"] = "QQQ"
    with pytest.raises(ValueError, match="exactly TLT, GLD, and SPY"):
        notebook_section.predictive_conclusion(frame)


def test_render_predictive_extension_reads_only_canonical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table_dir = tmp_path / "reports/predictive/tables"
    generated_dir = tmp_path / "reports/predictive/generated"
    figure_dir = tmp_path / "reports/predictive/figures"
    table_dir.mkdir(parents=True)
    generated_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)

    selected = {
        "family": "markov",
        "n_states": 2,
        "switch_hurdle_bps": 5.0,
        "transaction_cost_bps": 5.0,
    }
    (generated_dir / "selected_strategy.json").write_text(json.dumps(selected), encoding="utf-8")
    pd.DataFrame(
        {
            "family": ["markov"],
            "n_states": [2],
            "switch_hurdle_bps": [5.0],
            "selected": [True],
        }
    ).to_csv(table_dir / "candidate_validation_summary.csv", index=False)
    pd.DataFrame({"portfolio": ["selected_predictive_net"], "annualized_return": [0.12]}).to_csv(
        table_dir / "selected_test_performance.csv", index=False
    )
    _dominance((0.01, 0.02, 0.03)).to_csv(table_dir / "test_asset_dominance.csv", index=False)
    (figure_dir / "cumulative_performance_all_instruments.png").write_bytes(b"png")
    (figure_dir / "regime_forecast_probabilities.png").write_bytes(b"png")

    captured: notebook_section.DisplayPayload = []

    def capture(payload: notebook_section.DisplayPayload) -> None:
        captured.extend(payload)

    monkeypatch.setattr(notebook_section, "_display_payload", capture)
    notebook_section.render_predictive_extension(tmp_path)

    rendered_text = "\n".join(str(value) for _, value in captured)
    assert "additive research extension" in rendered_text
    assert "Selected validation configuration" in rendered_text
    assert "beat TLT, GLD, and SPY" in rendered_text
    assert "cumulative_performance_all_instruments.png" in rendered_text
    assert "regime_forecast_probabilities.png" in rendered_text
