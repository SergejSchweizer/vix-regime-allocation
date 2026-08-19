from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd

from vix_regime_allocation.sensitivity import build_state_count_sensitivity

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
STEP1 = ROOT / "data/processed/step1_data.csv"
SELECTED = ROOT / "reports/generated/step3_selected_model.json"
ALLOCATION = ROOT / "reports/tables/step4_allocation_mapping.csv"
OUT_TABLE = ROOT / "reports/tables/step5_state_count_sensitivity.csv"
OUT_MANIFEST = ROOT / "reports/generated/step5_manifest.json"

selected = json.loads(SELECTED.read_text(encoding="utf-8"))
family = selected["family"]
if family not in {"markov", "hmm"}:
    raise SystemExit("Unexpected preferred family")

step1 = pd.read_csv(STEP1, parse_dates=["Date"]).set_index("Date")
step1.index = pd.DatetimeIndex(step1.index, name="Date")
states_by_k: dict[int, pd.Series] = {}
for k in (2, 3):
    path = ROOT / f"reports/tables/step2_{family}_{k}_states.csv"
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index, name="Date")
    states_by_k[k] = frame["state"].astype(int).rename("state")

sensitivity = build_state_count_sensitivity(step1, family, states_by_k)
OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
sensitivity.to_csv(OUT_TABLE, index=False)

nb = nbformat.read(NOTEBOOK, as_version=4)
heading = "### Step 5 — State-count sensitivity and final takeaway"
works_index = next(
    (i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and "## Works Cited" in c.source),
    None,
)
if works_index is None:
    raise SystemExit("Works Cited not found")
old = next(
    (i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and heading in c.source),
    None,
)
if old is not None:
    nb.cells = nb.cells[:old] + nb.cells[works_index:]
    works_index = old

md = nbformat.v4.new_markdown_cell(
    """### Step 5 — State-count sensitivity and final takeaway

The final sensitivity check compares exactly two and three states **within the already selected model family**. It reuses the canonical persisted state paths, then delegates state-conditional ETF statistics, the deterministic 100% allocation rule, the one-observed-trading-row execution lag, and all five performance metrics to the shared project functions. No model is refitted and no state path is redecoded for this sensitivity analysis.

**Greek letters used in this section:** none.

The two rows are evaluated on the common lagged return-date intersection, so their cumulative return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, and observation count are directly comparable on the same dates.

This remains a **full-sample descriptive sensitivity**, not an out-of-sample experiment. The practical implication is therefore conditional: the state count changes the historical allocation path and its risk/return profile, but a production claim would require rolling or expanding estimation, one-sided state inference, decision-time-only allocation estimation, transaction costs, and further robustness checks."""
)
code = nbformat.v4.new_code_cell(
    r'''from vix_regime_allocation.sensitivity import build_state_count_sensitivity

preferred_family = selected_model["family"]
states_by_k = {}
for n_states in (2, 3):
    states_path = repo_root / f"reports/tables/step2_{preferred_family}_{n_states}_states.csv"
    state_frame = pd.read_csv(states_path, parse_dates=["Date"]).set_index("Date")
    state_frame.index = pd.DatetimeIndex(state_frame.index, name="Date")
    states_by_k[n_states] = state_frame["state"].astype(int).rename("state")

state_count_sensitivity = build_state_count_sensitivity(
    step1_data,
    preferred_family,
    states_by_k,
)
sensitivity_path = repo_root / "reports/tables/step5_state_count_sensitivity.csv"
state_count_sensitivity.to_csv(sensitivity_path, index=False)
display(Markdown("### K=2 versus K=3 sensitivity within the preferred family"))
display(state_count_sensitivity)'''
)
interp = nbformat.v4.new_markdown_cell(
    """### Final Step 5 interpretation

The state-count table is a robustness description of the same historical sample. The preferred two-state specification remains the primary strategy because state-count selection was fixed earlier; the three-state row is a sensitivity comparison rather than a replacement selected by Step 5 performance. The benchmark and sensitivity results should not be read as causal evidence of future excess performance.

For implementation, the strongest next validation would use an expanding or rolling training window, one-sided state inference, allocation means estimated only from information available at each decision date, and explicit turnover/transaction-cost modeling. That design would address the principal look-ahead limitation that remains even after applying the required one-row execution lag."""
)
nb.cells[works_index:works_index] = [md, code, interp]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)

step1_sha = hashlib.sha256(STEP1.read_bytes()).hexdigest()
tables = [
    "reports/tables/step5_daily_returns.csv",
    "reports/tables/step5_performance_summary.csv",
    "reports/tables/step5_state_count_sensitivity.csv",
]
figures = ["reports/figures/step5_cumulative_performance.png"]
for rel in tables + figures:
    path = ROOT / rel
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Missing canonical Step5 artifact: {rel}")
manifest = {
    "schema_version": 1,
    "input_data_path": "data/processed/step1_data.csv",
    "input_data_sha256": step1_sha,
    "selected_model_path": "reports/generated/step3_selected_model.json",
    "selected_states_path": selected["selected_states_path"],
    "allocation_path": "reports/tables/step4_allocation_mapping.csv",
    "notebook_path": "notebooks/gwp2_vix_regime_allocation.ipynb",
    "tables": sorted(tables),
    "figures": sorted(figures),
}
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
