# VIX Regime Allocation

Regime-based allocation project for **MScFE 622: Stochastic Modeling - Group Work Project #2**.

The project studies whether VIX-driven volatility regimes can support a transparent rotation strategy across:

- `TLT` - long-duration U.S. Treasury ETF
- `GLD` - gold ETF
- `SPY` - S&P 500 ETF
- `^VIX` / `VIX` - volatility-state input

## Current status

| Area | Status |
|---|---|
| Assignment template | Added and populated with team names |
| Step 1 backlog | Defined in [`BACKLOG.md`](BACKLOG.md) |
| Steps 2-4 backlog | Defined in [`BACKLOG_STEPS_2_4.md`](BACKLOG_STEPS_2_4.md) |
| Python package scaffold | Bootstrapped |
| Push quality gates | Configured |
| Pull-request quality gates | Configured |
| Coverage threshold | 90% combined source coverage |
| `main` merge protection | Repository branch rule still required |
| Step 1 implementation | Not started |
| Steps 2-4 implementation | Planned; PR-06 through PR-21 not yet implemented |
| Canonical notebook | Planned at `notebooks/gwp2_vix_regime_allocation.ipynb` |
| Final synchronized PDF report | Planned at `reports/Stochastic_Modeling_GWP2_Report.pdf` |

No Step 2-4 numerical result is claimed in this README until the canonical notebook has been executed successfully.

## Assignment implementation plan

### Step 1 - Data Preparation and Exploration

Step 1 must:

1. Download daily adjusted close prices for `TLT`, `GLD`, `SPY`, and `^VIX` from Yahoo Finance using the maximum available history.
2. Restrict the data to the maximum common sample period.
3. Compute daily log returns for `TLT`, `GLD`, and `SPY`.
4. Compute daily VIX change as `VIX_t - VIX_{t-1}`.
5. Align all derived series on common dates and remove missing values.
6. Produce one plot containing the three ETF return series.
7. Produce one plot containing VIX change.
8. Leave a clean dataset reusable by later assignment steps.

The exact Step 1 PR contracts are in [`BACKLOG.md`](BACKLOG.md).

### Step 2 - Modeling VIX Regimes

Both assignment approaches will be implemented:

- discrete Markov chains with **2 and 3 quantile-defined states**;
- Gaussian Hidden Markov Models with **2 and 3 states**, estimated by EM.

Step 2 will display:

- Markov quantile thresholds;
- transition matrices;
- stationary distributions;
- HMM fitted means, variances, initial probabilities and transition matrices;
- most likely/Viterbi state sequences;
- HMM smoothed/posterior probabilities;
- VIX plots with color-coded Markov and HMM states.

### Step 3 - State Selection and Interpretation

The analysis will compute log-likelihood, AIC and BIC for all four candidates.

AIC/BIC will be used **within each model family** because the Markov-chain likelihood is defined on a discretized state sequence while the HMM likelihood is defined on continuous `VIX_change` observations. The preferred method is then selected using the deterministic validity/interpretability rule fixed in [`BACKLOG_STEPS_2_4.md`](BACKLOG_STEPS_2_4.md).

For the preferred state sequence, the project will compute daily mean log return, daily standard deviation and observation count for `TLT`, `GLD`, and `SPY` by state and visualize those results.

### Step 4 - Designing the Rotation Strategy

For every preferred-model state, the strategy will allocate **100% to the ETF with the highest historical mean daily log return in that state**.

The mapping is deterministic. Exact ties use this order:

```text
TLT -> GLD -> SPY
```

The optional 60/40 extension is deliberately excluded from Steps 2-4. The later Step 5 backtest must apply a one-trading-day execution lag.

The complete atomic implementation contracts for Steps 2-4 are in [`BACKLOG_STEPS_2_4.md`](BACKLOG_STEPS_2_4.md).

## Canonical analysis and sidecars

### Canonical notebook

The primary assignment analysis will live in:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

Testable numerical logic lives in `src/vix_regime_allocation/`, but the notebook is the canonical analysis artifact: it calls those functions, displays their outputs, shows equations, shows plots and tables, and gives precise scientific-paper-style explanations and limitations.

The notebook must be committed with successful stored outputs.

### README sidecar policy

`README.md` is a **sidecar document** and must describe the repository as it actually exists.

After Steps 2-4 are executed, the README will contain a generated analysis block delimited by:

```text
<!-- BEGIN NOTEBOOK ANALYSIS OUTPUT -->
<!-- END NOTEBOOK ANALYSIS OUTPUT -->
```

That block must show the same canonical equations, model comparison, model decision, figures, state statistics and allocation mapping as the notebook. README synchronization must not refit a model or independently calculate a result.

Update the README in the same PR whenever a change affects:

- project scope or current implementation status;
- public functions or execution commands;
- repository structure or canonical paths;
- dependencies or Python requirements;
- generated outputs;
- CI/quality-gate behavior;
- coverage requirements;
- assignment workflow or report locations.

For purely internal changes that leave documented contracts unchanged, avoid unnecessary README edits so parallel weak-agent PRs do not conflict.

### PDF report sidecar

The final no-code report is fixed at:

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
```

It will use **page 1 only** of `reports/Template_Stochastic_Modeling_Group_Work_Project.pdf` as its cover. The template's instruction page 2 must not appear in the final report.

The PDF must show the same canonical equations, numerical tables, figures, selected model and state-to-allocation mapping as the executed notebook. It must consume the same generated artifacts rather than independently re-estimating the models.

### Same-output contract

The notebook, README and PDF report must use the same canonical outputs under:

```text
reports/tables/
reports/figures/
reports/generated/
```

The final Steps 2-4 CI work adds an `analysis-sidecars` gate that detects missing/stale notebook, README or PDF outputs.

## Fixed Step 1 outputs

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

## Planned canonical Step 2-4 outputs

Key generated artifacts will include:

```text
reports/tables/step3_model_comparison.csv
reports/tables/step3_state_asset_statistics.csv
reports/tables/step4_allocation_mapping.csv
reports/figures/step2_markov_vix_states.png
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
reports/figures/step3_state_asset_statistics.png
reports/generated/steps_2_4_manifest.json
reports/generated/step3_selected_model.json
```

The exhaustive path contract is defined in `BACKLOG_STEPS_2_4.md`.

## Repository layout

Current repository structure:

```text
.
├── .github/
│   └── workflows/
│       └── quality-gates.yml
├── reports/
│   ├── Template_Stochastic_Modeling_Group_Work_Project.md
│   └── Template_Stochastic_Modeling_Group_Work_Project.pdf
├── scripts/
│   └── check_readme_sidecar.py
├── src/
│   └── vix_regime_allocation/
│       └── __init__.py
├── tests/
│   ├── integration/
│   │   └── test_repository_smoke.py
│   └── test_bootstrap.py
├── BACKLOG.md
├── BACKLOG_STEPS_2_4.md
├── README.md
└── pyproject.toml
```

The structure grows only through the atomic PRs in the two backlog files.

## Development setup

Python `3.11+` is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Quality gates

The repository uses `.github/workflows/quality-gates.yml` on both **push** and **pull request** events.

The four core checks start independently and run in parallel:

| Gate | Command | Requirement |
|---|---|---|
| Lint | `ruff check .` and `ruff format --check .` | Must pass |
| Type check | `mypy src` | Must pass |
| Unit tests | `coverage run -m pytest -m "not integration"` | Must pass |
| Integration tests | `coverage run -m pytest -m integration` | Must pass |

After unit and integration jobs finish, coverage data is combined.

### Coverage threshold

The quality gate requires **at least 90% combined line coverage** for `src/vix_regime_allocation`.

```toml
[tool.coverage.report]
fail_under = 90
```

The `coverage` job fails below 90%.

### Aggregate gate

The current `quality-gate` requires:

- `lint`
- `type-check`
- `unit-tests`
- `integration-tests`
- `coverage`
- `readme-sidecar`

PR-21 in the Steps 2-4 backlog will also require `analysis-sidecars` after the notebook/README/PDF synchronization machinery exists.

### Required `main` branch rule

The GitHub Actions workflow runs checks, but blocking an invalid merge requires a repository branch rule for `main`:

- require a pull request before merging;
- require status checks before merging;
- require `quality-gate`;
- require the branch to be up to date;
- block force pushes;
- block branch deletion.

## Local quality commands

```bash
ruff check .
ruff format --check .
mypy src
coverage erase
coverage run --data-file=.coverage.unit -m pytest -q -m "not integration"
coverage run --data-file=.coverage.integration -m pytest -q -m integration
coverage combine
coverage report --fail-under=90
python scripts/check_readme_sidecar.py
```

## PR rules for weak parallel agents

Agents must follow the relevant backlog literally:

- Step 1: [`BACKLOG.md`](BACKLOG.md)
- Steps 2-4: [`BACKLOG_STEPS_2_4.md`](BACKLOG_STEPS_2_4.md)

In particular:

- work only on files assigned to the PR;
- never rename fixed interfaces;
- do not implement work belonging to a later PR;
- keep tests deterministic and offline;
- keep each PR atomic;
- satisfy every numbered task and its matching acceptance criterion;
- pass the complete `quality-gate` before merge;
- update README only when the sidecar policy requires it.

## Team

- Umuhoza Denyse Graine
- Opeyemi Waliyilah Oladipupo
- Sergej Schweizer

The populated report template is available in `reports/`.
