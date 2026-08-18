# VIX Regime Allocation

Regime-based allocation project for **MScFE 622: Stochastic Modeling — Group Work Project #2**.

The project studies whether VIX-driven volatility regimes can support a transparent rotation strategy across:

- `TLT` — long-duration U.S. Treasury ETF
- `GLD` — gold ETF
- `SPY` — S&P 500 ETF
- `^VIX` / `VIX` — volatility-state input

The assignment workflow is implemented step by step. The current backlog covers **Step 1: Data Preparation and Exploration** only.

## Current status

| Area | Status |
|---|---|
| Assignment template | Added |
| Step 1 backlog | Defined in `BACKLOG.md` |
| Python package scaffold | Bootstrapped |
| Push quality gates | Configured |
| Pull-request quality gates | Configured |
| `main` merge protection | Repository branch rule still required |
| Step 1 implementation | Not started |
| Step 2+ implementation | Out of scope for the current backlog |

## Step 1 scope

Step 1 must:

1. Download daily adjusted close prices for `TLT`, `GLD`, `SPY`, and `^VIX` from Yahoo Finance using the maximum available history.
2. Restrict the data to the maximum common sample period.
3. Compute daily log returns for `TLT`, `GLD`, and `SPY`.
4. Compute daily VIX change as `VIX_t - VIX_{t-1}`.
5. Align all derived series on common dates and remove missing values.
6. Produce one plot containing the three ETF return series.
7. Produce one plot containing VIX change.
8. Leave a clean dataset reusable by later assignment steps.

The detailed implementation contract, exact schemas, file ownership, PR dependencies, tasks, and acceptance criteria are defined in [`BACKLOG.md`](BACKLOG.md).

## Fixed Step 1 outputs

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

## Repository layout

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
├── README.md
└── pyproject.toml
```

The structure will grow only through the atomic PRs defined in the backlog.

## Development setup

Python `3.11+` is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Quality gates

The repository uses `.github/workflows/quality-gates.yml` on both **push** and **pull request** events.

The four core checks start independently and therefore run in parallel:

| Gate | Command | Requirement |
|---|---|---|
| Lint | `ruff check .` and `ruff format --check .` | Must pass |
| Type check | `mypy src` | Must pass |
| Unit tests | `coverage run -m pytest -m "not integration"` | Must pass |
| Integration tests | `coverage run -m pytest -m integration` | Must pass |

After the unit and integration jobs finish, their coverage data is combined into one repository coverage result.

### Coverage threshold

The quality gate requires **at least 90% combined line coverage** for `src/vix_regime_allocation`.

The threshold is defined in `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 90
```

The `coverage` job fails if the combined unit + integration coverage is below 90%.

### Aggregate gate

The final `quality-gate` job succeeds only when all of these have succeeded:

- `lint`
- `type-check`
- `unit-tests`
- `integration-tests`
- `coverage`
- `readme-sidecar`

### Required `main` branch rule

The GitHub Actions workflow can run the checks, but blocking an invalid merge requires a repository branch rule for `main`.

Configure `main` with these requirements:

- require a pull request before merging;
- require status checks to pass before merging;
- require the `quality-gate` status check;
- require the branch to be up to date before merging;
- block force pushes;
- block branch deletion.

Once that rule is enabled, a PR cannot merge unless the aggregate `quality-gate` succeeds.

## Local quality commands

Run the same checks locally before pushing:

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

## README sidecar policy

`README.md` is a **sidecar document** and must describe the repository as it actually exists.

Update the README in the same PR whenever a change affects any of the following:

- project scope or current implementation status;
- public functions or user-facing execution commands;
- repository structure or canonical paths;
- dependencies or Python requirements;
- generated outputs;
- CI/quality-gate behavior;
- coverage requirements;
- assignment workflow or report locations.

Do **not** edit the README for a purely internal change that leaves all documented contracts unchanged. This avoids unnecessary merge conflicts between parallel weak-agent PRs.

`scripts/check_readme_sidecar.py` protects the non-negotiable README contracts, including the 90% coverage threshold and the required CI jobs.

## PR rules for weak parallel agents

Agents must follow `BACKLOG.md` literally:

- work only on the files assigned to the PR;
- do not rename fixed interfaces;
- do not implement later assignment steps early;
- keep tests deterministic and offline unless a PR explicitly requires a live integration boundary;
- keep each PR atomic;
- satisfy every task and every acceptance criterion;
- pass the complete `quality-gate` before merge;
- update this README only when the sidecar policy above requires it.

## Team

- Umuhoza Denyse Graine
- Opeyemi Waliyilah Oladipupo
- Sergej Schweizer

The populated report template is available in `reports/`.
