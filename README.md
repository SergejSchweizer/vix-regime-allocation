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
| Canonical Steps 1-5 backlog | Audited, consolidated and defined in [`BACKLOG.md`](BACKLOG.md) |
| Python package scaffold | Bootstrapped |
| Push quality gates | Configured |
| Pull-request quality gates | Configured |
| Coverage threshold | 90% combined source coverage |
| `main` merge protection | Repository branch/ruleset still required |
| Step 1 implementation | Not started |
| Steps 2-4 implementation | Planned as atomic PR-06 through PR-32; not started |
| Step 5 implementation | Planned as atomic PR-33 through PR-47; not started |
| Canonical notebook | Planned at `notebooks/gwp2_vix_regime_allocation.ipynb` |
| Notebook duplicate | Planned at `reports/gwp2_vix_regime_allocation.html` |
| Standalone report sidecar | Planned at `reports/Stochastic_Modeling_GWP2_Report.pdf` |

No uncomputed assignment result is claimed in this README.

## Backlog audit outcome

The single canonical backlog was reviewed against the assignment tasks, submission requirements, template instructions and grading rubric. The audit fixes behavior that weak agents must not reinterpret:

- yfinance adjusted-close semantics are explicit rather than dependent on library defaults;
- every task has a one-to-one numbered acceptance criterion;
- Markov quantile boundaries, stationary-distribution handling and HMM restart/relabeling behavior are deterministic;
- Markov/HMM information criteria select state count only within family because the likelihoods use different observation spaces;
- Step 3 statistics and chart definitions and the Step 4 deterministic allocation rule are fixed;
- Step 5 uses a strict one-observed-trading-day lag and does not redefine the Step 3 model or Step 4 allocation rule;
- the monthly equal-weight benchmark has an explicit monthly-reset/intra-month-drift convention;
- all Step 5 portfolios use identical comparison dates and simple returns for wealth/performance calculations;
- the five Step 5 metric formulas, 252-day annualization and zero-risk-free Sharpe assumption are fixed;
- sensitivity is defined as 2-versus-3 states within the preferred model family;
- the in-sample allocation-map limitation remains explicit and the one-day lag is not described as an out-of-sample design;
- canonical notebook edits are serialized while independent source PRs are parallelized;
- README, notebook HTML and standalone PDF have explicit parity contracts;
- MLA citations, graph labels/scales and template instruction-page exclusion remain explicit acceptance requirements.

All PR definitions, dependencies, file ownership, tasks, acceptance criteria, schedules and Definitions of Done for Steps 1-5 live in **one file: [`BACKLOG.md`](BACKLOG.md)**.

## Assignment implementation plan

### Step 1 - Data Preparation and Exploration

Step 1 uses daily adjusted-close prices for TLT, GLD, SPY and VIX with explicit yfinance arguments and `auto_adjust=False`; it extracts `Adj Close` directly. The final sample is the date intersection on which all four adjusted-close values are present, with no forward/backward filling or interpolation.

ETF returns are daily log returns:

```text
X_log_return_t = ln(X_t / X_(t-1))
```

VIX is modeled using daily first differences:

```text
VIX_change_t = VIX_t - VIX_(t-1)
```

Step 1 produces:

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

The exact PR contracts are in [`BACKLOG.md`](BACKLOG.md), PR-01 through PR-05.

### Step 2 - Modeling VIX Regimes

Both assignment approaches will be implemented:

- 2-state and 3-state discrete Markov chains using empirical VIX-change quantiles;
- 2-state and 3-state univariate Gaussian Hidden Markov Models estimated by EM.

Step 2 will display and persist quantile thresholds; Markov transition matrices and stationary distributions; HMM parameters and diagnostics; Viterbi state sequences; smoothed/posterior probabilities; and VIX-level plots color-coded by states.

### Step 3 - State Selection and Interpretation

All four candidates will have log-likelihood, AIC and BIC reported. The Markov likelihood is the conditional likelihood of the discretized transition sequence, whereas the HMM likelihood is the likelihood of continuous `VIX_change` observations. Raw AIC/BIC values are therefore **not treated as a valid direct cross-family ranking**. BIC selects 2 versus 3 states within each family, followed by the deterministic preferred-method rule in [`BACKLOG.md`](BACKLOG.md).

For the preferred state sequence, Step 3 computes for TLT, GLD and SPY:

```text
mean daily log return
sample standard deviation (ddof=1)
observation count
```

The state-conditional bar chart uses mean log return as bar height and state-conditional standard deviation as its error bar.

### Step 4 - Designing the Rotation Strategy

For each preferred-model state, the rule allocates **100% to the ETF with the highest historical mean daily log return in that state**. Exact ties use:

```text
TLT -> GLD -> SPY
```

The optional 60/40 allocation is not implemented. The state-conditioned mean-return mapping is estimated in-sample, which remains an explicit limitation.

The complete atomic Step 2-4 contracts are in [`BACKLOG.md`](BACKLOG.md), PR-06 through PR-32.

### Step 5 - Backtesting and Evaluation

Step 5 is fully specified in the single canonical [`BACKLOG.md`](BACKLOG.md) as PR-33 through PR-47. It will backtest the Step 4 mapping with a one-observed-trading-day execution lag; compute Cumulative Return, Annualized Return, Annualized Volatility, Sharpe Ratio and Max Drawdown; compare against monthly rebalanced equal-weight TLT/GLD/SPY and buy-and-hold SPY; plot cumulative performance; and evaluate 2-versus-3-state sensitivity within the preferred model family.

The fixed implementation converts Step 1 ETF log returns to simple returns for wealth and performance calculations. Annualization uses 252 trading days. Sharpe uses a zero risk-free rate because the assignment supplies no risk-free series. All three portfolios use the same valid lagged comparison dates. The required backtest is gross of transaction costs, slippage and taxes, and it is **not** described as fully out-of-sample because the Step 4 allocation mapping is estimated using the analysis sample.

Planned canonical Step 5 outputs are:

```text
reports/tables/step5_daily_returns.csv
reports/tables/step5_performance_summary.csv
reports/tables/step5_state_count_sensitivity.csv
reports/figures/step5_cumulative_performance.png
reports/generated/step5_manifest.json
```

## Canonical technical notebook

The primary technical analysis artifact will be:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

Testable numerical logic lives in `src/vix_regime_allocation/`, but the notebook is where the assignment analysis is presented. It must explicitly identify each step/question number; call tested project functions; show stored function outputs, equations, tables and plots; list/pronounce Greek letters before equations in which they appear; provide precise scientific-paper-style methodology, interpretation and limitations; contain MLA-formatted citations for sources actually used; and be executed from top to bottom before commit.

Notebook edits for Steps 2-4 are serialized across PR-21 through PR-25. Step 5 notebook edits are likewise serialized across PR-40 through PR-42. Parallel edits to the canonical `.ipynb` are forbidden for the weak-agent workflow.

## Sidecar contracts

### README technical sidecar policy

`README.md` describes the repository as it actually exists. After analysis execution, it contains exactly one generated block:

```text
<!-- BEGIN NOTEBOOK ANALYSIS OUTPUT -->
<!-- END NOTEBOOK ANALYSIS OUTPUT -->
```

That block must have **technical-result parity** with the notebook. Synchronization reads canonical files and never refits models or independently recalculates a second analysis. Until numerical artifacts exist, README remains descriptive and must not invent placeholder results.

### Executed-notebook HTML duplicate

The assignment requires a duplicate version of the executable notebook in PDF or HTML format. This project will generate:

```text
reports/gwp2_vix_regime_allocation.html
```

The HTML is exported from the committed notebook's stored outputs without re-executing or refitting models. PR-46 regenerates it from the final Step 1-5 notebook.

### Standalone PDF non-technical sidecar

The no-code report path is:

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
```

It uses **page 1 only** of:

```text
reports/Template_Stochastic_Modeling_Group_Work_Project.pdf
```

as its cover. Template page 2 must never appear in the report. The PDF has **decision-result parity** with the notebook rather than copying every technical parameter table. It contains no source code, preserves the populated team names, leaves unknown fields blank, uses sources actually consulted, and is rendered to images for visual QA before merge. PR-45 regenerates the final report through Step 5.

### Parity levels

```text
Notebook <-> README: exact technical-result parity
Notebook <-> HTML: exact executed-notebook duplicate
Notebook <-> standalone PDF: decision-result parity with non-technical wording
```

PR-32 establishes the initial Steps 1-4 `analysis-sidecars` gate. PR-47 extends the same gate to final Steps 1-5 parity after all Step 5 sidecars exist.

## Planned canonical Steps 2-5 artifacts

Key outputs include:

```text
reports/tables/step2_markov_2_thresholds.csv
reports/tables/step2_markov_2_transition.csv
reports/tables/step2_markov_2_stationary.csv
reports/tables/step2_markov_3_thresholds.csv
reports/tables/step2_markov_3_transition.csv
reports/tables/step2_markov_3_stationary.csv
reports/tables/step2_hmm_2_parameters.csv
reports/tables/step2_hmm_2_transition.csv
reports/tables/step2_hmm_3_parameters.csv
reports/tables/step2_hmm_3_transition.csv
reports/tables/step3_model_comparison.csv
reports/tables/step3_state_asset_statistics.csv
reports/tables/step4_allocation_mapping.csv
reports/figures/step2_markov_vix_states.png
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
reports/figures/step3_state_asset_statistics.png
reports/generated/step3_selected_model.json
reports/generated/steps_2_4_manifest.json
reports/tables/step5_daily_returns.csv
reports/tables/step5_performance_summary.csv
reports/tables/step5_state_count_sensitivity.csv
reports/figures/step5_cumulative_performance.png
reports/generated/step5_manifest.json
```

The exhaustive schema/path contracts are in [`BACKLOG.md`](BACKLOG.md).

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

The repository uses `.github/workflows/quality-gates.yml` on both **push** and **pull request** events. The four core jobs start independently and therefore run in parallel:

| Gate | Command | Requirement |
|---|---|---|
| Lint | `ruff check .` and `ruff format --check .` | Must pass |
| Type check | `mypy src` | Must pass |
| Unit tests | `coverage run -m pytest -m "not integration"` | Must pass |
| Integration tests | `coverage run -m pytest -m integration` | Must pass |

After unit and integration jobs finish, their coverage data is combined.

### Coverage threshold

The repository requires **at least 90% combined line coverage** for `src/vix_regime_allocation`.

```toml
[tool.coverage.report]
fail_under = 90
```

The workflow additionally enforces `coverage report --fail-under=90`.

### Aggregate quality-gate

The current aggregate `quality-gate` requires `lint`, `type-check`, `unit-tests`, `integration-tests`, `coverage`, and `readme-sidecar`. PR-32 adds the initial `analysis-sidecars` requirement after the Steps 1-4 analysis artifacts exist; PR-47 extends that gate through Step 5 while preserving the same parallel core jobs and 90% coverage threshold.

### Required `main` branch rule

GitHub Actions can run checks, but technical merge blocking requires a repository rule/ruleset for `main`: require a pull request before merging; require status checks and `quality-gate`; require the branch to be up to date; block force pushes; and block branch deletion. Until that repository setting is enabled, the owner must apply these merge rules manually.

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

After the parity checker is implemented, final local validation also includes:

```text
python scripts/check_analysis_sidecars.py
```

## PR rules for weak parallel agents

Agents follow the **single canonical backlog**: [`BACKLOG.md`](BACKLOG.md).

Core rules:

- work only on files owned by the PR;
- never rename fixed interfaces, schemas or paths;
- never implement a later PR early;
- keep tests deterministic and offline;
- keep each PR atomic;
- satisfy every numbered task and its identically numbered acceptance criterion;
- start only after dependencies are merged to `main`;
- pass complete `quality-gate` before merge.

## Team

- Umuhoza Denyse Graine
- Opeyemi Waliyilah Oladipupo
- Sergej Schweizer

The populated report template is available in `reports/`.
