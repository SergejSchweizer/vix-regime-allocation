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
| Canonical Steps 1-4 backlog | Audited, consolidated and defined in [`BACKLOG.md`](BACKLOG.md) |
| Python package scaffold | Bootstrapped |
| Push quality gates | Configured |
| Pull-request quality gates | Configured |
| Coverage threshold | 90% combined source coverage |
| `main` merge protection | Repository branch/ruleset still required |
| Step 1 implementation | Not started |
| Steps 2-4 implementation | Planned as atomic PR-06 through PR-32; not started |
| Canonical notebook | Planned at `notebooks/gwp2_vix_regime_allocation.ipynb` |
| Notebook duplicate | Planned at `reports/gwp2_vix_regime_allocation.html` |
| Standalone report sidecar | Planned at `reports/Stochastic_Modeling_GWP2_Report.pdf` |

No uncomputed assignment result is claimed in this README.

## Backlog audit outcome

The single canonical backlog was reviewed against the assignment tasks, submission requirements, template instructions and grading rubric. The audit introduced several explicit safeguards:

- yfinance adjusted-close semantics are fixed explicitly rather than depending on library defaults;
- every task has a one-to-one numbered acceptance criterion;
- Markov quantile boundary behavior is deterministic;
- Markov stationary distributions must be numerically unique rather than returning an arbitrary eigenvector;
- HMM restart and equal-likelihood tie behavior are deterministic;
- HMM state relabeling is deterministic even under equal fitted means;
- Markov and HMM information criteria are displayed together but state-count selection occurs within model family because the likelihoods are defined on different observation spaces;
- the Step 3 bar chart is explicitly defined as mean-return bars with state-conditional standard-deviation error bars;
- the in-sample allocation-map/lookahead limitation is explicitly documented before Step 5;
- the canonical notebook is split into small sequential integration PRs instead of one oversized notebook PR;
- README, notebook HTML and standalone PDF have explicit parity contracts;
- the standalone PDF is treated as a non-technical report, in line with the rubric, rather than duplicating technical model mechanics;
- MLA citations, graph labels/scales and the template instruction-page exclusion are explicit acceptance requirements.

All PR definitions, dependencies, file ownership, tasks, acceptance criteria, schedules and Definitions of Done for Steps 1-4 now live in **one file: [`BACKLOG.md`](BACKLOG.md)**.

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

Step 2 will display and persist:

- quantile thresholds;
- Markov transition matrices and stationary distributions;
- HMM means, variances, initial probabilities and transition matrices;
- HMM restart/convergence diagnostics;
- most-likely/Viterbi state sequences;
- smoothed/posterior probabilities;
- VIX-level plots color-coded by Markov and HMM states.

### Step 3 - State Selection and Interpretation

All four candidates will have log-likelihood, AIC and BIC reported.

The Markov likelihood is the conditional likelihood of the discretized transition sequence, whereas the HMM likelihood is the likelihood of the continuous `VIX_change` observations. Therefore raw AIC/BIC values are **not treated as a valid direct cross-family ranking**. BIC selects 2 vs 3 states within each family. The preferred family then follows the deterministic validity/interpretability rule fixed in [`BACKLOG.md`](BACKLOG.md).

For the preferred state sequence, Step 3 computes for TLT, GLD and SPY:

```text
mean daily log return
sample standard deviation (ddof=1)
observation count
```

The required state-conditional bar chart uses mean log return as bar height and one state-conditional standard deviation as the error bar.

### Step 4 - Designing the Rotation Strategy

For each preferred-model state, the rule allocates **100% to the ETF with the highest historical mean daily log return in that state**.

Exact ties use:

```text
TLT -> GLD -> SPY
```

The optional 60/40 allocation is not implemented.

The analysis explicitly notes that the state-conditioned mean-return mapping is estimated in-sample. Step 5 must therefore be interpreted carefully and must at minimum apply the assignment-required one-trading-day lag to the state-driven position; a fully out-of-sample implementation would require rolling or expanding re-estimation.

The complete atomic Step 2-4 contracts are in [`BACKLOG.md`](BACKLOG.md), PR-06 through PR-32.

## Canonical technical notebook

The primary technical analysis artifact will be:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

Testable numerical logic lives in `src/vix_regime_allocation/`, but the notebook is where the assignment analysis is actually presented. It must:

- explicitly identify each step/question number;
- call tested project functions;
- show function outputs;
- show equations and parameter definitions;
- list/pronounce Greek letters before equations in which they appear;
- show plots and tables with readable axes, labels and scales;
- provide precise scientific-paper-style methodology, interpretation and limitations;
- contain in-text citations and an MLA-formatted bibliography for sources actually consulted;
- be executed from top to bottom with outputs stored before commit.

Notebook edits for Steps 2-4 are deliberately serialized across PR-21 through PR-25 because parallel edits to one `.ipynb` file are unsafe for weak agents.

## Sidecar contracts

### README technical sidecar policy

`README.md` describes the repository as it actually exists. After the analysis is executed, it will contain exactly one generated block:

```text
<!-- BEGIN NOTEBOOK ANALYSIS OUTPUT -->
<!-- END NOTEBOOK ANALYSIS OUTPUT -->
```

That block must have **technical-result parity** with the notebook: same equations, model-comparison numbers, preferred result, state statistics, allocation mapping and canonical figures. Synchronization reads canonical files and never refits a model.

Until those numerical artifacts exist, the README must remain descriptive and must not invent placeholder results.

For purely internal implementation changes that do not change a documented user-facing/canonical contract, unnecessary README edits are avoided to prevent merge conflicts between parallel weak-agent PRs.

### Executed-notebook HTML duplicate

The assignment requires a duplicate version of the executable notebook in PDF or HTML format. This project will generate:

```text
reports/gwp2_vix_regime_allocation.html
```

The HTML is exported from the committed notebook's stored outputs without re-executing or refitting the models. It must be regenerated after Step 5 before final submission.

### Standalone PDF non-technical sidecar

The no-code report path is:

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
```

It uses **page 1 only** of:

```text
reports/Template_Stochastic_Modeling_Group_Work_Project.pdf
```

as its cover. Template page 2 contains submission instructions and must never appear in the report.

The PDF has **decision-result parity** with the notebook rather than copying every technical parameter table. It contains the same state interpretation, state-conditional ETF statistics, decision-relevant figures and allocation mapping, but its prose avoids model names, algorithm names, library names and unnecessary technical mechanics in accordance with the non-technical-report rubric.

The report must contain no source code, must preserve the three populated team names, must leave unknown fields blank, must use MLA citations/bibliography and must be rendered to images for visual QA before merge. It covers Steps 1-4 initially and must be regenerated after Step 5 before final submission.

### Parity levels

```text
Notebook <-> README: exact technical-result parity
Notebook <-> HTML: exact executed-notebook duplicate
Notebook <-> standalone PDF: decision-result parity with non-technical wording
```

The final Steps 1-4 CI work adds an `analysis-sidecars` job that validates those contracts.

## Planned canonical Step 2-4 artifacts

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
```

The exhaustive schema/path contract is in [`BACKLOG.md`](BACKLOG.md).

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

The four core jobs start independently and therefore run in parallel:

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

The coverage job also enforces:

```text
coverage report --fail-under=90
```

### Aggregate quality-gate

The current aggregate `quality-gate` requires:

- `lint`
- `type-check`
- `unit-tests`
- `integration-tests`
- `coverage`
- `readme-sidecar`

PR-32 in the unified [`BACKLOG.md`](BACKLOG.md) adds `analysis-sidecars` after notebook/README/HTML/PDF synchronization exists.

### Required `main` branch rule

GitHub Actions can run the checks, but technical merge blocking requires a repository rule/ruleset for `main`:

- require a pull request before merging;
- require status checks before merging;
- require `quality-gate`;
- require the branch to be up to date;
- block force pushes;
- block branch deletion.

The connected GitHub tooling currently cannot enable that repository rule, so until it is configured in GitHub settings the owner must enforce the merge rule manually.

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

After PR-32, local final validation also includes:

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
