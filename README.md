# VIX Regime Allocation

Regime-based allocation project for **MScFE 622: Stochastic Modeling — Group Work Project #2**.

The assignment studies a VIX-driven allocation rule across `TLT`, `GLD`, and `SPY` using discrete Markov chains and Gaussian Hidden Markov Models, then backtests the chosen regime policy against monthly equal-weight and buy-and-hold SPY benchmarks.

## Current repository status

| Area | Status |
|---|---|
| Report template | Added and populated with known team names |
| Canonical implementation backlog | Fully audited in [`BACKLOG.md`](BACKLOG.md), PR-01 through PR-49 |
| Backlog structural validator | `scripts/check_backlog_contract.py` |
| Python package scaffold | Bootstrapped |
| Push / pull-request quality gates | Configured |
| Combined source coverage threshold | 90% |
| Step 1 implementation | Complete: PR-01 through PR-05 merged; canonical dataset, figures, notebook, and scientific references available |
| Steps 2–4 implementation | Not started |
| Step 5 implementation | Not started |
| Final submission bundle | Planned in PR-48/PR-49 |
| `main` branch protection | GitHub ruleset still required |

No uncomputed assignment result is claimed in this README.

## Canonical backlog

`BACKLOG.md` is the **single canonical planning source**. It fixes PR dependencies, file ownership, public interfaces, schemas, numerical conventions, tie rules, test evidence, notebook serialization, sidecar parity, Step 5 backtesting semantics, scientific-citation integrity, final submission packaging, and the Git branch/status/commit contract for every PR.

The backlog is deliberately optimized for two weak coding agents. Every PR has explicit lower-numbered dependencies (or `none`), a complete write set, contiguous numbered tasks, one matching acceptance criterion for every task, an exact feature-branch name, an explicit `git status --short --branch` clean-tree check, and an exact commit message containing the PR number and PR name. `scripts/check_backlog_contract.py` verifies PR-01..PR-49, backward-only dependencies, contiguous task IDs, one-to-one acceptance coverage, and this Git metadata.

## Git workflow per backlog PR

Every PR section in `BACKLOG.md` declares these three fields explicitly:

```text
Git branch
Git status
Commit message
```

The required status command is:

```bash
git status --short --branch
```

Immediately before commit and immediately before merge, it must show the branch declared by that PR and no staged, modified, or untracked files. Branch names start with the matching lowercase PR identifier (for example `pr-01-...`), and commit messages start with the matching uppercase PR identifier and exact PR name (for example `PR-01 — Yahoo adjusted-close loader`).

## Scientific citation policy

The technical notebook and standalone PDF report must both contain **verifiable scientific source attribution**. `reports/references.bib` is the canonical bibliography registry and is created in PR-05, then maintained only by serialized notebook PRs when a new source is required. These citation requirements are implementation requirements, not optional guidance.

The required citation standard is **MLA 9**: in-text citations are placed adjacent to externally sourced definitions, equations, methodological claims, and interpretations, and each artifact ends with a **Works Cited** section. Peer-reviewed papers and scholarly books/textbooks provide the academic support for Markov chains, HMM/EM/decoding, information criteria, performance metrics, and backtesting limitations. Official primary sources may additionally document Yahoo/Cboe/index/data definitions, but a bare URL or data-provider page does not substitute for scholarly support of theory or methodology.

Every notebook/PDF citation must resolve to `reports/references.bib`; every entry rendered in an artifact's Works Cited must be cited in that artifact. Duplicate keys, invented metadata, unresolved citations, bibliography-only orphan entries, and URL-only pseudo-citations are invalid. Figures and tables include concise source notes distinguishing the team's own calculations from external data or methodology.

The standalone PDF remains non-technical in its narrative. Bibliographic titles may naturally contain technical terminology; the no-model/no-algorithm wording rule applies to report narrative, not to Works Cited metadata. PR-31 introduces Step1–4 citation-integrity checks and PR-47 extends them through the final Step1–5 notebook, HTML, and PDF.

## Assignment implementation plan

### Step 1 — Data Preparation and Exploration

Use Yahoo Finance adjusted closes for `TLT`, `GLD`, `SPY`, and `^VIX`, maximum common dates, no imputation, ETF daily log returns, and daily VIX first difference. PR-01 implements the deterministic adjusted-close loader with explicit Yahoo arguments, `Adj Close` extraction, `^VIX`→`VIX` renaming, canonical column order, timezone-naive sorted unique `Date` index, positive/finite non-missing price validation, and mocked offline tests. Missing price observations are intentionally preserved for PR-02, which constructs the common-date sample.

Canonical Step 1 outputs generated across PR-01 through PR-05 are:

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

### Step 2 — Modeling VIX Regimes

Implement both assignment families with exactly two and three states:

- quantile-discretized Markov chains with transition matrices and stationary distributions;
- univariate Gaussian HMMs with deterministic restarts, fitted parameters, Viterbi states, and smoothed probabilities.

All four candidate state sequences are persisted as canonical `Date,state` CSVs instead of being refit/redecoded later merely to recover an existing state path.

### Step 3 — State Selection and Interpretation

Report log-likelihood, AIC, and BIC. Because Markov and HMM likelihoods are defined on different observation spaces, state count is selected by BIC **within family**, not by raw cross-family AIC/BIC comparison. The preferred-method rule in `BACKLOG.md` then selects a valid HMM or Markov fallback transparently.

The preferred state sequence is persisted at:

```text
reports/tables/step3_selected_states.csv
```

State-conditional TLT/GLD/SPY mean daily log return, sample standard deviation (`ddof=1`), and count are computed and visualized.

### Step 4 — Rotation Strategy

For every preferred state, allocate 100% to the ETF with the largest historical mean daily log return in that state. Exact ties use:

```text
TLT -> GLD -> SPY
```

The optional 60/40 variant is not used.

### Step 5 — Backtesting and Evaluation

The required backtest uses one **observed-trading-row** execution lag. ETF log returns are converted to simple returns before portfolio arithmetic. The comparison benchmarks are exactly:

- 1/3 TLT + 1/3 GLD + 1/3 SPY, rebalanced on the first observed comparison trading date of each calendar month and allowed to drift intra-month;
- buy-and-hold SPY.

All three portfolios use identical comparison dates. Required metrics are cumulative return, annualized return, annualized volatility, zero-risk-free Sharpe ratio, and maximum drawdown. Maximum drawdown explicitly includes initial wealth `W_0 = 1` in the running peak so an initial loss is not incorrectly treated as zero drawdown.

Sensitivity compares two versus three states **within the preferred model family** on common dates.

### Important in-sample qualification

The assignment-required one-day lag delays execution but does **not** make this implementation causal or out-of-sample. The backlog requires the notebook/report to disclose that regime thresholds/model parameters are fitted on the full sample, HMM Viterbi states are full-sequence decoded when HMM is preferred, and state-conditional allocation means are full-sample estimates. A stronger validation would require rolling/expanding estimation, one-sided/filtered state inference, and decision-time-only allocation estimation; that is documented as future work rather than silently claimed as completed.

## Canonical analysis artifacts

Primary technical notebook:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

Executed-notebook duplicate:

```text
reports/gwp2_vix_regime_allocation.html
```

Standalone no-code report:

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
```

Canonical scientific-source registry:

```text
reports/references.bib
```

The PDF uses page 1 of `reports/Template_Stochastic_Modeling_Group_Work_Project.pdf` as its cover and excludes the template instruction page. It is non-technical: decision results, recommended action, portfolio-impact factors, limitations, and practical takeaways without model/algorithm/library prose. It nevertheless includes MLA 9 in-text scholarly citations, source notes, and a final Works Cited derived from the canonical registry.

Parity policy:

```text
Notebook <-> README: exact technical-result parity
Notebook <-> HTML: exact executed-notebook duplicate
Notebook <-> standalone PDF: decision-result parity with non-technical wording
Notebook/PDF citations -> reports/references.bib: resolved citation and Works-Cited integrity
```

## Final submission package

The assignment requires a ZIP containing the executable notebook and its PDF/HTML duplicate, while the no-code PDF is uploaded separately. PR-48/PR-49 therefore produce:

```text
dist/MScFE_622_GWP2_submission.zip
reports/generated/submission_manifest.json
```

The ZIP contains the notebook, HTML duplicate, README, `pyproject.toml`, canonical `reports/references.bib`, Step 1 processed data, and the local `src/vix_regime_allocation` Python package needed to keep the notebook executable. The standalone PDF is explicitly excluded from the ZIP and remains a separate upload. The bundle is deterministic and hash-manifested.

## Development setup

Python `3.11+`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Quality gates

`.github/workflows/quality-gates.yml` runs on push and pull requests. Core jobs start independently and therefore remain parallel:

| Gate | Command | Requirement |
|---|---|---|
| Lint | `ruff check .` + `ruff format --check .` | pass |
| Type check | `mypy src` | pass |
| Unit tests (`unit-tests`) | `coverage run -m pytest -m "not integration"` | pass |
| Integration tests (`integration-tests`) | `coverage run -m pytest -m integration` | pass |
| README sidecar | `python scripts/check_readme_sidecar.py` | pass |
| Backlog contract | `python scripts/check_backlog_contract.py` | pass |
| Coverage | combined unit + integration | `>=90%` |

The aggregate `quality-gate` requires all current jobs. PR-32 later adds the first `analysis-sidecars` parity job after the corresponding notebook/README/HTML/PDF artifacts exist; PR-47 extends that checker through Step 5.

Local planning-stage checks:

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
python scripts/check_backlog_contract.py
```

## Main-branch rule

The workflow can evaluate commits, but GitHub must still be configured with a branch/ruleset for `main` to technically require pull requests/status checks and block force pushes/deletion. Until that repository setting is enabled, the backlog merge rules must be applied administratively.

## Team

- Umuhoza Denyse Graine
- Opeyemi Waliyilah Oladipupo
- Sergej Schweizer
