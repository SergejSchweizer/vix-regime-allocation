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
| Auto-complete after successful Quality Gates | Configured in `.github/workflows/auto-complete.yml` |
| Combined source coverage threshold | 90% |
| Step 1 implementation | Complete: PR-01 through PR-05 merged; canonical dataset, figures, notebook, and scientific references available |
| Step 2 implementation | Complete: PR-06 through PR-16 and PR-21/PR-22 merged; Markov/HMM tables, figures, canonical state paths, executed notebook, and scientific references available |
| Step 3 implementation | Complete: PR-17 through PR-19 and PR-23/PR-24 merged; model comparison, selected-state provenance, state-conditional ETF statistics, figure, executed notebook, and scientific references available |
| Step 4 implementation | Not started |
| Step 5 implementation | Not started |
| Final submission bundle | Planned in PR-48/PR-49 |
| `main` branch protection | Repository ruleset still must be enabled in GitHub settings |

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

Implemented both assignment families with exactly two and three states:

- quantile-discretized Markov chains with transition matrices and stationary distributions;
- univariate Gaussian HMMs with deterministic restarts, fitted parameters, Viterbi states, and smoothed probabilities.

All four candidate state sequences are persisted as canonical `Date,state` CSVs instead of being refit/redecoded later merely to recover an existing state path.

### Step 3 — State Selection and Interpretation

Implemented log-likelihood, AIC, and BIC comparison. Because Markov and HMM likelihoods are defined on different observation spaces, state count is selected by BIC **within family**, not by raw cross-family AIC/BIC comparison. The preferred-method rule selected **Markov K=2**: HMM K=3 won within the HMM family by BIC but failed the fixed minimum 5% decoded-state-occupancy diagnostic, so the deterministic Markov fallback was used.

The preferred state sequence is persisted at:

```text
reports/tables/step3_selected_states.csv
```

State-conditional TLT/GLD/SPY mean daily log return, sample standard deviation (`ddof=1`), and count are computed and visualized in `reports/tables/step3_state_asset_statistics.csv` and `reports/figures/step3_state_asset_statistics.png`.

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
| Lint | `ruff check .` + `ruff format --check src tests scripts` | pass |
| Type check | `mypy src` | pass |
| Unit tests (`unit-tests`) | `coverage run -m pytest -m "not integration"` | pass |
| Integration tests (`integration-tests`) | `coverage run -m pytest -m integration` | pass |
| README sidecar | `python scripts/check_readme_sidecar.py` | pass |
| Backlog contract | `python scripts/check_backlog_contract.py` | pass |
| Coverage | combined unit + integration | `>=90%` |

Ruff lint still examines supported Python and notebook code across the repository. The formatter gate is intentionally limited to the Python source, test, and script trees so the formatter does not rewrite the executed scientific notebook as a side effect of CI.

The aggregate `quality-gate` requires all current jobs. PR-32 later adds the first `analysis-sidecars` parity job after the corresponding notebook/README/HTML/PDF artifacts exist; PR-47 extends that checker through Step 5.

`.github/workflows/auto-complete.yml` listens only to completed **Quality Gates** runs originating from pull requests. After successful Quality Gates, **Auto Complete** merges only an open, non-draft PR targeting `main` whose current head SHA is exactly the SHA that passed validation. If `main` advanced after that validation, the workflow updates the PR branch and waits for a fresh Quality Gates run instead of merging stale validation. A successful auto-complete uses a merge commit and requests deletion of the feature branch. The workflow does not check out or execute PR code and therefore keeps its write-scoped token isolated from untrusted test execution.

Local planning-stage checks:

```bash
ruff check .
ruff format --check src tests scripts
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

The intended `main` ruleset is:

- require changes through a pull request;
- require the `quality-gate` status check before merge;
- require the PR branch to be up to date with `main` before merge;
- require zero approving reviews so fully automated backlog PRs can complete without a human approval bottleneck;
- block force pushes;
- block branch deletion.

The Auto Complete workflow is already configured to merge only after successful Quality Gates and to refresh a stale PR before a new validation run. GitHub repository settings must still enable the `main` branch/ruleset above for server-side enforcement; until that setting is enabled, the workflow discipline is implemented but direct privileged pushes remain technically possible.

## Team

- Umuhoza Denyse Graine
- Opeyemi Waliyilah Oladipupo
- Sergej Schweizer
