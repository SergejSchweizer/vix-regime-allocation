# VIX Regime Allocation

Regime-based allocation project for **MScFE 622: Stochastic Modeling — Group Work Project #2**.

The project classifies daily VIX changes into volatility regimes, studies how TLT, GLD, and SPY behaved inside those regimes, converts the preferred regime specification into a deterministic allocation rule, and evaluates that rule against monthly equal-weight and SPY buy-and-hold benchmarks.

## Current repository status

| Area | Status |
|---|---|
| Report template | Added and populated with known team names |
| Canonical implementation backlog | Fully audited in [`BACKLOG.md`](BACKLOG.md), PR-01 through PR-49 |
| Backlog structural validator | `scripts/check_backlog_contract.py` |
| Python package | Implemented under `src/vix_regime_allocation` |
| Push / pull-request quality gates | Configured |
| Auto-complete after successful Quality Gates | Configured in `.github/workflows/auto-complete.yml` |
| Combined source coverage threshold | 90% |
| Step 1 implementation | Complete: canonical common-sample dataset, return transformations, figures, notebook output, and scientific references |
| Step 2 implementation | Complete: Markov K=2/K=3 and Gaussian-HMM K=2/K=3 estimates, state paths, transition outputs, diagnostics, and figures |
| Step 3 implementation | Complete: within-family AIC/BIC analysis, deterministic preferred-model rule, selected-state provenance, and state-conditional ETF statistics |
| Step 4 implementation | Complete: deterministic 100% state-to-ETF allocation mapping |
| Step 5 implementation | Complete: one-observed-row lagged rotation, monthly equal-weight and SPY benchmarks, performance metrics, K=2/K=3 sensitivity, cumulative/drawdown figure, and Step 5 manifest |
| Numerical artifact consistency | Checked by `scripts/check_analysis_consistency.py` and the `analysis-consistency` quality-gate job |
| PDF report | Template-based rendering of the complete executed notebook; notebook SHA-256 embedded in PDF metadata |
| Final submission bundle | Planned in PR-48/PR-49 |
| `main` branch protection | Repository ruleset still must be enabled in GitHub settings |

The repository does not claim an uncomputed result. Canonical numerical artifacts are now recomputed and cross-checked in CI rather than merely checked for file existence.

## Verified empirical result

The preferred specification remains **Markov K=2**. Within the HMM family, K=3 has the lower BIC, but its least-populated Viterbi state contains 259 of 5,465 observations, or 4.739249771271729%, below the project's fixed 5% minimum-occupancy diagnostic. The fallback is therefore a deterministic project rule for avoiding a very small decoded state; it is not a statistical proof that the HMM is intrinsically invalid.

The resulting Step 4 mapping is:

| State | Interpretation from daily VIX change | Selected ETF |
|---|---|---|
| 0 | Lower daily VIX-change regime | SPY |
| 1 | Higher daily VIX-change regime | TLT |

The required one-observed-row lagged backtest produces these canonical results over 5,464 comparison observations:

| Portfolio | Cumulative return | Annualized return | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| Regime rotation | 84.9021% | 2.8754% | 16.0369% | 0.2572 | -53.7600% |
| Equal weight, monthly reset | 542.0849% | 8.9548% | 9.7112% | 0.9319 | -23.0437% |
| SPY buy and hold | 879.8148% | 11.0994% | 18.8845% | 0.6520 | -55.1894% |

The poor rotation result is **not an arithmetic failure discovered in Step 5**. Independent recomputation reproduces the lagged daily returns and all five performance metrics. The important modeling distinction is that Step 3 measures a **contemporaneous** relation between `VIX_change_t` and ETF return `r_t`, whereas Step 5 must use state `t-1` to select the asset earning return `t`. A strong same-day VIX/ETF association therefore need not contain useful next-observation predictive information.

The preferred-family state-count sensitivity reinforces rather than reverses the result: Markov K=2 has cumulative return 84.9021% and Sharpe 0.2572, while Markov K=3 has cumulative return 76.2162%, Sharpe 0.2422, and maximum drawdown -65.0533% on the same 5,464 dates.

## Core mathematical conventions

### Step 1 — ETF log return and VIX first difference

For ETF `i` and observed trading row `t`:

```text
r[i,t] = ln(P[i,t] / P[i,t-1])
```

For the VIX observation used by both model families:

```text
VIX_change[t] = VIX[t] - VIX[t-1]
```

The common sample is formed **before** lagged quantities are calculated; there is no forward fill, backward fill, or interpolation.

### Step 2 — discrete transition probabilities

For transition counts `N[i,j]`:

```text
P[i,j] = N[i,j] / sum_j N[i,j]
```

The stationary row distribution is the normalized non-negative solution of:

**Greek letter used below:** π — *pi*, pronounced “pie”.

```text
π P = π
```

### Step 3 — information criteria

For maximized log-likelihood `log L`, free-parameter count `k`, and observation count `n`:

```text
AIC = 2k - 2 log L
BIC = k ln(n) - 2 log L
```

Because the quantile-state Markov likelihood and Gaussian-HMM likelihood are defined on different observation spaces, raw AIC/BIC values are **not** used for cross-family ranking. State count is selected by BIC within each family, after which the fixed HMM-validity/fallback rule is applied.

### Step 4 — allocation rule

For each selected state, the strategy assigns 100% weight to the ETF with the largest historical state-conditional mean daily log return. Exact ties use the fixed priority `TLT -> GLD -> SPY`. The optional 60/40 rule is not used.

### Step 5 — execution, compounding, and metrics

The decision from observed state `t-1` determines the portfolio weights applied to ETF simple returns at row `t`. Log returns are converted before portfolio arithmetic:

```text
simple_return[i,t] = exp(r[i,t]) - 1
portfolio_return[t] = sum_i weight[i,t-1] * simple_return[i,t]
```

Cumulative wealth starts at `W_0 = 1` and compounds simple returns. The project uses 252 trading days and a zero risk-free rate for annualized volatility and Sharpe. Maximum drawdown includes initial wealth in the running peak, so a loss at the first comparison observation cannot be incorrectly treated as zero drawdown.

**Greek letter used below:** σ — *sigma*, pronounced “SIG-muh”.

```text
annualized volatility = σ_daily * sqrt(252)
Sharpe = mean(daily simple return) / σ_daily * sqrt(252)
```

## Why the result must remain qualified

The required one-row execution lag prevents trading on a state observed on the same return row, but it **does not make this implementation causal or out-of-sample**. Regime thresholds/model parameters, the selected full-sample state path, and the state-conditional means used for the allocation map are estimated from the full historical sample. A genuinely predictive experiment would require rolling or expanding estimation, one-sided state inference, allocation estimates using decision-time information only, and explicit turnover/transaction-cost modeling.

This qualification is central to the interpretation: the current project is a deterministic, reproducible assignment backtest, not evidence of a production-ready trading edge.

## Canonical artifacts

Processed data:

```text
data/processed/step1_data.csv
```

Primary technical notebook:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

Executed-notebook duplicate:

```text
reports/gwp2_vix_regime_allocation.html
```

Template-based notebook PDF report:

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
reports/Template_Stochastic_Modeling_Group_Work_Project.pdf
```

Selected-state provenance and allocation:

```text
reports/generated/step3_selected_model.json
reports/tables/step3_selected_states.csv
reports/tables/step3_state_asset_statistics.csv
reports/tables/step4_allocation_mapping.csv
```

Step 5 outputs:

```text
reports/tables/step5_daily_returns.csv
reports/tables/step5_performance_summary.csv
reports/tables/step5_state_count_sensitivity.csv
reports/figures/step5_cumulative_performance.png
reports/generated/step5_manifest.json
```

Canonical scientific-source registry:

```text
reports/references.bib
```

## Scientific citation policy

The technical notebook and PDF contain **verifiable scientific source attribution**. `reports/references.bib` is the canonical bibliography registry. The required citation standard is **MLA 9**: in-text citations are adjacent to externally sourced definitions, equations, methodological claims, and interpretations, and the analysis ends with a **Works Cited** section.

Peer-reviewed papers and scholarly books/textbooks support Markov chains, HMM/EM/decoding, information criteria, performance metrics, and backtesting limitations. Official primary sources may additionally document Yahoo/Cboe/index/data definitions, but a bare provider URL does not replace scholarly support for theory or methodology.

Every notebook/PDF citation must resolve to `reports/references.bib`; bibliography entries rendered in an artifact must actually be cited. Duplicate keys, invented metadata, unresolved citations, bibliography-only orphan entries, and URL-only pseudo-citations are invalid.

Parity policy:

```text
Notebook <-> README: exact technical-result parity
Notebook <-> HTML: exact executed-notebook duplicate
Notebook <-> standalone PDF: exact rendered-notebook content parity
Notebook/PDF citations -> reports/references.bib: resolved citation and Works-Cited integrity
```

The PDF uses page 1 of the supplied template as the course/group cover and excludes the template instruction page. `scripts/build_pdf_report.py` records the canonical notebook SHA-256 in PDF metadata as `/NotebookSHA256` so stale rendering can be detected.

## Complete numerical verification

`scripts/check_analysis_consistency.py` is deliberately broader than the ordinary unit tests. On every quality-gate run it:

- reconstructs all Step 1 quantities that can be checked from the persisted common sample;
- recomputes Markov K=2/K=3 state paths, thresholds, transitions, stationary distributions, likelihoods, AIC, and BIC;
- refits deterministic Gaussian-HMM K=2/K=3 candidates and reconciles persisted Viterbi paths, transitions, parameters, and diagnostics;
- recomputes Step 3 model selection, state-conditional ETF statistics, and Step 4 allocation;
- recomputes Step 5 lagged rotation, both benchmarks, performance summary, sensitivity table, and artifact manifest;
- independently reconstructs the lagged rotation and the five required performance metrics without delegating those checks back to the backtest/summary functions.

This gate is intended to detect stale generated files and numerical drift even when individual source-level unit tests still pass.

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

`.github/workflows/quality-gates.yml` runs on pushes and pull requests. Independent jobs remain parallel where possible.

| Gate | Command | Requirement |
|---|---|---|
| Lint | `ruff check .` + `ruff format --check src tests scripts` | pass |
| Type check | `mypy src` | pass |
| Unit tests (`unit-tests`) | `coverage run -m pytest -m "not integration"` | pass |
| Integration tests (`integration-tests`) | `coverage run -m pytest -m integration` | pass |
| README sidecar | `python scripts/check_readme_sidecar.py` | pass |
| Backlog contract | `python scripts/check_backlog_contract.py` | pass |
| Repository hygiene | `python scripts/check_repository_hygiene.py` | pass |
| Analysis consistency | `python scripts/check_analysis_consistency.py` | pass |
| Coverage | combined unit + integration | `>=90%` |

The aggregate `quality-gate` requires every job above. The numerical audit is therefore part of the merge gate rather than an optional review workflow.

Local verification:

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
python scripts/check_repository_hygiene.py
python scripts/check_analysis_consistency.py
```

## Canonical backlog and Git workflow

`BACKLOG.md` is the **single canonical planning source**. It specifies PR dependencies, file ownership, interfaces, schemas, numerical conventions, notebook serialization, sidecar parity, backtesting semantics, citation integrity, final packaging, and Git contracts.

### Git workflow per backlog PR

Each backlog PR declares its branch, clean-tree requirement, and exact commit message. The required check is:

```bash
git status --short --branch
```

For example, PR-01 starts with the exact commit name `PR-01 — Yahoo adjusted-close loader`. Immediately before commit and merge, the declared feature branch must have no staged, modified, or untracked files.

## Auto Complete and main-branch rule

`.github/workflows/auto-complete.yml` listens to completed **Quality Gates** runs associated with pull requests. After successful Quality Gates, **Auto Complete** verifies the tested head SHA, rejects draft/stale/wrong-base PRs, updates a branch when `main` advanced, and only then merges the validated PR.

The intended server-side `main` ruleset is still:

- require changes through a pull request;
- require `quality-gate` before merge;
- require the PR branch to be up to date with `main`;
- require zero approving reviews for the automated backlog workflow;
- block force pushes;
- block branch deletion.

GitHub repository settings must still enable that ruleset for server-side enforcement; workflow discipline alone does not technically prevent a privileged direct push.

## Final submission package

The planned final package is:

```text
dist/MScFE_622_GWP2_submission.zip
reports/generated/submission_manifest.json
```

The ZIP is intended to contain the executable notebook, its HTML duplicate, README, `pyproject.toml`, `reports/references.bib`, processed Step 1 data, and the local Python package needed for notebook execution. The template-based PDF remains a separate submission artifact.

## Team

- Umuhoza Denyse Graine
- Opeyemi Waliyilah Oladipupo
- Sergej Schweizer
