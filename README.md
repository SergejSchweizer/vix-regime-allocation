# VIX Regime Allocation

Regime-based allocation project for **MScFE 622: Stochastic Modeling — Group Work Project #2**.

The active assignment analysis uses **Gaussian Hidden Markov Models (HMMs) only**. Daily `VIX_change` is modeled with HMM candidates `K=2` and `K=3`; the preferred valid HMM is selected by BIC, translated into deterministic state-conditional ETF rankings, and evaluated with both required allocation rules: **100% Keep** and **60/40 Spread**.

## Current repository status

| Area | Status |
|---|---|
| Canonical implementation backlog | `BACKLOG.md`, revision PR-50 through PR-68 |
| Backlog structural validator | `scripts/check_backlog_contract.py` |
| Python package | Implemented under `src/vix_regime_allocation` |
| Assignment regime model | Gaussian HMM only, `K=2` and `K=3` candidates |
| Selected assignment model | HMM `K=2` |
| Step 1 | Canonical common-sample dataset is fixed in `data/processed/step1_data.csv` |
| Step 2 | HMM `K=2`/`K=3` parameters, transitions, Viterbi states, and diagnostics are canonical |
| Step 3 | HMM-only validity checks and valid-candidate BIC selection are canonical |
| Step 4 | Both `100_keep` and `60_40_spread` allocations are canonical |
| Step 5 | Both HMM strategies plus monthly equal-weight and SPY buy-and-hold benchmarks are canonical |
| Numerical artifact consistency | `scripts/check_analysis_consistency.py` and artifact-provenance checks |
| Notebook | `notebooks/gwp2_vix_regime_allocation.ipynb`, fully executed with stored outputs |
| HTML sidecar | `reports/gwp2_vix_regime_allocation.html` |
| PDF sidecar | `reports/Stochastic_Modeling_GWP2_Report.pdf`, derived from the executed notebook |
| Scientific references | Canonical registry `reports/references.bib`, MLA 9 |
| Combined source coverage threshold | 90% |

The repository does not claim an uncomputed assignment result. Canonical numerical artifacts are generated from the committed Step 1 dataset and are reconciled by independent consistency/provenance gates.

## Verified empirical result

The selected assignment specification is **HMM K=2**. HMM `K=3` has the lower raw BIC (`18039.55561963191` versus `18672.251234140676`) but fails the fixed project validity rule because its minimum Viterbi-state occupancy is `0.047392497712717294`, below the required `0.05`. HMM `K=2` is valid and is therefore selected among valid HMM candidates. The 5% occupancy threshold is a project stability rule, not a statistical theorem.

The resulting state rankings produce these two canonical Step 4 mappings:

### 100% Keep

| State | Rank 1 | Rank 2 | TLT weight | GLD weight | SPY weight |
|---:|---|---|---:|---:|---:|
| 0 | SPY | GLD | 0.00 | 0.00 | 1.00 |
| 1 | TLT | GLD | 1.00 | 0.00 | 0.00 |

### 60/40 Spread

| State | Rank 1 | Rank 2 | TLT weight | GLD weight | SPY weight |
|---:|---|---|---:|---:|---:|
| 0 | SPY | GLD | 0.00 | 0.40 | 0.60 |
| 1 | TLT | GLD | 0.60 | 0.40 | 0.00 |

The required one-observed-row lagged comparison contains exactly 5,464 common return observations:

| Portfolio | Cumulative return | Annualized return | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| `hmm_100_keep` | 4662.9595% | 19.5044% | 14.2125% | 1.3252 | -19.5403% |
| `hmm_60_40_spread` | 2667.4627% | 16.5491% | 11.9317% | 1.3435 | -16.4250% |
| `equal_weight_monthly` | 542.0849% | 8.9548% | 9.7112% | 0.9319 | -23.0437% |
| `spy_buy_hold` | 879.8148% | 11.0994% | 18.8845% | 0.6520 | -55.1894% |

The HMM state-count × allocation sensitivity table uses the same 5,464 return dates for all four rows:

| HMM states | Method | Cumulative return | Sharpe | Maximum drawdown |
|---:|---|---:|---:|---:|
| 2 | `100_keep` | 4662.9595% | 1.3252 | -19.5403% |
| 2 | `60_40_spread` | 2667.4627% | 1.3435 | -16.4250% |
| 3 | `100_keep` | 3589.4722% | 1.1528 | -24.0096% |
| 3 | `60_40_spread` | 2015.3970% | 1.2235 | -19.6571% |

Sensitivity is diagnostic only. It does not override Step 3 selection and it does not turn the full-sample assignment analysis into a causal trading experiment.

## Core mathematical conventions

### Step 1 — ETF log return and VIX first difference

For ETF `i` and observed trading row `t`:

```text
r[i,t] = ln(P[i,t] / P[i,t-1])
```

The HMM observation is:

```text
VIX_change[t] = VIX[t] - VIX[t-1]
```

The common sample is formed before lagged quantities are calculated; there is no forward fill, backward fill, or interpolation.

### Step 2 — Gaussian Hidden Markov Model

The latent regime process is first-order Markov. Conditional on state `j`, the observed `VIX_change` is Gaussian with a state-specific mean and variance. The fitted parameter set contains initial-state probabilities, the transition matrix, state means, and state variances.

Expectation-Maximization / Baum-Welch alternates between posterior-responsibility calculation with forward-backward probabilities and parameter re-estimation. Because EM can converge to local optima, the project uses deterministic multi-start fitting with fixed seeds and chooses the greatest converged finite log-likelihood for each `K`.

Viterbi decoding returns the most likely **joint state path**. Smoothed posterior probabilities are pointwise full-sample posterior probabilities; they use future observations relative to historical dates and are therefore non-causal.

### Step 3 — information criteria and validity

For maximized log-likelihood `log L`, free-parameter count `k`, and observation count `n`:

```text
AIC = 2k - 2 log L
BIC = k ln(n) - 2 log L
```

For an HMM with `K` states, the project uses:

```text
k = K^2 + 2K - 1
```

Only valid HMM candidates participate in selection. A valid candidate must be converged and finite, have finite ordered means, positive finite variances, normalized initial/transition/smoothed probabilities, valid Viterbi labels, and minimum Viterbi occupancy at least 5%. The smallest BIC among valid candidates wins; ties within `1e-12` select the lower `K`.

### Step 4 — allocation rules

Within each selected HMM state, TLT, GLD, and SPY are ranked by descending historical state-conditional mean daily log return. Exact equal means use fixed priority `TLT -> GLD -> SPY`.

```text
100_keep:
  rank 1 = 1.00
  rank 2 = 0.00
  rank 3 = 0.00

60_40_spread:
  rank 1 = 0.60
  rank 2 = 0.40
  rank 3 = 0.00
```

### Step 5 — execution, compounding, and metrics

The state observed on row `t-1` determines weights applied to ETF returns on row `t`. Log returns are converted to simple returns before portfolio arithmetic:

```text
simple_return[i,t] = exp(r[i,t]) - 1
portfolio_return[t] = sum_i weight[i,t-1] * simple_return[i,t]
```

Cumulative wealth starts at `W_0 = 1` and compounds simple returns. Metrics use 252 trading days, zero risk-free rate, and sample standard deviation with `ddof=1`.

**Greek letter used below:** σ — *sigma*, pronounced “SIG-muh”.

```text
annualized volatility = σ_daily * sqrt(252)
Sharpe = mean(daily simple return) / σ_daily * sqrt(252)
```

Maximum drawdown includes the initial wealth point in the running peak.

## Look-ahead limitation

The required one-row execution lag avoids same-row execution, but **does not make the assignment backtest causal or out-of-sample**. HMM parameters are estimated on the full sample, Viterbi decoding and smoothing use the full sequence, and state-conditional ETF rankings use full-sample state assignments. Earlier historical decisions therefore indirectly depend on later observations.

The assignment result is a deterministic, reproducible full-sample backtest. It is not evidence of a live trading edge. Any retained predictive extension is evaluated separately with one-sided/expanding inference and remains HMM-only after the revision.

## Canonical artifacts

Processed data:

```text
data/processed/step1_data.csv
```

Step 2 HMM artifacts:

```text
reports/tables/step2_hmm_2_parameters.csv
reports/tables/step2_hmm_3_parameters.csv
reports/tables/step2_hmm_2_transition.csv
reports/tables/step2_hmm_3_transition.csv
reports/tables/step2_hmm_2_states.csv
reports/tables/step2_hmm_3_states.csv
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
```

Step 3 selection/statistics:

```text
reports/tables/step3_model_comparison.csv
reports/tables/step3_selected_states.csv
reports/tables/step3_state_asset_statistics.csv
reports/figures/step3_state_asset_statistics.png
reports/generated/step3_selected_model.json
```

Step 4 allocations:

```text
reports/tables/step4_allocation_100_keep.csv
reports/tables/step4_allocation_60_40_spread.csv
```

Step 5 outputs:

```text
reports/tables/step5_daily_returns.csv
reports/tables/step5_performance_summary.csv
reports/tables/step5_state_count_sensitivity.csv
reports/figures/step5_cumulative_performance.png
reports/generated/step5_manifest.json
```

Primary technical notebook and sidecars:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/gwp2_vix_regime_allocation.html
reports/Stochastic_Modeling_GWP2_Report.pdf
reports/Template_Stochastic_Modeling_Group_Work_Project.pdf
```

Canonical scientific-source registry:

```text
reports/references.bib
```

## Scientific citation policy

The technical notebook and PDF use **MLA 9** source attribution. `reports/references.bib` is the canonical bibliography registry. In-text citations are placed next to externally sourced definitions, equations, methodological claims, and interpretations, and the analysis ends with a cited-only **Works Cited** section.

Peer-reviewed papers and scholarly books/textbooks support HMM theory, EM/Baum-Welch estimation, Viterbi decoding, information criteria, performance metrics, and backtesting limitations. Every notebook/PDF citation must resolve to `reports/references.bib`; duplicate keys, invented metadata, unresolved citations, orphan bibliography entries, and URL-only pseudo-citations are invalid.

Parity policy:

```text
Notebook <-> README: exact technical-result parity
Notebook <-> HTML: exact executed-notebook duplicate
Notebook <-> standalone PDF: exact rendered-notebook content parity
Notebook/PDF citations -> reports/references.bib: resolved citation and Works-Cited integrity
```

The PDF uses page 1 of the supplied template as the course/group cover and excludes the template instruction page. `scripts/build_pdf_report.py` records the canonical notebook SHA-256 in PDF metadata as `/NotebookSHA256`.

## Complete numerical verification

`scripts/check_analysis_consistency.py` independently reconciles the canonical assignment artifacts. The HMM-only audit:

- validates the committed Step 1 input schema and hash;
- refits deterministic Gaussian HMM `K=2` and `K=3` candidates;
- reconciles HMM parameters, transitions, Viterbi paths, smoothed probabilities, likelihood diagnostics, AIC, BIC, and validity;
- verifies HMM-only Step 3 selection and selected-state provenance;
- recomputes state-conditional ETF statistics and both Step 4 allocation mappings;
- rebuilds both lagged HMM strategies and both required benchmarks on identical dates;
- recomputes all five performance metrics and the four-row state-count × allocation sensitivity table;
- verifies artifact hashes and manifest membership.

No Markov-family fallback participates in the active assignment analysis.

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

`.github/workflows/quality-gates.yml` runs independent jobs in parallel where possible.

| Gate | Command | Requirement |
|---|---|---|
| Lint | `ruff check .` + `ruff format --check src tests scripts` | pass |
| Type check | `mypy src` | pass |
| Unit tests | `coverage run -m pytest -m "not integration"` | pass |
| Integration tests | `coverage run -m pytest -m integration` | pass |
| README sidecar | `python scripts/check_readme_sidecar.py` | pass |
| Backlog contract | `python scripts/check_backlog_contract.py` | pass |
| Repository hygiene | `python scripts/check_repository_hygiene.py` | pass |
| Analysis consistency | `python scripts/check_analysis_consistency.py` | pass |
| Artifact provenance | `python scripts/check_artifact_provenance.py` | pass |
| Coverage | combined unit + integration | `>=90%` |

The aggregate `quality-gate` requires every applicable job above.

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
python scripts/check_artifact_provenance.py
```

## Canonical backlog and Git workflow

`BACKLOG.md` is the single canonical planning source. The active revision is PR-50 through PR-68, with explicit dependencies and disjoint file ownership for safe parallel work where possible.

Each implementation PR uses the declared branch and commit message, starts from current `main` after its dependencies, and modifies only its owned paths. The required clean-tree check is:

```bash
git status --short --branch
```

For example, PR-50 uses branch `pr-50-hmm-only-model-selection` and commit message `PR-50 — HMM-only model selection`.

## Auto Complete and main-branch rule

`.github/workflows/auto-complete.yml` listens to completed **Quality Gates** runs associated with pull requests. After successful Quality Gates, **Auto Complete** validates the tested head SHA and base, updates the branch when necessary, and merges the validated PR.

The intended server-side `main` ruleset is:

- require changes through a pull request;
- require `quality-gate` before merge;
- require the PR branch to be up to date with `main`;
- block force pushes;
- block branch deletion.

## Team

- Umuhoza Denyse Graine
- Opeyemi Waliyilah Oladipupo
- Sergej Schweizer
