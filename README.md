# VIX Regime Allocation

Regime-based allocation project for **MScFE 622: Stochastic Modeling — Group Work Project #2**.

The project studies whether changes in the CBOE VIX can be converted into useful allocation states for rotating among `TLT`, `GLD`, and `SPY`. It implements a quantile-based discrete Markov model and Gaussian Hidden Markov Models, selects a project-preferred state specification under fixed deterministic rules, constructs a state-to-ETF allocation map, and evaluates the resulting strategy with a one-observation execution lag against monthly equal-weight and SPY buy-and-hold benchmarks.

The canonical implementation plan is `BACKLOG.md`. It defines **PR-01 through PR-49**, file ownership, dependencies, acceptance criteria, submission artifacts, and the Git workflow per backlog PR. The historical PR title sequence begins with `PR-01 — Yahoo adjusted-close loader`.

## Current status

| Item | Status |
|---|---|
| Canonical backlog | Complete and consolidated in `BACKLOG.md` |
| Step 1 implementation | Complete |
| Steps 2–4 implementation | Complete |
| Step 5 computational implementation | Complete |
| Full Step 1–5 executed notebook | Complete at `notebooks/gwp2_vix_regime_allocation.ipynb` |
| Step 5 sensitivity + manifest | Complete |
| Whole-project code/plot/explanation review | Complete |
| Final HTML/PDF/submission bundle sidecars | Still governed by the remaining publication/release items in `BACKLOG.md` |

The repository now contains computed Step 1–5 results. No result in this README should be inferred from an unexecuted notebook or an uncomputed placeholder.

## Data contract

The raw download uses Yahoo Finance through `yfinance` with the exact symbols:

```python
TICKERS = {"TLT": "TLT", "GLD": "GLD", "SPY": "SPY", "VIX": "^VIX"}
```

The download contract explicitly requests `period="max"`, `interval="1d"`, `auto_adjust=False`, `back_adjust=False`, `actions=False`, and `progress=False`, then extracts **Adjusted Close** rather than `Close`.

The common sample is the exact intersection of dates on which all four adjusted-close series are present. No forward fill, backward fill, or interpolation is used. The committed Step 1 sample spans **2005-01-03 through 2026-08-17** and contains **5,465 return/change observations** after the first common price row is removed by lag construction.

For ETF `i`, the daily log return is

$$
r_{i,t}=\ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right).
$$

**Greek letters used in the following equation:** `Δ` — delta, pronounced **“DEL-tuh”**.

The modeling variable is the first difference of the VIX level in index points:

$$
\Delta VIX_t = VIX_t - VIX_{t-1}.
$$

This distinction is central: the regime models are fitted to **daily `ΔVIX`**, not to the VIX level and not to a percentage VIX return. The state figures display the VIX level only as economic context; their titles and legends explicitly state that the colors correspond to states defined from daily `ΔVIX`.

Canonical Step 1 output:

- `data/processed/step1_data.csv`
- `reports/figures/step1_etf_log_returns.png`
- `reports/figures/step1_vix_change.png`

The reviewed Step 1 return chart uses percentage-formatted return axes and concise date labels; the VIX-change chart labels the unit explicitly as VIX index points.

## Step 2 — Regime models

### Quantile Markov states

For `K=2`, the `ΔVIX` sample is split at the median. For `K=3`, it is split at the one-third and two-thirds quantiles using NumPy's linear quantile rule. Exact threshold hits are assigned to the higher state through `numpy.searchsorted(..., side="right")`.

The transition matrix uses observed one-step transition counts without smoothing:

$$
P_{ij}=\frac{N_{ij}}{\sum_j N_{ij}}.
$$

**Greek letters used in the following equation:** `π` — pi, pronounced **“pie”**.

The stationary distribution is the unique normalized nonnegative solution of

$$
\pi P=\pi,\qquad \sum_i \pi_i=1.
$$

If the stationary subspace is non-unique, the implementation fails rather than returning an arbitrary eigenvector.

For the selected Markov `K=2` candidate, the fitted threshold is:

```text
State 0: ΔVIX < -0.0999994277954101
State 1: ΔVIX >= -0.0999994277954101
```

### Gaussian HMM

The HMM implementation fits `K in {2,3}` with five deterministic restart seeds `(42,43,44,45,46)`. Each restart uses diagonal covariance, `n_iter=500`, `tol=1e-6`, and `min_covar=1e-6`. A numerical failure in one initialization is now isolated to that restart; the remaining seeds are still evaluated. Among converged finite fits, the highest log-likelihood is selected, with the smallest seed breaking ties inside the configured likelihood tolerance.

**Greek letters used in the following equation:** `Δ` — delta, pronounced **“DEL-tuh”**; `μ` — mu, pronounced **“mew”**; `σ` — sigma, pronounced **“SIG-muh”**.

The conditional emission model is

$$
\Delta VIX_t\mid S_t=k \sim \mathcal N(\mu_k,\sigma_k^2).
$$

HMM state labels are relabeled by increasing fitted mean `ΔVIX`, so state numbers have a deterministic interpretation. Posterior probabilities, start probabilities, transition rows, and variances are explicitly validated.

The HMM probability figure now uses percentage-formatted posterior probabilities and clearly states that the probabilities refer to states of daily `ΔVIX`.

## Step 3 — Model selection and state-conditional ETF behavior

AIC and BIC use

$$
AIC=2k-2\ell,
$$

$$
BIC=k\ln(n)-2\ell,
$$

where `k` is the candidate parameter count, `n` is its observation count, and `ℓ` is the candidate log-likelihood.

The Markov likelihood is a conditional likelihood of the **discretized transition sequence**, whereas the HMM likelihood is a continuous Gaussian observation likelihood. Raw AIC/BIC values are therefore **not interpreted as directly comparable across the two model families**. BIC selects the state count inside each family; the fixed project-validity rule then chooses the preferred method.

The current preferred model is **Markov, K=2**. Within the HMM family BIC prefers `K=3`, but the smallest decoded HMM state contains exactly **259 of 5,465 observations = 4.739249771271729%**, which is **0.260750228728271 percentage points below** the fixed 5% occupancy threshold. The fallback is therefore a deterministic project-governance rule designed to avoid a very small decoded state; it is **not** a formal statistical proof that the HMM specification is invalid.

The canonical selected state path is persisted at `reports/tables/step3_selected_states.csv`; later stages do not refit or re-decode a model merely to recover the selected states.

### State-conditional ETF statistics

The selected `K=2` states produce these daily log-return statistics:

| State | Asset | Mean log return | Sample SD | Observations |
|---:|---|---:|---:|---:|
| 0 | TLT | -0.0012574270377546316 | 0.008853984508564897 | 2725 |
| 0 | GLD | 0.0008601630089008522 | 0.01087795494221207 | 2725 |
| 0 | SPY | 0.006701839934347453 | 0.009336952833126482 | 2725 |
| 1 | TLT | 0.0014777461049552123 | 0.00924861558331816 | 2740 |
| 1 | GLD | -0.00004804268342020831 | 0.012055995914238896 | 2740 |
| 1 | SPY | -0.005836313479436862 | 0.010843248722125581 | 2740 |

The reviewed figure `reports/figures/step3_state_asset_statistics.png` deliberately separates **mean daily return** from **sample daily standard deviation** into two basis-point panels. Sample standard deviation is dispersion of individual returns, not a standard error or confidence interval, so it is no longer drawn as a misleading error bar around the mean.

A crucial interpretation is that these are **contemporaneous** conditional statistics: the ETF return at date `t` is grouped by the `ΔVIX` state at the same date `t`. This is descriptive association, not yet evidence of a tradable one-step-ahead signal.

## Step 4 — Allocation rule

For each selected state, the strategy allocates 100% to the ETF with the highest historical state-conditional mean log return. Exact ties use the deterministic priority `TLT -> GLD -> SPY`.

The current mapping is:

```text
State 0 -> SPY
State 1 -> TLT
```

This mapping is estimated from the full sample, so it is in-sample. A genuinely causal/out-of-sample implementation would estimate the state model and state-conditional ETF means using only information available before each decision date.

## Step 5 — Lagged backtest

The backtest converts ETF log returns to simple returns and uses a **one-observation lag**. The state observed at `t-1` determines the position whose return is realized at `t`.

The rotation return is

$$
R_t^{rot}=\sum_i w_{i,S_{t-1}}R_{i,t}.
$$

This removes same-row execution look-ahead, but it does **not** make this implementation causal or out-of-sample because the regime estimates, selected path, and state-to-ETF mapping still use the full historical sample.

Required comparators are:

1. **Equal weight, monthly reset:** `1/3` in TLT, GLD, and SPY on the first comparison date of each calendar month, with intra-month weight drift from realized simple returns.
2. **SPY buy and hold:** simple SPY return on exactly the same comparison dates.

All three portfolios use exactly **5,464** common lagged return observations.

### Performance equations

Wealth starts at `W_0=1`:

$$
W_t=W_{t-1}(1+R_t).
$$

Cumulative return is

$$
R_{cum}=W_n-1.
$$

Annualized return is

$$
R_{ann}=W_n^{252/n}-1.
$$

With sample daily standard deviation `s`, annualized volatility is

$$
V_{ann}=s\sqrt{252}.
$$

With the assignment's zero-risk-free convention, the Sharpe ratio is

$$
S=\frac{\bar R}{s}\sqrt{252}.
$$

Drawdown is measured against the running wealth peak **including the initial wealth observation**:

$$
D_t=\frac{W_t}{\max_{0\le u\le t}W_u}-1,\qquad D_{max}=\min_t D_t.
$$

### Exact performance results

Canonical source: `reports/tables/step5_performance_summary.csv`.

| Portfolio | Cumulative return | Annualized return | Annualized volatility | Sharpe | Max drawdown | Obs. |
|---|---:|---:|---:|---:|---:|---:|
| Regime rotation | 0.8490206769205004 | 0.028753604683711353 | 0.1603694640864778 | 0.2572041096657541 | -0.5376002883101039 | 5464 |
| Equal weight, monthly reset | 5.4208488584380214 | 0.08954760643466542 | 0.09711173952437627 | 0.9318555957186669 | -0.2304366417516469 | 5464 |
| SPY buy and hold | 8.798147826014363 | 0.11099375758132557 | 0.18884509477036146 | 0.6519558427305885 | -0.5518943426401182 | 5464 |

In percentage terms, the regime rotation returns **84.9020676920500400% cumulatively** and **2.875360468371135300% annualized**, versus **542.0848858438021400% / 8.95476064346654200%** for monthly equal weight and **879.814782601436300% / 11.09937575813255700%** for SPY. The rotation Sharpe ratio is lower than both benchmarks. Its maximum drawdown is much worse than equal weight and only modestly less negative than SPY's despite substantially lower return.

The reviewed `reports/figures/step5_cumulative_performance.png` now combines cumulative growth with a second drawdown panel, making the path-risk difference visible rather than relying on terminal wealth alone.

### State-count sensitivity

Canonical source: `reports/tables/step5_state_count_sensitivity.csv`.

| Family | K | Cumulative return | Annualized return | Annualized volatility | Sharpe | Max drawdown | Obs. |
|---|---:|---:|---:|---:|---:|---:|---:|
| markov | 2 | 0.8490206769207176 | 0.028753604683716905 | 0.1603694640864783 | 0.25720410966575935 | -0.5376002883101008 | 5464 |
| markov | 3 | 0.7621616533844695 | 0.02647326922337867 | 0.16304291295689982 | 0.24216662666378957 | -0.6505326601311805 | 5464 |

Within the preferred Markov family, increasing the state count from `K=2` to `K=3` does not improve the lagged strategy: cumulative and annualized return decline, volatility rises, Sharpe falls, and maximum drawdown becomes materially more negative. The sensitivity table is a robustness description, not a second model-selection pass.

### Economic/statistical interpretation

Step 3 identifies a strong same-date pattern: state 0 has the highest contemporaneous mean SPY return, while state 1 has the highest contemporaneous mean TLT return. Step 5 asks a harder question: **does yesterday's state predict which ETF should be held for today's return?** The weak lagged backtest indicates that the contemporaneous `ΔVIX`/ETF-return separation does not translate into a sufficiently strong next-observation trading signal under this fixed mapping.

This is the main modeling lesson of the project: **state description and state prediction are not the same problem**. A regime can explain cross-sectional return behavior on the same date yet have weak value for next-date allocation.

For stronger causal validation, a future extension should use rolling or expanding estimation, one-sided state inference, allocation means estimated only from past information, and explicit turnover/transaction costs.

## Canonical Step 5 artifacts

- `reports/tables/step5_daily_returns.csv`
- `reports/tables/step5_performance_summary.csv`
- `reports/tables/step5_state_count_sensitivity.csv`
- `reports/figures/step5_cumulative_performance.png`
- `reports/generated/step5_manifest.json`
- `notebooks/gwp2_vix_regime_allocation.ipynb`

## Scientific citation policy

Scientific and methodological claims in the notebook/report use a canonical registry at `reports/references.bib` and MLA 9 in-text/source-note/Works Cited conventions. Peer-reviewed papers are preferred for model/statistical methodology; authoritative primary sources are used for data definitions where appropriate.

Required provenance contract:

```text
Notebook/PDF citations -> reports/references.bib
```

The report template remains `reports/Template_Stochastic_Modeling_Group_Work_Project.pdf`.

## Sidecar parity contract

The remaining publication/release backlog preserves three distinct parity levels:

- **Notebook <-> README: exact technical-result parity**
- **Notebook <-> HTML: exact executed-notebook duplicate**
- **Notebook <-> standalone PDF: decision-result parity**

Planned/final publication paths are:

- `reports/gwp2_vix_regime_allocation.html`
- `reports/Stochastic_Modeling_GWP2_Report.pdf`

The standalone report is intentionally nontechnical in narrative style; the executed notebook is the canonical technical record.

## Quality gates

`.github/workflows/quality-gates.yml` runs independent jobs and an aggregate gate:

| Job | Purpose |
|---|---|
| `lint` | `ruff check .` plus `ruff format --check src tests scripts` |
| `type-check` | `mypy src` |
| `unit-tests` | Offline unit suite with coverage data |
| `integration-tests` | Offline integration suite with coverage data |
| `readme-sidecar` | README/CI contract validation |
| `backlog-contract` | Backlog contract validation |
| `repository-hygiene` | Reject leaked temporary build/diagnostic files |
| `coverage` | Combined branch coverage with a **90%** minimum |
| `quality-gate` | Requires every preceding job to succeed |

Local checks:

```bash
ruff check .
ruff format --check src tests scripts
mypy src
pytest -q
coverage run -m pytest -q
coverage report --fail-under=90
python scripts/check_readme_sidecar.py
python scripts/check_backlog_contract.py
python scripts/check_repository_hygiene.py
```

**Backlog contract:** `scripts/check_backlog_contract.py` validates the canonical PR numbering/dependencies/ownership rules in `BACKLOG.md`.

The repository also contains `.github/workflows/auto-complete.yml` named **Auto Complete**. It reacts only to **successful Quality Gates** triggered by pull requests, revalidates the verified head/base SHA pair, updates the branch when safe, and merges/deletes the branch only after the required safety checks. Draft PRs are not auto-completed.

Main branch protection/ruleset configuration is an external repository setting and should not be inferred from these workflow files.

## Development workflow

For each backlog PR:

1. branch from current `main` after dependencies are merged;
2. modify only the files owned by that PR;
3. prove all matching acceptance criteria;
4. run the current quality gates;
5. update from `main` and rerun checks;
6. immediately before commit/merge, `git status --short --branch` must show the intended branch and a clean tree;
7. merge only after the aggregate `quality-gate` succeeds.

This is the **Git workflow per backlog PR** and remains the governing implementation discipline in `BACKLOG.md`.

## Submission package contract

The final release backlog targets:

- executable submission ZIP: `dist/MScFE_622_GWP2_submission.zip`
- deterministic submission manifest: `reports/generated/submission_manifest.json`
- standalone PDF uploaded separately from the ZIP.

The bundle must include the executable notebook and canonical scientific citation registry while excluding forbidden temporary files and the separately submitted standalone PDF.

## Team

- Umuhoza Denyse Graine
- Opeyemi Waliyilah Oladipupo
- Sergej Schweizer
