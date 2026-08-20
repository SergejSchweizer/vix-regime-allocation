# BACKLOG — MScFE 622 GWP2: Steps 1–5 + Final Submission

Single canonical implementation backlog for MScFE 622 Stochastic Modeling GWP2. It is optimized for **two weak coding agents**: all modeling choices, paths, schemas, dependencies, file ownership, and acceptance evidence are fixed here.

## Global PR contract

1. PR-01..PR-49 are contiguous. Every PR has exactly one Agent lane, explicit Dependencies, complete Files owned, Tasks, and Acceptance criteria.
2. Dependencies name only lower-numbered PRs or `none`; start only after all dependencies are merged to current `main`.
3. Task IDs are contiguous `Txx.1..n`; each has exactly one `ACxx.n` verifying the same responsibility. Acceptance may not add unstated work.
4. Files owned is the complete write set, including generated artifacts. If another file/interface/model choice is needed, stop and revise this backlog instead of guessing.
5. Parallel PRs must have disjoint write sets. All notebook PRs are serialized.
6. Fixed names, schemas, paths, tolerances, tie rules, and formulas are immutable unless this backlog is explicitly audited again.
7. Source PRs have deterministic offline tests. Unit tests never require Yahoo/network access. Numerical outputs are never fabricated.
8. Notebook technical prose explains finance/statistics rather than narrating Python library calls. The PDF is a rendered sidecar of the canonical executed notebook and therefore preserves the same notebook explanations, equations, code cells, stored outputs, tables, figures, citations, and Works Cited; PDF-only analytical narrative is forbidden.
9. Sidecars read canonical artifacts only; they do not refit, redecode, or recompute a second analysis.
10. Before equations containing Greek symbols, list each Greek letter and pronunciation. The technical notebook must support external theoretical, methodological, and empirical claims with verifiable scholarly sources using MLA 9 in-text citations and a Works Cited section; the PDF sidecar inherits that exact cited content from the notebook and may not diverge. Official primary sources may supplement scholarly literature for data/index definitions but do not replace academic support. Every citation must resolve to the canonical `reports/references.bib`; bare URLs are not citations and fabricated bibliographic metadata is forbidden.
11. Every PR passes the full current quality suite; combined source coverage remains `>=90%`.
12. `python scripts/check_backlog_contract.py` must pass before merge.

## Assignment coverage

- Step 1: PR-01..05.
- Step 2: PR-06..16, PR-21..22.
- Step 3: PR-17..19, PR-23..24.
- Step 4: PR-20, PR-25.
- Step 5: PR-33..47.
- Required notebook/HTML/separate PDF/final ZIP: PR-28..32, PR-44..49.

# Fixed contracts

## Step 1

Yahoo tickers: `TLT`, `GLD`, `SPY`, `^VIX`. `yfinance.download`: `period="max"`, `interval="1d"`, `auto_adjust=False`, `back_adjust=False`, `actions=False`, `progress=False`. Extract **Adj Close**; rename `^VIX`→`VIX`; timezone-naive sorted unique `DatetimeIndex` named `Date`; columns exactly `TLT,GLD,SPY,VIX`; non-missing prices finite and `>0`.

Common sample: drop dates missing any of the four prices; **no fill/interpolation**. ETF return: `ln(P_t/P_(t-1))`. VIX observation: `VIX_change_t=VIX_t-VIX_(t-1)`. Remove the first lag-induced row. Final columns exactly:

```text
TLT,GLD,SPY,VIX,TLT_log_return,GLD_log_return,SPY_log_return,VIX_change
```

No missing/non-finite final values. Canonical outputs:

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

## Step 2 models

Observation is exclusively `VIX_change`; K exactly 2 and 3.

**Markov:** NumPy linear quantiles: K=2 `q(.5)`, K=3 `q(1/3),q(2/3)`; cuts strictly increasing; state assignment `searchsorted(cuts,x,side="right")`. Transition `P_ij=N_ij/sum_j N_ij`, no pseudocounts; zero-outgoing expected state invalid. Unique stationary row vector satisfies `pi@P=pi`, nonnegative, sums 1 within `1e-10`. Conditional likelihood `sum log(P[s_t,s_(t+1)])`; `n=len(states)-1`; `k=K(K-1)`.

**Gaussian HMM:** univariate diagonal; `n_iter=500`, `tol=1e-6`, `min_covar=1e-6`, seeds `(42,43,44,45,46)`. Choose greatest-likelihood converged fit; tie within `1e-12`→smallest seed; none converged→`RuntimeError`. Relabel all state-dependent quantities by increasing fitted VIX-change mean, original component index breaking equal-mean ties. Posterior columns exact `state_i`, finite, rows sum 1 within `1e-8`. `k=K^2+2K-1`, `n=len(VIX_change)`.

`AIC=2k-2logL`; `BIC=k ln(n)-2logL`. Because Markov likelihood uses discrete transitions while HMM likelihood uses continuous observations, AIC/BIC select K **within family only**. Lower K breaks BIC ties within `1e-12`. Preferred-method rule: use within-family-BIC-selected HMM only if converged/finite, variances positive, initial/transition/posterior probabilities valid, and each Viterbi state occupancy `>=.05`; otherwise use selected Markov. This is an explicit project decision rule, not a cross-family IC proof.

Persist candidate/preferred state paths, schema `Date,state`, exact Step-1 dates, integer state `0..K-1`:

```text
reports/tables/step2_markov_2_states.csv
reports/tables/step2_markov_3_states.csv
reports/tables/step2_hmm_2_states.csv
reports/tables/step2_hmm_3_states.csv
reports/tables/step3_selected_states.csv
```

Later steps **load** these artifacts; they do not refit/redecode merely to recover an existing state sequence.

Canonical Step 2–4 files:

```text
reports/tables/step2_markov_2_thresholds.csv
reports/tables/step2_markov_3_thresholds.csv
reports/tables/step2_markov_2_transition.csv
reports/tables/step2_markov_3_transition.csv
reports/tables/step2_markov_2_stationary.csv
reports/tables/step2_markov_3_stationary.csv
reports/tables/step2_hmm_2_parameters.csv
reports/tables/step2_hmm_3_parameters.csv
reports/tables/step2_hmm_2_transition.csv
reports/tables/step2_hmm_3_transition.csv
reports/tables/step2_markov_2_states.csv
reports/tables/step2_markov_3_states.csv
reports/tables/step2_hmm_2_states.csv
reports/tables/step2_hmm_3_states.csv
reports/tables/step3_model_comparison.csv
reports/tables/step3_selected_states.csv
reports/tables/step3_state_asset_statistics.csv
reports/tables/step4_allocation_mapping.csv
reports/figures/step2_markov_vix_states.png
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
reports/figures/step3_state_asset_statistics.png
reports/generated/step3_selected_model.json
reports/generated/steps_2_4_manifest.json
```

Schemas: thresholds `state,lower_bound,upper_bound`; transition `from_state,state_0,state_1[,state_2]`; stationary `state,stationary_probability`; HMM parameters `state,mean_vix_change,variance_vix_change,start_probability,viterbi_observations,viterbi_occupancy,posterior_mean_probability`; model comparison `family,n_states,log_likelihood,n_parameters,n_observations,aic,bic,converged,criterion_scope`; Step3 stats `state,asset,mean_log_return,std_log_return,observations`; Step4 mapping `state,selected_asset,selection_mean_log_return,TLT_weight,GLD_weight,SPY_weight`.

Selected-model JSON keys exactly: `family,n_states,state_source,selection_reason,markov_best_n_states,hmm_best_n_states,input_data_sha256,selected_states_path`; selected path is `reports/tables/step3_selected_states.csv`.

Step3 uses mean daily ETF log return, sample std `ddof=1`, count, asset order `TLT,GLD,SPY`; no annualization. Step4 chooses greatest state mean, tie priority `TLT -> GLD -> SPY`; winner weight 1, others 0; no optional 60/40.

Steps2–4 manifest keys: `schema_version,input_data_path,input_data_sha256,notebook_path,selected_model_path,tables,figures`; version 1; repository-relative POSIX paths; every canonical Step2–4 table/figure exactly once; no timestamp.

## Lookahead qualification

The required one-trading-row execution lag does **not** make this exercise causal/out-of-sample. Explicitly disclose: (1) Markov thresholds or HMM parameters use the full sample; (2) preferred HMM Viterbi path, if used, is a full-sequence decode and can use future observations relative to earlier dates; (3) allocation means use the full sample. Stronger causal validation would require rolling/expanding re-estimation, one-sided/filtered state inference, and decision-time-only allocation estimation; that is future work, not implemented here.

## Step 5

Convert ETF log returns to simple returns: `r=exp(ell)-1`. Rotation return at t uses **previous observed trading row** state: `r_rotation,t=w(S_(t-1))'r_t`. First row excluded; no fill. Rotation detail columns: `decision_date,decision_state,selected_asset,TLT_weight,GLD_weight,SPY_weight,regime_rotation_return`.

Equal-weight benchmark: target 1/3 each; reset before first comparison return and before first observed comparison date of each new calendar month; intra-month weights drift via `w_i+ = w_i(1+r_i)/(1+r_p)`; no costs. SPY benchmark is SPY simple return. Comparison columns exact `regime_rotation,equal_weight_monthly,spy_buy_hold`; identical dates for all three.

Metrics use simple daily returns, 252 trading days, risk-free=0:

```text
W_0=1
W_t=product_(j=1..t)(1+r_j)
M_t=max(W_0,...,W_t)
Cumulative=W_N-1
Annualized=W_N^(252/N)-1
Volatility=sample_std(r,ddof=1)*sqrt(252)
Sharpe=mean(r)/sample_std(r,ddof=1)*sqrt(252)
MaxDrawdown=min(0,min_t(W_t/M_t-1))
```

`W_0=1` must enter drawdown peak so a first-period loss counts. Require returns `>-1`; at least 2 observations for sample volatility/Sharpe; zero-volatility Sharpe fails clearly. Summary columns: `portfolio,cumulative_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown,observations`; row order rotation, equal-weight, SPY.

Sensitivity: K=2 vs3 **within preferred family**, load canonical candidate states, rebuild Step3 stats/Step4 mapping with shared functions, same lag, common date intersection, same five metrics; do not switch family/window; remain in-sample.

Canonical Step5 files: `reports/tables/step5_daily_returns.csv`, `step5_performance_summary.csv`, `step5_state_count_sensitivity.csv`, `reports/figures/step5_cumulative_performance.png`, `reports/generated/step5_manifest.json`. Step5 manifest keys: `schema_version,input_data_path,input_data_sha256,selected_model_path,selected_states_path,allocation_path,notebook_path,tables,figures`; version1; no timestamp.

## Scientific citation contract

`reports/references.bib` is the single canonical source registry. PR-05 creates it; serialized notebook PRs may extend it only when a new cited source is required. Entries use stable citation keys and complete verifiable metadata: author(s), title, venue or publisher, year, and DOI when one exists; otherwise ISBN for scholarly books or a stable official URL for primary data/index documentation. No citation metadata may be invented.

Scientific support is mandatory, not optional. Peer-reviewed journal/conference papers and scholarly books/textbooks support Markov-chain theory, HMM/EM/Viterbi/posterior inference, information criteria, performance metrics, backtesting limitations, and other methodological claims. Official primary sources such as index methodology or data-provider documentation may support definitions and provenance only. If a claim cannot be supported, omit it or label it explicitly as a project assumption/decision rule.

The notebook uses MLA 9 parenthetical citations adjacent to externally sourced definitions, equations, methodological claims, and interpretations, plus a final **Works Cited** section rendered from the canonical registry. Each major method section has at least one relevant scholarly citation. Tables and figures include concise source notes distinguishing project calculations from external data/method sources.

The PDF report is a rendered sidecar of the canonical executed notebook. It inherits the notebook's MLA 9 in-text citations, source notes, equations, code cells, stored outputs, tables, figures, interpretations, limitations, and final **Works Cited** in notebook order. The supplied template contributes page 1 as the cover; template instruction page 2 is excluded. No separately authored PDF narrative or independent analysis is permitted.

Citation integrity is deterministic: every in-text citation in notebook/PDF must resolve to `reports/references.bib`; every entry printed in an artifact's Works Cited must be cited in that artifact; duplicate keys, unresolved cites, bibliography-only orphan entries in rendered Works Cited, and URL-only pseudo-citations fail validation. PR-31 establishes Step1–4 citation parity checks and PR-47 extends them through Step5.

## Reports/submission

Canonical technical notebook: `notebooks/gwp2_vix_regime_allocation.ipynb`. Each step visibly has question/step number, project-function calls, stored code output, equations/definitions, tables/plots, interpretation/recommendation, limitations, mandatory MLA 9 scholarly citations/source notes, and a final Works Cited rendered from `reports/references.bib`; execute top-to-bottom before commit.

README has exact technical parity from canonical artifacts. HTML `reports/gwp2_vix_regime_allocation.html` is exported from stored notebook outputs without execution/refitting and preserves the notebook citations/Works Cited. Separate PDF sidecar `reports/Stochastic_Modeling_GWP2_Report.pdf` uses provided template **page1 only**, excludes instruction page2, preserves known team names/blank unknown fields, and then renders the canonical executed notebook in order. The PDF stores the source notebook path and exact notebook SHA-256 for stale-sidecar detection; render every final page for visual QA.

Final ZIP `dist/MScFE_622_GWP2_submission.zip` contains exactly notebook, HTML, README, `pyproject.toml`, `reports/references.bib`, `data/processed/step1_data.csv`, and `src/vix_regime_allocation/**/*.py`. It excludes the separately uploaded PDF sidecar, `.git`, `.github`, tests, rendered QA, caches, coverage, `.env*`, keys. Sorted POSIX members, no symlinks/traversal, timestamp `1980-01-01`. Submission manifest keys remain `schema_version,zip_path,zip_sha256,standalone_pdf_path,standalone_pdf_sha256,included_files,member_sha256`; no timestamp. The PDF sidecar is uploaded separately.

---
## PR-01 — Yahoo adjusted-close loader

**Agent lane:** A

**Dependencies:** none

**Git branch:** `pr-01-yahoo-adjusted-close-loader`

**Git status:** `git status --short --branch` must show `pr-01-yahoo-adjusted-close-loader` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-01 — Yahoo adjusted-close loader`

**Files owned:**

```text
src/vix_regime_allocation/data.py
tests/test_data.py
```

### Public interface

```python
def download_adjusted_close() -> pandas.DataFrame: ...
```

### Tasks

- [ ] T01.1 Implement exact Yahoo ticker/argument/Adj-Close contract and output schema/index validation.
- [ ] T01.2 Add mocked offline tests for arguments, extraction/rename/order/index, duplicate dates, and invalid prices.

### Acceptance criteria

- [ ] AC01.1 (`T01.1`) Mock proves all fixed download arguments/tickers; output schema/index/prices satisfy contract and invalid input fails.
- [ ] AC01.2 (`T01.2`) All loader tests pass without network access.

---

## PR-02 — Step 1 common-sample transformation

**Agent lane:** B

**Dependencies:** none

**Git branch:** `pr-02-step-1-common-sample-transformation`

**Git status:** `git status --short --branch` must show `pr-02-step-1-common-sample-transformation` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-02 — Step 1 common-sample transformation`

**Files owned:**

```text
src/vix_regime_allocation/transform.py
tests/test_transform.py
```

### Public interface

```python
def prepare_step1_data(prices: pandas.DataFrame) -> pandas.DataFrame: ...
```

### Tasks

- [ ] T02.1 Validate raw schema; drop incomplete common dates; compute exact three ETF log returns and VIX first difference; return exact clean schema with only lag-first-row removal.
- [ ] T02.2 Add hand-computable tests including interior missing date, exact calculations, schema/index errors, and non-finite output.

### Acceptance criteria

- [ ] AC02.1 (`T02.1`) Output has exact dates/columns/calculations, no imputation/missing/non-finite values; malformed input fails.
- [ ] AC02.2 (`T02.2`) All transformation tests pass offline.

---

## PR-03 — Step 1 exploratory plots

**Agent lane:** A

**Dependencies:** PR-02

**Git branch:** `pr-03-step-1-exploratory-plots`

**Git status:** `git status --short --branch` must show `pr-03-step-1-exploratory-plots` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-03 — Step 1 exploratory plots`

**Files owned:**

```text
src/vix_regime_allocation/plots.py
tests/test_plots.py
```

### Public interface

```python
def plot_etf_log_returns(data: pandas.DataFrame, output_path: pathlib.Path) -> None: ...
def plot_vix_change(data: pandas.DataFrame, output_path: pathlib.Path) -> None: ...
```

### Tasks

- [ ] T03.1 Implement validated ETF-log-return and VIX-change figures with fixed series, titles, axes, scales/ticks, ETF legend, save/close behavior.
- [ ] T03.2 Add deterministic tests for plotted data, labels, non-empty files, and figure closure.

### Acceptance criteria

- [ ] AC03.1 (`T03.1`) ETF plot has exactly TLT/GLD/SPY log returns; VIX plot has exactly VIX_change; all required presentation elements exist.
- [ ] AC03.2 (`T03.2`) All plot tests pass offline with no figure leak.

---

## PR-04 — Executable Step 1 pipeline

**Agent lane:** B

**Dependencies:** PR-01, PR-02, PR-03

**Git branch:** `pr-04-executable-step-1-pipeline`

**Git status:** `git status --short --branch` must show `pr-04-executable-step-1-pipeline` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-04 — Executable Step 1 pipeline`

**Files owned:**

```text
scripts/run_step1.py
tests/test_run_step1.py
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

### Tasks

- [ ] T04.1 Delegate exactly once to loader/transformer and both plotters; write canonical CSV/PNGs; print start/end/count.
- [ ] T04.2 Add mocked offline orchestration test using temporary output root.

### Acceptance criteria

- [ ] AC04.1 (`T04.1`) Generated paths/schema/stdout match contract and no numerical logic is duplicated.
- [ ] AC04.2 (`T04.2`) Integration/orchestration test passes with no network or out-of-temp writes.

---

## PR-05 — Canonical notebook Step 1

**Agent lane:** A

**Dependencies:** PR-04

**Git branch:** `pr-05-canonical-notebook-step-1`

**Git status:** `git status --short --branch` must show `pr-05-canonical-notebook-step-1` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-05 — Canonical notebook Step 1`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
README.md
```

### Tasks

- [ ] T05.1 Add complete Step1 technical section using project functions: data checks, equations/notation, both plots, assumptions/limitations, interpretation, MLA 9 in-text citations for external claims, and source notes for external data/figures.
- [ ] T05.2 Create `reports/references.bib` as the canonical verified source registry with unique keys, complete scholarly metadata, and official primary data/index sources kept distinct from scholarly support; render the notebook Works Cited from cited entries only.
- [ ] T05.3 Execute notebook with stored outputs and synchronize README Step1 status/paths/scientific-citation policy without recomputation.

### Acceptance criteria

- [ ] AC05.1 (`T05.1`) Notebook covers all Step1 deliverables, uses shared functions/correct notation, has evidence-bounded interpretation, source notes, and resolved MLA 9 citations adjacent to external claims.
- [ ] AC05.2 (`T05.2`) Canonical registry contains no duplicate keys or fabricated metadata; every rendered Works Cited entry is cited and every notebook citation resolves to the registry.
- [ ] AC05.3 (`T05.3`) Notebook has no failed/unexecuted cells; README accurately references canonical Step1 artifacts/source policy and claims no later-step result.

---

## PR-06 — Model/notebook/report dependencies

**Agent lane:** setup

**Dependencies:** PR-05

**Git branch:** `pr-06-model-notebook-report-dependencies`

**Git status:** `git status --short --branch` must show `pr-06-model-notebook-report-dependencies` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-06 — Model/notebook/report dependencies`

**Files owned:**

```text
pyproject.toml
```

### Tasks

- [ ] T06.1 Add fixed modeling (`scipy`,`hmmlearn`), notebook (`nbformat`,`nbclient`,`nbconvert`), and PDF (`reportlab`,`pypdf`,`pymupdf`) minimum dependencies without removing existing quality dependencies.

### Acceptance criteria

- [ ] AC06.1 (`T06.1`) Every fixed dependency appears exactly once and existing dev/quality configuration remains intact.

---

## PR-07 — Immutable model configuration

**Agent lane:** setup

**Dependencies:** PR-06

**Git branch:** `pr-07-immutable-model-configuration`

**Git status:** `git status --short --branch` must show `pr-07-immutable-model-configuration` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-07 — Immutable model configuration`

**Files owned:**

```text
src/vix_regime_allocation/model_config.py
tests/test_model_config.py
```

### Tasks

- [ ] T07.1 Define exact K values, seeds, HMM settings, stationary/probability/likelihood tolerances, and .05 occupancy threshold.
- [ ] T07.2 Add tests asserting every constant exactly.

### Acceptance criteria

- [ ] AC07.1 (`T07.1`) All constants equal fixed contracts.
- [ ] AC07.2 (`T07.2`) Tests fail on any configuration drift and pass offline.

---

## PR-08 — Markov quantile states

**Agent lane:** A

**Dependencies:** PR-07

**Git branch:** `pr-08-markov-quantile-states`

**Git status:** `git status --short --branch` must show `pr-08-markov-quantile-states` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-08 — Markov quantile states`

**Files owned:**

```text
src/vix_regime_allocation/markov_states.py
tests/test_markov_states.py
```

### Public interface

```python
def discretize_vix_change(vix_change: pandas.Series, n_states: int) -> tuple[pandas.Series, pandas.DataFrame]: ...
```

### Tasks

- [ ] T08.1 Implement validated K=2/3 linear-quantile cuts, duplicate-cut rejection, right-side boundary assignment, state Series and threshold table.
- [ ] T08.2 Test K=2/3, exact cut membership, invalid inputs, and duplicate cuts.

### Acceptance criteria

- [ ] AC08.1 (`T08.1`) State/index/name/range and threshold schema/bounds match contract; invalid cases fail.
- [ ] AC08.2 (`T08.2`) All tests pass offline.

---

## PR-09 — Deterministic Gaussian HMM fitter

**Agent lane:** B

**Dependencies:** PR-07

**Git branch:** `pr-09-deterministic-gaussian-hmm-fitter`

**Git status:** `git status --short --branch` must show `pr-09-deterministic-gaussian-hmm-fitter` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-09 — Deterministic Gaussian HMM fitter`

**Files owned:**

```text
src/vix_regime_allocation/hmm_model.py
tests/test_hmm_model.py
```

### Public interface

```python
def fit_gaussian_hmm(vix_change: pandas.Series, n_states: int) -> HMMFitResult: ...
```

### Tasks

- [ ] T09.1 Fit all fixed seeds/settings; select converged max likelihood with tie rule; deterministic mean-order relabel all outputs; validate Viterbi/posteriors.
- [ ] T09.2 Test settings/seeds, convergence/tie selection, relabel consistency, K=2/3 shapes, posterior normalization, no-converged failure.

### Acceptance criteria

- [ ] AC09.1 (`T09.1`) Returned fit obeys every fixed fitting/relabel/index/probability contract and invalid/no-converged cases fail.
- [ ] AC09.2 (`T09.2`) All HMM fitter tests pass offline.

---

## PR-10 — Markov transition and stationary distribution

**Agent lane:** A

**Dependencies:** PR-08

**Git branch:** `pr-10-markov-transition-and-stationary-distribution`

**Git status:** `git status --short --branch` must show `pr-10-markov-transition-and-stationary-distribution` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-10 — Markov transition and stationary distribution`

**Files owned:**

```text
src/vix_regime_allocation/markov_chain.py
tests/test_markov_chain.py
```

### Public interface

```python
def estimate_transition_matrix(states: pandas.Series, n_states: int) -> pandas.DataFrame: ...
def stationary_distribution(transition: pandas.DataFrame) -> pandas.Series: ...
```

### Tasks

- [ ] T10.1 Implement consecutive transition counts/row normalization/no-pseudocount validation and unique stationary solver with fixed tolerance.
- [ ] T10.2 Add hand-computable transition/stationary tests plus zero-outgoing/nonunique/malformed failures.

### Acceptance criteria

- [ ] AC10.1 (`T10.1`) Transition and stationary outputs equal manual values and satisfy all fixed validity rules.
- [ ] AC10.2 (`T10.2`) All tests pass offline.

---

## PR-11 — Information-criterion helpers

**Agent lane:** B

**Dependencies:** PR-07

**Git branch:** `pr-11-information-criterion-helpers`

**Git status:** `git status --short --branch` must show `pr-11-information-criterion-helpers` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-11 — Information-criterion helpers`

**Files owned:**

```text
src/vix_regime_allocation/information_criteria.py
tests/test_information_criteria.py
```

### Public interface

```python
def aic(log_likelihood: float, n_parameters: int) -> float: ...
def bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float: ...
```

### Tasks

- [ ] T11.1 Implement validated AIC/BIC plus exact Markov/HMM parameter-count formulas for K=2/3.
- [ ] T11.2 Add exact numerical and invalid-input tests.

### Acceptance criteria

- [ ] AC11.1 (`T11.1`) AIC/BIC and parameter counts (Markov 2/6; HMM 7/14) equal contract.
- [ ] AC11.2 (`T11.2`) All tests pass offline.

---

## PR-12 — Markov candidate evaluation

**Agent lane:** A

**Dependencies:** PR-10, PR-11

**Git branch:** `pr-12-markov-candidate-evaluation`

**Git status:** `git status --short --branch` must show `pr-12-markov-candidate-evaluation` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-12 — Markov candidate evaluation`

**Files owned:**

```text
src/vix_regime_allocation/markov_evaluation.py
tests/test_markov_evaluation.py
```

### Public interface

```python
def evaluate_markov_candidate(vix_change: pandas.Series, n_states: int) -> dict[str, object]: ...
```

### Tasks

- [ ] T12.1 Implement conditional transition likelihood and candidate assembly by delegating to shared states/transition/stationary/IC helpers with exact family/count/key semantics.
- [ ] T12.2 Add hand-computable K=2/3 likelihood/AIC/BIC, impossible-transition, and delegation tests.

### Acceptance criteria

- [ ] AC12.1 (`T12.1`) Candidate values/key set/count conventions match fixed contract with no duplicate math.
- [ ] AC12.2 (`T12.2`) All tests pass offline.

---

## PR-13 — HMM candidate evaluation

**Agent lane:** B

**Dependencies:** PR-09, PR-11

**Git branch:** `pr-13-hmm-candidate-evaluation`

**Git status:** `git status --short --branch` must show `pr-13-hmm-candidate-evaluation` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-13 — HMM candidate evaluation`

**Files owned:**

```text
src/vix_regime_allocation/hmm_evaluation.py
tests/test_hmm_evaluation.py
```

### Public interface

```python
def evaluate_hmm_candidate(vix_change: pandas.Series, n_states: int) -> dict[str, object]: ...
```

### Tasks

- [ ] T13.1 Call fitter once, use shared HMM count/AIC/BIC, and map immutable fit result to exact candidate keys.
- [ ] T13.2 Add mocked K=2/3 math/delegation tests.

### Acceptance criteria

- [ ] AC13.1 (`T13.1`) Candidate values/key set/counts equal shared fit/helpers with exactly one fit.
- [ ] AC13.2 (`T13.2`) All tests pass offline.

---

## PR-14 — Markov VIX-state figure

**Agent lane:** A

**Dependencies:** PR-12

**Git branch:** `pr-14-markov-vix-state-figure`

**Git status:** `git status --short --branch` must show `pr-14-markov-vix-state-figure` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-14 — Markov VIX-state figure`

**Files owned:**

```text
src/vix_regime_allocation/markov_plots.py
tests/test_markov_plots.py
```

### Public interface

```python
def plot_markov_vix_states(vix: pandas.Series, states_2: pandas.Series, states_3: pandas.Series, output_path: pathlib.Path) -> None: ...
```

### Tasks

- [ ] T14.1 Implement exact-index two-panel K=2/3 VIX-level state-colored figure with complete labels/scales/legends and save/close.
- [ ] T14.2 Add deterministic index/data/presentation/file tests.

### Acceptance criteria

- [ ] AC14.1 (`T14.1`) Figure uses supplied VIX level/states only and satisfies all presentation/output rules; mismatch fails.
- [ ] AC14.2 (`T14.2`) All tests pass offline.

---

## PR-15 — HMM VIX-state figure

**Agent lane:** B

**Dependencies:** PR-13

**Git branch:** `pr-15-hmm-vix-state-figure`

**Git status:** `git status --short --branch` must show `pr-15-hmm-vix-state-figure` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-15 — HMM VIX-state figure`

**Files owned:**

```text
src/vix_regime_allocation/hmm_state_plot.py
tests/test_hmm_state_plot.py
```

### Public interface

```python
def plot_hmm_vix_states(vix: pandas.Series, states_2: pandas.Series, states_3: pandas.Series, output_path: pathlib.Path) -> None: ...
```

### Tasks

- [ ] T15.1 Implement exact-index two-panel K=2/3 VIX-level Viterbi-colored figure with complete labels/scales/legends and save/close.
- [ ] T15.2 Add deterministic index/data/presentation/file tests.

### Acceptance criteria

- [ ] AC15.1 (`T15.1`) Figure uses supplied VIX level/Viterbi states only and satisfies output rules; mismatch fails.
- [ ] AC15.2 (`T15.2`) All tests pass offline.

---

## PR-16 — HMM smoothed-probability figure

**Agent lane:** B

**Dependencies:** PR-13

**Git branch:** `pr-16-hmm-smoothed-probability-figure`

**Git status:** `git status --short --branch` must show `pr-16-hmm-smoothed-probability-figure` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-16 — HMM smoothed-probability figure`

**Files owned:**

```text
src/vix_regime_allocation/hmm_probability_plot.py
tests/test_hmm_probability_plot.py
```

### Public interface

```python
def plot_hmm_smoothed_probabilities(probabilities_2: pandas.DataFrame, probabilities_3: pandas.DataFrame, output_path: pathlib.Path) -> None: ...
```

### Tasks

- [ ] T16.1 Validate K=2/3 posterior frames and plot all state probabilities once in two panels with y=[0,1], labels/scales/legends, save/close.
- [ ] T16.2 Add tests for malformed probabilities, plotted columns, limits/presentation, and file closure.

### Acceptance criteria

- [ ] AC16.1 (`T16.1`) Invalid/non-normalized input fails; correct figure contains every posterior exactly once with fixed limits.
- [ ] AC16.2 (`T16.2`) All tests pass offline.

---

## PR-17 — Model comparison and preferred selection

**Agent lane:** A

**Dependencies:** PR-12, PR-13

**Git branch:** `pr-17-model-comparison-and-preferred-selection`

**Git status:** `git status --short --branch` must show `pr-17-model-comparison-and-preferred-selection` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-17 — Model comparison and preferred selection`

**Files owned:**

```text
src/vix_regime_allocation/model_selection.py
tests/test_model_selection.py
```

### Public interface

```python
def build_model_comparison(markov_candidates: list[dict[str, object]], hmm_candidates: list[dict[str, object]]) -> pandas.DataFrame: ...
def select_preferred_model(comparison: pandas.DataFrame, markov_candidates: list[dict[str, object]], hmm_candidates: list[dict[str, object]]) -> dict[str, object]: ...
```

### Tasks

- [ ] T17.1 Build exact four-row comparison; choose K by within-family BIC/tie rule; validate HMM conditions; return valid-HMM-or-Markov preferred result with exact state/source/reason fields.
- [ ] T17.2 Test malformed candidates, BIC/ties, valid HMM, and each individual fallback condition; prove no cross-family IC ranking.

### Acceptance criteria

- [ ] AC17.1 (`T17.1`) Comparison/selection semantics and exact result fields follow fixed project rule.
- [ ] AC17.2 (`T17.2`) All selection branches/tests pass offline.

---

## PR-18 — Preferred-state ETF statistics

**Agent lane:** B

**Dependencies:** PR-02

**Git branch:** `pr-18-preferred-state-etf-statistics`

**Git status:** `git status --short --branch` must show `pr-18-preferred-state-etf-statistics` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-18 — Preferred-state ETF statistics`

**Files owned:**

```text
src/vix_regime_allocation/state_statistics.py
tests/test_state_statistics.py
```

### Public interface

```python
def compute_state_asset_statistics(data: pandas.DataFrame, states: pandas.Series) -> pandas.DataFrame: ...
```

### Tasks

- [ ] T18.1 Validate exact index/return/state data and compute state×asset mean log return, sample std ddof=1, count in fixed order/schema.
- [ ] T18.2 Add manual calculations and malformed/index/finite/insufficient-state tests.

### Acceptance criteria

- [ ] AC18.1 (`T18.1`) All table values/order/schema equal contract and invalid input fails.
- [ ] AC18.2 (`T18.2`) All tests pass offline.

---

## PR-19 — Step 3 state-return bar chart

**Agent lane:** A

**Dependencies:** PR-18

**Git branch:** `pr-19-step-3-state-return-bar-chart`

**Git status:** `git status --short --branch` must show `pr-19-step-3-state-return-bar-chart` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-19 — Step 3 state-return bar chart`

**Files owned:**

```text
src/vix_regime_allocation/state_statistics_plot.py
tests/test_state_statistics_plot.py
```

### Public interface

```python
def plot_state_asset_statistics(statistics: pandas.DataFrame, output_path: pathlib.Path) -> None: ...
```

### Tasks

- [ ] T19.1 Implement validated grouped mean bars for TLT/GLD/SPY with state std error bars, zero line, complete presentation, save/close.
- [ ] T19.2 Test heights/error bars/schema/presentation/file closure.

### Acceptance criteria

- [ ] AC19.1 (`T19.1`) Chart exactly reflects canonical statistics and all required visual elements.
- [ ] AC19.2 (`T19.2`) All tests pass offline.

---

## PR-20 — Step 4 state-to-allocation mapping

**Agent lane:** B

**Dependencies:** PR-18

**Git branch:** `pr-20-step-4-state-to-allocation-mapping`

**Git status:** `git status --short --branch` must show `pr-20-step-4-state-to-allocation-mapping` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-20 — Step 4 state-to-allocation mapping`

**Files owned:**

```text
src/vix_regime_allocation/allocation.py
tests/test_allocation.py
```

### Public interface

```python
def build_state_allocation(statistics: pandas.DataFrame) -> pandas.DataFrame: ...
```

### Tasks

- [ ] T20.1 Validate complete state×asset means; select maximum with exact tie priority; return exact 100/0/0 mapping schema/order.
- [ ] T20.2 Test all winners, two-way/three-way ties, and malformed input.

### Acceptance criteria

- [ ] AC20.1 (`T20.1`) Selection/weights/schema/ties exactly match fixed contract.
- [ ] AC20.2 (`T20.2`) All tests pass offline.

---

## PR-21 — Notebook Step 2 Markov + canonical states

**Agent lane:** A

**Dependencies:** PR-05, PR-12, PR-14

**Git branch:** `pr-21-notebook-step-2-markov-canonical-states`

**Git status:** `git status --short --branch` must show `pr-21-notebook-step-2-markov-canonical-states` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-21 — Notebook Step 2 Markov + canonical states`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
reports/tables/step2_markov_2_thresholds.csv
reports/tables/step2_markov_3_thresholds.csv
reports/tables/step2_markov_2_transition.csv
reports/tables/step2_markov_3_transition.csv
reports/tables/step2_markov_2_stationary.csv
reports/tables/step2_markov_3_stationary.csv
reports/tables/step2_markov_2_states.csv
reports/tables/step2_markov_3_states.csv
reports/figures/step2_markov_vix_states.png
```

### Tasks

- [ ] T21.1 Add complete K=2/K=3 Markov technical analysis with equations/assumptions/limitations, canonical state figure using shared functions, scholarly Markov-chain citations from `reports/references.bib`, and figure/data source notes.
- [ ] T21.2 Serialize six model tables plus both exact candidate Date,state sequences; execute entire notebook and store outputs.

### Acceptance criteria

- [ ] AC21.1 (`T21.1`) Notebook outputs/equations/figure are complete, correct, evidence-bounded, and all external Markov claims/caption source notes have resolved scholarly/primary citations.
- [ ] AC21.2 (`T21.2`) All eight CSVs match displayed/shared outputs and exact Step1 dates; notebook has no failed/unexecuted cell.

---

## PR-22 — Notebook Step 2 HMM + canonical states

**Agent lane:** B

**Dependencies:** PR-13, PR-15, PR-16, PR-21

**Git branch:** `pr-22-notebook-step-2-hmm-canonical-states`

**Git status:** `git status --short --branch` must show `pr-22-notebook-step-2-hmm-canonical-states` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-22 — Notebook Step 2 HMM + canonical states`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
reports/tables/step2_hmm_2_parameters.csv
reports/tables/step2_hmm_3_parameters.csv
reports/tables/step2_hmm_2_transition.csv
reports/tables/step2_hmm_3_transition.csv
reports/tables/step2_hmm_2_states.csv
reports/tables/step2_hmm_3_states.csv
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
```

### Tasks

- [ ] T22.1 Add complete K=2/K=3 HMM parameters/diagnostics/EM-Viterbi-posterior explanation and both canonical figures, citing scholarly HMM/EM/decoding/posterior sources from `reports/references.bib` with figure/data source notes.
- [ ] T22.2 Serialize four model tables plus both exact Viterbi Date,state files without second fit/decode; execute full notebook.

### Acceptance criteria

- [ ] AC22.1 (`T22.1`) Notebook technical outputs/explanation/figures satisfy fixed contracts and every externally sourced HMM/EM/decoding/posterior claim resolves to scholarly bibliography metadata.
- [ ] AC22.2 (`T22.2`) All six CSVs equal fitted outputs/Step1 dates and notebook has no failed/unexecuted cell.

---

## PR-23 — Notebook Step 3 model selection + selected-state provenance

**Agent lane:** A

**Dependencies:** PR-17, PR-22

**Git branch:** `pr-23-notebook-step-3-model-selection-selected-state-provenance`

**Git status:** `git status --short --branch` must show `pr-23-notebook-step-3-model-selection-selected-state-provenance` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-23 — Notebook Step 3 model selection + selected-state provenance`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
reports/tables/step3_model_comparison.csv
reports/tables/step3_selected_states.csv
reports/generated/step3_selected_model.json
```

### Tasks

- [ ] T23.1 Add IC equations/four-row table, within-family comparison caveat/winners, preferred-method result/reason, and MLA 9 citations to scholarly information-criterion sources from `reports/references.bib`.
- [ ] T23.2 Copy selected candidate state Series to canonical Date,state CSV; write exact selected-model JSON with Step1 SHA/path; execute full notebook.

### Acceptance criteria

- [ ] AC23.1 (`T23.1`) Displayed/CSV comparison and selected decision match shared functions with no cross-family IC claim; AIC/BIC definitions and external selection claims have resolved scholarly citations.
- [ ] AC23.2 (`T23.2`) Selected states/JSON/hash/path match candidate and Step1 exactly; no refit/decode; notebook fully executed.

---

## PR-24 — Notebook Step 3 state-conditional ETF analysis

**Agent lane:** B

**Dependencies:** PR-18, PR-19, PR-23

**Git branch:** `pr-24-notebook-step-3-state-conditional-etf-analysis`

**Git status:** `git status --short --branch` must show `pr-24-notebook-step-3-state-conditional-etf-analysis` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-24 — Notebook Step 3 state-conditional ETF analysis`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
reports/tables/step3_state_asset_statistics.csv
reports/figures/step3_state_asset_statistics.png
```

### Tasks

- [ ] T24.1 Load/validate canonical selected states (no fit/decode); compute/display/save shared mean/std/count table and grouped bar chart.
- [ ] T24.2 Interpret states from displayed evidence; state units, ddof=1, non-annualization/sample-size limitations; cite every external statistical/financial interpretation from `reports/references.bib`; execute full notebook.

### Acceptance criteria

- [ ] AC24.1 (`T24.1`) Selected-state provenance/dates are exact; table/chart match shared outputs.
- [ ] AC24.2 (`T24.2`) Interpretation is evidence-based with all fixed statistical caveats; external interpretation claims have resolved scholarly citations and notebook is fully executed.

---

## PR-25 — Notebook Step 4 allocation + Steps2–4 manifest

**Agent lane:** A

**Dependencies:** PR-20, PR-24

**Git branch:** `pr-25-notebook-step-4-allocation-steps2-4-manifest`

**Git status:** `git status --short --branch` must show `pr-25-notebook-step-4-allocation-steps2-4-manifest` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-25 — Notebook Step 4 allocation + Steps2–4 manifest`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
reports/tables/step4_allocation_mapping.csv
reports/generated/steps_2_4_manifest.json
```

### Tasks

- [ ] T25.1 Add/display/save shared Step4 decision mapping/equation/justification/100%+tie rule; state no 60/40 and all fixed lookahead limitations.
- [ ] T25.2 Add evidence-supported practical takeaways with MLA 9 citations, refresh `reports/references.bib`, render a complete Step1–4 Works Cited from cited entries only, verify citation/source-note integrity, write the exact deterministic Steps2–4 manifest, and execute/verify full Step1–4 notebook/artifacts.

### Acceptance criteria

- [ ] AC25.1 (`T25.1`) Mapping/justifications/rules and non-OOS caveats match canonical evidence.
- [ ] AC25.2 (`T25.2`) All notebook citations resolve to verified bibliography entries, every rendered Works Cited entry is cited, source notes are present, manifest/hash/path coverage is exact, and notebook/artifacts are fully consistent.

---

## PR-26 — Deterministic README analysis synchronizer

**Agent lane:** A

**Dependencies:** PR-25

**Git branch:** `pr-26-deterministic-readme-analysis-synchronizer`

**Git status:** `git status --short --branch` must show `pr-26-deterministic-readme-analysis-synchronizer` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-26 — Deterministic README analysis synchronizer`

**Files owned:**

```text
scripts/sync_readme_analysis.py
tests/test_sync_readme_analysis.py
```

### Tasks

- [ ] T26.1 Generate one deterministic marker-bounded technical block solely from canonical artifacts, including equations/cautions/comparison/selection/stats/allocation/figures; no estimation path.
- [ ] T26.2 Add fixture tests for exact parity, Greek notation, missing/marker failures, no estimation calls, and idempotence.

### Acceptance criteria

- [ ] AC26.1 (`T26.1`) Generated block exactly reflects canonical files and refuses missing/invalid markers without fitting.
- [ ] AC26.2 (`T26.2`) All synchronizer tests pass offline; second run byte-identical.

---

## PR-27 — Synchronize README through Step 4

**Agent lane:** A

**Dependencies:** PR-26

**Git branch:** `pr-27-synchronize-readme-through-step-4`

**Git status:** `git status --short --branch` must show `pr-27-synchronize-readme-through-step-4` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-27 — Synchronize README through Step 4`

**Files owned:**

```text
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T27.1 Run synchronizer and update factual Step1–4 status/commands/paths while marking Step5 unimplemented; preserve quality/coverage documentation.
- [ ] T27.2 Extend checker for one marker pair, notebook/selected-model/selected-states/manifest paths and unified backlog references.

### Acceptance criteria

- [ ] AC27.1 (`T27.1`) README has exact Step1–4 technical parity and no Step5 result claim.
- [ ] AC27.2 (`T27.2`) Checker passes valid README and fails required-path/marker/backlog mutations.

---

## PR-28 — Notebook PDF sidecar builder

**Agent lane:** B

**Dependencies:** PR-25

**Git branch:** `pr-28-notebook-pdf-sidecar-builder`

**Git status:** `git status --short --branch` must show `pr-28-notebook-pdf-sidecar-builder` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-28 — Notebook PDF sidecar builder`

**Files owned:**

```text
scripts/build_pdf_report.py
tests/test_build_pdf_report.py
```

### Tasks

- [ ] T28.1 Build from template page1 plus the canonical executed notebook; reject notebook error outputs; render notebook cells/outputs in order; exclude template page2; preserve `reports/references.bib`-resolved citations/Works Cited; embed sidecar role, source notebook path, and exact notebook SHA-256; forbid PDF-only analysis.
- [ ] T28.2 Add offline tests for cover/page2 exclusion, notebook error rejection, notebook-SHA/source-path metadata, non-empty renderability, known names, inherited `reports/references.bib` citation integrity, and rejection of any stale sidecar hash.

### Acceptance criteria

- [ ] AC28.1 (`T28.1`) Generated fixture PDF is a notebook-derived sidecar with correct cover/no page2, exact notebook provenance metadata, no independent analysis path, and the notebook's resolved scholarly citations/source notes/Works Cited.
- [ ] AC28.2 (`T28.2`) All PDF sidecar, provenance, stale-hash, renderability, and citation-integrity tests pass offline.

---

## PR-29 — Generate/visually verify Step1–4 PDF sidecar

**Agent lane:** B

**Dependencies:** PR-28

**Git branch:** `pr-29-generate-visually-verify-step1-4-pdf-sidecar`

**Git status:** `git status --short --branch` must show `pr-29-generate-visually-verify-step1-4-pdf-sidecar` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-29 — Generate/visually verify Step1–4 PDF sidecar`

**Files owned:**

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
reports/rendered/Stochastic_Modeling_GWP2_Report/*.png
```

### Tasks

- [ ] T29.1 Generate the Step1–4 PDF sidecar from the committed executed notebook and template page1 only; preserve notebook equations, project-function cells, stored outputs, figures, explanations, limitations, MLA 9 citations, source notes, and Works Cited resolved through `reports/references.bib`; add no PDF-only prose.
- [ ] T29.2 Render/inspect every page for clipping/overlap/glyph/blank/split defects, verify the stored notebook SHA-256 metadata, and mark the sidecar interim until Step5.

### Acceptance criteria

- [ ] AC29.1 (`T29.1`) PDF content is derived from the executed notebook in order, includes the notebook's technical content and resolved citations/source notes/Works Cited, has the correct cover/no page2, and contains no independently authored analytical section.
- [ ] AC29.2 (`T29.2`) Every page passes visual QA, notebook provenance metadata matches exactly, and Step5 sidecar regeneration requirement is explicit.

---

## PR-30 — Executed-notebook HTML exporter

**Agent lane:** A

**Dependencies:** PR-25

**Git branch:** `pr-30-executed-notebook-html-exporter`

**Git status:** `git status --short --branch` must show `pr-30-executed-notebook-html-exporter` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-30 — Executed-notebook HTML exporter`

**Files owned:**

```text
scripts/export_notebook_html.py
tests/test_export_notebook_html.py
reports/gwp2_vix_regime_allocation.html
```

### Tasks

- [ ] T30.1 Export stored-output notebook to canonical HTML without execution/refit; reject failed/unexecuted/missing expected output or missing final Works Cited; preserve MLA citations/source notes rendered from `reports/references.bib`; embed notebook SHA marker.
- [ ] T30.2 Add offline fixture tests and generate actual Step1–4 HTML.

### Acceptance criteria

- [ ] AC30.1 (`T30.1`) HTML contains Step1–4 stored outputs/current hash plus notebook citations/source notes/Works Cited and exporter has no execution path.
- [ ] AC30.2 (`T30.2`) All exporter tests pass offline and canonical HTML is non-empty.

---

## PR-31 — Step1–4 sidecar parity checker

**Agent lane:** B

**Dependencies:** PR-27, PR-29, PR-30

**Git branch:** `pr-31-step1-4-sidecar-parity-checker`

**Git status:** `git status --short --branch` must show `pr-31-step1-4-sidecar-parity-checker` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-31 — Step1–4 sidecar parity checker`

**Files owned:**

```text
scripts/check_analysis_sidecars.py
tests/test_analysis_sidecars.py
```

### Tasks

- [ ] T31.1 Validate manifest/input hash/artifacts plus notebook/README exact technical parity, HTML notebook hash, PDF notebook-content parity and exact notebook-SHA/source-path provenance, and citation integrity against `reports/references.bib`: resolved in-text cites, cited-only Works Cited entries, required scholarly support, and source notes.
- [ ] T31.2 Add deterministic stale/missing/hash/value failure tests for every sidecar/artifact class plus duplicate bibliography keys, unresolved citations, orphan Works Cited entries, missing scholarly support, and URL-only pseudo-citations.

### Acceptance criteria

- [ ] AC31.1 (`T31.1`) Any missing/stale/mismatched canonical technical artifact, notebook/PDF provenance mismatch, or citation/source-note/Works-Cited defect fails at the correct parity level.
- [ ] AC31.2 (`T31.2`) All parity tests pass offline.

---

## PR-32 — Step1–4 sidecar + backlog CI gates

**Agent lane:** A

**Dependencies:** PR-31

**Git branch:** `pr-32-step1-4-sidecar-backlog-ci-gates`

**Git status:** `git status --short --branch` must show `pr-32-step1-4-sidecar-backlog-ci-gates` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-32 — Step1–4 sidecar + backlog CI gates`

**Files owned:**

```text
.github/workflows/quality-gates.yml
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T32.1 Add independent `analysis-sidecars` and `backlog-contract` jobs; aggregate quality-gate requires them plus existing jobs while core lint/type/unit/integration remain parallel and coverage>=90%.
- [ ] T32.2 Update README/checker to document/require new jobs, Step1–4 parity policy, canonical `reports/references.bib`, MLA 9 scholarly-citation requirements, and citation-integrity enforcement.

### Acceptance criteria

- [ ] AC32.1 (`T32.1`) Workflow dependencies/jobs/coverage/parallelism exactly match contract.
- [ ] AC32.2 (`T32.2`) README/checker accurately enforce current parity/CI paths and scientific-citation policy.

---

## PR-33 — One-day-lag rotation engine

**Agent lane:** A

**Dependencies:** PR-20, PR-23

**Git branch:** `pr-33-one-day-lag-rotation-engine`

**Git status:** `git status --short --branch` must show `pr-33-one-day-lag-rotation-engine` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-33 — One-day-lag rotation engine`

**Files owned:**

```text
src/vix_regime_allocation/backtest.py
tests/test_backtest.py
```

### Public interface

```python
def build_rotation_returns(data: pandas.DataFrame, states: pandas.Series, allocation: pandas.DataFrame) -> pandas.DataFrame: ...
```

### Tasks

- [ ] T33.1 Validate Step1/state/allocation; convert log→simple; apply state t-1 weights to return t; exclude first row; return exact decision-detail schema.
- [ ] T33.2 Add manual lag/decision/return and invalid allocation/index/return tests.

### Acceptance criteria

- [ ] AC33.1 (`T33.1`) A state change affects only next trading row; schema/dates/weights/simple returns exactly match contract.
- [ ] AC33.2 (`T33.2`) All backtest tests pass offline.

---

## PR-34 — Required benchmark engines

**Agent lane:** B

**Dependencies:** PR-02

**Git branch:** `pr-34-required-benchmark-engines`

**Git status:** `git status --short --branch` must show `pr-34-required-benchmark-engines` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-34 — Required benchmark engines`

**Files owned:**

```text
src/vix_regime_allocation/benchmarks.py
tests/test_benchmarks.py
```

### Tasks

- [ ] T34.1 Implement validated comparison-index simple returns, exact monthly 1/3 reset + intra-month drift, and exact SPY return/names.
- [ ] T34.2 Test across >=2 months for reset/drift/no-daily-reset/SPY/index/invalid return.

### Acceptance criteria

- [ ] AC34.1 (`T34.1`) Both benchmark series exactly satisfy date/weight/simple-return conventions.
- [ ] AC34.2 (`T34.2`) All benchmark tests pass offline.

---

## PR-35 — Required performance metrics

**Agent lane:** A

**Dependencies:** PR-02

**Git branch:** `pr-35-required-performance-metrics`

**Git status:** `git status --short --branch` must show `pr-35-required-performance-metrics` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-35 — Required performance metrics`

**Files owned:**

```text
src/vix_regime_allocation/performance.py
tests/test_performance.py
```

### Public interface

```python
TRADING_DAYS=252
def cumulative_wealth(returns: pandas.Series) -> pandas.Series: ...
def performance_metrics(returns: pandas.Series) -> dict[str, float | int]: ...
```

### Tasks

- [ ] T35.1 Implement validated shared wealth and all five fixed metrics, including W0=1 drawdown and zero-volatility Sharpe failure.
- [ ] T35.2 Add exact manual tests including first-period loss, invalid <=-1, insufficient N, and zero volatility.

### Acceptance criteria

- [ ] AC35.1 (`T35.1`) Every metric equals fixed formula/sign/observation convention.
- [ ] AC35.2 (`T35.2`) All performance tests pass offline.

---

## PR-36 — Aligned comparison + performance summary

**Agent lane:** B

**Dependencies:** PR-33, PR-34, PR-35

**Git branch:** `pr-36-aligned-comparison-performance-summary`

**Git status:** `git status --short --branch` must show `pr-36-aligned-comparison-performance-summary` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-36 — Aligned comparison + performance summary`

**Files owned:**

```text
src/vix_regime_allocation/backtest_summary.py
tests/test_backtest_summary.py
```

### Tasks

- [ ] T36.1 Require exact date equality; build exact three-column comparison and exact three-row summary by delegating to shared metrics.
- [ ] T36.2 Test mismatch failure, schema/order/value mapping, and no duplicate metric math.

### Acceptance criteria

- [ ] AC36.1 (`T36.1`) Comparison/summary have exact dates/names/order/values and mismatched dates fail.
- [ ] AC36.2 (`T36.2`) All summary tests pass offline.

---

## PR-37 — Cumulative-performance figure

**Agent lane:** A

**Dependencies:** PR-35

**Git branch:** `pr-37-cumulative-performance-figure`

**Git status:** `git status --short --branch` must show `pr-37-cumulative-performance-figure` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-37 — Cumulative-performance figure`

**Files owned:**

```text
src/vix_regime_allocation/backtest_plot.py
tests/test_backtest_plot.py
```

### Tasks

- [ ] T37.1 Validate comparison and plot exactly three cumulative-return curves using shared `cumulative_wealth`, with title/axes/scales/zero line/legend/save/close.
- [ ] T37.2 Test helper delegation, curve y-values, series set, presentation, file closure.

### Acceptance criteria

- [ ] AC37.1 (`T37.1`) Curves equal shared compounded wealth-1 and exactly three required portfolios appear.
- [ ] AC37.2 (`T37.2`) All plot tests pass offline.

---

## PR-38 — K=2 vs K=3 sensitivity

**Agent lane:** B

**Dependencies:** PR-18, PR-20, PR-33, PR-35

**Git branch:** `pr-38-k-2-vs-k-3-sensitivity`

**Git status:** `git status --short --branch` must show `pr-38-k-2-vs-k-3-sensitivity` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-38 — K=2 vs K=3 sensitivity`

**Files owned:**

```text
src/vix_regime_allocation/sensitivity.py
tests/test_sensitivity.py
```

### Tasks

- [ ] T38.1 Validate preferred family and K=2/3 states; delegate to shared Step3 stats/Step4 mapping/backtest/metrics; use common return-date intersection; return exact sorted schema.
- [ ] T38.2 Test validation, all delegations, common observations/dates, and metric values.

### Acceptance criteria

- [ ] AC38.1 (`T38.1`) Sensitivity uses only fixed preferred family/shared rules/common dates and exact schema.
- [ ] AC38.2 (`T38.2`) All sensitivity tests pass offline.

---

## PR-39 — Step5 source integration test

**Agent lane:** A

**Dependencies:** PR-33, PR-34, PR-35, PR-36, PR-37, PR-38

**Git branch:** `pr-39-step5-source-integration-test`

**Git status:** `git status --short --branch` must show `pr-39-step5-source-integration-test` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-39 — Step5 source integration test`

**Files owned:**

```text
tests/integration/test_step5_pipeline.py
```

### Tasks

- [ ] T39.1 Create hand-checkable >=2-month synthetic fixture and run rotation, both benchmarks, comparison, metrics, plot, K sensitivity end-to-end.
- [ ] T39.2 Assert all schemas/date contracts/lag/monthly-reset/common-sensitivity rules and no network/model fit.

### Acceptance criteria

- [ ] AC39.1 (`T39.1`) Fixture exercises all required Step5 source paths and exact dates/counts.
- [ ] AC39.2 (`T39.2`) Integration test passes entirely offline.

---

## PR-40 — Notebook Step5 backtest + benchmarks

**Agent lane:** A

**Dependencies:** PR-25, PR-39

**Git branch:** `pr-40-notebook-step5-backtest-benchmarks`

**Git status:** `git status --short --branch` must show `pr-40-notebook-step5-backtest-benchmarks` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-40 — Notebook Step5 backtest + benchmarks`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
reports/tables/step5_daily_returns.csv
```

### Tasks

- [ ] T40.1 Load/verify Step1 hash, selected-model+selected-state CSV, Step4 mapping without fit/decode; show simple-return/one-row-lag equations and decision examples; call shared rotation/benchmarks; save daily comparison.
- [ ] T40.2 Explain monthly benchmark convention, gross costs, and all full-sample/Viterbi/allocation lookahead caveats with scholarly backtesting/portfolio-method citations from `reports/references.bib`; maintain source notes and execute full notebook.

### Acceptance criteria

- [ ] AC40.1 (`T40.1`) Daily CSV/displayed decisions prove exact lag/benchmarks/state provenance and shared-function use.
- [ ] AC40.2 (`T40.2`) All assumptions/non-OOS caveats are explicit, externally sourced methodological claims have resolved scholarly citations/source notes, and the full notebook has no failed/unexecuted cell.

---

## PR-41 — Notebook Step5 metrics + cumulative comparison

**Agent lane:** B

**Dependencies:** PR-36, PR-37, PR-40

**Git branch:** `pr-41-notebook-step5-metrics-cumulative-comparison`

**Git status:** `git status --short --branch` must show `pr-41-notebook-step5-metrics-cumulative-comparison` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-41 — Notebook Step5 metrics + cumulative comparison`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
reports/tables/step5_performance_summary.csv
reports/figures/step5_cumulative_performance.png
```

### Tasks

- [ ] T41.1 Show five formulas/assumptions (including W0=1) with scholarly citations for externally sourced performance-metric definitions, build/display/save exact shared summary and three-curve figure, and maintain source notes from `reports/references.bib`.
- [ ] T41.2 Interpret all five metrics versus both benchmarks and state risk-adjusted conclusion as full-sample descriptive/in-sample; execute full notebook.

### Acceptance criteria

- [ ] AC41.1 (`T41.1`) Summary/figure/equations exactly match canonical shared calculations/presentation and every external metric definition has a resolved scholarly citation.
- [ ] AC41.2 (`T41.2`) Interpretation covers both benchmarks/all metrics, makes no causal/OOS claim, notebook fully executed.

---

## PR-42 — Notebook Step5 sensitivity + manifest

**Agent lane:** A

**Dependencies:** PR-38, PR-41

**Git branch:** `pr-42-notebook-step5-sensitivity-manifest`

**Git status:** `git status --short --branch` must show `pr-42-notebook-step5-sensitivity-manifest` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-42 — Notebook Step5 sensitivity + manifest`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
reports/tables/step5_state_count_sensitivity.csv
reports/generated/step5_manifest.json
```

### Tasks

- [ ] T42.1 Load preferred-family canonical K=2/3 state files (no fit/decode), run/display/save shared sensitivity, discuss all five metrics and in-sample scope.
- [ ] T42.2 Add final practical takeaway/limitations/future causal validation with scholarly support, refresh `reports/references.bib`, verify all MLA 9 citations/source notes, render the final cited-only Works Cited, write exact Step5 manifest, and execute/verify full Step1–5 notebook.

### Acceptance criteria

- [ ] AC42.1 (`T42.1`) Sensitivity table/provenance/common dates and discussion match fixed contract without OOS claim.
- [ ] AC42.2 (`T42.2`) Takeaway/citations/source notes/Works Cited have complete verified scholarly provenance with no unresolved/orphan entries; manifest/hash/path coverage is exact and final notebook/artifacts are fully consistent.

---

## PR-43 — README Step5 synchronization

**Agent lane:** B

**Dependencies:** PR-27, PR-42

**Git branch:** `pr-43-readme-step5-synchronization`

**Git status:** `git status --short --branch` must show `pr-43-readme-step5-synchronization` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-43 — README Step5 synchronization`

**Files owned:**

```text
scripts/sync_readme_analysis.py
tests/test_sync_readme_analysis.py
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T43.1 Extend artifact-only synchronizer/generated block/status to Step5 equations/assumptions/summary/sensitivity/figure and no-OOS caveat; no recomputation.
- [ ] T43.2 Extend tests/checker for exact Step5 parity, required paths/assumptions, `reports/references.bib`/MLA 9 citation policy, missing/stale failures, and idempotence; regenerate README.

### Acceptance criteria

- [ ] AC43.1 (`T43.1`) README has exact Step1–5 technical parity with canonical files and factual status.
- [ ] AC43.2 (`T43.2`) All sync/checker tests pass offline and repeated sync is byte-identical.

---

## PR-44 — Extend PDF sidecar builder through Step5

**Agent lane:** A

**Dependencies:** PR-28, PR-42

**Git branch:** `pr-44-extend-pdf-sidecar-builder-through-step5`

**Git status:** `git status --short --branch` must show `pr-44-extend-pdf-sidecar-builder-through-step5` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-44 — Extend PDF sidecar builder through Step5`

**Files owned:**

```text
scripts/build_pdf_report.py
tests/test_build_pdf_report.py
```

### Tasks

- [ ] T44.1 Extend PDF-sidecar validation through the final Step5 notebook sections and `reports/references.bib`; require the benchmark comparison, summary, sensitivity, recommendation, limitations, figures, equations, stored outputs, resolved MLA 9 citations/source notes/Works Cited to come from the executed notebook with no second analysis.
- [ ] T44.2 Add tests for Step5 stale/missing notebook content, exact notebook-SHA/source-path metadata, final citation integrity, template cover/page2 exclusion, known names, and rejection of PDF-only analytical content.

### Acceptance criteria

- [ ] AC44.1 (`T44.1`) Builder requires the exact final executed notebook as its Step5 analytical source and a valid canonical scientific-source registry with resolved rendered citations.
- [ ] AC44.2 (`T44.2`) All extended PDF-builder tests pass offline.

---

## PR-45 — Final Step1–5 PDF sidecar + visual QA

**Agent lane:** A

**Dependencies:** PR-29, PR-44

**Git branch:** `pr-45-final-step1-5-pdf-sidecar-visual-qa`

**Git status:** `git status --short --branch` must show `pr-45-final-step1-5-pdf-sidecar-visual-qa` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-45 — Final Step1–5 PDF sidecar + visual QA`

**Files owned:**

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
reports/rendered/Stochastic_Modeling_GWP2_Report/*.png
```

### Tasks

- [ ] T45.1 Generate the final Step1–5 PDF sidecar from the fully executed notebook with canonical metrics, cumulative comparison figure, sensitivity, both-benchmark conclusion, recommendation, costs, full-sample/non-OOS limitations, equations, code/output cells, MLA 9 citations, official-primary source notes, and final Works Cited from `reports/references.bib`; use template page1 only and no PDF-only narrative.
- [ ] T45.2 Render and inspect every final page for all fixed visual defects.

### Acceptance criteria

- [ ] AC45.1 (`T45.1`) Final PDF is an exact rendered-notebook sidecar after the template cover, has matching notebook SHA-256/source-path metadata, includes the notebook's technical content and limitations, and preserves resolved scholarly citations/source notes with cited-only Works Cited entries.
- [ ] AC45.2 (`T45.2`) Every rendered page passes visual QA and final PDF is non-empty.

---

## PR-46 — Final executed-notebook HTML

**Agent lane:** B

**Dependencies:** PR-30, PR-42

**Git branch:** `pr-46-final-executed-notebook-html`

**Git status:** `git status --short --branch` must show `pr-46-final-executed-notebook-html` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-46 — Final executed-notebook HTML`

**Files owned:**

```text
reports/gwp2_vix_regime_allocation.html
```

### Tasks

- [ ] T46.1 Run existing exporter on final executed notebook; verify Step5 daily returns/summary/figure/sensitivity/interpretation, MLA 9 citations/source notes/final Works Cited from `reports/references.bib`, and exact notebook hash.

### Acceptance criteria

- [ ] AC46.1 (`T46.1`) Final HTML is non-empty, contains all Step1–5 stored outputs plus citations/source notes/Works Cited, and hash equals canonical notebook without re-execution.

---

## PR-47 — Final Step1–5 parity CI

**Agent lane:** B

**Dependencies:** PR-31, PR-32, PR-43, PR-45, PR-46

**Git branch:** `pr-47-final-step1-5-parity-ci`

**Git status:** `git status --short --branch` must show `pr-47-final-step1-5-parity-ci` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-47 — Final Step1–5 parity CI`

**Files owned:**

```text
scripts/check_analysis_sidecars.py
tests/test_analysis_sidecars.py
.github/workflows/quality-gates.yml
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T47.1 Extend checker/tests to Step5 manifest/artifacts, notebook/README technical parity, current HTML, PDF exact notebook-sidecar parity with notebook-SHA/source-path provenance, and final citation integrity against `reports/references.bib` for notebook-derived HTML/PDF artifacts.
- [ ] T47.2 Keep `analysis-sidecars`/`backlog-contract`/aggregate wiring, parallel core jobs, coverage>=90%; finalize README/checker Step1–5 status.

### Acceptance criteria

- [ ] AC47.1 (`T47.1`) Every Step5 stale/missing/hash/value sidecar defect, notebook/PDF provenance mismatch, and every unresolved/duplicate/orphan/URL-only citation or missing Works Cited/source-note defect fails at the correct parity level.
- [ ] AC47.2 (`T47.2`) CI/coverage/parallelism and final README/checker contracts remain exact.

---

## PR-48 — Deterministic final submission bundle builder

**Agent lane:** A

**Dependencies:** PR-47

**Git branch:** `pr-48-deterministic-final-submission-bundle-builder`

**Git status:** `git status --short --branch` must show `pr-48-deterministic-final-submission-bundle-builder` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-48 — Deterministic final submission bundle builder`

**Files owned:**

```text
scripts/build_submission_bundle.py
tests/test_submission_bundle.py
```

### Public interface

```python
def build_submission_bundle(repository_root: pathlib.Path, output_zip: pathlib.Path, manifest_path: pathlib.Path) -> None: ...
```

### Tasks

- [ ] T48.1 Implement exact allowlist including `reports/references.bib`, exclusions, non-symlink/path safety, deterministic sorted normalized ZIP, separate-PDF requirement, exact member/ZIP/PDF hashes and no-timestamp manifest, post-build reinspection; reject a missing/empty scientific citation registry.
- [ ] T48.2 Test byte-identical rebuild, exact members/hashes, missing/empty/traversal/symlink/forbidden failures, and PDF exclusion.

### Acceptance criteria

- [ ] AC48.1 (`T48.1`) Builder output/manifest exactly satisfy fixed submission contract and include the canonical non-empty scientific citation registry.
- [ ] AC48.2 (`T48.2`) All bundle tests pass offline.

---

## PR-49 — Generate/validate final release artifacts

**Agent lane:** B

**Dependencies:** PR-48

**Git branch:** `pr-49-generate-validate-final-release-artifacts`

**Git status:** `git status --short --branch` must show `pr-49-generate-validate-final-release-artifacts` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-49 — Generate/validate final release artifacts`

**Files owned:**

```text
dist/MScFE_622_GWP2_submission.zip
reports/generated/submission_manifest.json
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T49.1 Generate final ZIP/manifest from post-PR47 canonical files; inspect exact members including `reports/references.bib`, citation-registry byte parity, forbidden exclusions, and separately hashed PDF.
- [ ] T49.2 Update README/checker with exact ZIP path, separate PDF path, citation-registry contents/rebuild/upload instructions, and notebook/PDF scholarly-source requirements; run backlog/README/sidecar/full quality suite.

### Acceptance criteria

- [ ] AC49.1 (`T49.1`) ZIP/manifest/member hashes, included canonical citation registry, and separate-PDF exclusion exactly match fixed contract.
- [ ] AC49.2 (`T49.2`) README/checker are actionable/exact and all final quality checks pass.

---
# Parallel schedule (two weak agents)

```text
W1  A PR-01 | B PR-02
W2  A PR-03 | B wait
W3  B PR-04
W4  A PR-05
W5  PR-06 then PR-07 sequential
W6  A PR-08 | B PR-09
W7  A PR-10 | B PR-11
W8  A PR-12 | B PR-13
W9  A PR-14 | B PR-15
W10 A PR-17 | B PR-16
W11 B PR-18
W12 A PR-19 | B PR-20
Notebook serialized: W13 PR-21, W14 PR-22, W15 PR-23, W16 PR-24, W17 PR-25
W18 A PR-26 | B PR-28
W19 A PR-27 | B PR-29
W20 A PR-30
W21 B PR-31
W22 A PR-32
W23 A PR-33 | B PR-34
W24 A PR-35
W25 A PR-37 | B PR-36
W26 B PR-38
W27 A PR-39
Notebook serialized: W28 PR-40, W29 PR-41, W30 PR-42
W31 A PR-44 | B PR-43
W32 A PR-45 | B PR-46
W33 B PR-47
W34 A PR-48
W35 B PR-49
```

No two PRs that write the same path may be open concurrently; notebook PRs never overlap.

# Merge rules

For every PR: branch from current main after dependencies; modify only owned files; implement every task and no later PR; prove every matching AC; run current lint/type/unit/integration/coverage/readme/backlog/parity gates; update from main and rerun; merge only after aggregate quality-gate; delete branch. A GitHub branch/ruleset is still required to technically block privileged direct pushes.

# Final Definition of Done

- [ ] PR-01..PR-49 merged to main.
- [ ] All assignment Steps1–5 outputs, metrics, benchmarks, sensitivity, plots/tables and interpretations satisfy fixed contracts.
- [ ] All candidate/preferred state sequences are canonical artifacts; no later refit/redecode is used merely to recover them.
- [ ] One-row execution lag, W0=1 drawdown, zero-RF Sharpe, monthly benchmark, and identical comparison dates are verified.
- [ ] Full-sample regime/Viterbi/allocation lookahead limitations, gross-cost assumption, and non-OOS status are explicit.
- [ ] Final notebook fully executed with resolved MLA 9 scholarly citations/source notes/final Works Cited; README exact technical parity; HTML exact duplicate; separate PDF exact rendered-notebook sidecar with matching notebook SHA-256/source-path provenance, template page1 only, and visual QA.
- [ ] `analysis-sidecars`, `backlog-contract`, lint/type/unit/integration, combined coverage>=90%, and aggregate quality-gate pass.
- [ ] Deterministic final ZIP contains exact executable allowlist including `reports/references.bib` and excludes standalone PDF/forbidden files; submission manifest hashes exact final bytes.
- [ ] README gives exact rebuild and upload instructions, including separate PDF upload.
