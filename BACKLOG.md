# BACKLOG — MScFE 622 GWP2: Steps 1–5

This is the **single canonical implementation backlog** for MScFE 622 Stochastic Modeling GWP2 through Step 5.

It consolidates the previously separate Step 1 and Steps 2–4 backlogs without changing the established PR numbering. PR-01 through PR-05 cover Step 1. PR-06 through PR-32 cover Steps 2–4, notebook integration, synchronized sidecars, and Steps 1–4 parity CI. PR-33 through PR-47 cover complete Step 5 backtesting/evaluation and final Steps 1–5 sidecar parity.

The backlog has been audited against the assignment brief, submission instructions, report template, and grading rubric. It is intentionally optimized for **two weak coding agents working in parallel**: PRs have explicit dependencies, fixed file ownership, deterministic interfaces, numbered tasks, and one-to-one matching acceptance criteria.

---

# Part I — Step 1: Data Preparation and Exploration

The assignment requires daily adjusted-close data for TLT, GLD, SPY and VIX, the maximum common sample, ETF daily log returns, a VIX change/return series, common-date alignment, missing-value removal, one ETF-return plot and one VIX-change plot.

## Step 1 non-negotiable rules

1. Every task has a task ID `Txx.n`.
2. Every task has exactly one matching acceptance criterion `ACxx.n`.
3. An acceptance criterion may test several observable facts only when those facts belong to the single matching task.
4. Agents must modify only files listed under **Files owned** for their PR.
5. Agents must not implement work assigned to a later PR.
6. Tests must be deterministic and offline unless a PR explicitly states otherwise.
7. Every PR must pass the repository `quality-gate`, including the >=90% combined source-coverage requirement.
8. The README sidecar is updated only when a PR changes a user-facing/canonical contract. Internal source-file additions alone do not require README edits.
9. The canonical analysis artifact is `notebooks/gwp2_vix_regime_allocation.ipynb`; Step 1 creates its first section and later PRs in this same backlog extend the same notebook.
10. No numerical result may be invented or copied from an external example.

---

## Fixed Step 1 contracts

### Tickers

Use exactly:

```python
TICKERS = {
    "TLT": "TLT",
    "GLD": "GLD",
    "SPY": "SPY",
    "VIX": "^VIX",
}
```

### Yahoo Finance download contract

Use `yfinance.download` with these explicit semantics:

```text
tickers = ["TLT", "GLD", "SPY", "^VIX"]
period = "max"
interval = "1d"
auto_adjust = False
back_adjust = False
actions = False
progress = False
```

The implementation must extract **`Adj Close`**, not `Close`.

Do not rely on the current yfinance default for `auto_adjust`; pass the argument explicitly so a library-default change cannot silently change the dataset definition.

A live network request is used only when the notebook/pipeline is intentionally executed. Unit tests must mock the downloader.

### Raw data schema

The raw adjusted-close table must:

- be a `pandas.DataFrame`;
- use a timezone-naive `DatetimeIndex` named `Date`;
- contain unique dates;
- be sorted strictly ascending by date;
- contain exactly these columns, in this order:

```text
TLT
GLD
SPY
VIX
```

Each non-missing raw price must be finite and strictly positive.

### Maximum common sample definition

The common raw sample is the **intersection of dates on which all four adjusted-close values are present**.

Implementation rule:

```text
common_prices = raw_prices.dropna(subset=["TLT", "GLD", "SPY", "VIX"])
```

Do not forward-fill, backward-fill or interpolate any price.

After common-date restriction, the first retained date must be the earliest date with all four values available and the final retained date must be the latest retained common date.

### Clean Step 1 dataset schema

The final clean table must:

- use a timezone-naive `DatetimeIndex` named `Date`;
- contain unique dates;
- be sorted ascending;
- contain only finite values;
- contain no missing values;
- contain exactly these columns, in this order:

```text
TLT
GLD
SPY
VIX
TLT_log_return
GLD_log_return
SPY_log_return
VIX_change
```

### Return definitions

For each ETF `X in {TLT, GLD, SPY}`:

```text
X_log_return_t = ln(X_t / X_{t-1})
```

For VIX:

```text
VIX_change_t = VIX_t - VIX_{t-1}
```

Step 1 uses **VIX change**, not a percentage/simple VIX return.

Because lagged calculations require the prior common-date observation, the first row of the common raw sample is removed from the final clean dataset.

### Fixed output paths

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

The CSV and figures are generated artifacts. They must never be hand-edited.

---

## PR-01 — Implement Yahoo Finance adjusted-close loader

**Agent lane:** A

**Dependencies:** repository scaffold only

**Files owned:**

```text
src/vix_regime_allocation/data.py
tests/test_data.py
```

### Public interface

```python
def download_adjusted_close() -> pandas.DataFrame:
    ...
```

### Tasks

- [ ] T01.1 Create `data.py` and define the exact `TICKERS` mapping.
- [ ] T01.2 Implement `download_adjusted_close()` using the exact Yahoo/yfinance call semantics in this backlog.
- [ ] T01.3 Extract `Adj Close` only and rename the `^VIX` output column to `VIX`.
- [ ] T01.4 Normalize the result to exact column order `TLT, GLD, SPY, VIX` and a timezone-naive `DatetimeIndex` named `Date`.
- [ ] T01.5 Sort dates ascending and reject duplicate dates.
- [ ] T01.6 Reject any non-missing raw adjusted-close value that is non-finite or <=0.
- [ ] T01.7 Add mocked offline tests for ticker request arguments, adjusted-close extraction, renaming, schema, index normalization, sorting, duplicate rejection and invalid-price rejection.

### Acceptance criteria

- [ ] AC01.1 (`T01.1`) `data.py` exists and `TICKERS` equals the fixed mapping exactly.
- [ ] AC01.2 (`T01.2`) a mocked call proves `period="max"`, `interval="1d"`, `auto_adjust=False`, `back_adjust=False`, `actions=False` and `progress=False` are passed explicitly for exactly the four required Yahoo tickers.
- [ ] AC01.3 (`T01.3`) tests prove returned values come from `Adj Close` and the external `^VIX` label is exposed internally as `VIX`.
- [ ] AC01.4 (`T01.4`) returned columns/index type/index name/timezone match the contract exactly.
- [ ] AC01.5 (`T01.5`) unsorted input is returned sorted and duplicate-date input fails with `ValueError`.
- [ ] AC01.6 (`T01.6`) zero, negative and infinite non-missing prices fail with `ValueError`; NaNs remain allowed for PR-02 common-date handling.
- [ ] AC01.7 (`T01.7`) all loader tests pass without network access.

---

## PR-02 — Implement deterministic Step 1 transformation

**Agent lane:** B

**Dependencies:** none; develop against the fixed raw-data schema

**Files owned:**

```text
src/vix_regime_allocation/transform.py
tests/test_transform.py
```

### Public interface

```python
def prepare_step1_data(prices: pandas.DataFrame) -> pandas.DataFrame:
    ...
```

### Tasks

- [ ] T02.1 Validate the exact raw columns, `DatetimeIndex`, `Date` index name, unique dates and ascending ordering.
- [ ] T02.2 Restrict to the exact common-date intersection with `dropna` across all four price columns; do not impute.
- [ ] T02.3 Compute TLT, GLD and SPY daily log returns with `ln(P_t/P_{t-1})` on the common sample.
- [ ] T02.4 Compute `VIX_change = VIX_t - VIX_{t-1}` on the same common sample.
- [ ] T02.5 Remove only rows made incomplete by the lagged transformations and reject any remaining non-finite derived value.
- [ ] T02.6 Return the exact clean schema/order/index contract.
- [ ] T02.7 Add hand-computable deterministic tests covering normal calculation, an interior missing raw date, first-row removal, invalid schema and non-finite derived output.

### Acceptance criteria

- [ ] AC02.1 (`T02.1`) malformed columns/index/name/order/duplicates fail clearly rather than being silently repaired.
- [ ] AC02.2 (`T02.2`) a synthetic date containing one missing instrument is absent before lagged calculations, and no fill/interpolation is performed.
- [ ] AC02.3 (`T02.3`) all three ETF log returns equal hand-calculated `ln(P_t/P_{t-1})` values within numerical tolerance.
- [ ] AC02.4 (`T02.4`) every VIX change equals the hand-calculated first difference on the identical common-date sequence.
- [ ] AC02.5 (`T02.5`) final output contains no NaN/inf and contains exactly one fewer row than a complete common raw sample.
- [ ] AC02.6 (`T02.6`) columns, index name, unique/sorted index and finite/no-missing guarantees match the fixed clean-data contract.
- [ ] AC02.7 (`T02.7`) all transformation tests pass offline.

---

## PR-03 — Implement the two required Step 1 plots

**Agent lane:** A after PR-01; may be developed from the fixed clean schema without PR-02 code

**Dependencies:** fixed clean-data contract

**Files owned:**

```text
src/vix_regime_allocation/plots.py
tests/test_plots.py
```

### Public interfaces

```python
def plot_etf_log_returns(data: pandas.DataFrame, output_path: pathlib.Path) -> None:
    ...


def plot_vix_change(data: pandas.DataFrame, output_path: pathlib.Path) -> None:
    ...
```

### Tasks

- [ ] T03.1 Validate the exact required input columns and aligned `Date` index before plotting.
- [ ] T03.2 Implement one ETF-return figure containing exactly `TLT_log_return`, `GLD_log_return` and `SPY_log_return` over time.
- [ ] T03.3 Give the ETF-return figure a non-empty title, date x-axis label, log-return y-axis label, visible scale/ticks and legend naming all three ETFs.
- [ ] T03.4 Implement one VIX-change figure containing exactly `VIX_change` over time.
- [ ] T03.5 Give the VIX-change figure a non-empty title, date x-axis label, VIX-change y-axis label and visible scale/ticks.
- [ ] T03.6 Create output parent directories, save non-empty PNGs and close each created figure.
- [ ] T03.7 Add deterministic synthetic/offline tests for plotted series, labels, file creation and figure closure.

### Acceptance criteria

- [ ] AC03.1 (`T03.1`) missing/misaligned required plot inputs fail clearly.
- [ ] AC03.2 (`T03.2`) the ETF figure contains exactly the three required ETF return series and no VIX series.
- [ ] AC03.3 (`T03.3`) title, axes, scale/ticks and three-ETF legend are present.
- [ ] AC03.4 (`T03.4`) the second figure plots `VIX_change`, not VIX level or VIX percentage return.
- [ ] AC03.5 (`T03.5`) VIX figure title, axes and scale/ticks are present.
- [ ] AC03.6 (`T03.6`) both requested files are created and non-empty and no created figure remains open.
- [ ] AC03.7 (`T03.7`) all plotting tests pass offline.

---

## PR-04 — Wire the executable Step 1 pipeline

**Agent lane:** B

**Dependencies:** PR-01, PR-02, PR-03

**Files owned:**

```text
scripts/run_step1.py
tests/test_run_step1.py
```

Generated but never hand-edited by this PR:

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

### Tasks

- [ ] T04.1 Create `scripts/run_step1.py` that calls `download_adjusted_close()` exactly once.
- [ ] T04.2 Pass the downloaded table to `prepare_step1_data()` exactly once and do not duplicate transformation logic.
- [ ] T04.3 Save the clean table to `data/processed/step1_data.csv` with the `Date` index serialized.
- [ ] T04.4 Call both plotting functions with the exact canonical figure paths.
- [ ] T04.5 Print the clean sample start date, end date and row count.
- [ ] T04.6 Add an offline orchestration test using mocks for the downloader and plotting calls plus a temporary output root.

### Acceptance criteria

- [ ] AC04.1 (`T04.1`) orchestration test proves exactly one loader call occurs.
- [ ] AC04.2 (`T04.2`) orchestration delegates to the shared transformation exactly once and contains no duplicate return/difference implementation.
- [ ] AC04.3 (`T04.3`) generated CSV has `Date` plus the exact eight clean columns, no missing values and non-zero rows.
- [ ] AC04.4 (`T04.4`) both canonical plot functions are invoked with their exact required output paths.
- [ ] AC04.5 (`T04.5`) captured stdout contains parseable start date, end date and integer row count matching the generated data.
- [ ] AC04.6 (`T04.6`) orchestration tests pass without Yahoo/network access and without writing outside temporary test paths.

---

## PR-05 — Create and execute the canonical notebook Step 1 section

**Agent lane:** A or B after PR-04

**Dependencies:** PR-04

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
README.md
```

### Required notebook structure

```text
Step 1: Data Preparation and Exploration
  1.1 Data definition and adjusted-close convention
  1.2 Common-sample construction
  1.3 ETF log-return equation
  1.4 VIX-change equation
  1.5 Data-quality checks
  1.6 Exploratory plots
  1.7 Interpretation and limitations
```

### Tasks

- [ ] T05.1 Create the Step 1 notebook section with the explicit heading/question number and concise scientific purpose.
- [ ] T05.2 Execute project functions rather than duplicate loader/transformation/plot implementations in notebook cells.
- [ ] T05.3 Display the cleaned-data shape, first/last date, first rows, exact column list and missing/non-finite validation.
- [ ] T05.4 Show the ETF log-return equation and define every symbol; if a Greek symbol is used, list its name/pronunciation before the equation.
- [ ] T05.5 Show the VIX-change equation and list `Δ — delta` before the equation.
- [ ] T05.6 Display the two canonical Step 1 plots with titles, axes, scales and legend where applicable.
- [ ] T05.7 Add precise interpretation describing only observed data preparation/volatility dynamics; do not claim regime or strategy results.
- [ ] T05.8 Add assumptions/limitations: adjusted-close convention, common-date filtering, use of VIX change and no imputation.
- [ ] T05.9 Add in-text citations and an MLA-formatted bibliography only for sources actually consulted; do not copy assignment-question wording verbatim.
- [ ] T05.10 Execute every Step 1 cell successfully and store outputs in the committed notebook.
- [ ] T05.11 Update README status and Step 1 sidecar section so it references the same canonical data/figures and does not independently recalculate results.

### Acceptance criteria

- [ ] AC05.1 (`T05.1`) notebook visibly identifies Step 1 and its purpose without reproducing the assignment prompt verbatim.
- [ ] AC05.2 (`T05.2`) notebook imports/calls project functions and contains no second implementation of download/return/difference/plot logic.
- [ ] AC05.3 (`T05.3`) all requested data-quality outputs are visibly stored and internally consistent.
- [ ] AC05.4 (`T05.4`) ETF-return equation is mathematically correct and every symbol/Greek pronunciation rule is satisfied.
- [ ] AC05.5 (`T05.5`) `Δ — delta` appears before the correct first-difference equation.
- [ ] AC05.6 (`T05.6`) both required figures are visible in stored notebook output and match canonical figure files.
- [ ] AC05.7 (`T05.7`) interpretation is evidence-based and contains no Step 2+ result claim.
- [ ] AC05.8 (`T05.8`) all four fixed Step 1 assumptions/limitations are explicitly stated.
- [ ] AC05.9 (`T05.9`) citations/bibliography use MLA format, refer only to actually consulted sources and no assignment questions are copied verbatim.
- [ ] AC05.10 (`T05.10`) Step 1 contains no failed or unexecuted code cell and all expected outputs are stored.
- [ ] AC05.11 (`T05.11`) README status is factually current and references the exact same generated Step 1 artifacts without recomputation.

### Step 1 parallel execution schedule

```text
Wave 1 - parallel
Agent A: PR-01 loader
Agent B: PR-02 transformation

Wave 2
Agent A: PR-03 plots
Agent B: wait until PR-01/02/03 are merged

Wave 3
Agent B: PR-04 pipeline

Wave 4
Either agent: PR-05 canonical notebook Step 1 + README sidecar
```

No agent may start a PR before its listed dependencies are merged to `main`.

### Step 1 Definition of Done

- [ ] PR-01 through PR-05 are merged to `main`.
- [ ] Yahoo data request semantics are explicit and adjusted close is unambiguous.
- [ ] TLT, GLD, SPY and VIX use the maximum common date intersection without imputation.
- [ ] ETF daily log returns and daily VIX change are calculated on the common sample.
- [ ] The final clean dataset satisfies the exact schema and contains no missing/non-finite values.
- [ ] `data/processed/step1_data.csv` exists and is reproducible.
- [ ] Both required Step 1 figures exist, are non-empty and are visible in the notebook.
- [ ] The notebook Step 1 section contains code calls, stored outputs, equations, scientific interpretation, limitations, citations and MLA bibliography entries as applicable.
- [ ] README is synchronized with the Step 1 canonical artifacts.
- [ ] No Step 2+ implementation is included before its PR dependencies permit it.
- [ ] Combined source coverage is >=90%.
- [ ] Final `quality-gate` passes.

---

# Part II — Steps 2–4: Regime Modeling, Selection and Allocation

This part covers the **complete implementation of Steps 2, 3 and 4**. Source-code work is parallelized wherever file ownership permits; edits to the single canonical notebook are intentionally serialized to avoid merge conflicts and stale notebook outputs.

Both assignment model families are implemented:

- discrete Markov chains with 2 and 3 quantile-defined states;
- Gaussian Hidden Markov Models with 2 and 3 states estimated by EM.

The notebook is the technical report; README is its technical sidecar; the standalone PDF uses the official template cover and is the non-technical sidecar required by the rubric.

## Steps 2–4 non-negotiable rules

1. Every task has a task ID `Txx.n`.
2. Every task has exactly one matching acceptance criterion `ACxx.n`.
3. No task is considered complete because a nearby acceptance criterion "implicitly" covers it; task/acceptance numbering is one-to-one.
4. Agents modify only files listed under **Files owned** for their PR.
5. Agents do not implement work assigned to a later PR.
6. Agents do not rename specified functions, columns, paths, JSON keys, state-ordering rules or model-selection rules.
7. Agents stop if a listed dependency is not merged to `main`.
8. All source changes have deterministic offline tests.
9. Every PR passes the repository `quality-gate`; combined source coverage remains >=90%.
10. Lint, type-check, unit-test and integration-test CI jobs remain independent and parallel.
11. Numerical outputs are never fabricated. Narrative claims are written only after the corresponding numerical outputs exist.
12. Assignment questions are not copied verbatim into notebook, README or report; use step/question numbers and original section titles instead.
13. In-text citations and bibliography use MLA format and refer only to sources actually consulted.
14. Before any equation containing a Greek letter, list every Greek letter used and its pronunciation.

---

## Artifact roles and rubric contract

### Canonical technical notebook

Canonical path:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

The notebook is the primary technical analysis artifact. It must contain, for each step:

1. the step/question number;
2. executable code calling tested project functions;
3. stored function output;
4. equations and parameter definitions;
5. tables/plots;
6. precise interpretation and recommended action supported by the results.

The narrative may name the statistical models and estimation methods, but must explain theory rather than narrate Python library/function names. Library names may appear naturally in executable code or environment metadata, not as a substitute for methodological explanation.

### README technical sidecar

`README.md` must show the **same canonical technical results** as the executed notebook after PR-27:

- same equations;
- same candidate comparison values;
- same selected family/state count;
- same state statistics;
- same allocation mapping;
- same canonical figures.

README synchronization must read canonical files. It must never refit a model or recompute a second independent analysis.

### Standalone PDF non-technical sidecar

Fixed path:

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
```

The PDF must preserve **decision parity**, not duplicate every technical parameter table. It must contain the same decision-relevant numerical results and figures as the notebook, including:

- data/sample summary needed to understand the analysis;
- final regime interpretation;
- state-conditional ETF mean/std results;
- selected state count/result without unnecessary algorithm detail;
- state-to-allocation mapping;
- practical recommendation and limitations.

To comply with the rubric, its prose must avoid model/library/algorithm names and unnecessary estimation detail. It contains no source code. The technical transition matrices, EM mechanics and parameter-count derivations remain in the notebook/README.

The report must use page 1 of:

```text
reports/Template_Stochastic_Modeling_Group_Work_Project.pdf
```

as its cover and must never include template page 2, which contains instructions the template explicitly says to delete.

Unknown group number, country and email fields remain blank until supplied.

### Executed-notebook duplicate

The submission instructions require an executable notebook **and a duplicate version in PDF or HTML format**. For deterministic generation without a LaTeX dependency, this backlog creates:

```text
reports/gwp2_vix_regime_allocation.html
```

It must be exported from the committed executed notebook and contain its stored outputs. Step 5 must regenerate this HTML after the notebook is extended later.

### Parity levels

```text
Notebook <-> README: exact technical-result parity
Notebook <-> HTML: exact executed-notebook duplicate
Notebook <-> standalone PDF: decision-result parity, non-technical wording
```

---

## Fixed modeling contracts

### Step 1 input

All Step 2-4 work uses:

```text
data/processed/step1_data.csv
```

with exactly:

```text
Date
TLT
GLD
SPY
VIX
TLT_log_return
GLD_log_return
SPY_log_return
VIX_change
```

The modeling observation is exclusively:

```text
X_t = VIX_change_t
```

Do not switch to VIX level or percentage VIX return.

Before modeling, notebook code must verify unique sorted dates, zero missing/non-finite values and the exact required columns.

### Candidate state counts

Implement exactly 2 and 3 states for each family.

```python
SUPPORTED_STATE_COUNTS = (2, 3)
```

### Deterministic state ordering

Returned state identifiers are ordered by increasing state mean `VIX_change`.

```text
State 0 = lowest mean VIX_change
State 1 = next higher mean VIX_change
State 2 = highest mean VIX_change  # 3-state only
```

If two HMM component means are numerically equal, break the ordering tie by the component's original pre-relabel index. This makes relabeling deterministic.

Do not hard-code semantic labels such as `calm` or `stress`. Any label/interpretation must be justified from displayed state statistics.

### Discrete Markov quantile states

Quantile cut points are empirical quantiles of finite `VIX_change` values using NumPy's linear quantile method.

For 2 states:

```text
cut points = q(0.50)
```

For 3 states:

```text
cut points = q(1/3), q(2/3)
```

State assignment uses:

```python
numpy.searchsorted(cut_points, values, side="right")
```

Therefore a value exactly equal to a cut point belongs to the higher-numbered state.

The threshold table has exactly:

```text
state
lower_bound
upper_bound
```

with conceptual intervals `[lower_bound, upper_bound)`, except the final interval includes all values above its lower bound. Use `-inf` and `+inf` as outer bounds.

Required cut points must be strictly increasing. Duplicate cut points raise `ValueError`; never silently merge bins.

### Markov transition estimate

For observed transition counts `N_ij`:

```text
P_ij = N_ij / sum_j N_ij
```

No smoothing/pseudocounts.

Only consecutive rows of the already aligned Step 1 sequence are counted.

If an expected state has zero outgoing transitions, raise `ValueError`.

### Markov stationary distribution

Return a unique stationary row vector satisfying:

```text
pi @ P = pi
sum(pi) = 1
pi_i >= 0
```

Validation tolerance:

```text
1e-10
```

If the transition matrix does not admit a numerically unique stationary distribution at that tolerance, raise `ValueError`; do not return an arbitrary eigenvector from a non-unique stationary subspace.

### Markov conditional likelihood

Use the conditional likelihood of the observed discretized state transitions:

```text
log L_MC = sum_t log(P[state_t, state_(t+1)])
```

The first state probability is not modeled in this conditional likelihood.

For AIC/BIC:

```text
n_MC = number of transitions = len(states) - 1
k_MC = K * (K - 1)
```

The notebook must state this definition explicitly.

### Gaussian HMM fitting

Use `hmmlearn.hmm.GaussianHMM` with exactly:

```text
covariance_type = "diag"
n_iter = 500
tol = 1e-6
min_covar = 1e-6
restart seeds = [42, 43, 44, 45, 46]
```

Fit all five seeds for each state count.

Select the converged fit with greatest training log-likelihood. If several converged fits have log-likelihoods equal within `1e-12`, select the smallest restart seed.

If no restart converges, raise `RuntimeError`.

Relabel means, variances, start probabilities, transition rows/columns, Viterbi states and posterior columns using the fixed state ordering.

Posterior columns are exactly `state_0`, `state_1`, and `state_2` when applicable. Every posterior row must be finite and sum to 1 within `1e-8`.

### HMM likelihood and parameter count

For univariate diagonal Gaussian HMM with K states:

```text
k_HMM = (K - 1)            # initial probabilities
      + K*(K - 1)          # transition probabilities
      + K                  # means
      + K                  # variances
      = K^2 + 2*K - 1
```

Use:

```text
n_HMM = len(VIX_change)
```

### Information criteria

```text
AIC = 2*k - 2*log_likelihood
BIC = k*ln(n) - 2*log_likelihood
```

The four candidates appear together in one comparison table, but the notebook must state that their likelihoods are defined on different observation spaces:

- Markov: discretized state-transition sequence;
- HMM: continuous `VIX_change` observations.

Therefore AIC/BIC select **2 vs 3 states within each family**. The notebook/README must not claim that a raw cross-family difference in AIC/BIC establishes statistical superiority.

### Preferred-method rule

1. Inside the Markov family, select the lowest-BIC candidate.
2. Inside the HMM family, select the lowest-BIC converged candidate.
3. The selected HMM is valid only when all conditions hold:
   - convergence flag is true;
   - log-likelihood is finite;
   - means and variances are finite;
   - every variance is >0;
   - start probabilities are finite, nonnegative and sum to 1 within `1e-8`;
   - transition probabilities are finite/nonnegative and every row sums to 1 within `1e-8`;
   - posterior probabilities are finite and every row sums to 1 within `1e-8`;
   - every state has at least 5% Viterbi occupancy.
4. If the selected HMM is valid, use it as the preferred method because it models the continuous observation directly and provides posterior regime probabilities.
5. Otherwise use the selected Markov candidate.

This is an explicit project decision rule, **not** a claim that HMM wins a cross-family information-criterion test. The notebook must say so.

Preferred HMM state sequence = Viterbi sequence. Preferred Markov state sequence = quantile state sequence.

### Step 3 state-conditional ETF statistics

For the preferred state sequence and fixed asset order `TLT, GLD, SPY`, compute:

```text
mean daily log return
sample standard deviation of daily log return (ddof=1)
number of observations
```

Do not annualize these Step 3 statistics.

### Step 3 bar chart

Use a grouped bar chart:

- x-axis: state;
- grouped bars: TLT, GLD, SPY;
- bar height: mean daily log return;
- error bars: one state-conditional sample standard deviation;
- horizontal zero line;
- title, axis labels, scales/ticks and legend.

### Step 4 allocation

For each preferred state, choose the ETF with greatest state-conditional mean daily log return.

```text
winner weight = 1.0
other weights = 0.0
```

Exact tie-break order:

```text
TLT -> GLD -> SPY
```

Do not implement the optional 60/40 variant.

Do not implement Step 5 backtesting here. The notebook must note that Step 5 applies a one-trading-day lag to the state-driven position. It must also explicitly identify the in-sample allocation-map/lookahead limitation: state-conditional means are estimated using the analysis sample, so a truly out-of-sample trading evaluation would require a rolling/expanding estimation design.

---

## Canonical Step 2–4 output schemas

### Markov threshold CSVs

```text
state,lower_bound,upper_bound
```

```text
reports/tables/step2_markov_2_thresholds.csv
reports/tables/step2_markov_3_thresholds.csv
```

### Transition CSVs

All transition tables serialize as:

```text
from_state,state_0,state_1[,state_2]
```

```text
reports/tables/step2_markov_2_transition.csv
reports/tables/step2_markov_3_transition.csv
reports/tables/step2_hmm_2_transition.csv
reports/tables/step2_hmm_3_transition.csv
```

### Markov stationary CSVs

```text
state,stationary_probability
```

```text
reports/tables/step2_markov_2_stationary.csv
reports/tables/step2_markov_3_stationary.csv
```

### HMM parameter CSVs

```text
state
mean_vix_change
variance_vix_change
start_probability
viterbi_observations
viterbi_occupancy
posterior_mean_probability
```

```text
reports/tables/step2_hmm_2_parameters.csv
reports/tables/step2_hmm_3_parameters.csv
```

### Model comparison CSV

```text
family
n_states
log_likelihood
n_parameters
n_observations
aic
bic
converged
criterion_scope
```

Path: `reports/tables/step3_model_comparison.csv`.

`family` values are exactly `markov` and `hmm`. `criterion_scope` values are exactly `within_markov_family` and `within_hmm_family`.

### State-asset statistics CSV

```text
state
asset
mean_log_return
std_log_return
observations
```

Path: `reports/tables/step3_state_asset_statistics.csv`.

### Allocation CSV

```text
state
selected_asset
selection_mean_log_return
TLT_weight
GLD_weight
SPY_weight
```

Path: `reports/tables/step4_allocation_mapping.csv`.

### Figures

```text
reports/figures/step2_markov_vix_states.png
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
reports/figures/step3_state_asset_statistics.png
```

### Selected-model JSON

Path: `reports/generated/step3_selected_model.json`.

Exact keys:

```text
family
n_states
state_source
selection_reason
markov_best_n_states
hmm_best_n_states
input_data_sha256
```

`state_source` is exactly `quantile_states` or `viterbi_states`.

### Manifest JSON

Path: `reports/generated/steps_2_4_manifest.json`.

Exact top-level keys:

```text
schema_version
input_data_path
input_data_sha256
notebook_path
selected_model_path
tables
figures
```

Requirements:

- `schema_version` = `1`;
- paths are repository-relative POSIX strings;
- `tables` lists every canonical Step 2-4 CSV exactly once;
- `figures` lists every canonical Step 2-4 PNG exactly once;
- no timestamp is stored, so regenerating identical analysis does not create a meaningless manifest diff.

---

## PR-06 — Add Steps 2–4 dependencies

**Agent lane:** setup / sequential

**Dependencies:** repository scaffold

**Files owned:** `pyproject.toml`

### Tasks

- [ ] T06.1 Add `scipy>=1.13` and `hmmlearn>=0.3.3` as runtime dependencies.
- [ ] T06.2 Add `nbformat>=5.10`, `nbclient>=0.10` and `nbconvert>=7.16` for notebook validation/export.
- [ ] T06.3 Add `reportlab>=4.2`, `pypdf>=5.0` and `pymupdf>=1.24` for deterministic report generation/text inspection/render validation.

### Acceptance criteria

- [ ] AC06.1 (`T06.1`) both modeling dependencies appear exactly once in project dependencies.
- [ ] AC06.2 (`T06.2`) all three notebook dependencies appear exactly once.
- [ ] AC06.3 (`T06.3`) all three report dependencies appear exactly once and the existing development/quality dependencies remain intact.

---

## PR-07 — Add immutable model configuration

**Agent lane:** setup / sequential after PR-06

**Dependencies:** PR-06

**Files owned:**

```text
src/vix_regime_allocation/model_config.py
tests/test_model_config.py
```

### Tasks

- [ ] T07.1 Define `SUPPORTED_STATE_COUNTS = (2, 3)`.
- [ ] T07.2 Define restart seeds exactly `(42, 43, 44, 45, 46)`.
- [ ] T07.3 Define HMM settings exactly: diagonal covariance, 500 iterations, `1e-6` tolerance and `1e-6` minimum covariance.
- [ ] T07.4 Define numerical tolerances exactly: stationary `1e-10`, probability rows `1e-8`, restart-likelihood tie `1e-12`, minimum Viterbi occupancy `0.05`.
- [ ] T07.5 Add tests asserting every constant exactly.

### Acceptance criteria

- [ ] AC07.1 (`T07.1`) supported state counts equal `(2, 3)` exactly.
- [ ] AC07.2 (`T07.2`) restart seeds equal the five fixed seeds exactly and preserve order.
- [ ] AC07.3 (`T07.3`) every HMM setting equals the fixed contract.
- [ ] AC07.4 (`T07.4`) all four tolerances/thresholds equal the fixed values.
- [ ] AC07.5 (`T07.5`) tests fail if any fixed configuration value changes.

---

## PR-08 — Implement Markov quantile discretization

**Agent lane:** A

**Dependencies:** PR-07

**Files owned:**

```text
src/vix_regime_allocation/markov_states.py
tests/test_markov_states.py
```

### Public interface

```python
def discretize_vix_change(
    vix_change: pandas.Series,
    n_states: int,
) -> tuple[pandas.Series, pandas.DataFrame]:
    ...
```

### Tasks

- [ ] T08.1 Validate state count is exactly 2 or 3 and input is a non-empty finite Series with unique index.
- [ ] T08.2 Calculate exact linear-method empirical quantile cut points.
- [ ] T08.3 Reject non-strictly-increasing/duplicate required cut points.
- [ ] T08.4 Assign states with `searchsorted(..., side="right")` and preserve the input index.
- [ ] T08.5 Return state Series named `state` with integer values `0..K-1`.
- [ ] T08.6 Return exact threshold schema `state, lower_bound, upper_bound` using infinite outer bounds.
- [ ] T08.7 Add deterministic tests for 2 states, 3 states, exact-boundary membership, invalid input and duplicate cut points.

### Acceptance criteria

- [ ] AC08.1 (`T08.1`) unsupported counts, empty input, NaN/inf and duplicate index fail clearly.
- [ ] AC08.2 (`T08.2`) test cut points equal NumPy linear quantiles at 0.5 or 1/3 and 2/3.
- [ ] AC08.3 (`T08.3`) duplicate cut points raise `ValueError` and are never merged.
- [ ] AC08.4 (`T08.4`) values exactly on cut points enter the higher state and output index equals input index.
- [ ] AC08.5 (`T08.5`) state name/dtype/value set match the contract.
- [ ] AC08.6 (`T08.6`) threshold columns/order/bounds match the contract exactly.
- [ ] AC08.7 (`T08.7`) all focused tests pass offline.

---

## PR-09 — Implement deterministic Gaussian HMM fitter

**Agent lane:** B

**Dependencies:** PR-07

**Files owned:**

```text
src/vix_regime_allocation/hmm_model.py
tests/test_hmm_model.py
```

### Public result

```python
@dataclass(frozen=True)
class HMMFitResult:
    n_states: int
    selected_seed: int
    log_likelihood: float
    converged: bool
    means: numpy.ndarray
    variances: numpy.ndarray
    start_probabilities: numpy.ndarray
    transition_matrix: numpy.ndarray
    viterbi_states: pandas.Series
    posterior_probabilities: pandas.DataFrame
```

### Public interface

```python
def fit_gaussian_hmm(vix_change: pandas.Series, n_states: int) -> HMMFitResult:
    ...
```

### Tasks

- [ ] T09.1 Validate state count, non-empty finite observations and unique index.
- [ ] T09.2 Fit one univariate diagonal Gaussian HMM for each fixed restart seed with the exact fixed settings.
- [ ] T09.3 Discard non-converged fits and raise `RuntimeError` if none converge.
- [ ] T09.4 Select maximum-likelihood converged restart; break a `1e-12` likelihood tie by smallest seed.
- [ ] T09.5 Relabel states by increasing fitted mean, with original component index as deterministic equal-mean tie break.
- [ ] T09.6 Apply the same relabel permutation to every state-dependent parameter, transition row/column, Viterbi state and posterior column.
- [ ] T09.7 Return Viterbi states on original index named `state`.
- [ ] T09.8 Return finite posterior probabilities on original index with exact `state_i` columns and normalized rows.
- [ ] T09.9 Add deterministic mocked/synthetic tests for settings, restart selection, tie selection, relabel consistency, 2/3-state shapes and failure when no restart converges.

### Acceptance criteria

- [ ] AC09.1 (`T09.1`) invalid state count/empty/non-finite/duplicate-index inputs fail clearly.
- [ ] AC09.2 (`T09.2`) tests prove exactly five fixed seeds and all fixed HMM settings are used.
- [ ] AC09.3 (`T09.3`) non-converged fits cannot be selected and all-non-converged raises `RuntimeError`.
- [ ] AC09.4 (`T09.4`) selected seed is highest-likelihood converged fit or smallest seed under the fixed equality tolerance.
- [ ] AC09.5 (`T09.5`) relabeled means are nondecreasing and equal-mean ordering is deterministic.
- [ ] AC09.6 (`T09.6`) tests prove means/variances/start probabilities/transition axes/Viterbi/posteriors use one consistent permutation.
- [ ] AC09.7 (`T09.7`) Viterbi index/name/state values match the contract.
- [ ] AC09.8 (`T09.8`) posterior columns/index are exact and rows are finite/sum to 1 within `1e-8`.
- [ ] AC09.9 (`T09.9`) all fitter tests pass offline.

---

## PR-10 — Implement Markov transition and unique stationary distribution

**Agent lane:** A

**Dependencies:** PR-08

**Files owned:**

```text
src/vix_regime_allocation/markov_chain.py
tests/test_markov_chain.py
```

### Public interfaces

```python
def estimate_transition_matrix(states: pandas.Series, n_states: int) -> pandas.DataFrame:
    ...


def stationary_distribution(transition: pandas.DataFrame) -> pandas.Series:
    ...
```

### Tasks

- [ ] T10.1 Validate states/index/state count and count exactly consecutive `state_t -> state_(t+1)` transitions.
- [ ] T10.2 Row-normalize observed counts with no pseudocounts.
- [ ] T10.3 Raise `ValueError` when an expected state has no outgoing transitions.
- [ ] T10.4 Validate transition matrix finiteness, non-negativity and row sums before stationary calculation.
- [ ] T10.5 Solve and normalize the stationary distribution.
- [ ] T10.6 Reject a non-unique stationary solution and validate `pi @ P = pi` within `1e-10`.
- [ ] T10.7 Add hand-computable tests for counts, probabilities, stationary vector, zero-outgoing state and non-unique stationary case.

### Acceptance criteria

- [ ] AC10.1 (`T10.1`) transition counts match a manually enumerated path exactly.
- [ ] AC10.2 (`T10.2`) every returned transition row sums to 1 and unobserved cells remain zero.
- [ ] AC10.3 (`T10.3`) zero-outgoing expected state raises `ValueError`.
- [ ] AC10.4 (`T10.4`) malformed transition matrices fail before stationary solving.
- [ ] AC10.5 (`T10.5`) returned stationary probabilities are finite/nonnegative and sum to 1.
- [ ] AC10.6 (`T10.6`) stationary equation passes at `1e-10` and non-unique stationary subspace raises `ValueError`.
- [ ] AC10.7 (`T10.7`) all tests pass offline.

---

## PR-11 — Implement information-criterion helpers

**Agent lane:** B

**Dependencies:** PR-07

**Files owned:**

```text
src/vix_regime_allocation/information_criteria.py
tests/test_information_criteria.py
```

### Public interfaces

```python
def aic(log_likelihood: float, n_parameters: int) -> float: ...
def bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float: ...
def markov_parameter_count(n_states: int) -> int: ...
def gaussian_hmm_parameter_count(n_states: int) -> int: ...
```

### Tasks

- [ ] T11.1 Implement exact AIC formula and finite/positive-parameter validation.
- [ ] T11.2 Implement exact BIC formula and finite/positive parameter/observation validation.
- [ ] T11.3 Implement Markov parameter count `K*(K-1)` for supported K only.
- [ ] T11.4 Implement HMM parameter count `K^2 + 2*K - 1` for supported K only.
- [ ] T11.5 Add exact numerical tests for K=2 and K=3 plus invalid inputs.

### Acceptance criteria

- [ ] AC11.1 (`T11.1`) AIC equals hand calculations and invalid inputs fail clearly.
- [ ] AC11.2 (`T11.2`) BIC equals hand calculations and invalid inputs fail clearly.
- [ ] AC11.3 (`T11.3`) Markov counts are exactly 2 and 6 for K=2/3.
- [ ] AC11.4 (`T11.4`) HMM counts are exactly 7 and 14 for K=2/3.
- [ ] AC11.5 (`T11.5`) all criterion tests pass offline.

---

## PR-12 — Implement Markov candidate evaluation

**Agent lane:** A

**Dependencies:** PR-10, PR-11

**Files owned:**

```text
src/vix_regime_allocation/markov_evaluation.py
tests/test_markov_evaluation.py
```

### Public interfaces

```python
def markov_log_likelihood(states: pandas.Series, transition: pandas.DataFrame) -> float: ...

def evaluate_markov_candidate(vix_change: pandas.Series, n_states: int) -> dict[str, object]: ...
```

Exact candidate keys:

```text
family
n_states
log_likelihood
n_parameters
n_observations
aic
bic
converged
states
thresholds
transition
stationary
```

### Tasks

- [ ] T12.1 Implement the conditional state-transition log-likelihood exactly as fixed above.
- [ ] T12.2 Reject an externally supplied transition matrix that assigns zero/non-finite probability to an observed transition.
- [ ] T12.3 Build a complete candidate by delegating to PR-08/10/11 functions without duplicating their math.
- [ ] T12.4 Set `n_observations = len(states)-1`, `family="markov"` and `converged=True`.
- [ ] T12.5 Return exactly the fixed candidate keys.
- [ ] T12.6 Add hand-computable likelihood/AIC/BIC tests for both state counts.

### Acceptance criteria

- [ ] AC12.1 (`T12.1`) log-likelihood equals a manual sum of observed `log(P_ij)` terms and excludes an initial-state probability term.
- [ ] AC12.2 (`T12.2`) impossible/non-finite observed transition probabilities fail clearly.
- [ ] AC12.3 (`T12.3`) candidate construction calls shared discretization/transition/stationary/criterion functions rather than reimplementing them.
- [ ] AC12.4 (`T12.4`) family/convergence/observation count match the contract exactly.
- [ ] AC12.5 (`T12.5`) returned key set matches exactly.
- [ ] AC12.6 (`T12.6`) all evaluation tests pass offline.

---

## PR-13 — Implement HMM candidate evaluation

**Agent lane:** B

**Dependencies:** PR-09, PR-11

**Files owned:**

```text
src/vix_regime_allocation/hmm_evaluation.py
tests/test_hmm_evaluation.py
```

### Public interface

```python
def evaluate_hmm_candidate(vix_change: pandas.Series, n_states: int) -> dict[str, object]: ...
```

Exact candidate keys:

```text
family
n_states
selected_seed
log_likelihood
n_parameters
n_observations
aic
bic
converged
means
variances
start_probabilities
transition
states
posterior_probabilities
```

### Tasks

- [ ] T13.1 Call `fit_gaussian_hmm()` exactly once for one candidate evaluation.
- [ ] T13.2 Use shared HMM parameter count and `n_observations=len(vix_change)`.
- [ ] T13.3 Compute AIC/BIC with shared helpers from the fitted log-likelihood.
- [ ] T13.4 Map the `HMMFitResult` into exactly the fixed candidate keys without mutating fitted arrays/dataframes.
- [ ] T13.5 Add mocked deterministic tests for 2/3-state evaluation math and delegation.

### Acceptance criteria

- [ ] AC13.1 (`T13.1`) mock asserts exactly one fitter call per evaluation.
- [ ] AC13.2 (`T13.2`) parameter/observation counts are exact for K=2/3.
- [ ] AC13.3 (`T13.3`) AIC/BIC equal shared-helper results exactly within floating tolerance.
- [ ] AC13.4 (`T13.4`) returned key set and values map to the fit result without hidden refitting/relabeling.
- [ ] AC13.5 (`T13.5`) all evaluation tests pass offline.

---

## PR-14 — Implement Markov VIX-state figure

**Agent lane:** A

**Dependencies:** PR-12

**Files owned:**

```text
src/vix_regime_allocation/markov_plots.py
tests/test_markov_plots.py
```

### Public interface

```python
def plot_markov_vix_states(
    vix: pandas.Series,
    states_2: pandas.Series,
    states_3: pandas.Series,
    output_path: pathlib.Path,
) -> None: ...
```

### Tasks

- [ ] T14.1 Validate exact index equality between VIX and both supplied state series.
- [ ] T14.2 Build one two-panel figure: 2-state candidate and 3-state candidate.
- [ ] T14.3 Plot VIX **level** over time and color every observation by supplied state.
- [ ] T14.4 Add per-panel state-count titles, date/VIX axes, visible scales/ticks and complete state legends.
- [ ] T14.5 Create parent directories, save non-empty PNG and close the figure.
- [ ] T14.6 Add deterministic plotting tests.

### Acceptance criteria

- [ ] AC14.1 (`T14.1`) any index mismatch fails clearly.
- [ ] AC14.2 (`T14.2`) output has exactly two candidate panels.
- [ ] AC14.3 (`T14.3`) y-data are supplied VIX levels and colors derive only from supplied states.
- [ ] AC14.4 (`T14.4`) each panel has non-empty title/axes/scales/complete legend.
- [ ] AC14.5 (`T14.5`) output path exists/non-empty and no created figure remains open.
- [ ] AC14.6 (`T14.6`) tests pass offline.

---

## PR-15 — Implement HMM VIX-state figure

**Agent lane:** B

**Dependencies:** PR-13

**Files owned:**

```text
src/vix_regime_allocation/hmm_state_plot.py
tests/test_hmm_state_plot.py
```

### Public interface

```python
def plot_hmm_vix_states(
    vix: pandas.Series,
    states_2: pandas.Series,
    states_3: pandas.Series,
    output_path: pathlib.Path,
) -> None: ...
```

### Tasks

- [ ] T15.1 Validate exact index equality between VIX and both HMM state series.
- [ ] T15.2 Build one two-panel figure for 2-state and 3-state HMM candidates.
- [ ] T15.3 Plot VIX level and color observations only from supplied Viterbi states.
- [ ] T15.4 Add state-count titles, date/VIX axes, visible scales/ticks and complete legends.
- [ ] T15.5 Create directories, save non-empty PNG and close the figure.
- [ ] T15.6 Add deterministic synthetic tests.

### Acceptance criteria

- [ ] AC15.1 (`T15.1`) any index mismatch fails clearly.
- [ ] AC15.2 (`T15.2`) output has exactly two HMM candidate panels.
- [ ] AC15.3 (`T15.3`) y-data are VIX levels and state coloring matches supplied Viterbi states.
- [ ] AC15.4 (`T15.4`) every panel has required titles/axes/scales/legends.
- [ ] AC15.5 (`T15.5`) non-empty image is created and figure is closed.
- [ ] AC15.6 (`T15.6`) tests pass offline.

---

## PR-16 — Implement HMM smoothed-probability figure

**Agent lane:** B after PR-15

**Dependencies:** PR-13

**Files owned:**

```text
src/vix_regime_allocation/hmm_probability_plot.py
tests/test_hmm_probability_plot.py
```

### Public interface

```python
def plot_hmm_smoothed_probabilities(
    probabilities_2: pandas.DataFrame,
    probabilities_3: pandas.DataFrame,
    output_path: pathlib.Path,
) -> None: ...
```

### Tasks

- [ ] T16.1 Validate exact expected posterior column names, finite values and rows summing to 1 within `1e-8`.
- [ ] T16.2 Build one two-panel figure for 2-state and 3-state posterior probabilities over time.
- [ ] T16.3 Plot every posterior state column exactly once.
- [ ] T16.4 Set probability y-limits to `[0,1]` and add titles, date/probability axes, scales/ticks and complete legends.
- [ ] T16.5 Create directories, save non-empty PNG and close figure.
- [ ] T16.6 Add deterministic synthetic tests.

### Acceptance criteria

- [ ] AC16.1 (`T16.1`) malformed/non-normalized/non-finite posterior frames fail clearly.
- [ ] AC16.2 (`T16.2`) output contains exactly the two required state-count panels.
- [ ] AC16.3 (`T16.3`) no posterior state column is omitted or duplicated.
- [ ] AC16.4 (`T16.4`) both y-axes are bounded `[0,1]` and all labels/scales/legends are present.
- [ ] AC16.5 (`T16.5`) non-empty image is created and figure is closed.
- [ ] AC16.6 (`T16.6`) tests pass offline.

---

## PR-17 — Implement Step 3 model comparison and deterministic selection

**Agent lane:** A

**Dependencies:** PR-12, PR-13

**Files owned:**

```text
src/vix_regime_allocation/model_selection.py
tests/test_model_selection.py
```

### Public interfaces

```python
def build_model_comparison(
    markov_candidates: list[dict[str, object]],
    hmm_candidates: list[dict[str, object]],
) -> pandas.DataFrame: ...


def select_preferred_model(
    comparison: pandas.DataFrame,
    markov_candidates: list[dict[str, object]],
    hmm_candidates: list[dict[str, object]],
) -> dict[str, object]: ...
```

Exact selected-result keys:

```text
family
n_states
states
state_source
selection_reason
markov_best_n_states
hmm_best_n_states
```

### Tasks

- [ ] T17.1 Validate exactly one 2-state and one 3-state candidate exist for each family and build the fixed four-row comparison schema.
- [ ] T17.2 Assign exact family-specific `criterion_scope` values and do not implement cross-family AIC/BIC ranking.
- [ ] T17.3 Select minimum-BIC state count independently within Markov and HMM families with deterministic lower-state-count tie break when BIC is equal within `1e-12`.
- [ ] T17.4 Implement every HMM validity condition from the fixed preferred-method contract.
- [ ] T17.5 Select valid HMM or fallback Markov exactly according to the fixed preferred-method contract.
- [ ] T17.6 Return exact selected-result keys, correct `state_source`, exact selected state Series and non-empty deterministic selection reason.
- [ ] T17.7 Add tests for BIC choice/tie, valid HMM selection and each individual HMM fallback condition.

### Acceptance criteria

- [ ] AC17.1 (`T17.1`) malformed candidate sets fail; valid input yields exactly four rows and fixed columns.
- [ ] AC17.2 (`T17.2`) code contains no raw cross-family IC winner calculation and criterion scopes are exact.
- [ ] AC17.3 (`T17.3`) each family's state count follows minimum BIC and fixed tie rule.
- [ ] AC17.4 (`T17.4`) tests independently trigger every HMM validity condition.
- [ ] AC17.5 (`T17.5`) preferred family follows the fixed project rule in all tested branches.
- [ ] AC17.6 (`T17.6`) returned keys/state source/state sequence/reason match the contract.
- [ ] AC17.7 (`T17.7`) all selection tests pass offline.

---

## PR-18 — Implement preferred-state ETF statistics

**Agent lane:** A or B

**Dependencies:** Step 1 data contract; independent of PR-17 implementation

**Files owned:**

```text
src/vix_regime_allocation/state_statistics.py
tests/test_state_statistics.py
```

### Public interface

```python
def compute_state_asset_statistics(
    data: pandas.DataFrame,
    states: pandas.Series,
) -> pandas.DataFrame: ...
```

### Tasks

- [ ] T18.1 Validate required Step 1 return columns, exact index equality with states, no missing/non-finite values and at least two observations per state.
- [ ] T18.2 Compute mean daily log return for TLT, GLD and SPY by state.
- [ ] T18.3 Compute sample standard deviation with `ddof=1` for each state/asset.
- [ ] T18.4 Count observations for each state/asset.
- [ ] T18.5 Return exact tidy schema sorted by state then fixed asset order `TLT, GLD, SPY`.
- [ ] T18.6 Add hand-computable deterministic tests including index mismatch and insufficient-state-observation failure.

### Acceptance criteria

- [ ] AC18.1 (`T18.1`) malformed/misaligned/non-finite/too-small-state input fails clearly.
- [ ] AC18.2 (`T18.2`) every mean equals hand calculation.
- [ ] AC18.3 (`T18.3`) every std equals `ddof=1` hand calculation.
- [ ] AC18.4 (`T18.4`) observation counts are exact and equal across assets within each state for complete Step 1 data.
- [ ] AC18.5 (`T18.5`) schema and state/asset order match exactly.
- [ ] AC18.6 (`T18.6`) all statistics tests pass offline.

---

## PR-19 — Implement Step 3 grouped bar chart

**Agent lane:** A

**Dependencies:** PR-18

**Files owned:**

```text
src/vix_regime_allocation/state_statistics_plot.py
tests/test_state_statistics_plot.py
```

### Public interface

```python
def plot_state_asset_statistics(
    statistics: pandas.DataFrame,
    output_path: pathlib.Path,
) -> None: ...
```

### Tasks

- [ ] T19.1 Validate exact state-statistics schema and one row per state/asset.
- [ ] T19.2 Plot grouped state bars for TLT/GLD/SPY with mean log return as bar height.
- [ ] T19.3 Use state-conditional standard deviation as bar error bars and add horizontal zero line.
- [ ] T19.4 Add title, state x-axis, daily-log-return y-axis, visible scales/ticks and complete asset legend.
- [ ] T19.5 Create directories, save non-empty PNG and close the figure.
- [ ] T19.6 Add deterministic plotting tests.

### Acceptance criteria

- [ ] AC19.1 (`T19.1`) malformed/duplicate/incomplete statistics fail clearly.
- [ ] AC19.2 (`T19.2`) each state contains exactly three bars whose heights equal canonical means.
- [ ] AC19.3 (`T19.3`) error bars equal canonical standard deviations and zero line is present.
- [ ] AC19.4 (`T19.4`) title/axes/scales/legend are complete.
- [ ] AC19.5 (`T19.5`) non-empty image is created and figure is closed.
- [ ] AC19.6 (`T19.6`) tests pass offline.

---

## PR-20 — Implement Step 4 state-to-allocation mapping

**Agent lane:** B

**Dependencies:** PR-18

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

- [ ] T20.1 Validate every state has exactly one row for each fixed asset and finite mean returns.
- [ ] T20.2 Select the asset with maximum mean daily log return in every state.
- [ ] T20.3 Apply exact tie rule `TLT -> GLD -> SPY`.
- [ ] T20.4 Set selected weight to `1.0` and other weights to `0.0`.
- [ ] T20.5 Return exact allocation schema sorted by state.
- [ ] T20.6 Add deterministic tests for each possible winner, two-way ties and three-way tie.

### Acceptance criteria

- [ ] AC20.1 (`T20.1`) incomplete/duplicate/non-finite state-asset inputs fail clearly.
- [ ] AC20.2 (`T20.2`) selected asset equals maximum mean for every non-tied test state.
- [ ] AC20.3 (`T20.3`) every exact tie follows the fixed priority order.
- [ ] AC20.4 (`T20.4`) each row contains only 0/1 weights and sums exactly to 1.
- [ ] AC20.5 (`T20.5`) columns/order/state sorting match the canonical schema.
- [ ] AC20.6 (`T20.6`) all allocation tests pass offline.

---

## Canonical notebook integration PRs

Only one notebook PR may be open at a time. Each notebook PR starts from the executed notebook on current `main`, appends/updates only its assigned section, executes the full notebook from top to bottom and commits stored outputs.

### PR-21 — Add notebook Step 2 Markov analysis

**Agent lane:** A

**Dependencies:** completed Step 1 notebook/data, PR-12, PR-14

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/step2_markov_2_thresholds.csv
reports/tables/step2_markov_2_transition.csv
reports/tables/step2_markov_2_stationary.csv
reports/tables/step2_markov_3_thresholds.csv
reports/tables/step2_markov_3_transition.csv
reports/tables/step2_markov_3_stationary.csv
reports/figures/step2_markov_vix_states.png
```

### Tasks

- [ ] T21.1 Add `Step 2: Modeling VIX Regimes` and `2.1 Observation definition and assumptions` with `X_t = VIX_change_t` and precise definitions.
- [ ] T21.2 Add 2-state Markov subsection calling project functions and displaying thresholds, transition matrix, stationary distribution and likelihood/AIC/BIC.
- [ ] T21.3 Add 3-state Markov subsection with the same complete outputs.
- [ ] T21.4 Show transition-probability, stationary-distribution and conditional-likelihood equations; list/pronounce Greek letters before equations.
- [ ] T21.5 Explain quantile discretization, conditional-likelihood convention, state ordering and limitations without narrating library code.
- [ ] T21.6 Display and save the canonical two-panel Markov VIX-state figure with axes/scales/legend.
- [ ] T21.7 Serialize all six Markov CSVs using the exact canonical schemas.
- [ ] T21.8 Execute the complete notebook successfully and store all outputs.

### Acceptance criteria

- [ ] AC21.1 (`T21.1`) Step 2 observation is visibly/exclusively `VIX_change` and assumptions are explicit.
- [ ] AC21.2 (`T21.2`) all required MC2 numerical/function outputs are visible and internally consistent.
- [ ] AC21.3 (`T21.3`) all required MC3 numerical/function outputs are visible and internally consistent.
- [ ] AC21.4 (`T21.4`) equations match project definitions and Greek-letter pronunciation rule is satisfied.
- [ ] AC21.5 (`T21.5`) methodology/limitations are precise and contain no unsupported interpretation.
- [ ] AC21.6 (`T21.6`) canonical figure is visible in notebook and exists non-empty at the fixed path.
- [ ] AC21.7 (`T21.7`) all six Markov CSVs exist, are non-empty and match fixed schemas.
- [ ] AC21.8 (`T21.8`) notebook has no failed/unexecuted cell and stores outputs.

---

### PR-22 — Add notebook Step 2 HMM analysis

**Agent lane:** B

**Dependencies:** PR-21, PR-13, PR-15, PR-16

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/step2_hmm_2_parameters.csv
reports/tables/step2_hmm_2_transition.csv
reports/tables/step2_hmm_3_parameters.csv
reports/tables/step2_hmm_3_transition.csv
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
```

### Tasks

- [ ] T22.1 Add 2-state HMM subsection displaying selected restart seed, log-likelihood, fitted mean, variance, start probability, transition matrix, Viterbi occupancy and posterior summary.
- [ ] T22.2 Add 3-state HMM subsection with the same complete outputs.
- [ ] T22.3 Show Gaussian emission equation and HMM transition/initial-probability notation; list/pronounce every Greek letter before use.
- [ ] T22.4 Explain EM estimation, Viterbi decoding and smoothed/posterior probabilities precisely, while keeping library names out of explanatory prose.
- [ ] T22.5 Explain deterministic restarts/state relabeling and the role of convergence/occupancy diagnostics.
- [ ] T22.6 Display/save HMM state-colored VIX figure.
- [ ] T22.7 Display/save HMM smoothed-probability figure.
- [ ] T22.8 Serialize four HMM CSVs using the exact canonical schemas.
- [ ] T22.9 Execute the full notebook successfully and store all outputs.

### Acceptance criteria

- [ ] AC22.1 (`T22.1`) all fixed HMM2 parameters/diagnostics are visibly displayed and consistent with project output.
- [ ] AC22.2 (`T22.2`) all fixed HMM3 parameters/diagnostics are visibly displayed and consistent with project output.
- [ ] AC22.3 (`T22.3`) equations/notation are correct and every Greek symbol is introduced/pronounced first.
- [ ] AC22.4 (`T22.4`) explanation clearly distinguishes estimation, decoded most-likely states and smoothed probabilities.
- [ ] AC22.5 (`T22.5`) reproducibility/convergence/state-order rules are documented accurately.
- [ ] AC22.6 (`T22.6`) HMM state figure is visible and canonical file exists non-empty.
- [ ] AC22.7 (`T22.7`) posterior figure is visible and canonical file exists non-empty.
- [ ] AC22.8 (`T22.8`) all four HMM CSVs exist/non-empty and match exact schemas.
- [ ] AC22.9 (`T22.9`) full notebook executes without failed/unexecuted cells and stores outputs.

---

### PR-23 — Add notebook Step 3 model comparison and preferred-model decision

**Agent lane:** A

**Dependencies:** PR-22, PR-17

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/step3_model_comparison.csv
reports/generated/step3_selected_model.json
```

### Tasks

- [ ] T23.1 Add `Step 3: State Selection and Interpretation` with log-likelihood/AIC/BIC equations and explicit K=2/3 parameter counts.
- [ ] T23.2 Display one four-row candidate comparison table and save the canonical CSV.
- [ ] T23.3 Explain observation-space difference and why IC ranking is within-family only.
- [ ] T23.4 Display each within-family BIC winner including deterministic tie behavior if relevant.
- [ ] T23.5 Apply `select_preferred_model()` and display preferred family/state count/state source and exact selection reason.
- [ ] T23.6 Explain that HMM preference when valid is a project decision rule based on continuous-observation modeling/posterior information, not a cross-family IC proof.
- [ ] T23.7 Compute SHA-256 of the exact Step 1 CSV bytes and save selected-model JSON with exact required keys.
- [ ] T23.8 Execute the full notebook successfully and store outputs.

### Acceptance criteria

- [ ] AC23.1 (`T23.1`) formulas/parameter counts are visible, correct and satisfy Greek-symbol pronunciation rules.
- [ ] AC23.2 (`T23.2`) comparison table has exactly four rows/fixed columns and canonical CSV matches displayed values.
- [ ] AC23.3 (`T23.3`) cross-family IC non-comparability is explicit and no raw cross-family IC winner is claimed.
- [ ] AC23.4 (`T23.4`) displayed family winners match code selection exactly.
- [ ] AC23.5 (`T23.5`) displayed preferred result exactly matches selected-result function output.
- [ ] AC23.6 (`T23.6`) method-level rationale is scientifically qualified exactly as specified.
- [ ] AC23.7 (`T23.7`) JSON keys are exact and `input_data_sha256` equals the actual Step 1 file hash.
- [ ] AC23.8 (`T23.8`) full notebook executes without failed/unexecuted cells and stores outputs.

---

### PR-24 — Add notebook Step 3 state-conditional ETF analysis

**Agent lane:** B

**Dependencies:** PR-23, PR-18, PR-19

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/step3_state_asset_statistics.csv
reports/figures/step3_state_asset_statistics.png
```

### Tasks

- [ ] T24.1 Retrieve exactly the preferred state sequence from the selected candidate; do not recompute an alternative state sequence.
- [ ] T24.2 Display mean/std/count table for every preferred state x TLT/GLD/SPY and save canonical CSV.
- [ ] T24.3 Display/save the grouped mean-return bar chart with standard-deviation error bars.
- [ ] T24.4 Interpret every preferred state using displayed VIX-change/state evidence and ETF statistics; avoid unsupported causal labels.
- [ ] T24.5 Explain daily-log-return units, `ddof=1`, non-annualization and state-sample-size limitations.
- [ ] T24.6 Execute the full notebook successfully and store outputs.

### Acceptance criteria

- [ ] AC24.1 (`T24.1`) state Series identity/source matches `step3_selected_model.json` and selected candidate exactly.
- [ ] AC24.2 (`T24.2`) table contains every state x asset combination, canonical schema and values matching project function output.
- [ ] AC24.3 (`T24.3`) bar chart is visible, canonical file exists and bars/error bars match table means/stds.
- [ ] AC24.4 (`T24.4`) state interpretation is explicitly tied to displayed evidence and contains no unsupported causal claim.
- [ ] AC24.5 (`T24.5`) all four statistical interpretation points are explicit.
- [ ] AC24.6 (`T24.6`) full notebook executes without failed/unexecuted cells and stores outputs.

---

### PR-25 — Add notebook Step 4 allocation and finalize Steps 2–4 manifest

**Agent lane:** A

**Dependencies:** PR-24, PR-20

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/step4_allocation_mapping.csv
reports/generated/steps_2_4_manifest.json
```

### Tasks

- [ ] T25.1 Add `Step 4: Designing the Rotation Strategy` and show the state-wise `argmax` decision equation with all symbols defined/Greek pronunciations listed first.
- [ ] T25.2 Call `build_state_allocation()` on the canonical Step 3 statistics and display/save exact allocation mapping.
- [ ] T25.3 Justify each state choice by explicitly pointing to the corresponding maximum displayed historical mean return.
- [ ] T25.4 State the exact 100/0/0 rule and deterministic tie rule; state that the optional 60/40 rule is not used.
- [ ] T25.5 Explain the in-sample mapping/lookahead limitation and state that Step 5 must lag state-driven positions by one trading day.
- [ ] T25.6 Add concise economic interpretation/practical takeaways and identify relevant factors affecting SPY/TLT/GLD only when supported by authoritative sources actually cited.
- [ ] T25.7 Add/refresh MLA in-text citations and bibliography for all sources actually used in Steps 1-4; do not copy assignment prompt wording.
- [ ] T25.8 Create manifest with exact keys/schema, exact canonical paths and Step 1 SHA-256 matching selected-model JSON.
- [ ] T25.9 Execute the entire notebook from first cell to last cell, store outputs and verify all canonical Step 1-4 files remain consistent.

### Acceptance criteria

- [ ] AC25.1 (`T25.1`) decision equation is correct, symbols are defined and Greek pronunciation rule is satisfied.
- [ ] AC25.2 (`T25.2`) displayed allocation equals canonical CSV and has one row per selected state with row weights summing exactly to 1.
- [ ] AC25.3 (`T25.3`) every selection justification cites the actual highest displayed mean for that state.
- [ ] AC25.4 (`T25.4`) 100% rule/tie rule/no-60-40 decision are explicit and consistent with code.
- [ ] AC25.5 (`T25.5`) both in-sample-lookahead limitation and future one-day execution lag are explicit; no Step 5 backtest is implemented.
- [ ] AC25.6 (`T25.6`) practical takeaways/factor discussion are supported by cited authoritative sources and do not exceed observed evidence.
- [ ] AC25.7 (`T25.7`) notebook has MLA-formatted citations/bibliography and no fabricated source/copied assignment question.
- [ ] AC25.8 (`T25.8`) manifest keys/paths are exact, every canonical Step 2-4 artifact appears exactly once and input hashes agree.
- [ ] AC25.9 (`T25.9`) complete notebook contains no failed/unexecuted cell and all canonical outputs match the stored notebook analysis.

---

## README, report, HTML and parity PRs

### PR-26 — Implement deterministic README analysis synchronizer

**Agent lane:** A

**Dependencies:** PR-25

**Files owned:**

```text
scripts/sync_readme_analysis.py
tests/test_sync_readme_analysis.py
```

Fixed markers:

```text
<!-- BEGIN NOTEBOOK ANALYSIS OUTPUT -->
<!-- END NOTEBOOK ANALYSIS OUTPUT -->
```

### Tasks

- [ ] T26.1 Implement a synchronizer that reads only manifest/canonical CSV/JSON/PNG paths and never fits/recomputes a model.
- [ ] T26.2 Generate one Markdown block containing the same technical equations/cautions, comparison values, preferred result, state statistics, allocation mapping and all four figure links.
- [ ] T26.3 Apply Greek-letter name/pronunciation rule before generated equations.
- [ ] T26.4 Make replacement deterministic between exactly one pair of fixed markers and idempotent on repeated execution.
- [ ] T26.5 Add fixture-based offline tests for content parity, no estimation imports/calls, missing artifacts, marker errors and idempotence.

### Acceptance criteria

- [ ] AC26.1 (`T26.1`) synchronizer has no model-fitting path and fails if required canonical artifacts are missing.
- [ ] AC26.2 (`T26.2`) generated block contains every fixed technical result/figure category and values equal fixture canonical files.
- [ ] AC26.3 (`T26.3`) generated equations satisfy the Greek pronunciation rule.
- [ ] AC26.4 (`T26.4`) exactly one marker block is replaced and a second identical run creates no content change.
- [ ] AC26.5 (`T26.5`) all synchronizer tests pass offline.

---

### PR-27 — Synchronize README sidecar to executed notebook

**Agent lane:** A

**Dependencies:** PR-26

**Files owned:**

```text
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T27.1 Ensure README contains exactly one generated-analysis marker pair and run the synchronizer against canonical Step 2-4 artifacts.
- [ ] T27.2 Update repository status/commands/artifact paths so README describes actual implemented state through Step 4 and explicitly says Step 5 is not implemented.
- [ ] T27.3 Preserve quality-gate documentation and >=90% coverage contract.
- [ ] T27.4 Extend `check_readme_sidecar.py` to require the marker pair, Steps 2-4 manifest, selected-model JSON and executed notebook path without requiring artifacts that belong to later PRs.

### Acceptance criteria

- [ ] AC27.1 (`T27.1`) README has exactly one generated block and its technical numerical content/figure links equal canonical notebook artifacts.
- [ ] AC27.2 (`T27.2`) status/commands/paths are factually current and no Step 5 result is claimed.
- [ ] AC27.3 (`T27.3`) README still documents parallel lint/type/unit/integration, coverage and aggregate quality gate correctly.
- [ ] AC27.4 (`T27.4`) sidecar checker passes current repository and fails fixture mutations removing required markers/paths.

---

### PR-28 — Implement non-technical PDF report builder

**Agent lane:** B

**Dependencies:** PR-25

**Files owned:**

```text
scripts/build_pdf_report.py
tests/test_build_pdf_report.py
```

### Tasks

- [ ] T28.1 Implement builder that takes the populated template page 1 plus a Markdown/text report body and produces a PDF; never copy template page 2.
- [ ] T28.2 Implement canonical-artifact readers for decision-relevant tables/figures without model refitting/recalculation.
- [ ] T28.3 Enforce a report-body validation that rejects code fences/source-code listings and verifies required decision-result sections exist.
- [ ] T28.4 Implement PDF text inspection verifying the three team names and absence of distinctive instruction-page text.
- [ ] T28.5 Implement page rendering to PNG images with PyMuPDF for later visual QA.
- [ ] T28.6 Add fixture-based offline tests for cover page, page-2 exclusion, required sections, artifact embedding, non-empty output and renderability.

### Acceptance criteria

- [ ] AC28.1 (`T28.1`) generated fixture PDF begins with populated template page 1 and never contains template page 2.
- [ ] AC28.2 (`T28.2`) builder reads canonical artifact values/files and contains no model estimation path.
- [ ] AC28.3 (`T28.3`) source-code/code-fence body fails and a complete non-technical body passes.
- [ ] AC28.4 (`T28.4`) tests detect missing team names or leaked instruction-page text.
- [ ] AC28.5 (`T28.5`) every generated PDF page can be rendered to a non-empty image.
- [ ] AC28.6 (`T28.6`) all report-builder tests pass offline.

---

### PR-29 — Generate and visually verify Steps 1–4 non-technical PDF sidecar

**Agent lane:** B

**Dependencies:** PR-28

**Files owned:**

```text
reports/Stochastic_Modeling_GWP2_Report.md
reports/Stochastic_Modeling_GWP2_Report.pdf
reports/rendered/Stochastic_Modeling_GWP2_Report/*.png
```

### Tasks

- [ ] T29.1 Write original non-technical report prose covering results through Step 4 without copying assignment questions.
- [ ] T29.2 Avoid model names, algorithm names, library names and unnecessary parameter mechanics in report prose, as required by the rubric.
- [ ] T29.3 Include the same decision-relevant state statistics, selected regime result and allocation mapping as canonical notebook files.
- [ ] T29.4 Include canonical decision-relevant figures with axes/labels/scales and explanations of how to read them; do not split a chart/table across report pages.
- [ ] T29.5 Include clear recommendations, practical takeaways, limitations and factors affecting each portfolio, supported by MLA in-text citations/bibliography to sources actually consulted.
- [ ] T29.6 Generate fixed-path PDF using populated template page 1 and preserve the three team names/blank unknown fields.
- [ ] T29.7 Render every final PDF page to PNG and manually/visually verify no clipping, overlap, broken glyphs, blank figures, split tables/charts or instruction page.
- [ ] T29.8 State in the report source/README workflow that this report covers Steps 1-4 and must be regenerated after Step 5 before final submission.

### Acceptance criteria

- [ ] AC29.1 (`T29.1`) prose is original, covers Step 1-4 answers/results and contains no copied assignment-question text.
- [ ] AC29.2 (`T29.2`) non-technical prose contains no model/algorithm/library naming that violates the rubric.
- [ ] AC29.3 (`T29.3`) decision-result numbers/tables equal canonical notebook artifacts exactly.
- [ ] AC29.4 (`T29.4`) included figures are canonical, readable, explained, and no chart/table is split across pages.
- [ ] AC29.5 (`T29.5`) report contains recommendation/takeaways/limitations/portfolio factors plus MLA citations/bibliography with no fabricated source.
- [ ] AC29.6 (`T29.6`) PDF exists/non-empty, first page preserves populated cover and unknown fields remain blank.
- [ ] AC29.7 (`T29.7`) all rendered pages are inspected and free of the listed visual defects/instruction page.
- [ ] AC29.8 (`T29.8`) interim-through-Step4 status and required Step5 regeneration are explicit.

---

### PR-30 — Export executed notebook duplicate to HTML

**Agent lane:** A

**Dependencies:** PR-25

**Files owned:**

```text
scripts/export_notebook_html.py
tests/test_export_notebook_html.py
reports/gwp2_vix_regime_allocation.html
```

### Tasks

- [ ] T30.1 Implement deterministic HTML export from the committed notebook using its stored outputs; do not execute/refit during export.
- [ ] T30.2 Fail if the notebook contains a failed cell, unexecuted code cell or missing stored output expected by the Step 1-4 sections.
- [ ] T30.3 Export to exact fixed path `reports/gwp2_vix_regime_allocation.html`.
- [ ] T30.4 Validate HTML contains Step 1, Step 2, Step 3 and Step 4 headings plus references/embedded representations of canonical figures/tables.
- [ ] T30.5 Add fixture-based offline export/failure tests and generate the actual HTML.

### Acceptance criteria

- [ ] AC30.1 (`T30.1`) exporter reads the committed notebook and performs no notebook execution/model fitting.
- [ ] AC30.2 (`T30.2`) failed/unexecuted/missing-output fixture notebooks are rejected.
- [ ] AC30.3 (`T30.3`) fixed-path HTML exists and is non-empty.
- [ ] AC30.4 (`T30.4`) HTML visibly contains all four step headings and stored analysis outputs.
- [ ] AC30.5 (`T30.5`) tests pass offline and actual HTML is generated from current notebook.

---

### PR-31 — Implement artifact/sidecar parity checker

**Agent lane:** B

**Dependencies:** PR-27, PR-29, PR-30

**Files owned:**

```text
scripts/check_analysis_sidecars.py
tests/test_analysis_sidecars.py
```

### Tasks

- [ ] T31.1 Validate manifest schema, Step 1 SHA-256 and existence/non-emptiness of every listed canonical artifact.
- [ ] T31.2 Validate notebook references/displays every manifest table/figure and selected-model/allocation result.
- [ ] T31.3 Validate README generated block has exact technical parity for model comparison, preferred result, state statistics, allocation and all four figures.
- [ ] T31.4 Validate HTML is a duplicate export of current notebook by checking notebook content hash stored/embedded by exporter plus required step/output markers.
- [ ] T31.5 Validate standalone PDF has decision parity: selected state-count/result wording, every allocation row, state statistics and required decision figures; do not require technical model names/transition matrices in the non-technical report.
- [ ] T31.6 Add deterministic failure tests for stale/missing/hash-mismatched notebook, README, HTML, PDF, table, figure and input data.

### Acceptance criteria

- [ ] AC31.1 (`T31.1`) any manifest/schema/hash/missing/empty artifact defect causes checker failure.
- [ ] AC31.2 (`T31.2`) omission of a canonical technical output from notebook causes failure.
- [ ] AC31.3 (`T31.3`) any technical-result mismatch/omission in README generated block causes failure.
- [ ] AC31.4 (`T31.4`) stale HTML from a different notebook revision causes failure.
- [ ] AC31.5 (`T31.5`) stale/missing decision result in PDF causes failure without incorrectly requiring forbidden technical prose.
- [ ] AC31.6 (`T31.6`) all parity failure-mode tests pass offline.

---

### PR-32 — Add final Steps 1–4 sidecar CI gate

**Agent lane:** A

**Dependencies:** PR-31

**Files owned:**

```text
.github/workflows/quality-gates.yml
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T32.1 Add independent CI job named exactly `analysis-sidecars` that runs `scripts/check_analysis_sidecars.py`.
- [ ] T32.2 Make aggregate `quality-gate` depend on successful `analysis-sidecars`.
- [ ] T32.3 Preserve independent parallel lint, type-check, unit-test and integration-test jobs.
- [ ] T32.4 Preserve combined source coverage threshold >=90%.
- [ ] T32.5 Update README quality/status text to document notebook/README/HTML/PDF parity levels and current Steps 1-4 completion state.
- [ ] T32.6 Extend README checker so workflow must contain `analysis-sidecars` and README must reference the notebook HTML/report/manifest/parity policy.

### Acceptance criteria

- [ ] AC32.1 (`T32.1`) workflow contains an independent job key/name `analysis-sidecars` executing the checker.
- [ ] AC32.2 (`T32.2`) `quality-gate` cannot succeed when `analysis-sidecars` fails/skips.
- [ ] AC32.3 (`T32.3`) lint/type/unit/integration jobs have no dependency on one another and remain parallel-start capable.
- [ ] AC32.4 (`T32.4`) `fail_under=90` and workflow `--fail-under=90` remain unchanged.
- [ ] AC32.5 (`T32.5`) README accurately documents parity/status with no Step 5 result claim.
- [ ] AC32.6 (`T32.6`) README checker verifies the new workflow/path contracts and passes on current repository.

---

## Steps 2–4 parallel execution schedule

Exactly two weak agents are assumed.

```text
Wave 0 - sequential foundation
PR-06 dependencies
PR-07 fixed configuration

Wave 1 - parallel
Agent A: PR-08 Markov discretization
Agent B: PR-09 HMM fitter

Wave 2 - parallel
Agent A: PR-10 Markov transition/stationary
Agent B: PR-11 information criteria

Wave 3 - parallel
Agent A: PR-12 Markov evaluation
Agent B: PR-13 HMM evaluation

Wave 4 - parallel
Agent A: PR-14 Markov state figure
Agent B: PR-15 HMM state figure

Wave 5 - parallel
Agent A: PR-17 model comparison/selection
Agent B: PR-16 HMM posterior figure

Wave 6
Either agent: PR-18 state statistics

Wave 7 - parallel after PR-18
Agent A: PR-19 state-statistics bar chart
Agent B: PR-20 allocation mapping

Notebook waves - serialized because one file is canonical
Wave 8  Agent A: PR-21 notebook Markov
Wave 9  Agent B: PR-22 notebook HMM
Wave 10 Agent A: PR-23 notebook selection
Wave 11 Agent B: PR-24 notebook state statistics
Wave 12 Agent A: PR-25 notebook allocation + manifest + full execution

Sidecar tooling - parallel
Wave 13
Agent A: PR-26 README synchronizer
Agent B: PR-28 PDF builder

Sidecar generation - parallel
Wave 14
Agent A: PR-27 README synchronization
Agent B: PR-29 PDF generation/visual QA

Wave 15
Agent A: PR-30 notebook HTML export

Wave 16
Agent B: PR-31 parity checker

Wave 17
Agent A: PR-32 CI integration/final sidecar documentation
```

No notebook PR may be parallelized with another notebook PR.

---

# Global merge rules

For every PR from PR-01 through PR-47:

1. start from current `main` after dependencies are merged;
2. change only PR-owned files;
3. run the complete repository quality suite;
4. verify every `Txx.n` has its matching `ACxx.n` and both are satisfied;
5. verify no later-step feature was added;
6. update from current `main` before final validation;
7. merge only after `quality-gate` succeeds;
8. delete the feature branch after merge.

If GitHub `main` branch protection is not enabled, the repository owner must still apply these merge rules manually. The workflow itself cannot prevent a privileged direct push without a branch/ruleset configuration.

---

# Steps 1–4 Definition of Done

Steps 1–4 are complete only when all conditions are true:

- [ ] PR-01 through PR-32 are merged to `main`.
- [ ] Step 1 clean data and both exploratory figures are complete and reproducible.
- [ ] Both 2-state and 3-state discrete Markov candidates are implemented from `VIX_change` quantiles.
- [ ] Both Markov transition matrices and unique stationary distributions are computed/displayed.
- [ ] Both 2-state and 3-state Gaussian HMMs are fit with deterministic restarts/relabeling.
- [ ] HMM fitted parameters, Viterbi states and smoothed probabilities are displayed.
- [ ] VIX-level color-coded state figures exist for both model families.
- [ ] HMM posterior-probability figure exists.
- [ ] Log-likelihood/AIC/BIC are computed for all four candidates with explicit parameter counts/observation definitions.
- [ ] AIC/BIC state-count selection occurs within family only and the methodological reason is stated.
- [ ] One preferred family/state count/state sequence is selected by the fixed project rule.
- [ ] Mean/std/count of ETF log returns are computed for every preferred state and TLT/GLD/SPY.
- [ ] Required grouped bar chart with std error bars exists.
- [ ] One 100%-allocation ETF is selected for every preferred state with deterministic ties.
- [ ] Step 4 mapping and economic justification are visible.
- [ ] In-sample mapping/lookahead limitation and future Step 5 one-day execution lag are explicit.
- [ ] Canonical notebook is fully executed, stores outputs and contains technical equations, function output, parameters, plots, interpretation, limitations, citations and MLA bibliography.
- [ ] README has exact technical-result parity with the notebook canonical outputs.
- [ ] `reports/gwp2_vix_regime_allocation.html` duplicates the executed notebook through Step 4 and is marked for regeneration after Step 5.
- [ ] `reports/Stochastic_Modeling_GWP2_Report.pdf` uses populated template page 1, excludes instruction page 2, contains no code and has non-technical decision parity with notebook results.
- [ ] Standalone report avoids model/algorithm/library names in non-technical prose, includes recommendation/portfolio factors and MLA citations.
- [ ] Report pages have been rendered/visually checked; graphs/tables are not split and all axes/labels/scales are readable.
- [ ] Manifest and selected-model JSON hashes/paths are valid.
- [ ] `analysis-sidecars` parity CI passes.
- [ ] Combined source coverage remains >=90%.
- [ ] Lint/type/unit/integration remain parallel.
- [ ] Aggregate `quality-gate` passes on final `main`.
- [ ] Step 5 is not implemented by PR-01 through PR-32; complete Step 5 implementation is defined in Part III (PR-33 through PR-47).

---

# Part III — Step 5: Backtesting and Evaluation

Step 5 completes the assignment by backtesting the Step 4 regime-rotation rule over the common sample with a **one-trading-day execution lag**, computing the five required performance metrics, comparing against the two required benchmarks, plotting cumulative performance, summarizing results in a table, and discussing sensitivity. The required benchmarks are **equal-weight TLT/GLD/SPY with monthly rebalancing** and **buy-and-hold SPY**.

The Step 5 implementation does **not** redefine Steps 3 or 4. It consumes the preferred Step 3 state sequence and the Step 4 state-to-allocation rule. Any Step 5 function that needs states or an allocation mapping must receive those objects explicitly or reconstruct them through the already-fixed deterministic Step 2–4 project functions; it must not invent a second model-selection or allocation rule.

## Step 5 non-negotiable rules

1. PR-33 through PR-47 continue the one-task/one-acceptance numbering rule: every `Txx.n` has exactly one matching `ACxx.n`.
2. Every PR modifies only its **Files owned** list.
3. Source tests are deterministic and offline; no Step 5 unit/integration test may download market data.
4. Combined source coverage remains `>=90%`.
5. Lint, type-check, unit-test and integration-test CI jobs remain independent and parallel.
6. The canonical notebook remains `notebooks/gwp2_vix_regime_allocation.ipynb`; only one notebook PR may be open at a time.
7. README, notebook HTML and standalone PDF are regenerated after Step 5; no sidecar may independently refit a model or recompute a second analysis.
8. No transaction cost, tax, slippage, leverage or financing assumption may be silently added. The required assignment backtest is gross of such costs; their omission is stated as a limitation.
9. All three portfolios are evaluated on the **same final Step 5 comparison dates** so metric differences cannot be caused by different sample lengths.
10. All wealth/performance calculations use **simple returns**, derived deterministically from the Step 1 ETF log returns.
11. The risk-free rate for the required Sharpe ratio is fixed at `0.0` because the assignment does not provide a risk-free series; this assumption must be stated explicitly in notebook/README/report.
12. Step 5 sensitivity uses **number of states** rather than a new sample-window design, satisfying the assignment's “number of states or sample window” requirement without introducing a second estimation-window methodology.
13. No result may be described as out-of-sample. The one-day lag removes same-day signal execution, but Step 4 state-conditional mean mappings are still estimated in-sample unless a later, separately scoped walk-forward study is added.

---

## Fixed Step 5 mathematical and data contracts

### Input data

Step 5 uses the exact Step 1 file:

```text
data/processed/step1_data.csv
```

with the fixed columns defined in Part I. It also consumes the preferred model/state decision from Step 3 and the canonical Step 4 mapping:

```text
reports/generated/step3_selected_model.json
reports/tables/step4_allocation_mapping.csv
```

The executable notebook may reconstruct the preferred state Series by rerunning the deterministic project model functions on the unchanged Step 1 data. Sidecars may not refit models.

### Simple-return conversion

If `ell_i,t` is the Step 1 log return for ETF `i`, the simple return used in Step 5 is:

```text
r_i,t = exp(ell_i,t) - 1
```

The code must never sum log returns and label the result a simple cumulative return.

### One-day execution lag

Let `S_t` be the state observed/decoded at trading date `t` and `w(S_t)` the Step 4 portfolio weights. The return earned on date `t` uses the **previous trading date's** state:

```text
r_rotation,t = w(S_(t-1))' r_t
```

Consequences:

- the first state date has no executable prior-state decision and is excluded from Step 5 comparison returns;
- the lag is by one **observed trading row**, not one calendar day;
- no backward fill is allowed;
- all benchmark returns are restricted to the resulting strategy return index before comparison.

### Rotation daily-return schema

`build_rotation_returns()` returns a DataFrame indexed by execution `Date` with columns exactly:

```text
decision_date
decision_state
selected_asset
TLT_weight
GLD_weight
SPY_weight
regime_rotation_return
```

`decision_date` is the immediately preceding Step 1 trading date. `decision_state` is the state on that date. Weights must match the corresponding Step 4 mapping row exactly.

### Equal-weight monthly benchmark

The required benchmark holds TLT, GLD and SPY with target weights `1/3` each and rebalances monthly.

Use this exact convention:

1. initialize target weights `1/3, 1/3, 1/3` immediately before the first comparison-date return;
2. on the first observed comparison trading date of each new calendar month, reset all three weights to `1/3` **before applying that date's return**;
3. on all other dates, allow weights to drift after returns according to portfolio arithmetic;
4. no intra-month rebalance is allowed;
5. no transaction costs are deducted.

After applying date `t` asset simple returns, drift weights for the next date as:

```text
w_i,t+ = w_i,t * (1 + r_i,t) / (1 + r_portfolio,t)
```

The benchmark return on date `t` is:

```text
r_equal_weight,t = sum_i w_i,t * r_i,t
```

### Buy-and-hold SPY benchmark

The SPY benchmark return is exactly the SPY adjusted-close simple return on each comparison date:

```text
r_spy,t = exp(SPY_log_return_t) - 1
```

No rebalance logic is applied to SPY.

### Common Step 5 comparison table

The exact daily comparison return columns are:

```text
regime_rotation
equal_weight_monthly
spy_buy_hold
```

All three columns use the same sorted unique `DatetimeIndex` named `Date`, contain no missing/non-finite values, and use exactly the rotation strategy's valid one-day-lag dates.

### Performance metrics

Use `TRADING_DAYS = 252` and simple daily returns.

For a return series `r_1, ..., r_N`, wealth is:

```text
W_t = product_(j=1..t) (1 + r_j)
```

Required metrics are fixed as:

```text
Cumulative Return     = W_N - 1
Annualized Return     = W_N^(252/N) - 1
Annualized Volatility = sample_std(r_t, ddof=1) * sqrt(252)
Sharpe Ratio          = mean(r_t) / sample_std(r_t, ddof=1) * sqrt(252)
Max Drawdown          = min_t(W_t / max_(u<=t) W_u - 1)
```

Sharpe uses zero risk-free return. `Max Drawdown` is stored as a non-positive decimal; `0.0` means no drawdown. Reject any daily simple return `<= -1` because wealth would become non-positive/invalid for these formulas.

### Performance-summary schema

Canonical table columns exactly:

```text
portfolio
cumulative_return
annualized_return
annualized_volatility
sharpe_ratio
max_drawdown
observations
```

Portfolio rows exactly and in this order:

```text
regime_rotation
equal_weight_monthly
spy_buy_hold
```

### Cumulative-performance figure

Plot cumulative return, not raw daily return:

```text
C_t = W_t - 1
```

One figure contains exactly the three required portfolios. It must have a non-empty title, date x-axis, cumulative-return y-axis, visible ticks/scales, zero reference line, and complete legend.

### Sensitivity contract

Sensitivity is evaluated for **2 versus 3 states within the Step 3 preferred model family**.

For each `K in {2, 3}`:

1. use that family's already-defined K-state candidate state sequence on the same Step 1 sample;
2. recompute state-conditional ETF statistics with the fixed Step 3 function;
3. rebuild the state-to-allocation mapping with the fixed Step 4 function;
4. backtest with the exact same one-trading-day lag;
5. compare the K-state rotation variant on the common sensitivity date intersection;
6. compute the same five performance metrics.

Do not switch model family in this sensitivity check. Do not introduce a new sample window. The sensitivity discussion must explicitly state that both mappings are in-sample and that this is a robustness comparison, not an out-of-sample test.

Sensitivity table columns exactly:

```text
family
n_states
cumulative_return
annualized_return
annualized_volatility
sharpe_ratio
max_drawdown
observations
```

Rows are ordered by `n_states` ascending.

---

## Canonical Step 5 artifacts

```text
reports/tables/step5_daily_returns.csv
reports/tables/step5_performance_summary.csv
reports/tables/step5_state_count_sensitivity.csv
reports/figures/step5_cumulative_performance.png
reports/generated/step5_manifest.json
```

`step5_daily_returns.csv` serializes `Date` plus the exact three common comparison return columns.

`step5_manifest.json` exact top-level keys:

```text
schema_version
input_data_path
input_data_sha256
selected_model_path
allocation_path
notebook_path
tables
figures
```

Requirements:

- `schema_version = 1`;
- `input_data_sha256` equals the Step 1 SHA-256 already recorded by Step 3;
- `tables` lists the three canonical Step 5 CSVs exactly once;
- `figures` lists the Step 5 cumulative-performance PNG exactly once;
- all paths are repository-relative POSIX paths;
- no timestamp is stored.

---

## PR-33 — Implement one-day-lag rotation backtest engine

**Agent lane:** A

**Dependencies:** PR-20 (Step 4 mapping contract)

**Files owned:**

```text
src/vix_regime_allocation/backtest.py
tests/test_backtest.py
```

### Public interface

```python
def build_rotation_returns(
    data: pandas.DataFrame,
    states: pandas.Series,
    allocation: pandas.DataFrame,
) -> pandas.DataFrame: ...
```

### Tasks

- [ ] T33.1 Validate Step 1 ETF log-return columns, exact data/state index equality, unique sorted dates, finite values, and complete allocation rows for every state used.
- [ ] T33.2 Convert current-date ETF log returns to simple returns with `exp(log_return)-1`.
- [ ] T33.3 Apply the Step 4 weights from state `t-1` to ETF returns at trading row `t`; do not use state `t` for return `t`.
- [ ] T33.4 Exclude exactly the first state/data row because it has no prior trading-state decision.
- [ ] T33.5 Return the exact rotation daily schema, including immediately preceding `decision_date`, `decision_state`, selected asset, weights, and `regime_rotation_return`.
- [ ] T33.6 Reject any missing allocation state, non-0/1 Step 4 weight, weight row not summing to 1, or simple ETF return `<=-1`.
- [ ] T33.7 Add hand-computable deterministic tests proving one-row lag, correct asset return, correct decision dates, and failure cases.

### Acceptance criteria

- [ ] AC33.1 (`T33.1`) malformed/misaligned/duplicate/non-finite/incomplete inputs fail clearly.
- [ ] AC33.2 (`T33.2`) synthetic ETF simple returns equal hand-computed `exp(ell)-1` values.
- [ ] AC33.3 (`T33.3`) a state change on date `t` affects date `t+1` return and never date `t` return.
- [ ] AC33.4 (`T33.4`) output has exactly `len(data)-1` rows for complete inputs and starts on the second trading row.
- [ ] AC33.5 (`T33.5`) columns/order/index/decision dates/state/selected asset/weights match the fixed schema exactly.
- [ ] AC33.6 (`T33.6`) every listed invalid allocation/return condition raises a clear error.
- [ ] AC33.7 (`T33.7`) all backtest tests pass offline.

---

## PR-34 — Implement required benchmark return engines

**Agent lane:** B

**Dependencies:** Step 1 clean-data contract only

**Files owned:**

```text
src/vix_regime_allocation/benchmarks.py
tests/test_benchmarks.py
```

### Public interfaces

```python
def build_equal_weight_monthly_returns(
    data: pandas.DataFrame,
    comparison_index: pandas.DatetimeIndex,
) -> pandas.Series: ...


def build_spy_buy_hold_returns(
    data: pandas.DataFrame,
    comparison_index: pandas.DatetimeIndex,
) -> pandas.Series: ...
```

Series names exactly:

```text
equal_weight_monthly
spy_buy_hold
```

### Tasks

- [ ] T34.1 Validate the comparison index is a non-empty sorted unique subset of the Step 1 dates and required ETF return columns are finite.
- [ ] T34.2 Convert ETF log returns to simple returns without using price-level shortcuts that change the sample.
- [ ] T34.3 Implement exact monthly `1/3` target-weight reset on the first observed comparison trading date of each calendar month before that date's return.
- [ ] T34.4 Implement exact intra-month weight drift with no additional rebalance.
- [ ] T34.5 Implement SPY buy-and-hold simple returns on exactly the supplied comparison index.
- [ ] T34.6 Return finite Series with the exact supplied index and fixed names; reject simple returns `<=-1`.
- [ ] T34.7 Add hand-computable tests spanning at least two months and verifying reset, drift, no intra-month reset, SPY equality and index failures.

### Acceptance criteria

- [ ] AC34.1 (`T34.1`) invalid/non-subset/duplicate/unsorted/empty comparison indexes and malformed data fail clearly.
- [ ] AC34.2 (`T34.2`) converted asset returns equal `exp(log_return)-1` hand calculations.
- [ ] AC34.3 (`T34.3`) first comparison day and first observed day of every new month use exactly `1/3` starting weights.
- [ ] AC34.4 (`T34.4`) intra-month benchmark returns/weights match hand-calculated drift and are not daily rebalanced.
- [ ] AC34.5 (`T34.5`) SPY benchmark equals SPY simple returns exactly on every comparison date.
- [ ] AC34.6 (`T34.6`) both output Series have exact names/index and finite valid returns.
- [ ] AC34.7 (`T34.7`) all benchmark tests pass offline.

---

## PR-35 — Implement required performance metrics

**Agent lane:** A

**Dependencies:** none; develop against the fixed Step 5 metric contract

**Files owned:**

```text
src/vix_regime_allocation/performance.py
tests/test_performance.py
```

### Public interfaces

```python
TRADING_DAYS = 252


def cumulative_wealth(returns: pandas.Series) -> pandas.Series: ...
def cumulative_return(returns: pandas.Series) -> float: ...
def annualized_return(returns: pandas.Series) -> float: ...
def annualized_volatility(returns: pandas.Series) -> float: ...
def sharpe_ratio(returns: pandas.Series) -> float: ...
def max_drawdown(returns: pandas.Series) -> float: ...
def performance_metrics(returns: pandas.Series) -> dict[str, float | int]: ...
```

### Tasks

- [ ] T35.1 Validate non-empty finite simple returns with unique sorted index, at least two observations for sample-volatility/Sharpe metrics, and every return `>-1`.
- [ ] T35.2 Implement cumulative wealth and cumulative return exactly from compounded simple returns.
- [ ] T35.3 Implement annualized return with `252/N` exponent exactly.
- [ ] T35.4 Implement annualized sample volatility with `ddof=1` and `sqrt(252)`.
- [ ] T35.5 Implement zero-risk-free annualized Sharpe using arithmetic mean daily simple return divided by sample daily standard deviation times `sqrt(252)`; reject zero volatility.
- [ ] T35.6 Implement max drawdown as the minimum wealth/running-peak minus one, returning a value in `[-1,0]`.
- [ ] T35.7 Return exact metric dictionary keys: `cumulative_return, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, observations`.
- [ ] T35.8 Add exact hand-calculated tests for all metrics and invalid/zero-volatility cases.

### Acceptance criteria

- [ ] AC35.1 (`T35.1`) every invalid-input condition fails clearly before metric calculation.
- [ ] AC35.2 (`T35.2`) wealth and cumulative return equal manual compound products.
- [ ] AC35.3 (`T35.3`) annualized return equals the exact fixed exponent formula.
- [ ] AC35.4 (`T35.4`) annualized volatility equals `ddof=1` manual standard deviation times `sqrt(252)`.
- [ ] AC35.5 (`T35.5`) Sharpe equals the zero-risk-free fixed formula and zero-volatility series fails clearly.
- [ ] AC35.6 (`T35.6`) max drawdown equals a manually identified peak-to-trough drawdown and is non-positive.
- [ ] AC35.7 (`T35.7`) metric keys/types and observation count match exactly.
- [ ] AC35.8 (`T35.8`) all performance tests pass offline.

---

## PR-36 — Assemble common strategy/benchmark returns and summary table

**Agent lane:** B

**Dependencies:** PR-33, PR-34, PR-35

**Files owned:**

```text
src/vix_regime_allocation/backtest_summary.py
tests/test_backtest_summary.py
```

### Public interfaces

```python
def build_comparison_returns(
    rotation: pandas.DataFrame,
    equal_weight: pandas.Series,
    spy: pandas.Series,
) -> pandas.DataFrame: ...


def build_performance_summary(comparison_returns: pandas.DataFrame) -> pandas.DataFrame: ...
```

### Tasks

- [ ] T36.1 Validate exact index equality among rotation/benchmark inputs; do not silently inner-join different date ranges.
- [ ] T36.2 Build exact daily comparison columns `regime_rotation, equal_weight_monthly, spy_buy_hold` with finite values.
- [ ] T36.3 Build one performance-summary row per fixed portfolio by delegating to `performance_metrics()`.
- [ ] T36.4 Return exact summary schema/order and fixed portfolio row order.
- [ ] T36.5 Add mocked/hand-computable tests proving no metric reimplementation and exact alignment behavior.

### Acceptance criteria

- [ ] AC36.1 (`T36.1`) any date mismatch fails instead of being silently dropped/aligned.
- [ ] AC36.2 (`T36.2`) comparison frame columns/order/index are exact and values equal supplied source returns.
- [ ] AC36.3 (`T36.3`) summary delegates each portfolio to the shared performance helper exactly once.
- [ ] AC36.4 (`T36.4`) summary columns/portfolio rows/order match the canonical contract exactly.
- [ ] AC36.5 (`T36.5`) all summary tests pass offline.

---

## PR-37 — Implement cumulative-performance figure

**Agent lane:** A

**Dependencies:** PR-35; develop against fixed comparison-return schema

**Files owned:**

```text
src/vix_regime_allocation/backtest_plot.py
tests/test_backtest_plot.py
```

### Public interface

```python
def plot_cumulative_performance(
    comparison_returns: pandas.DataFrame,
    output_path: pathlib.Path,
) -> None: ...
```

### Tasks

- [ ] T37.1 Validate exact three comparison columns, sorted unique Date index, finite values and returns `>-1`.
- [ ] T37.2 Compute each cumulative-return curve as `(1+r).cumprod()-1` using shared wealth semantics.
- [ ] T37.3 Plot exactly three curves with fixed readable portfolio labels.
- [ ] T37.4 Add title, date x-axis, cumulative-return y-axis, visible ticks/scales, zero reference line and complete legend.
- [ ] T37.5 Create parent directories, save non-empty PNG and close the figure.
- [ ] T37.6 Add deterministic plotting tests verifying curve data, labels and figure closure.

### Acceptance criteria

- [ ] AC37.1 (`T37.1`) malformed comparison input fails clearly.
- [ ] AC37.2 (`T37.2`) plotted y-values equal hand-compounded cumulative returns.
- [ ] AC37.3 (`T37.3`) exactly the three required portfolios appear and no extra performance series is plotted.
- [ ] AC37.4 (`T37.4`) title/axes/scales/zero line/legend are all present.
- [ ] AC37.5 (`T37.5`) canonical requested file can be created non-empty and no created figure remains open.
- [ ] AC37.6 (`T37.6`) all plot tests pass offline.

---

## PR-38 — Implement 2-versus-3-state sensitivity evaluator

**Agent lane:** B

**Dependencies:** PR-18, PR-20, PR-33, PR-35

**Files owned:**

```text
src/vix_regime_allocation/sensitivity.py
tests/test_sensitivity.py
```

### Public interface

```python
def evaluate_state_count_sensitivity(
    data: pandas.DataFrame,
    family: str,
    states_by_count: dict[int, pandas.Series],
) -> pandas.DataFrame: ...
```

### Tasks

- [ ] T38.1 Accept family exactly `markov` or `hmm` and require exactly state-count keys `{2,3}` with index-aligned finite state Series.
- [ ] T38.2 For each K, delegate to existing Step 3 state-statistics and Step 4 allocation functions; do not implement alternative mapping math.
- [ ] T38.3 Backtest each K variant using `build_rotation_returns()` with the fixed one-day lag.
- [ ] T38.4 Restrict both K variants to their exact common return-date intersection before metrics.
- [ ] T38.5 Compute all five required metrics for each K using shared performance helpers.
- [ ] T38.6 Return exact sensitivity schema, family values and ascending 2/3 row order.
- [ ] T38.7 Add deterministic tests for delegation, common-index handling, same-family enforcement and metric outputs.

### Acceptance criteria

- [ ] AC38.1 (`T38.1`) invalid family/key/index/state inputs fail clearly.
- [ ] AC38.2 (`T38.2`) mocks prove shared Step 3/4 functions are used exactly once per state count and no alternative winner rule exists.
- [ ] AC38.3 (`T38.3`) both candidate strategies use the shared lagged backtest engine.
- [ ] AC38.4 (`T38.4`) both rows are computed over the same sensitivity date intersection/observation count.
- [ ] AC38.5 (`T38.5`) metric values equal shared performance helper outputs.
- [ ] AC38.6 (`T38.6`) schema/family/state-count ordering are exact.
- [ ] AC38.7 (`T38.7`) all sensitivity tests pass offline.

---

## PR-39 — Add deterministic Step 5 integration test

**Agent lane:** A or B after source PRs merge

**Dependencies:** PR-33 through PR-38

**Files owned:**

```text
tests/integration/test_step5_pipeline.py
```

### Tasks

- [ ] T39.1 Build a small deterministic synthetic Step 1-style dataset spanning at least two calendar months.
- [ ] T39.2 Supply a deterministic state sequence and valid Step 4 mapping and run the complete rotation-return path.
- [ ] T39.3 Build both required benchmarks on the rotation index.
- [ ] T39.4 Build common comparison returns and the full performance summary.
- [ ] T39.5 Run 2/3-state sensitivity with deterministic synthetic state sequences.
- [ ] T39.6 Assert no network/model fit occurs and all canonical schemas/row counts/alignment contracts hold end-to-end.

### Acceptance criteria

- [ ] AC39.1 (`T39.1`) fixture spans multiple months and has hand-checkable finite ETF returns.
- [ ] AC39.2 (`T39.2`) integration output proves one-day lag and valid strategy returns.
- [ ] AC39.3 (`T39.3`) both benchmark Series cover exactly the strategy dates and monthly reset behavior is exercised.
- [ ] AC39.4 (`T39.4`) three-portfolio comparison/summary schemas and observation counts are exact.
- [ ] AC39.5 (`T39.5`) sensitivity produces exactly K=2/3 rows on a common date set.
- [ ] AC39.6 (`T39.6`) integration test passes offline with no Yahoo/model-estimation dependency.

---

# Step 5 canonical notebook PRs

Only one of PR-40, PR-41 and PR-42 may be open at a time. Each starts from the current executed notebook on `main`, executes the full notebook from top to bottom, and commits stored outputs.

## PR-40 — Add notebook Step 5 backtest and benchmark construction

**Agent lane:** A

**Dependencies:** PR-39 and PR-25 canonical Steps 1–4 notebook

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/step5_daily_returns.csv
```

### Tasks

- [ ] T40.1 Add `Step 5: Backtesting and Evaluation` and state the assignment objective in original wording without copying the prompt verbatim.
- [ ] T40.2 Load/validate the unchanged Step 1 data, reconstruct exactly the Step 3 preferred state sequence, and load the canonical Step 4 allocation mapping.
- [ ] T40.3 Show the log-to-simple-return equation with every symbol defined and any Greek symbol listed/pronounced first.
- [ ] T40.4 Show the one-trading-day execution-lag equation and explicitly demonstrate with at least the first five rows that decision state/date precede execution return date.
- [ ] T40.5 Build the regime-rotation return Series by calling the shared backtest function; do not duplicate backtest math in cells.
- [ ] T40.6 Build equal-weight monthly and SPY benchmarks on exactly the rotation return index and explain the monthly reset/drift convention.
- [ ] T40.7 Display first rows, date range, row count and missing/non-finite checks of the exact three-column comparison returns.
- [ ] T40.8 Save `step5_daily_returns.csv` with Date plus the exact three canonical comparison-return columns.
- [ ] T40.9 State explicitly that returns are gross of transaction costs/slippage/taxes and that the allocation mapping remains in-sample despite the one-day execution lag.
- [ ] T40.10 Execute the full notebook successfully and store outputs.

### Acceptance criteria

- [ ] AC40.1 (`T40.1`) notebook visibly identifies Step 5 and its evaluation purpose without copied question text.
- [ ] AC40.2 (`T40.2`) selected family/state count/state source match Step 3 JSON and allocation matches canonical Step 4 CSV.
- [ ] AC40.3 (`T40.3`) simple-return equation is correct and notation/pronunciation rules are satisfied.
- [ ] AC40.4 (`T40.4`) equation and displayed rows visibly prove state `t-1` drives execution at `t`.
- [ ] AC40.5 (`T40.5`) notebook calls shared rotation function and contains no second implementation.
- [ ] AC40.6 (`T40.6`) benchmark definitions exactly match the fixed monthly/SPY contracts and share the rotation index.
- [ ] AC40.7 (`T40.7`) displayed comparison data quality/date/row outputs are internally consistent and contain no missing/non-finite values.
- [ ] AC40.8 (`T40.8`) canonical daily CSV exists/non-empty and exactly matches displayed comparison returns.
- [ ] AC40.9 (`T40.9`) gross-cost and in-sample-mapping limitations are explicit.
- [ ] AC40.10 (`T40.10`) full notebook contains no failed/unexecuted cell and stores outputs.

---

## PR-41 — Add notebook Step 5 metrics, benchmark comparison and performance plot

**Agent lane:** B

**Dependencies:** PR-40, PR-36, PR-37

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/step5_performance_summary.csv
reports/figures/step5_cumulative_performance.png
```

### Tasks

- [ ] T41.1 Show equations for cumulative return, annualized return, annualized volatility, Sharpe and max drawdown; define all symbols and list/pronounce Greek letters first.
- [ ] T41.2 State `252` trading days, `ddof=1`, simple-return compounding, zero risk-free Sharpe and negative max-drawdown sign convention.
- [ ] T41.3 Build/display the exact three-row performance summary using shared functions and save canonical CSV.
- [ ] T41.4 Display/save the exact three-curve cumulative-performance figure.
- [ ] T41.5 Interpret the rotation strategy versus **both** required benchmarks using displayed cumulative return, annualized return, volatility, Sharpe and max drawdown; do not cherry-pick only favorable metrics.
- [ ] T41.6 State whether the displayed evidence supports or does not support an improvement in risk-adjusted performance, qualified as in-sample evidence.
- [ ] T41.7 Execute the full notebook successfully and store outputs.

### Acceptance criteria

- [ ] AC41.1 (`T41.1`) all five assignment metric equations are visible/correct and notation rules are satisfied.
- [ ] AC41.2 (`T41.2`) every fixed metric assumption/convention is explicit and consistent with source code.
- [ ] AC41.3 (`T41.3`) summary has exact schema/row order and values equal shared performance functions.
- [ ] AC41.4 (`T41.4`) figure exists/non-empty, is visible, and contains exactly the three compounded cumulative-return curves with required labels/scales.
- [ ] AC41.5 (`T41.5`) interpretation references all five metrics for the rotation and both benchmarks and matches displayed values.
- [ ] AC41.6 (`T41.6`) risk-adjusted conclusion is evidence-based and explicitly limited to the in-sample experiment.
- [ ] AC41.7 (`T41.7`) full notebook contains no failed/unexecuted cell and stores outputs.

---

## PR-42 — Add notebook Step 5 sensitivity, final interpretation and manifest

**Agent lane:** A

**Dependencies:** PR-41, PR-38

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/step5_state_count_sensitivity.csv
reports/generated/step5_manifest.json
```

### Tasks

- [ ] T42.1 Reconstruct 2- and 3-state candidate state sequences **within the Step 3 preferred family only** and verify both use the unchanged Step 1 input.
- [ ] T42.2 Run/display the canonical state-count sensitivity table by calling the shared sensitivity function and save CSV.
- [ ] T42.3 Briefly discuss whether performance conclusions are stable or materially change between 2 and 3 states, referencing the displayed five metrics.
- [ ] T42.4 State explicitly that state-count sensitivity is not an out-of-sample test and does not test alternative sample windows.
- [ ] T42.5 Add final Step 5 practical takeaway covering benchmark comparison, risk-adjusted evidence, limitations, and what would be required for a stronger out-of-sample validation.
- [ ] T42.6 Add/refresh authoritative MLA citations/bibliography only where external factual/financial claims are made; do not invent sources.
- [ ] T42.7 Create exact `step5_manifest.json`, including Step 1 SHA-256 matching Step 3 metadata and all Step 5 canonical artifacts exactly once.
- [ ] T42.8 Execute the entire notebook from Step 1 through Step 5 successfully, store outputs, and verify canonical Step 1–5 artifacts remain consistent.

### Acceptance criteria

- [ ] AC42.1 (`T42.1`) sensitivity uses exactly K=2/3 states from the preferred family and the same Step 1 data hash.
- [ ] AC42.2 (`T42.2`) displayed sensitivity table and canonical CSV have exact schema/values and same observation count for both K variants.
- [ ] AC42.3 (`T42.3`) discussion references all five metrics and accurately reflects displayed K=2/3 differences.
- [ ] AC42.4 (`T42.4`) robustness scope and non-out-of-sample limitation are explicit.
- [ ] AC42.5 (`T42.5`) final takeaway is supported by displayed results and identifies a concrete stronger-validation direction without claiming it was performed.
- [ ] AC42.6 (`T42.6`) citations/bibliography remain MLA-formatted and contain no fabricated source.
- [ ] AC42.7 (`T42.7`) manifest keys/paths/hash are exact and every canonical Step 5 table/figure appears exactly once.
- [ ] AC42.8 (`T42.8`) complete notebook has no failed/unexecuted cells and stored outputs/artifacts are mutually consistent.

---

# Step 5 sidecar and final submission PRs

## PR-43 — Extend and regenerate README technical sidecar through Step 5

**Agent lane:** B

**Dependencies:** PR-42, PR-27

**Files owned:**

```text
scripts/sync_readme_analysis.py
tests/test_sync_readme_analysis.py
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T43.1 Extend the synchronizer to read Step 5 manifest/daily returns/summary/sensitivity/figure without executing notebook or fitting a model.
- [ ] T43.2 Extend the generated README analysis block with Step 5 return/metric equations and the same fixed metric assumptions.
- [ ] T43.3 Render the exact Step 5 performance-summary and sensitivity values and link/embed the canonical cumulative-performance figure.
- [ ] T43.4 Add concise Step 5 benchmark/sensitivity interpretation matching the notebook; do not independently infer different conclusions.
- [ ] T43.5 Update README status/artifact map from “Step 5 planned” to actual Step 5 completion only after canonical artifacts exist.
- [ ] T43.6 Extend README checker to require Step 5 manifest/table/figure paths and a single synchronized analysis block.
- [ ] T43.7 Add/extend deterministic fixture tests for Step 5 parity, missing/stale artifact failure and idempotent README regeneration.

### Acceptance criteria

- [ ] AC43.1 (`T43.1`) synchronizer has no model/backtest recalculation path and fails when required Step 5 artifact is missing.
- [ ] AC43.2 (`T43.2`) README equations/assumptions exactly match notebook fixed definitions.
- [ ] AC43.3 (`T43.3`) README Step 5 numbers/figure path equal canonical files exactly.
- [ ] AC43.4 (`T43.4`) README conclusion/sensitivity wording is consistent with notebook evidence.
- [ ] AC43.5 (`T43.5`) README implementation/status/artifact sections are factually current through Step 5.
- [ ] AC43.6 (`T43.6`) checker fails if any required Step 5 path/block disappears and passes current valid repository.
- [ ] AC43.7 (`T43.7`) all README synchronization tests pass offline and repeated generation is byte-identical.

---

## PR-44 — Extend non-technical PDF builder validation for Step 5

**Agent lane:** A

**Dependencies:** PR-42, PR-28

**Files owned:**

```text
scripts/build_pdf_report.py
tests/test_build_pdf_report.py
```

### Tasks

- [ ] T44.1 Extend canonical-artifact readers to accept Step 5 summary/sensitivity/performance figure without model/backtest recomputation.
- [ ] T44.2 Require a non-technical Step 5 section containing benchmark comparison, risk-adjusted conclusion, sensitivity and limitations.
- [ ] T44.3 Validate that report Step 5 decision numbers equal canonical summary/sensitivity artifacts.
- [ ] T44.4 Preserve existing prohibitions on code, model/algorithm/library prose, template instruction page and missing team names.
- [ ] T44.5 Add fixture-based offline tests for Step 5 content parity and stale/missing Step 5 report data.

### Acceptance criteria

- [ ] AC44.1 (`T44.1`) builder reads Step 5 canonical files only and contains no estimation/backtest engine path.
- [ ] AC44.2 (`T44.2`) incomplete Step 5 non-technical report body fails validation.
- [ ] AC44.3 (`T44.3`) stale/mismatched performance or sensitivity values fail validation.
- [ ] AC44.4 (`T44.4`) all previous non-technical/template/code/team-name safeguards remain enforced.
- [ ] AC44.5 (`T44.5`) all extended report-builder tests pass offline.

---

## PR-45 — Regenerate and visually verify final Step 1–5 PDF report

**Agent lane:** A

**Dependencies:** PR-44, PR-29

**Files owned:**

```text
reports/Stochastic_Modeling_GWP2_Report.md
reports/Stochastic_Modeling_GWP2_Report.pdf
reports/rendered/Stochastic_Modeling_GWP2_Report/*.png
```

### Tasks

- [ ] T45.1 Extend original non-technical prose through Step 5 without copying assignment questions and without model/algorithm/library names.
- [ ] T45.2 Add the same decision-relevant Step 5 performance summary, state-count sensitivity result and cumulative-performance figure as the notebook.
- [ ] T45.3 Explain whether the rotation improves risk-adjusted performance relative to **both** benchmarks based on the canonical metrics.
- [ ] T45.4 Include practical recommendation, sensitivity conclusion, gross-cost assumption, in-sample limitation and factors affecting the three portfolios.
- [ ] T45.5 Preserve populated template page 1 as cover, exclude template page 2, preserve team names and leave unknown fields blank.
- [ ] T45.6 Refresh MLA citations/bibliography only from sources actually used.
- [ ] T45.7 Generate final PDF, render every page to PNG and visually inspect every page for clipping, overlap, broken glyphs, blank figures, split tables/charts and instruction-page leakage.

### Acceptance criteria

- [ ] AC45.1 (`T45.1`) report prose is original/non-technical and covers all Step 5 answer components without prohibited technical naming.
- [ ] AC45.2 (`T45.2`) report Step 5 numbers/figure are identical to canonical notebook artifacts.
- [ ] AC45.3 (`T45.3`) benchmark/risk-adjusted conclusion addresses both required benchmarks and matches displayed metrics.
- [ ] AC45.4 (`T45.4`) recommendation/sensitivity/cost/in-sample/portfolio-factor discussion is complete and evidence-based.
- [ ] AC45.5 (`T45.5`) final PDF first page is the populated cover, instruction page is absent, names remain present and unknown fields remain blank.
- [ ] AC45.6 (`T45.6`) MLA citations/bibliography contain no fabricated source.
- [ ] AC45.7 (`T45.7`) every rendered page is manually verified free of the listed defects and final PDF is non-empty.

---

## PR-46 — Regenerate executed-notebook HTML duplicate through Step 5

**Agent lane:** B

**Dependencies:** PR-42, PR-30

**Files owned:**

```text
reports/gwp2_vix_regime_allocation.html
```

### Tasks

- [ ] T46.1 Run the existing deterministic exporter against the committed fully executed Step 1–5 notebook without re-execution/refitting.
- [ ] T46.2 Verify the HTML includes Step 5 heading, stored daily-return preview, performance summary, cumulative figure, sensitivity table and final interpretation.
- [ ] T46.3 Ensure the HTML corresponds to the exact current notebook revision/hash expected by the exporter/parity checker.

### Acceptance criteria

- [ ] AC46.1 (`T46.1`) generated HTML is non-empty and produced without notebook execution/model fitting.
- [ ] AC46.2 (`T46.2`) all required Step 5 stored outputs/heading are visibly present in the HTML.
- [ ] AC46.3 (`T46.3`) HTML notebook hash/revision marker matches the current committed Step 1–5 notebook.

---

## PR-47 — Extend final sidecar parity CI through Step 5

**Agent lane:** B after all sidecars

**Dependencies:** PR-43, PR-45, PR-46, PR-31, PR-32

**Files owned:**

```text
scripts/check_analysis_sidecars.py
tests/test_analysis_sidecars.py
.github/workflows/quality-gates.yml
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T47.1 Extend parity checker to validate Step 5 manifest schema/hash and existence/non-emptiness of every canonical Step 5 artifact.
- [ ] T47.2 Validate notebook and README exact technical parity for Step 5 performance summary, sensitivity result and cumulative figure.
- [ ] T47.3 Validate HTML is the current executed notebook duplicate through Step 5.
- [ ] T47.4 Validate standalone PDF has non-technical decision parity for both benchmark comparisons, Step 5 summary metrics, sensitivity result and recommendation without requiring prohibited technical prose.
- [ ] T47.5 Add deterministic stale/missing/mismatched failure tests for each Step 5 table, figure, manifest, notebook, README, HTML and PDF path.
- [ ] T47.6 Keep CI job name exactly `analysis-sidecars`, make it run the extended checker, and preserve aggregate `quality-gate` dependency.
- [ ] T47.7 Preserve independent parallel lint/type/unit/integration jobs and the combined `>=90%` coverage threshold.
- [ ] T47.8 Update README final status/parity documentation to Steps 1–5 and extend README checker accordingly.

### Acceptance criteria

- [ ] AC47.1 (`T47.1`) any Step 5 manifest/hash/missing/empty canonical artifact defect causes parity failure.
- [ ] AC47.2 (`T47.2`) any Step 5 notebook/README technical-result or figure mismatch causes failure.
- [ ] AC47.3 (`T47.3`) stale HTML from a pre-Step-5 notebook revision causes failure.
- [ ] AC47.4 (`T47.4`) stale/missing Step 5 decision content in PDF causes failure while non-technical wording remains allowed/required.
- [ ] AC47.5 (`T47.5`) all enumerated Step 5 stale/missing/mismatch failure tests pass offline.
- [ ] AC47.6 (`T47.6`) existing `analysis-sidecars` and aggregate `quality-gate` wiring remain correct and Step 5-aware.
- [ ] AC47.7 (`T47.7`) lint/type/unit/integration remain parallel-start capable and both configuration/workflow still enforce 90% coverage.
- [ ] AC47.8 (`T47.8`) README accurately states final Steps 1–5 completion/parity and checker passes current repository.

---

## Step 5 parallel execution schedule

Exactly two weak agents are assumed.

```text
Wave 18 - parallel source foundation
Agent A: PR-33 rotation backtest
Agent B: PR-34 benchmarks

Wave 19 - parallel
Agent A: PR-35 performance metrics
Agent B: wait for PR-33/34

Wave 20 - parallel
Agent A: PR-37 cumulative plot
Agent B: PR-36 comparison/summary

Wave 21
Agent B: PR-38 state-count sensitivity
Agent A: wait

Wave 22
Either agent: PR-39 Step 5 integration test

Notebook waves - strictly serialized
Wave 23 Agent A: PR-40 notebook backtest/benchmarks
Wave 24 Agent B: PR-41 notebook metrics/plot
Wave 25 Agent A: PR-42 notebook sensitivity/final manifest

Sidecar waves
Wave 26 - parallel
Agent B: PR-43 README synchronization
Agent A: PR-44 PDF builder extension

Wave 27 - parallel
Agent A: PR-45 final PDF generation/visual QA
Agent B: PR-46 notebook HTML regeneration

Wave 28
Agent B: PR-47 final Steps 1–5 parity CI
```

No agent may start a PR before all listed dependencies are merged to `main`. Notebook PRs may never run in parallel with one another.

---

## Step 5 Definition of Done

Step 5 is complete only when all conditions are true:

- [ ] PR-33 through PR-47 are merged to `main`.
- [ ] Rotation returns apply the Step 4 allocation with exactly one observed-trading-day lag.
- [ ] Equal-weight benchmark is exactly 1/3 TLT, 1/3 GLD, 1/3 SPY with monthly reset and intra-month drift.
- [ ] SPY benchmark is adjusted-close buy-and-hold return.
- [ ] All three portfolios use identical Step 5 comparison dates.
- [ ] Cumulative Return, Annualized Return, Annualized Volatility, zero-risk-free Sharpe Ratio and Max Drawdown are computed from simple returns using the fixed formulas.
- [ ] Performance summary contains exactly regime rotation, monthly equal weight and SPY buy-and-hold.
- [ ] Cumulative-performance plot contains exactly the three required portfolios with professional labels/axes/scales.
- [ ] Sensitivity compares 2 versus 3 states within the preferred model family and uses the same sample/common sensitivity dates.
- [ ] Notebook Step 5 contains code calls, stored output, equations, assumptions, benchmark comparison, sensitivity, interpretation, limitations and practical takeaway.
- [ ] One-day lag is not misrepresented as a fully out-of-sample design; in-sample mapping limitation is explicit.
- [ ] No transaction cost/slippage/tax result is fabricated; gross-return assumption is explicit.
- [ ] `step5_daily_returns.csv`, `step5_performance_summary.csv`, `step5_state_count_sensitivity.csv`, cumulative figure and Step 5 manifest exist and are mutually consistent.
- [ ] README has exact technical parity with Step 5 notebook outputs.
- [ ] Notebook HTML is regenerated from the final executed Step 1–5 notebook.
- [ ] Standalone PDF is regenerated through Step 5, remains non-technical/no-code, uses template page 1 only and has decision-result parity.
- [ ] Step 5 sidecar parity is enforced by the existing `analysis-sidecars` CI job.
- [ ] Combined source coverage remains `>=90%`.
- [ ] Lint, type, unit and integration jobs remain parallel.
- [ ] Final aggregate `quality-gate` passes.
