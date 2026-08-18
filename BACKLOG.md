# BACKLOG — Step 1: Data Preparation and Exploration

This backlog covers **only Step 1** of MScFE 622 Stochastic Modeling GWP2.

It has been audited against the assignment brief. The implementation is intentionally split into small PRs so that two weak coding agents can work with minimal ambiguity and minimal file overlap.

The assignment requires daily adjusted-close data for TLT, GLD, SPY and VIX, the maximum common sample, ETF daily log returns, a VIX change/return series, common-date alignment, missing-value removal, one ETF-return plot and one VIX-change plot.

---

# Non-negotiable backlog rules

1. Every task has a task ID `Txx.n`.
2. Every task has exactly one matching acceptance criterion `ACxx.n`.
3. An acceptance criterion may test several observable facts only when those facts belong to the single matching task.
4. Agents must modify only files listed under **Files owned** for their PR.
5. Agents must not implement work assigned to a later PR.
6. Tests must be deterministic and offline unless a PR explicitly states otherwise.
7. Every PR must pass the repository `quality-gate`, including the >=90% combined source-coverage requirement.
8. The README sidecar is updated only when a PR changes a user-facing/canonical contract. Internal source-file additions alone do not require README edits.
9. The canonical analysis artifact is `notebooks/gwp2_vix_regime_allocation.ipynb`; Step 1 creates its first section and later backlogs extend the same notebook.
10. No numerical result may be invented or copied from an external example.

---

# Fixed Step 1 contracts

## Tickers

Use exactly:

```python
TICKERS = {
    "TLT": "TLT",
    "GLD": "GLD",
    "SPY": "SPY",
    "VIX": "^VIX",
}
```

## Yahoo Finance download contract

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

## Raw data schema

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

## Maximum common sample definition

The common raw sample is the **intersection of dates on which all four adjusted-close values are present**.

Implementation rule:

```text
common_prices = raw_prices.dropna(subset=["TLT", "GLD", "SPY", "VIX"])
```

Do not forward-fill, backward-fill or interpolate any price.

After common-date restriction, the first retained date must be the earliest date with all four values available and the final retained date must be the latest retained common date.

## Clean Step 1 dataset schema

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

## Return definitions

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

## Fixed output paths

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

The CSV and figures are generated artifacts. They must never be hand-edited.

---

# PR backlog

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

---

# Parallel execution schedule

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

---

# Step 1 Definition of Done

Step 1 is complete only when all of the following are true:

- [ ] PR-01 through PR-05 are merged to `main`.
- [ ] Yahoo data request semantics are explicit and adjusted close is unambiguous.
- [ ] TLT, GLD, SPY and VIX use the maximum common date intersection without imputation.
- [ ] ETF daily log returns and daily VIX change are calculated on the common sample.
- [ ] The final clean dataset satisfies the exact schema and contains no missing/non-finite values.
- [ ] `data/processed/step1_data.csv` exists and is reproducible.
- [ ] Both required Step 1 figures exist, are non-empty and are visible in the notebook.
- [ ] The notebook Step 1 section contains code calls, stored outputs, equations, scientific interpretation, limitations, citations and MLA bibliography entries as applicable.
- [ ] README is synchronized with the Step 1 canonical artifacts.
- [ ] No Step 2+ implementation is included.
- [ ] Combined source coverage is >=90%.
- [ ] Final `quality-gate` passes.
