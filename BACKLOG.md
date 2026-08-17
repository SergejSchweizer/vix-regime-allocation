# BACKLOG — Step 1: Data Preparation and Exploration

This backlog covers **only Step 1** of MScFE 622 Stochastic Modeling GWP2.

The implementation is intentionally split into small, deterministic PRs so that two weak coding agents can work in parallel with minimal ambiguity and minimal file overlap.

## Step 1 scope

The completed Step 1 must:

1. Download daily adjusted close prices for `TLT`, `GLD`, `SPY`, and `VIX` from Yahoo Finance.
2. Use the maximum common sample period for which all four series are available.
3. Compute daily log returns for `TLT`, `GLD`, and `SPY`.
4. Compute daily VIX change, using `ΔVIX = VIX_t - VIX_{t-1}`.
5. Align all derived series on common dates and remove missing values.
6. Produce one plot containing the three ETF return series over time.
7. Produce one plot containing `ΔVIX` over time.
8. Leave a clean, reproducible dataset that later steps can reuse.

Out of scope for Step 1: Markov chains, HMMs, regime estimation, allocation rules, backtesting, performance metrics, model selection, AIC/BIC, and report writing.

---

## Fixed technical contracts

These contracts are part of the backlog. Agents must not rename them without a separate PR.

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

### Raw data schema

The raw price table must:

- use a `DatetimeIndex` named `Date`;
- be sorted ascending by date;
- contain exactly these columns, in this order:

```text
TLT
GLD
SPY
VIX
```

Each value represents the adjusted daily close for that instrument.

### Clean Step 1 dataset schema

The final clean table must:

- use a `DatetimeIndex` named `Date`;
- be sorted ascending by date;
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

For each ETF `X ∈ {TLT, GLD, SPY}`:

```text
X_log_return_t = ln(X_t / X_{t-1})
```

For VIX:

```text
VIX_change_t = VIX_t - VIX_{t-1}
```

Do not compute percentage VIX returns in Step 1.

### Output paths

Use exactly:

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

The CSV is a reproducible project output and may be regenerated; it is not a hand-edited source file.

---

# PR backlog

## PR-01 — Add Yahoo Finance adjusted-close loader

**Goal:** Implement one small module that downloads and normalizes the four required price series.

**Agent lane:** A

**Dependencies:** none

**Files owned by this PR:**

```text
src/vix_regime_allocation/data.py
tests/test_data.py
```

Do not modify files owned by another PR.

### Tasks

- [ ] Create `src/vix_regime_allocation/data.py`.
- [ ] Add the fixed `TICKERS` mapping from this backlog.
- [ ] Add `download_adjusted_close() -> pandas.DataFrame`.
- [ ] Download daily Yahoo Finance data for all four tickers using the longest available history.
- [ ] Extract only adjusted close values.
- [ ] Rename `^VIX` to `VIX`.
- [ ] Return only `TLT`, `GLD`, `SPY`, `VIX` in that exact order.
- [ ] Normalize the index to a timezone-naive `DatetimeIndex` named `Date`.
- [ ] Sort rows ascending by date.
- [ ] Add focused tests for the returned schema and index properties without requiring a live Yahoo request.

### Acceptance criteria

- [ ] `download_adjusted_close()` exists in `src/vix_regime_allocation/data.py`.
- [ ] The function requests `TLT`, `GLD`, `SPY`, and `^VIX` from Yahoo Finance.
- [ ] The request uses the maximum available history rather than a hard-coded start date.
- [ ] The returned object is a `pandas.DataFrame`.
- [ ] Returned columns are exactly `['TLT', 'GLD', 'SPY', 'VIX']` and in that order.
- [ ] Returned values are adjusted closing prices, not unadjusted close prices.
- [ ] The index is a `DatetimeIndex` named `Date`.
- [ ] The index is timezone-naive.
- [ ] The index is sorted ascending.
- [ ] The function does not compute returns, remove common-date rows, create plots, or write output files.
- [ ] Tests use mocked/synthetic download output and pass without internet access.
- [ ] Every task checkbox in PR-01 is satisfied by the implementation.

---

## PR-02 — Add deterministic Step 1 data transformation

**Goal:** Convert a raw four-column price table into the final aligned Step 1 dataset.

**Agent lane:** B

**Dependencies:** none; develop against the fixed raw-data schema above.

**Files owned by this PR:**

```text
src/vix_regime_allocation/transform.py
tests/test_transform.py
```

Do not modify files owned by another PR.

### Tasks

- [ ] Create `src/vix_regime_allocation/transform.py`.
- [ ] Add `prepare_step1_data(prices: pandas.DataFrame) -> pandas.DataFrame`.
- [ ] Validate that input columns are exactly `TLT`, `GLD`, `SPY`, `VIX`.
- [ ] Restrict the data to dates on which all four raw price series are present.
- [ ] Compute `TLT_log_return` using `ln(TLT_t / TLT_{t-1})`.
- [ ] Compute `GLD_log_return` using `ln(GLD_t / GLD_{t-1})`.
- [ ] Compute `SPY_log_return` using `ln(SPY_t / SPY_{t-1})`.
- [ ] Compute `VIX_change` using `VIX_t - VIX_{t-1}`.
- [ ] Remove rows made incomplete by lagged calculations.
- [ ] Return the final columns in the fixed clean-dataset order.
- [ ] Add deterministic unit tests with a tiny synthetic price table whose expected values can be calculated exactly.

### Acceptance criteria

- [ ] `prepare_step1_data()` exists in `src/vix_regime_allocation/transform.py`.
- [ ] The function accepts a DataFrame matching the fixed raw-data schema.
- [ ] Invalid/missing raw columns raise a clear `ValueError`.
- [ ] A date with a missing value in any of `TLT`, `GLD`, `SPY`, `VIX` is absent from the common raw sample used for calculations.
- [ ] `TLT_log_return` equals `ln(TLT_t / TLT_{t-1})` for every returned row.
- [ ] `GLD_log_return` equals `ln(GLD_t / GLD_{t-1})` for every returned row.
- [ ] `SPY_log_return` equals `ln(SPY_t / SPY_{t-1})` for every returned row.
- [ ] `VIX_change` equals `VIX_t - VIX_{t-1}` for every returned row.
- [ ] No percentage/simple VIX return column is created.
- [ ] Returned columns are exactly `['TLT', 'GLD', 'SPY', 'VIX', 'TLT_log_return', 'GLD_log_return', 'SPY_log_return', 'VIX_change']` in that order.
- [ ] The returned index is named `Date` and sorted ascending.
- [ ] The returned DataFrame contains no missing values.
- [ ] Tests verify exact expected calculations from synthetic input and pass without internet access.
- [ ] The function does not download data, create plots, or write files.
- [ ] Every task checkbox in PR-02 is satisfied by the implementation.

---

## PR-03 — Add Step 1 plotting functions

**Goal:** Implement only the two plots explicitly required by Step 1.

**Agent lane:** A

**Dependencies:** none; develop against the fixed clean-dataset schema above.

**Files owned by this PR:**

```text
src/vix_regime_allocation/plots.py
tests/test_plots.py
```

Do not modify files owned by another PR.

### Tasks

- [ ] Create `src/vix_regime_allocation/plots.py`.
- [ ] Add `plot_etf_log_returns(data, output_path)`.
- [ ] Plot `TLT_log_return`, `GLD_log_return`, and `SPY_log_return` together in one figure.
- [ ] Add a descriptive title, x-axis label, y-axis label, and legend.
- [ ] Add `plot_vix_change(data, output_path)`.
- [ ] Plot `VIX_change` over time in one figure.
- [ ] Add a descriptive title, x-axis label, and y-axis label.
- [ ] Ensure parent directories for output paths are created if missing.
- [ ] Save figures to the supplied output path and close figures after saving.
- [ ] Add tests using synthetic clean data and temporary output paths.

### Acceptance criteria

- [ ] `plot_etf_log_returns()` exists and accepts a clean Step 1 DataFrame plus an output path.
- [ ] The ETF figure contains exactly the three required return series: TLT, GLD, and SPY.
- [ ] The ETF figure has a non-empty title.
- [ ] The ETF figure has non-empty x- and y-axis labels.
- [ ] The ETF figure has a legend identifying all three ETFs.
- [ ] `plot_vix_change()` exists and accepts a clean Step 1 DataFrame plus an output path.
- [ ] The VIX figure plots `VIX_change`, not the VIX level and not a percentage VIX return.
- [ ] The VIX figure has a non-empty title.
- [ ] The VIX figure has non-empty x- and y-axis labels.
- [ ] Both functions create missing parent directories.
- [ ] Both functions save a non-empty image file at the requested path.
- [ ] Both functions close their figure after saving.
- [ ] Tests pass with synthetic data and do not require internet access.
- [ ] The module does not download data, transform data, or write CSV files.
- [ ] Every task checkbox in PR-03 is satisfied by the implementation.

---

## PR-04 — Add executable Step 1 pipeline and outputs

**Goal:** Wire the three previously defined components together without adding new analytics.

**Agent lane:** B

**Dependencies:** PR-01, PR-02, PR-03

**Files owned by this PR:**

```text
scripts/run_step1.py
tests/test_run_step1.py
```

This PR may also generate, but must not hand-edit:

```text
data/processed/step1_data.csv
reports/figures/step1_etf_log_returns.png
reports/figures/step1_vix_change.png
```

### Tasks

- [ ] Create `scripts/run_step1.py`.
- [ ] Call `download_adjusted_close()`.
- [ ] Pass its output to `prepare_step1_data()`.
- [ ] Save the clean result to `data/processed/step1_data.csv` with the `Date` index included.
- [ ] Call `plot_etf_log_returns()` with `reports/figures/step1_etf_log_returns.png`.
- [ ] Call `plot_vix_change()` with `reports/figures/step1_vix_change.png`.
- [ ] Print the common-sample start date, end date, and final row count.
- [ ] Add a focused orchestration test using mocks so the test does not require internet access.

### Acceptance criteria

- [ ] Running `python scripts/run_step1.py` performs the complete Step 1 workflow from download to outputs.
- [ ] The script uses the three public functions created in PR-01, PR-02, and PR-03 rather than duplicating their logic.
- [ ] `data/processed/step1_data.csv` is created.
- [ ] The saved CSV contains the `Date` column/index plus exactly the eight fixed clean-data columns.
- [ ] The saved dataset contains no missing values.
- [ ] `reports/figures/step1_etf_log_returns.png` is created and non-empty.
- [ ] `reports/figures/step1_vix_change.png` is created and non-empty.
- [ ] The script prints a start date.
- [ ] The script prints an end date.
- [ ] The script prints the final number of observations.
- [ ] No Markov-chain, HMM, regime, allocation, or backtesting logic is added.
- [ ] The orchestration test passes without internet access.
- [ ] Every task checkbox in PR-04 is satisfied by the implementation.

---

## PR-05 — Add Step 1 notebook presentation

**Goal:** Present the already implemented Step 1 workflow in the executable notebook format required for the course submission.

**Agent lane:** A or B after PR-04

**Dependencies:** PR-04

**Files owned by this PR:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

### Tasks

- [ ] Create `notebooks/gwp2_vix_regime_allocation.ipynb`.
- [ ] Add a markdown heading `Step 1: Data Preparation and Exploration`.
- [ ] Load or generate the clean Step 1 dataset using project code rather than reimplementing calculations in notebook cells.
- [ ] Display the first rows of the clean dataset.
- [ ] Display the common-sample start date, end date, and row count.
- [ ] Display a missing-value check for the final dataset.
- [ ] Display the ETF log-return plot.
- [ ] Display the VIX-change plot.
- [ ] Add a short markdown interpretation confirming what was prepared and what the plots show at a descriptive level only.
- [ ] Run all Step 1 cells so outputs are stored in the notebook.

### Acceptance criteria

- [ ] The notebook contains a clearly labeled Step 1 section.
- [ ] The notebook uses the project implementation and does not contain a second independent implementation of download, return, alignment, or plotting logic.
- [ ] The notebook visibly shows the clean dataset preview.
- [ ] The notebook visibly shows the common-sample start date, end date, and number of rows.
- [ ] The notebook visibly demonstrates that the final Step 1 dataset has no missing values.
- [ ] The notebook visibly contains one plot with all three ETF log-return series.
- [ ] The notebook visibly contains one plot of `VIX_change`.
- [ ] Both displayed plots have titles, axis labels, and the ETF plot has a legend.
- [ ] Notebook outputs are stored; the Step 1 section is not left unexecuted.
- [ ] The interpretation does not claim regime results, strategy performance, or causal conclusions that belong to later steps.
- [ ] Every task checkbox in PR-05 is satisfied by the implementation.

---

# Parallel execution plan

To reduce merge conflicts, use the following schedule.

```text
Wave 1 — parallel
Agent A: PR-01 data loader
Agent B: PR-02 transformation

Wave 2 — after each agent finishes its Wave 1 PR
Agent A: PR-03 plotting
Agent B: wait for PR-01 and PR-03 interfaces to be merged; then PR-04 pipeline

Wave 3
Either agent: PR-05 notebook
```

PR-01, PR-02, and PR-03 are intentionally based on fixed interfaces in this backlog, so their implementations do not need to edit the same source files.

---

# Step 1 Definition of Done

Step 1 is complete only when **all** of the following are true:

- [ ] PR-01 through PR-05 are merged.
- [ ] Daily adjusted close prices are sourced from Yahoo Finance for TLT, GLD, SPY, and VIX.
- [ ] The final sample is restricted to common dates across all four raw series.
- [ ] ETF daily log returns are calculated for TLT, GLD, and SPY.
- [ ] Daily VIX change is calculated as `VIX_t - VIX_{t-1}`.
- [ ] The final clean dataset contains no missing values.
- [ ] The final clean dataset is saved at `data/processed/step1_data.csv`.
- [ ] One combined ETF-return plot exists at `reports/figures/step1_etf_log_returns.png`.
- [ ] One VIX-change plot exists at `reports/figures/step1_vix_change.png`.
- [ ] Both required plots are visible in the executable notebook.
- [ ] The notebook stores the executed Step 1 outputs.
- [ ] Automated tests for loader schema, transformation calculations, plotting outputs, and orchestration pass.
- [ ] No Step 2+ implementation is included.
