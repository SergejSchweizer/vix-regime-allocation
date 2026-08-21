# BACKLOG — HMM-only GWP2 rebuild with 100% Keep and 60/40 Spread

This is the canonical **revision backlog** for MScFE 622 Group Work Project #2. It supersedes the earlier PR-01..PR-49 implementation plan, which remains available in Git history only. The revision is intentionally split into small, deterministic PRs for **two weak coding agents working in parallel**.

The target end state is strict: **Gaussian Hidden Markov Models are the only regime model used anywhere in the active analysis**, both assignment allocation rules are evaluated (**100% Keep** and **60/40 Spread**), all numerical artifacts are recomputed from the canonical Step 1 dataset, the notebook is rebuilt and fully executed, and the PDF is regenerated only from that executed notebook plus the supplied template cover.

## Global PR contract

1. This revision contains exactly PR-50..PR-68. Every PR has one Agent lane, explicit backward-only dependencies, an exhaustive write/delete set, tasks, and one-to-one acceptance criteria.
2. Every implementation branch starts from current `main` after all declared dependencies are merged. Agents never branch from the incomplete `hmm-only-rebuild` branch or from open PR #70.
3. Parallel PRs must have disjoint write sets. Any PR that writes the notebook is serialized. Any PR that writes the generated PDF is serialized after the notebook.
4. An agent modifies only the paths listed under **Files owned**. If another write is required, stop and revise this backlog before coding.
5. Source changes must have deterministic offline tests. Numerical outputs are recomputed by project code; no result may be copied from an earlier report or hard-coded from a previous run.
6. Transitional Markov files may remain temporarily only to keep intermediate commits green, but after PR-50 no new selection, strategy, artifact, notebook section, report section, predictive run, or validation decision may use a Markov model. PR-67 removes the legacy runtime and artifacts.
7. The technical notebook explains HMMs, the EM/Baum-Welch estimation procedure, Viterbi decoding, and smoothed posterior probabilities without naming Python libraries in explanatory prose.
8. Before every equation containing Greek symbols, the notebook lists every Greek letter used there and its pronunciation.
9. `reports/references.bib` remains the canonical scientific source registry. Notebook and PDF citations use MLA 9, every in-text citation resolves, and Works Cited contains cited entries only. No bibliographic metadata is invented.
10. The PDF remains a strict sidecar of the executed notebook: supplied template page 1 first, template instruction page excluded, then the notebook rendered in order. It introduces no independent calculations or conclusions.
11. Every PR passes the current lint, type, unit, integration, repository-hygiene, numerical-consistency, notebook-orchestration, README, and coverage gates that apply at that point. Combined source coverage remains `>=90%`.
12. `python scripts/check_backlog_contract.py` must pass before the revision backlog itself is merged and before every later implementation PR is merged.

# Fixed analytical contracts

## Step 1 — unchanged canonical input

The source of all recomputation remains:

```text
data/processed/step1_data.csv
```

Columns remain exactly:

```text
TLT,GLD,SPY,VIX,TLT_log_return,GLD_log_return,SPY_log_return,VIX_change
```

Step 1 is not re-downloaded by this revision. Later PRs read the committed canonical dataset and fail on schema, date, missing-value, or finite-value violations.

## Step 2 — HMM only

The only regime observation is `VIX_change`. Candidate state counts are exactly `K=2` and `K=3`.

Gaussian HMM settings remain fixed:

```text
covariance_type = diag
n_iter = 500
tol = 1e-6
min_covar = 1e-6
seeds = (42, 43, 44, 45, 46)
likelihood tie tolerance = 1e-12
probability tolerance = 1e-8
minimum Viterbi state occupancy = 0.05
```

For each K, fit every configured seed, retain only converged finite-likelihood fits, choose the greatest log-likelihood, and break a likelihood tie by the smallest seed. Relabel every state-dependent quantity by increasing fitted mean `VIX_change`, with original component index as the deterministic equal-mean tie break.

Canonical HMM outputs are:

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

No Step 2 Markov table or figure is canonical after this revision.

### HMM / EM documentation contract

The technical notebook must explain the following concepts before presenting fitted results:

- latent state process and first-order transition probabilities;
- Gaussian state-conditional emission distribution;
- initial-state probabilities, transition matrix, state means, and state variances;
- observed-data likelihood;
- **Expectation-Maximization / Baum-Welch**: E-step computes posterior state and transition responsibilities using forward-backward probabilities; M-step updates initial probabilities, transition probabilities, Gaussian means, and variances from those responsibilities; iterations continue to the configured convergence rule;
- **Viterbi** sequence as the most likely joint state path, distinct from pointwise smoothed posterior probabilities;
- smoothed probabilities as full-sample posterior probabilities and therefore non-causal for historical dates.

The explanation must use scholarly references from `reports/references.bib` and must state that EM can converge to local optima, which is why deterministic multi-start fitting is used.

## Step 3 — HMM state-count selection only

The comparison table contains exactly two HMM rows, K=2 then K=3. Schema:

```text
family,n_states,log_likelihood,n_parameters,n_observations,aic,bic,converged,min_viterbi_occupancy,valid
```

`family` is always `hmm`. HMM free-parameter count remains:

```text
k = K^2 + 2K - 1
AIC = 2k - 2 logL
BIC = k ln(n) - 2 logL
```

A candidate is `valid=True` only when all of these hold: converged; finite likelihood; finite ordered means; strictly positive finite variances; valid initial probabilities; valid row-normalized transition probabilities; valid row-normalized smoothed posterior probabilities; valid Viterbi labels; every Viterbi state occupancy `>=0.05`.

The preferred model is the **valid HMM candidate with the smallest BIC**. A BIC tie within `1e-12` selects the lower K. If no HMM candidate is valid, the pipeline fails explicitly; it never falls back to another model family. The 5% occupancy rule is documented as a project stability rule, not a statistical theorem.

Canonical outputs:

```text
reports/tables/step3_model_comparison.csv
reports/tables/step3_selected_states.csv
reports/tables/step3_state_asset_statistics.csv
reports/figures/step3_state_asset_statistics.png
reports/generated/step3_selected_model.json
```

Selected-model JSON keys are exactly:

```text
family,n_states,state_source,selection_reason,input_data_sha256,selected_states_path
```

`family` must equal `hmm` and `selected_states_path` must equal `reports/tables/step3_selected_states.csv`.

## Step 4 — both allocation methods are mandatory

For every selected HMM state, rank `TLT`, `GLD`, `SPY` by descending historical state-conditional mean daily log return. Exact equal means use fixed priority:

```text
TLT -> GLD -> SPY
```

Two allocation methods are produced from the **same ranking**:

```text
100_keep:
  rank 1 ETF = 1.00
  rank 2 ETF = 0.00
  rank 3 ETF = 0.00

60_40_spread:
  rank 1 ETF = 0.60
  rank 2 ETF = 0.40
  rank 3 ETF = 0.00
```

Each mapping uses schema:

```text
method,state,rank_1_asset,rank_2_asset,rank_1_mean_log_return,rank_2_mean_log_return,TLT_weight,GLD_weight,SPY_weight
```

Canonical outputs:

```text
reports/tables/step4_allocation_100_keep.csv
reports/tables/step4_allocation_60_40_spread.csv
```

Every row has non-negative finite weights summing exactly to 1 within `1e-12`. The old single-method `step4_allocation_mapping.csv` is non-canonical and is removed in PR-67.

## Step 5 — two HMM strategies plus two required benchmarks

ETF log returns are converted to simple returns before portfolio arithmetic:

```text
simple_return[i,t] = exp(log_return[i,t]) - 1
```

For both allocation methods, the state observed on row `t-1` determines weights applied to ETF returns on row `t`. The first row is excluded. There is no same-row execution and no fill.

The exact daily comparison columns are:

```text
hmm_100_keep,hmm_60_40_spread,equal_weight_monthly,spy_buy_hold
```

The equal-weight benchmark remains one-third TLT / one-third GLD / one-third SPY, reset before the first comparison return and before the first observed comparison date of each new calendar month, with intra-month weight drift. SPY remains buy-and-hold. All four return series use exactly identical dates.

Metrics remain based on simple daily returns, 252 trading days, zero risk-free rate, sample standard deviation with `ddof=1`, and initial wealth `W_0=1` included in drawdown peaks:

```text
Cumulative Return
Annualized Return
Annualized Volatility
Sharpe Ratio
Maximum Drawdown
```

Performance summary row order is exactly:

```text
hmm_100_keep
hmm_60_40_spread
equal_weight_monthly
spy_buy_hold
```

State-count sensitivity is exactly four rows: HMM K=2 and K=3 crossed with `100_keep` and `60_40_spread`. It uses the persisted HMM candidate state paths, common lagged return-date intersection, shared Step 3 statistics, shared Step 4 rankings, shared backtest, and shared performance metrics. K=3 may appear in sensitivity even if it fails the 5% main-selection rule; sensitivity does not override the selected main model.

Canonical Step 5 outputs:

```text
reports/tables/step5_daily_returns.csv
reports/tables/step5_performance_summary.csv
reports/tables/step5_state_count_sensitivity.csv
reports/figures/step5_cumulative_performance.png
reports/generated/step5_manifest.json
```

The cumulative-performance figure contains exactly the four comparison series above.

## Look-ahead and interpretation contract

The assignment-required one-row execution lag avoids same-row execution but does not make the descriptive full-sample analysis causal. The notebook must state clearly that full-sample HMM parameter estimation, full-sequence Viterbi decoding/smoothing, and full-sample state-conditional ranking use future information relative to earlier dates. Results are therefore assignment backtest results, not evidence of a live trading edge.

The existing causal predictive extension, if retained in the repository, is also HMM-only after this revision. It may use one-sided/expanding HMM inference, but it may not contain a Markov candidate, Markov forecast, Markov walk-forward path, or family-selection branch.

## Notebook, HTML, PDF, and provenance

Canonical technical notebook:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

The notebook contains no Markov modeling section or Markov result. Step 2 documents and displays HMM K=2/K=3; Step 3 selects only between those HMM candidates; Step 4 displays both allocation mappings; Step 5 compares both HMM allocation methods with both required benchmarks and shows the four-row sensitivity table.

Canonical executed-notebook duplicate:

```text
reports/gwp2_vix_regime_allocation.html
```

Canonical notebook-derived PDF:

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
```

The PDF is regenerated from the final executed notebook by `scripts/build_pdf_report.py`; it prepends supplied template page 1, excludes the template instruction page, preserves the group data, and records the final notebook SHA-256 in PDF metadata.

---

## PR-50 — HMM-only model selection

**Agent lane:** A

**Dependencies:** none

**Git branch:** `pr-50-hmm-only-model-selection`

**Git status:** `git status --short --branch` must show `pr-50-hmm-only-model-selection` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-50 — HMM-only model selection`

**Files owned:**

```text
src/vix_regime_allocation/model_selection.py
tests/test_model_selection.py
```

### Public interface

```python
def build_hmm_model_comparison(hmm_candidates: list[dict[str, object]]) -> pandas.DataFrame: ...
def select_preferred_hmm(comparison: pandas.DataFrame, hmm_candidates: list[dict[str, object]]) -> dict[str, object]: ...
```

### Tasks

- [ ] T50.1 Add the exact two-row HMM comparison, fixed validity diagnostics, valid-candidate BIC selection, lower-K BIC tie break, and explicit no-valid-HMM failure while keeping transitional legacy wrappers only if current callers require them.
- [ ] T50.2 Replace selection tests with deterministic K=2/K=3 fixtures covering schema, every validity failure, BIC winner, tie winner, and proof that no Markov candidate can be selected.

### Acceptance criteria

- [ ] AC50.1 (`T50.1`) Public HMM-only functions return the fixed schemas and can return only family `hmm`; no-valid-HMM raises clearly and no alternate-family fallback exists.
- [ ] AC50.2 (`T50.2`) All model-selection tests pass offline and independently exercise every selection branch.

---

## PR-51 — Predictive family lock to HMM

**Agent lane:** B

**Dependencies:** none

**Git branch:** `pr-51-predictive-family-lock-to-hmm`

**Git status:** `git status --short --branch` must show `pr-51-predictive-family-lock-to-hmm` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-51 — Predictive family lock to HMM`

**Files owned:**

```text
src/vix_regime_allocation/predictive/config.py
src/vix_regime_allocation/predictive/selection.py
src/vix_regime_allocation/predictive/artifacts.py
tests/predictive/test_selection.py
tests/predictive/test_artifacts.py
```

### Tasks

- [ ] T51.1 Restrict the predictive candidate grid and artifact coordinator to HMM K=2/K=3 only, remove family tie-breaking and Markov signal construction from active execution, and preserve the existing switch-hurdle, validation-window, holdout, and transaction-cost contracts.
- [ ] T51.2 Update predictive selection/artifact tests so exact candidate counts, selected configuration, persisted family fields, and manifest outputs accept only `hmm` and fail any Markov candidate input.

### Acceptance criteria

- [ ] AC51.1 (`T51.1`) Predictive execution constructs HMM signals only and every selected/persisted family value is exactly `hmm`.
- [ ] AC51.2 (`T51.2`) Predictive selection and artifact tests pass offline with deterministic HMM-only candidate counts and explicit Markov rejection.

---

## PR-52 — Dual Step 4 allocation engine

**Agent lane:** A

**Dependencies:** none

**Git branch:** `pr-52-dual-step-4-allocation-engine`

**Git status:** `git status --short --branch` must show `pr-52-dual-step-4-allocation-engine` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-52 — Dual Step 4 allocation engine`

**Files owned:**

```text
src/vix_regime_allocation/allocation.py
tests/test_allocation.py
```

### Public interface

```python
def build_state_allocation(statistics: pandas.DataFrame, method: str = "100_keep") -> pandas.DataFrame: ...
```

### Tasks

- [ ] T52.1 Implement deterministic top-three ranking with fixed TLT→GLD→SPY equality priority and exact `100_keep` plus `60_40_spread` weights/schema, retaining `100_keep` as the temporary default for compatibility.
- [ ] T52.2 Test all rank permutations, two-way and three-way equal means, both method schemas/weights, sum-to-one validation, unsupported method failure, and malformed statistics.

### Acceptance criteria

- [ ] AC52.1 (`T52.1`) Both methods use the same deterministic ranking, produce exact 1/0/0 or 0.6/0.4/0 weights, and return the fixed method-aware schema.
- [ ] AC52.2 (`T52.2`) Allocation tests pass offline and fail on every invalid method, ranking input, or weight invariant.

---

## PR-53 — Dual-method Step 5 comparison engine

**Agent lane:** B

**Dependencies:** PR-52

**Git branch:** `pr-53-dual-method-step-5-comparison-engine`

**Git status:** `git status --short --branch` must show `pr-53-dual-method-step-5-comparison-engine` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-53 — Dual-method Step 5 comparison engine`

**Files owned:**

```text
src/vix_regime_allocation/strategy_comparison.py
tests/test_strategy_comparison.py
```

### Public interface

```python
def build_dual_method_comparison(data: pandas.DataFrame, states: pandas.Series, statistics: pandas.DataFrame) -> tuple[pandas.DataFrame, dict[str, pandas.DataFrame]]: ...
```

### Tasks

- [ ] T53.1 Build both allocations through the shared allocation engine, call the existing one-row-lag rotation engine once per method, build both existing benchmarks on the same return dates, and return the exact four-column comparison plus both rotation-detail frames.
- [ ] T53.2 Add hand-checkable tests proving previous-row state timing, 100% and 60/40 portfolio arithmetic, exact identical indexes, exact four-column order, and delegation to existing benchmark logic.

### Acceptance criteria

- [ ] AC53.1 (`T53.1`) The returned comparison columns are exactly `hmm_100_keep,hmm_60_40_spread,equal_weight_monthly,spy_buy_hold` on identical sorted dates.
- [ ] AC53.2 (`T53.2`) Comparison tests pass offline and prove a state change affects both HMM strategies only on the following observed return row.

---

## PR-54 — Four-portfolio performance summary

**Agent lane:** A

**Dependencies:** PR-53

**Git branch:** `pr-54-four-portfolio-performance-summary`

**Git status:** `git status --short --branch` must show `pr-54-four-portfolio-performance-summary` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-54 — Four-portfolio performance summary`

**Files owned:**

```text
src/vix_regime_allocation/backtest_summary.py
tests/test_backtest_summary.py
```

### Tasks

- [ ] T54.1 Replace the canonical comparison schema with the exact four-series order and build a four-row summary by delegating every metric to `performance_metrics` without duplicating metric formulas.
- [ ] T54.2 Update tests for four-series schema/order, identical-date enforcement, exact portfolio-to-metric mapping, and rejection of the legacy three-series schema.

### Acceptance criteria

- [ ] AC54.1 (`T54.1`) Canonical summary contains exactly four rows in the fixed order and all values come from the shared performance function.
- [ ] AC54.2 (`T54.2`) Summary tests pass offline and any missing, extra, reordered, or date-misaligned comparison series fails.

---

## PR-55 — Four-curve cumulative performance figure

**Agent lane:** B

**Dependencies:** PR-53

**Git branch:** `pr-55-four-curve-cumulative-performance-figure`

**Git status:** `git status --short --branch` must show `pr-55-four-curve-cumulative-performance-figure` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-55 — Four-curve cumulative performance figure`

**Files owned:**

```text
src/vix_regime_allocation/backtest_plot.py
tests/test_backtest_plot.py
tests/test_cumulative_plot_series_contract.py
```

### Tasks

- [ ] T55.1 Update the canonical Step 5 figure to plot exactly the four fixed comparison series by delegating cumulative wealth to the shared performance helper, preserving readable title, axes, legend, zero reference, save, and close behavior.
- [ ] T55.2 Update plot tests to assert exact four labels, exact compounded y-values, no fifth series, non-empty output, and closed figures.

### Acceptance criteria

- [ ] AC55.1 (`T55.1`) The plot contains both HMM allocation methods and both required benchmarks exactly once with values equal to shared compounded wealth minus one.
- [ ] AC55.2 (`T55.2`) All cumulative-plot tests pass offline and the former three-series contract is rejected.

---

## PR-56 — HMM state-count and allocation sensitivity

**Agent lane:** A

**Dependencies:** PR-52, PR-53

**Git branch:** `pr-56-hmm-state-count-and-allocation-sensitivity`

**Git status:** `git status --short --branch` must show `pr-56-hmm-state-count-and-allocation-sensitivity` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-56 — HMM state-count and allocation sensitivity`

**Files owned:**

```text
src/vix_regime_allocation/sensitivity.py
tests/test_sensitivity.py
```

### Tasks

- [ ] T56.1 Replace family sensitivity with exactly HMM K=2/K=3 crossed with `100_keep`/`60_40_spread`, use persisted state paths only, recompute shared statistics and allocations, apply the same lag, intersect common return dates, and return the fixed four-row schema.
- [ ] T56.2 Test exact four combinations, common observation count, both allocation delegations, metric delegations, K=3 diagnostic inclusion, and rejection of family arguments or incomplete state dictionaries.

### Acceptance criteria

- [ ] AC56.1 (`T56.1`) Sensitivity has exactly four HMM rows ordered K=2 100%, K=2 60/40, K=3 100%, K=3 60/40 on one common date intersection.
- [ ] AC56.2 (`T56.2`) Sensitivity tests pass offline and no model fitting, decoding, Markov family switch, or duplicated metric math occurs.

---

## PR-57 — HMM-only rebuild and numerical audit

**Agent lane:** B

**Dependencies:** PR-50, PR-51, PR-54, PR-55, PR-56

**Git branch:** `pr-57-hmm-only-rebuild-and-numerical-audit`

**Git status:** `git status --short --branch` must show `pr-57-hmm-only-rebuild-and-numerical-audit` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-57 — HMM-only rebuild and numerical audit`

**Files owned:**

```text
scripts/rebuild_analysis_review.py
scripts/check_analysis_consistency.py
scripts/check_artifact_provenance.py
```

### Tasks

- [ ] T57.1 Refactor rebuild/audit paths to fit and validate only HMM K=2/K=3, select with the new HMM-only rule, construct both allocations and four-series Step 5 comparison, recompute all metrics/sensitivity/manifests, and remove every Markov calculation from these scripts.
- [ ] T57.2 Extend numerical assertions so persisted HMM parameters/states/transitions, selected-state provenance, both allocation maps, both HMM strategy returns, both benchmarks, all metrics, sensitivity rows, hashes, and manifest membership are independently reconciled.

### Acceptance criteria

- [ ] AC57.1 (`T57.1`) Rebuild/audit execution contains no Markov fit/discretization/import path and produces in-memory outputs for every new canonical artifact contract.
- [ ] AC57.2 (`T57.2`) Numerical audit fails on a one-value mutation in each HMM, allocation-method, strategy-return, performance, sensitivity, or hash class and passes on a consistent fixture.

---

## PR-58 — HMM-only notebook orchestration helpers

**Agent lane:** A

**Dependencies:** PR-50, PR-52, PR-54, PR-55, PR-56

**Git branch:** `pr-58-hmm-only-notebook-orchestration-helpers`

**Git status:** `git status --short --branch` must show `pr-58-hmm-only-notebook-orchestration-helpers` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-58 — HMM-only notebook orchestration helpers`

**Files owned:**

```text
src/vix_regime_allocation/notebook_helpers.py
src/vix_regime_allocation/notebook_sensitivity.py
scripts/check_notebook_orchestration.py
```

### Tasks

- [ ] T58.1 Remove the Markov notebook helper path and mixed-family selection path; expose presentation helpers for HMM K=2/K=3 diagnostics, HMM-only selection, state statistics, both Step 4 mappings, four-series Step 5 results, and four-row sensitivity using shared project functions only.
- [ ] T58.2 Update notebook-orchestration validation so code cells may call only presentation helpers/imports, required HMM/dual-method sections are present exactly once, and forbidden Markov helper calls or embedded analytical code fail.

### Acceptance criteria

- [ ] AC58.1 (`T58.1`) Notebook helper execution can produce every required HMM-only and dual-allocation display/artifact without invoking a Markov module.
- [ ] AC58.2 (`T58.2`) Orchestration checker passes the intended helper-only notebook structure and deterministically rejects Markov sections, missing allocation methods, or duplicated analytical code.

---

## PR-59 — HMM-only dual-method integration test

**Agent lane:** B

**Dependencies:** PR-57, PR-58

**Git branch:** `pr-59-hmm-only-dual-method-integration-test`

**Git status:** `git status --short --branch` must show `pr-59-hmm-only-dual-method-integration-test` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-59 — HMM-only dual-method integration test`

**Files owned:**

```text
tests/integration/test_hmm_only_dual_allocation_pipeline.py
```

### Tasks

- [ ] T59.1 Build a deterministic synthetic fixture and execute HMM candidate evaluation/selection, state statistics, both allocations, both lagged rotations, both benchmarks, four-row performance summary, four-curve plot path, and four-row sensitivity end-to-end.
- [ ] T59.2 Assert exact dates, schemas, weights, selected family, lag timing, common sensitivity dates, and absence of any Markov import or generated Markov artifact in the integration path.

### Acceptance criteria

- [ ] AC59.1 (`T59.1`) The integration test exercises every source-level contract required before canonical full-data regeneration and is fully offline.
- [ ] AC59.2 (`T59.2`) The test passes only when family is HMM, both allocation methods are present, all four comparison series align, and no Markov path is executed.

---

## PR-60 — Canonical HMM artifact rebuild

**Agent lane:** A

**Dependencies:** PR-59

**Git branch:** `pr-60-canonical-hmm-artifact-rebuild`

**Git status:** `git status --short --branch` must show `pr-60-canonical-hmm-artifact-rebuild` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-60 — Canonical HMM artifact rebuild`

**Files owned:**

```text
reports/tables/step2_hmm_2_parameters.csv
reports/tables/step2_hmm_3_parameters.csv
reports/tables/step2_hmm_2_transition.csv
reports/tables/step2_hmm_3_transition.csv
reports/tables/step2_hmm_2_states.csv
reports/tables/step2_hmm_3_states.csv
reports/tables/step3_model_comparison.csv
reports/tables/step3_selected_states.csv
reports/tables/step3_state_asset_statistics.csv
reports/tables/step4_allocation_100_keep.csv
reports/tables/step4_allocation_60_40_spread.csv
reports/tables/step5_daily_returns.csv
reports/tables/step5_performance_summary.csv
reports/tables/step5_state_count_sensitivity.csv
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
reports/figures/step3_state_asset_statistics.png
reports/figures/step5_cumulative_performance.png
reports/generated/step3_selected_model.json
reports/generated/steps_2_4_manifest.json
reports/generated/step5_manifest.json
```

### Tasks

- [ ] T60.1 Run the deterministic rebuild from the committed Step 1 CSV and persist every listed HMM/selection/statistics/dual-allocation/Step 5 table, figure, JSON, and manifest without editing generated values by hand.
- [ ] T60.2 Run the numerical consistency and provenance audits against the newly written files, verify selected family is `hmm`, verify both allocation maps and four comparison columns, and record no stale canonical Markov path in either manifest.

### Acceptance criteria

- [ ] AC60.1 (`T60.1`) Every owned artifact is non-empty, schema-valid, mutually consistent, derived from the current Step 1 hash, and reproducible by a second rebuild.
- [ ] AC60.2 (`T60.2`) Full numerical/provenance audits pass and manifests list only the new HMM/dual-method canonical artifacts for Steps 2–5.

---

## PR-61 — Canonical HMM notebook rebuild

**Agent lane:** B

**Dependencies:** PR-60

**Git branch:** `pr-61-canonical-hmm-notebook-rebuild`

**Git status:** `git status --short --branch` must show `pr-61-canonical-hmm-notebook-rebuild` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-61 — Canonical HMM notebook rebuild`

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/references.bib
```

### Tasks

- [ ] T61.1 Rebuild notebook Steps 2–5 to contain only HMM K=2/K=3, add the complete HMM and EM/Baum-Welch explanation plus Viterbi/smoothed-probability distinction with Greek-letter pronunciation declarations and resolved MLA 9 scholarly citations, then display HMM-only selection and both Step 4 allocation mappings.
- [ ] T61.2 Display both HMM allocation backtests with both required benchmarks, all five metrics, four cumulative curves, four-row K×allocation sensitivity, exact limitations, source notes, final cited-only Works Cited, and execute the entire notebook top-to-bottom with stored outputs.

### Acceptance criteria

- [ ] AC61.1 (`T61.1`) Notebook contains no Markov model/result/figure, explains HMM/EM/decoding/posteriors accurately without Python library names in prose, lists/pronounces all Greek symbols before equations, and every external claim has a resolved citation from `reports/references.bib`.
- [ ] AC61.2 (`T61.2`) Notebook outputs match PR-60 artifacts exactly, contains both `100_keep` and `60_40_spread`, contains all four comparison portfolios and sensitivity combinations, has no failed/unexecuted code cell, and states the full-sample look-ahead limitations explicitly.

---

## PR-62 — README HMM-only synchronization

**Agent lane:** A

**Dependencies:** PR-61

**Git branch:** `pr-62-readme-hmm-only-synchronization`

**Git status:** `git status --short --branch` must show `pr-62-readme-hmm-only-synchronization` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-62 — README HMM-only synchronization`

**Files owned:**

```text
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T62.1 Rewrite the factual analysis/status block from canonical artifacts so README describes HMM only, the selected HMM state count, both allocation methods, four-portfolio Step 5 results, four-row sensitivity, look-ahead qualification, notebook/PDF paths, and the `reports/references.bib` MLA 9 source policy without recomputing values.
- [ ] T62.2 Extend the README checker to require both allocation artifacts, HMM-only selected-model provenance, four Step 5 portfolios, current notebook/report paths, citation-registry references, and to reject canonical Markov-result claims or the legacy single-allocation path.

### Acceptance criteria

- [ ] AC62.1 (`T62.1`) README numerical statements and paths have exact artifact parity and no Markov model is presented as active analysis.
- [ ] AC62.2 (`T62.2`) README checker passes the synchronized file and fails stale selected-family, missing allocation-method, old path, or missing citation-policy mutations.

---

## PR-63 — Executed notebook HTML refresh

**Agent lane:** B

**Dependencies:** PR-61

**Git branch:** `pr-63-executed-notebook-html-refresh`

**Git status:** `git status --short --branch` must show `pr-63-executed-notebook-html-refresh` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-63 — Executed notebook HTML refresh`

**Files owned:**

```text
reports/gwp2_vix_regime_allocation.html
```

### Tasks

- [ ] T63.1 Export the already executed PR-61 notebook to HTML without executing or refitting it, preserving all stored HMM/EM explanations, outputs, both allocation methods, four comparison portfolios, citations, source notes, and Works Cited.
- [ ] T63.2 Verify the HTML is non-empty, has no Markov modeling/result section, contains both allocation methods and all four Step 5 portfolio labels, and visually exposes no failed notebook output.

### Acceptance criteria

- [ ] AC63.1 (`T63.1`) HTML is a content duplicate of the stored executed notebook and contains the `reports/references.bib`-derived citations/Works Cited without a second analysis execution.
- [ ] AC63.2 (`T63.2`) Static checks confirm current HMM-only/dual-method content and reject legacy Markov or single-method exported content.

---

## PR-64 — Notebook-derived PDF pipeline validation

**Agent lane:** A

**Dependencies:** PR-61

**Git branch:** `pr-64-notebook-derived-pdf-pipeline-validation`

**Git status:** `git status --short --branch` must show `pr-64-notebook-derived-pdf-pipeline-validation` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-64 — Notebook-derived PDF pipeline validation`

**Files owned:**

```text
scripts/build_pdf_report.py
tests/test_build_pdf_report.py
```

### Tasks

- [ ] T64.1 Preserve notebook-as-single-source behavior while validating that the source notebook is fully executed, HMM-only, contains both allocation methods and four Step 5 portfolios, then render it after supplied template page 1 with template instruction pages excluded and final notebook SHA-256 embedded in PDF metadata.
- [ ] T64.2 Add offline/fixture tests for failed or unexecuted notebook cells, missing HMM/EM section, Markov-result leakage, missing 100% or 60/40 section, wrong cover/page handling, missing citation/Works Cited content, empty output, and notebook-hash metadata mismatch.

### Acceptance criteria

- [ ] AC64.1 (`T64.1`) Builder performs no independent analysis and can only render a validated HMM-only dual-method executed notebook with the supplied cover and correct SHA metadata.
- [ ] AC64.2 (`T64.2`) PDF builder tests pass and each stale/malformed notebook, cover, citation, or metadata fixture fails deterministically.

---

## PR-65 — Final notebook-derived PDF regeneration

**Agent lane:** B

**Dependencies:** PR-64

**Git branch:** `pr-65-final-notebook-derived-pdf-regeneration`

**Git status:** `git status --short --branch` must show `pr-65-final-notebook-derived-pdf-regeneration` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-65 — Final notebook-derived PDF regeneration`

**Files owned:**

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
```

### Tasks

- [ ] T65.1 Generate the PDF from the final PR-61 notebook through the validated builder and verify template cover, group information, HMM-only notebook content, HMM/EM explanation, both allocation methods, four Step 5 portfolios, citations/Works Cited, and exact notebook SHA metadata.
- [ ] T65.2 Render and inspect every PDF page for clipping, overlap, malformed equations, raw LaTeX leakage, blank pages, missing plots/tables, and unreadable glyphs; regenerate only through the builder if a defect is found.

### Acceptance criteria

- [ ] AC65.1 (`T65.1`) Final PDF is non-empty, has exact notebook-content parity after the cover, and metadata identifies the exact committed notebook bytes.
- [ ] AC65.2 (`T65.2`) Every page passes visual QA and all HMM, allocation, performance, sensitivity, and citation content remains readable and complete.

---

## PR-66 — HMM-only sidecar and CI parity

**Agent lane:** A

**Dependencies:** PR-62, PR-63, PR-65

**Git branch:** `pr-66-hmm-only-sidecar-and-ci-parity`

**Git status:** `git status --short --branch` must show `pr-66-hmm-only-sidecar-and-ci-parity` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-66 — HMM-only sidecar and CI parity`

**Files owned:**

```text
.github/workflows/quality-gates.yml
.github/workflows/review-artifact-build.yml
.github/workflows/report-sync.yml
scripts/check_repository_hygiene.py
```

### Tasks

- [ ] T66.1 Update CI/review/report workflows to require HMM-only numerical audit, both Step 4 allocation artifacts, four-series Step 5 artifacts, final executed notebook, current HTML, current notebook-derived PDF, README parity, `reports/references.bib` citation integrity, and combined coverage `>=90%` while retaining independent parallel lint/type/unit/integration jobs.
- [ ] T66.2 Extend repository hygiene to reject active canonical Markov imports/outputs, legacy `step4_allocation_mapping.csv` references, missing dual-method paths, or manifests that list Markov artifacts, while allowing historical/bibliographic occurrences that are not executable or canonical analysis.

### Acceptance criteria

- [ ] AC66.1 (`T66.1`) Aggregate quality gate cannot pass with stale notebook/HTML/PDF, missing HMM/dual-method artifacts, citation drift, numerical drift, or coverage below 90%.
- [ ] AC66.2 (`T66.2`) Hygiene passes the intended transitional tree before PR-67 and fails any active/canonical Markov or legacy single-allocation reference covered by the fixed scope.

---

## PR-67 — Markov purge and repository hygiene

**Agent lane:** B

**Dependencies:** PR-66

**Git branch:** `pr-67-markov-purge-and-repository-hygiene`

**Git status:** `git status --short --branch` must show `pr-67-markov-purge-and-repository-hygiene` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-67 — Markov purge and repository hygiene`

**Files owned:**

```text
src/vix_regime_allocation/markov_states.py
src/vix_regime_allocation/markov_chain.py
src/vix_regime_allocation/markov_evaluation.py
src/vix_regime_allocation/markov_plots.py
src/vix_regime_allocation/predictive/markov_forecast.py
src/vix_regime_allocation/predictive/markov_walkforward.py
tests/test_markov_states.py
tests/test_markov_chain.py
tests/test_markov_evaluation.py
tests/test_markov_plots.py
tests/predictive/test_markov_forecast.py
tests/predictive/test_markov_walkforward.py
reports/tables/step2_markov_2_thresholds.csv
reports/tables/step2_markov_3_thresholds.csv
reports/tables/step2_markov_2_transition.csv
reports/tables/step2_markov_3_transition.csv
reports/tables/step2_markov_2_stationary.csv
reports/tables/step2_markov_3_stationary.csv
reports/tables/step2_markov_2_states.csv
reports/tables/step2_markov_3_states.csv
reports/figures/step2_markov_vix_states.png
reports/tables/step4_allocation_mapping.csv
src/vix_regime_allocation/information_criteria.py
tests/test_information_criteria.py
src/vix_regime_allocation/model_config.py
tests/test_model_config.py
```

### Tasks

- [ ] T67.1 Delete every listed Markov runtime/test/artifact path and the legacy single-allocation artifact, then remove Markov-only parameter-count/config constants from the shared information/config modules without changing HMM constants or formulas.
- [ ] T67.2 Run the complete test, numerical-audit, notebook-orchestration, README, report-sync, repository-hygiene, and coverage suite and verify no surviving source/test/workflow/manifest imports a deleted Markov module.

### Acceptance criteria

- [ ] AC67.1 (`T67.1`) Final repository contains no executable Markov implementation, predictive Markov path, Markov unit test, canonical Markov artifact, or legacy single-allocation mapping, while HMM K=2/K=3 behavior is unchanged.
- [ ] AC67.2 (`T67.2`) Full quality suite passes after deletion with coverage `>=90%`, all imports resolve, and canonical artifacts/notebook/HTML/PDF remain byte-consistent with HMM-only analysis.

---

## PR-68 — Final release artifact rebuild

**Agent lane:** A

**Dependencies:** PR-67

**Git branch:** `pr-68-final-release-artifact-rebuild`

**Git status:** `git status --short --branch` must show `pr-68-final-release-artifact-rebuild` and no staged, modified, or untracked files immediately before commit and merge.

**Commit message:** `PR-68 — Final release artifact rebuild`

**Files owned:**

```text
reports/predictive/tables/candidate_validation_summary.csv
reports/predictive/tables/selected_test_daily.csv
reports/predictive/tables/selected_test_performance.csv
reports/predictive/tables/test_asset_dominance.csv
reports/predictive/generated/selected_strategy.json
reports/predictive/generated/predictive_manifest.json
README.md
scripts/check_readme_sidecar.py
```

### Tasks

- [ ] T68.1 Recompute the retained predictive extension from the committed Step 1 data through the HMM-only coordinator, persist its tables/selected JSON/manifest, and verify every persisted family is `hmm`, all hashes are current, and no deleted Markov path appears in predictive provenance.
- [ ] T68.2 Refresh README/checker predictive status from the final HMM-only predictive artifacts, retain the `reports/references.bib` citation policy and final notebook/HTML/PDF paths, then run the entire final quality suite including backlog contract and coverage.

### Acceptance criteria

- [ ] AC68.1 (`T68.1`) Predictive artifacts are deterministic, current, HMM-only, hash-consistent, and contain no Markov candidate or selected family.
- [ ] AC68.2 (`T68.2`) README/checker reflect final HMM-only assignment and predictive state, citation-registry policy remains explicit, and every final quality gate passes with coverage `>=90%`.

---

# Parallel schedule for two weak agents

```text
W1   Agent A: PR-50                 | Agent B: PR-51
W2   Agent A: PR-52                 | Agent B: wait
W3   Agent A: wait                  | Agent B: PR-53
W4   Agent A: PR-54                 | Agent B: PR-55
W5   Agent A: PR-56                 | Agent B: wait
W6   Agent A: PR-58                 | Agent B: PR-57
W7   Agent A: wait                  | Agent B: PR-59
W8   Agent A: PR-60                 | Agent B: wait
W9   Agent A: wait                  | Agent B: PR-61
W10  Agent A: PR-64                 | Agent B: PR-62
W11  Agent A: wait                  | Agent B: PR-63
W12  Agent A: wait                  | Agent B: PR-65
W13  Agent A: PR-66                 | Agent B: wait
W14  Agent A: wait                  | Agent B: PR-67
W15  Agent A: PR-68                 | Agent B: wait
```

PR-54 and PR-55 are the primary safe parallel implementation pair after the dual-method comparison engine exists. PR-57 and PR-58 are the second safe parallel pair: rebuild/audit scripts versus notebook-helper orchestration. Notebook and PDF work remains serialized because those artifacts share a single source-of-truth chain.

# Merge rules

For every implementation PR: branch from current `main` after dependencies; modify only owned files; implement every listed task and no later-PR work; prove every matching acceptance criterion; update from `main`; rerun the full applicable quality suite; merge only after the aggregate quality gate succeeds; then delete the implementation branch.

Generated numerical files are merged only in PR-60, the notebook only in PR-61, HTML only in PR-63, and final PDF only in PR-65. This prevents weak agents from independently generating conflicting versions of the same analytical truth.

# Final Definition of Done

- [ ] PR-50..PR-68 are merged in dependency order and all quality gates pass.
- [ ] Active assignment analysis uses Gaussian HMM only; no executable/canonical Markov model, test, artifact, selection, notebook result, report result, or predictive branch remains.
- [ ] HMM K=2/K=3 are recomputed from `VIX_change`, evaluated with log-likelihood/AIC/BIC and fixed validity diagnostics, and exactly one valid HMM state count is selected without alternate-family fallback.
- [ ] Technical notebook accurately explains HMM, EM/Baum-Welch, Viterbi decoding, and smoothed posterior probabilities with Greek-letter pronunciations and resolved MLA 9 scholarly citations.
- [ ] Both state allocation methods are canonical: `100_keep` = 100% rank-1 ETF; `60_40_spread` = 60% rank-1 + 40% rank-2 ETF.
- [ ] Both HMM strategies use the exact one-observed-row execution lag and are compared on identical dates with monthly equal-weight and SPY buy-and-hold benchmarks.
- [ ] Performance summary contains cumulative return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, and observation count for exactly four portfolios.
- [ ] Sensitivity contains exactly HMM K=2/K=3 × 100%/60-40 on a common return-date intersection and is interpreted as diagnostic rather than a second model-selection stage.
- [ ] All HMM, state, allocation, return, metric, sensitivity, figure, JSON, and manifest artifacts are freshly recomputed and pass independent numerical/provenance checks.
- [ ] Final notebook executes top-to-bottom with stored outputs and no Markov section; README and HTML have current parity.
- [ ] `reports/Stochastic_Modeling_GWP2_Report.pdf` is regenerated from that exact notebook with the supplied template cover, excluded instruction page, readable math/figures/tables, resolved citations/Works Cited, and exact notebook SHA-256 metadata.
- [ ] Full-sample HMM/Viterbi/smoothing/allocation look-ahead limitations are explicit; no live-trading profitability claim is made from the assignment backtest.
- [ ] Combined source coverage is at least 90%, repository hygiene is clean, and `python scripts/check_backlog_contract.py` passes.
