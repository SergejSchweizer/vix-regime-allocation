# BACKLOG — Steps 2, 3 and 4: Regime Modeling, Model Selection and Allocation

This backlog covers **complete implementation of Steps 2, 3 and 4** of MScFE 622 Stochastic Modeling GWP2.

It is designed for **two weak coding agents working in parallel**. Every PR is intentionally small, has fixed file ownership, fixed public interfaces, explicit dependencies, deterministic behavior, and one-to-one task-to-acceptance traceability.

The assignment requires regime modeling from VIX, model comparison/selection, state-conditional ETF behavior, and a rule-based state-to-allocation mapping. Both the **discrete Markov-chain approach** and the **Gaussian Hidden Markov Model approach** must be implemented and presented even if only one is selected as the preferred model for Step 4.

---

# Non-negotiable project rules

## 1. Notebook is the canonical analysis artifact

The canonical analysis file is:

```text
notebooks/gwp2_vix_regime_allocation.ipynb
```

Core numerical logic must live in small tested functions under `src/vix_regime_allocation/`, but the **actual assignment analysis is executed, displayed, interpreted and explained in the notebook**.

The notebook must:

- call the tested project functions rather than duplicate their implementation;
- display function outputs directly;
- show all required equations before or next to the outputs they define;
- show all required tables and plots;
- provide precise, scientific-paper-style explanations of assumptions, methodology, interpretation and limitations;
- contain stored outputs from a successful execution;
- never fabricate a numerical result;
- use the frozen Step 1 dataset as the analysis input once Step 1 is complete;
- clearly distinguish observed quantities, estimated quantities and model-selection decisions.

## 2. Greek-letter rule for the notebook and report

Before an equation containing Greek letters, explicitly list every Greek letter used and its pronunciation.

Examples:

```text
Δ — delta
π — pi
μ — mu
σ — sigma
```

Do not use a Greek symbol in an equation without first identifying it in the surrounding explanatory text.

## 3. README is a synchronized sidecar

`README.md` is not a separate analysis. It must show the **same final equations, selected tables, figures, model decision and allocation mapping** as the executed notebook.

The README must not independently recompute results. It must consume or reference canonical outputs produced by the notebook workflow.

## 4. PDF report is a synchronized sidecar

The final report path is fixed:

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
```

The report must:

- use page 1 of `reports/Template_Stochastic_Modeling_Group_Work_Project.pdf` as the cover;
- never include page 2 of the template because page 2 contains submission instructions that the template explicitly says to remove;
- contain no source code;
- show the same equations, numerical tables, figures, selected model, state interpretation and Step 4 allocation mapping as the notebook;
- use the same canonical generated artifacts as the README;
- preserve the three already-filled team names on the cover;
- leave unknown group number, country and email fields unchanged until values are supplied;
- be rendered and visually verified before merge.

## 5. Same-output contract

The notebook, README and PDF report must derive their displayed numerical outputs from the same files under:

```text
reports/tables/
reports/figures/
reports/generated/
```

No sidecar may independently rerun model estimation to obtain a second copy of a result.

## 6. Quality contract

Every PR must pass the repository `quality-gate`.

Required combined coverage remains:

```text
>= 90%
```

All new source modules must have focused deterministic unit tests. Tests must not depend on Yahoo Finance or network access.

## 7. Weak-agent rule

An agent must:

- modify only files owned by its PR;
- not rename a specified function, column, path or state-ordering rule;
- not implement work assigned to a later PR;
- not add optional features not listed in the PR;
- not change the model-selection rule;
- not change the state-label ordering;
- not write narrative results before numerical outputs exist;
- stop if a required dependency PR is not merged.

---

# Fixed modeling contracts

## Step 1 input

All Step 2-4 modeling uses the Step 1 clean dataset with these columns:

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

The state-model observation is exactly:

```text
X_t = VIX_change_t
```

Do not switch from `VIX_change` to the VIX level or percentage VIX return.

## Candidate state counts

Implement exactly:

```text
2 states
3 states
```

for both model families.

## Deterministic state ordering

All returned state identifiers must be ordered by increasing mean `VIX_change`:

```text
State 0 = lowest mean VIX_change
State 1 = next higher mean VIX_change
State 2 = highest mean VIX_change  # only for 3-state models
```

This rule applies to both Markov-chain and HMM outputs.

Do not hard-code semantic labels such as `calm`, `normal`, or `stress` before the actual state statistics are displayed. Interpret the states from their estimated VIX-change behavior.

## Discrete Markov-chain state construction

Use empirical quantile bins.

For `n_states = 2`:

```text
quantile edges = [0, 0.5, 1.0]
```

For `n_states = 3`:

```text
quantile edges = [0, 1/3, 2/3, 1.0]
```

Use integer states `0..K-1` in ascending `VIX_change` order.

If quantile edges are not unique because the input lacks enough distinct values, raise a clear `ValueError`; do not silently merge bins.

## Markov transition estimate

For state transition counts `N_ij`, estimate:

```text
P_ij = N_ij / sum_j N_ij
```

No Laplace smoothing or pseudocounts are allowed.

If a state has zero outgoing transitions, raise a clear `ValueError`.

## Markov stationary distribution

Return a stationary row vector satisfying:

```text
pi @ P = pi
sum(pi) = 1
pi_i >= 0
```

Numerical tolerance for validation:

```text
1e-10
```

## Gaussian HMM contract

Use `hmmlearn.hmm.GaussianHMM` with:

```text
covariance_type = "diag"
n_iter = 500
tol = 1e-6
min_covar = 1e-6
random restart seeds = [42, 43, 44, 45, 46]
```

For each state count, fit all five restarts and select the converged fit with the highest training log-likelihood.

If none of the five fits converges, raise a clear `RuntimeError`.

After fitting, relabel all state-dependent parameters, Viterbi states and posterior-probability columns using the fixed increasing-mean state ordering.

## Information criteria

For every candidate model, compute:

```text
AIC = 2*k - 2*log_likelihood
BIC = k*ln(n) - 2*log_likelihood
```

where:

```text
k = number of free model parameters
n = number of observations used by that model likelihood
```

Markov-chain free-parameter count:

```text
k_MC = K * (K - 1)
```

Gaussian HMM free-parameter count for a univariate K-state model:

```text
k_HMM = (K - 1) + K*(K - 1) + K + K
      = K^2 + 2*K - 1
```

The notebook must state that the Markov-chain likelihood is a likelihood of the **discretized state sequence**, whereas the HMM likelihood is a likelihood of the **continuous VIX-change observations**. Therefore AIC/BIC values are used to choose the state count **within each model family**, not as a mathematically valid direct cross-family ranking.

## Preferred-model selection rule

Selection is deterministic and must not be changed by an agent.

1. Select the 2-state or 3-state Markov chain with the lowest Markov-family BIC.
2. Select the 2-state or 3-state HMM with the lowest HMM-family BIC among converged models.
3. Prefer the selected HMM if all of the following are true:
   - the selected HMM converged;
   - every posterior probability is finite;
   - every posterior-probability row sums to 1 within `1e-8`;
   - every state has at least 5% Viterbi occupancy;
   - every fitted state variance is finite and strictly positive;
   - the transition matrix is finite, nonnegative and every row sums to 1 within `1e-8`.
4. Otherwise use the selected Markov chain.

The scientific explanation must explicitly state that the method-level decision is based on model validity, probabilistic regime information and interpretability, rather than comparing non-comparable AIC/BIC values across families.

## Step 3 state-conditional ETF statistics

For the preferred state sequence, compute for every state and every ETF:

```text
mean daily log return
standard deviation of daily log return
number of observations
```

Assets are fixed and ordered:

```text
TLT
GLD
SPY
```

Do not annualize these Step 3 state-conditional statistics unless a later assignment step explicitly requires it.

## Step 4 allocation rule

For each preferred-model state, choose the ETF with the highest historical mean daily log return in that state.

Use exactly 100% allocation to the selected ETF:

```text
selected asset weight = 1.0
other two asset weights = 0.0
```

If two means are exactly equal, use the fixed deterministic tie-break order:

```text
TLT -> GLD -> SPY
```

Do not implement the optional 60/40 variant in Steps 2-4.

Do not implement the Step 5 backtest in this backlog. The later Step 5 backtest must apply a 1-day execution lag to the state-driven position.

---

# Canonical output contract

The executed notebook must create or refresh these files exactly.

## Tables

```text
reports/tables/step2_markov_2_thresholds.csv
reports/tables/step2_markov_2_transition.csv
reports/tables/step2_markov_2_stationary.csv
reports/tables/step2_markov_3_thresholds.csv
reports/tables/step2_markov_3_transition.csv
reports/tables/step2_markov_3_stationary.csv
reports/tables/step2_hmm_2_parameters.csv
reports/tables/step2_hmm_2_transition.csv
reports/tables/step2_hmm_3_parameters.csv
reports/tables/step2_hmm_3_transition.csv
reports/tables/step3_model_comparison.csv
reports/tables/step3_state_asset_statistics.csv
reports/tables/step4_allocation_mapping.csv
```

## Figures

```text
reports/figures/step2_markov_vix_states.png
reports/figures/step2_hmm_vix_states.png
reports/figures/step2_hmm_smoothed_probabilities.png
reports/figures/step3_state_asset_statistics.png
```

## Generated metadata

```text
reports/generated/steps_2_4_manifest.json
reports/generated/step3_selected_model.json
```

The manifest must contain the exact relative paths of every canonical Step 2-4 table and figure.

---

# PR backlog

## PR-06 — Add Step 2-4 modeling dependencies and fixed configuration

**Agent lane:** setup / sequential

**Dependencies:** Step 1 repository scaffold only

**Files owned:**

```text
pyproject.toml
src/vix_regime_allocation/model_config.py
tests/test_model_config.py
README.md
```

### Tasks

- [ ] T06.1 Add runtime dependencies `scipy>=1.13` and `hmmlearn>=0.3.3`.
- [ ] T06.2 Add development/runtime support needed later for notebook and sidecars: `nbformat>=5.10`, `nbclient>=0.10`, `reportlab>=4.2`, and `pypdf>=5.0`.
- [ ] T06.3 Create `model_config.py` with `SUPPORTED_STATE_COUNTS = (2, 3)`.
- [ ] T06.4 Add the exact HMM seeds `[42, 43, 44, 45, 46]` and fixed HMM numerical settings from this backlog.
- [ ] T06.5 Add tests asserting the fixed constants exactly.
- [ ] T06.6 Update README dependency/status text without claiming Step 2-4 results exist.

### Acceptance criteria

- [ ] AC06.1 (`T06.1`) `pyproject.toml` contains `scipy>=1.13` and `hmmlearn>=0.3.3`.
- [ ] AC06.2 (`T06.2`) the four notebook/report support dependencies are declared.
- [ ] AC06.3 (`T06.3`) `SUPPORTED_STATE_COUNTS` is exactly `(2, 3)`.
- [ ] AC06.4 (`T06.4`) HMM seeds and numerical settings exactly match this backlog.
- [ ] AC06.5 (`T06.5`) tests fail if any fixed setting changes.
- [ ] AC06.6 (`T06.6`) README remains factually current and does not present uncomputed results.

---

## PR-07 — Implement Markov quantile-state discretization

**Agent lane:** A

**Dependencies:** PR-06

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

Returned threshold table columns must be:

```text
state
lower_bound
upper_bound
```

### Tasks

- [ ] T07.1 Validate `n_states` is exactly 2 or 3.
- [ ] T07.2 Reject missing/non-finite observations with a clear `ValueError`.
- [ ] T07.3 Use the exact quantile edges defined in this backlog.
- [ ] T07.4 Raise `ValueError` if required quantile boundaries are duplicated.
- [ ] T07.5 Return integer states ordered from lowest to highest `VIX_change`.
- [ ] T07.6 Return the exact threshold table schema.
- [ ] T07.7 Preserve the original input index and state-series name `state`.
- [ ] T07.8 Add deterministic tests for 2-state, 3-state, invalid-state-count and duplicate-boundary cases.

### Acceptance criteria

- [ ] AC07.1 (`T07.1`) unsupported state counts raise `ValueError`.
- [ ] AC07.2 (`T07.2`) NaN/inf input raises `ValueError`.
- [ ] AC07.3 (`T07.3`) tests prove the exact 50% and tercile quantile rules are used.
- [ ] AC07.4 (`T07.4`) duplicated required boundaries are never silently merged.
- [ ] AC07.5 (`T07.5`) returned states are exactly `0..K-1` in increasing-value order.
- [ ] AC07.6 (`T07.6`) threshold columns are exactly `state, lower_bound, upper_bound`.
- [ ] AC07.7 (`T07.7`) index and series name are preserved exactly.
- [ ] AC07.8 (`T07.8`) all focused tests pass offline.

---

## PR-08 — Implement Markov transition matrix and stationary distribution

**Agent lane:** A

**Dependencies:** PR-07

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

Matrix index/columns must be integer states `0..K-1`.

Stationary series name must be `stationary_probability`.

### Tasks

- [ ] T08.1 Count only consecutive observed transitions `state_t -> state_{t+1}`.
- [ ] T08.2 Normalize every transition row by that row's outgoing transition count.
- [ ] T08.3 Add no pseudocounts or smoothing.
- [ ] T08.4 Raise `ValueError` if any expected state has zero outgoing transitions.
- [ ] T08.5 Solve for a nonnegative normalized stationary distribution.
- [ ] T08.6 Validate `pi @ P = pi` within `1e-10`.
- [ ] T08.7 Add exact tests using a hand-constructed state sequence with a known transition matrix and stationary distribution.

### Acceptance criteria

- [ ] AC08.1 (`T08.1`) transition counts match a manually enumerated sequence exactly.
- [ ] AC08.2 (`T08.2`) every returned row sums to 1.
- [ ] AC08.3 (`T08.3`) zero observed transitions remain zero.
- [ ] AC08.4 (`T08.4`) zero-outgoing states raise `ValueError`.
- [ ] AC08.5 (`T08.5`) stationary probabilities are finite, nonnegative and sum to 1.
- [ ] AC08.6 (`T08.6`) the stationary equation passes at the fixed tolerance.
- [ ] AC08.7 (`T08.7`) focused tests pass offline.

---

## PR-09 — Implement Gaussian HMM fitting with deterministic state ordering

**Agent lane:** B

**Dependencies:** PR-06

**Files owned:**

```text
src/vix_regime_allocation/hmm_model.py
tests/test_hmm_model.py
```

### Public result type

Create a frozen dataclass:

```python
@dataclass(frozen=True)
class HMMFitResult:
    n_states: int
    log_likelihood: float
    converged: bool
    means: numpy.ndarray
    variances: numpy.ndarray
    start_probabilities: numpy.ndarray
    transition_matrix: numpy.ndarray
    viterbi_states: pandas.Series
    posterior_probabilities: pandas.DataFrame
```

Posterior columns must be exactly:

```text
state_0
state_1
state_2
```

as applicable.

### Public interface

```python
def fit_gaussian_hmm(vix_change: pandas.Series, n_states: int) -> HMMFitResult:
    ...
```

### Tasks

- [ ] T09.1 Validate state count and finite input.
- [ ] T09.2 Fit one univariate diagonal-covariance Gaussian HMM for each fixed restart seed.
- [ ] T09.3 Use exactly the fixed iteration, tolerance and minimum-covariance values.
- [ ] T09.4 Discard non-converged restarts.
- [ ] T09.5 Select the converged restart with maximum log-likelihood.
- [ ] T09.6 Raise `RuntimeError` if no restart converges.
- [ ] T09.7 Relabel all state-dependent outputs by increasing fitted mean `VIX_change`.
- [ ] T09.8 Return Viterbi states on the original index.
- [ ] T09.9 Return posterior probabilities on the original index with rows summing to 1.
- [ ] T09.10 Add deterministic synthetic tests covering 2-state and 3-state shapes, state ordering and probability normalization.

### Acceptance criteria

- [ ] AC09.1 (`T09.1`) invalid state counts and non-finite input fail clearly.
- [ ] AC09.2 (`T09.2`) the implementation uses exactly five fixed restart seeds.
- [ ] AC09.3 (`T09.3`) HMM numerical settings equal the backlog contract.
- [ ] AC09.4 (`T09.4`) non-converged restarts cannot be selected.
- [ ] AC09.5 (`T09.5`) selected likelihood equals the maximum among converged restarts.
- [ ] AC09.6 (`T09.6`) all-non-converged behavior raises `RuntimeError`.
- [ ] AC09.7 (`T09.7`) returned means are strictly nondecreasing and every dependent state output is reordered consistently.
- [ ] AC09.8 (`T09.8`) Viterbi index equals the input index.
- [ ] AC09.9 (`T09.9`) posterior rows are finite and sum to 1 within `1e-8`.
- [ ] AC09.10 (`T09.10`) focused tests pass offline.

---

## PR-10 — Add common likelihood-information-criteria helpers

**Agent lane:** B

**Dependencies:** PR-06

**Files owned:**

```text
src/vix_regime_allocation/information_criteria.py
tests/test_information_criteria.py
```

### Public interfaces

```python
def aic(log_likelihood: float, n_parameters: int) -> float:
    ...


def bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    ...


def markov_parameter_count(n_states: int) -> int:
    ...


def gaussian_hmm_parameter_count(n_states: int) -> int:
    ...
```

### Tasks

- [ ] T10.1 Implement the exact AIC formula.
- [ ] T10.2 Implement the exact BIC formula.
- [ ] T10.3 Implement `K*(K-1)` Markov free-parameter count.
- [ ] T10.4 Implement `K^2 + 2*K - 1` HMM free-parameter count.
- [ ] T10.5 Reject invalid nonpositive observation counts and unsupported state counts.
- [ ] T10.6 Add exact numerical unit tests.

### Acceptance criteria

- [ ] AC10.1 (`T10.1`) AIC matches hand-calculated examples exactly within floating tolerance.
- [ ] AC10.2 (`T10.2`) BIC matches hand-calculated examples exactly within floating tolerance.
- [ ] AC10.3 (`T10.3`) Markov parameter counts are 2 for K=2 and 6 for K=3.
- [ ] AC10.4 (`T10.4`) HMM parameter counts are 7 for K=2 and 14 for K=3.
- [ ] AC10.5 (`T10.5`) invalid counts fail clearly.
- [ ] AC10.6 (`T10.6`) tests pass offline.

---

## PR-11 — Add Markov-chain likelihood and candidate result builder

**Agent lane:** A

**Dependencies:** PR-08, PR-10

**Files owned:**

```text
src/vix_regime_allocation/markov_evaluation.py
tests/test_markov_evaluation.py
```

### Public interfaces

```python
def markov_log_likelihood(states: pandas.Series, transition: pandas.DataFrame) -> float:
    ...


def evaluate_markov_candidate(
    vix_change: pandas.Series,
    n_states: int,
) -> dict[str, object]:
    ...
```

Candidate dictionary keys must be exactly:

```text
family
n_states
log_likelihood
n_parameters
n_observations
aic
bic
states
thresholds
transition
stationary
```

### Tasks

- [ ] T11.1 Compute state-sequence log-likelihood from observed transitions only.
- [ ] T11.2 Treat any observed transition with estimated probability zero as invalid and raise `ValueError`.
- [ ] T11.3 Build a complete 2- or 3-state Markov candidate using PR-07/08/10 functions.
- [ ] T11.4 Use `n_observations = number of observed transitions` for the Markov likelihood criteria.
- [ ] T11.5 Return exactly the fixed candidate keys.
- [ ] T11.6 Add tests with a hand-computable state path.

### Acceptance criteria

- [ ] AC11.1 (`T11.1`) log-likelihood equals the manual sum of `log(P_ij)` over the path.
- [ ] AC11.2 (`T11.2`) impossible observed transitions fail clearly.
- [ ] AC11.3 (`T11.3`) both 2-state and 3-state candidate builders call the shared components rather than duplicating them.
- [ ] AC11.4 (`T11.4`) Markov `n_observations` equals `len(states)-1`.
- [ ] AC11.5 (`T11.5`) returned keys match the contract exactly.
- [ ] AC11.6 (`T11.6`) tests pass offline.

---

## PR-12 — Add HMM candidate evaluation table helpers

**Agent lane:** B

**Dependencies:** PR-09, PR-10

**Files owned:**

```text
src/vix_regime_allocation/hmm_evaluation.py
tests/test_hmm_evaluation.py
```

### Public interface

```python
def evaluate_hmm_candidate(vix_change: pandas.Series, n_states: int) -> dict[str, object]:
    ...
```

Candidate keys must be exactly:

```text
family
n_states
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

- [ ] T12.1 Call `fit_gaussian_hmm()` exactly once per candidate evaluation.
- [ ] T12.2 Use the fixed HMM free-parameter formula from PR-10.
- [ ] T12.3 Use `n_observations = len(vix_change)`.
- [ ] T12.4 Compute AIC and BIC from the fitted log-likelihood.
- [ ] T12.5 Return exactly the fixed candidate keys.
- [ ] T12.6 Add mocked tests that verify the evaluation math without relying on stochastic fitting.

### Acceptance criteria

- [ ] AC12.1 (`T12.1`) evaluation delegates estimation to PR-09.
- [ ] AC12.2 (`T12.2`) parameter count is 7 for 2 states and 14 for 3 states.
- [ ] AC12.3 (`T12.3`) observation count equals the continuous input length.
- [ ] AC12.4 (`T12.4`) AIC/BIC equal shared helper results.
- [ ] AC12.5 (`T12.5`) returned keys match exactly.
- [ ] AC12.6 (`T12.6`) deterministic tests pass offline.

---

## PR-13 — Add Step 2 Markov visualization functions

**Agent lane:** A

**Dependencies:** PR-11

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
) -> None:
    ...
```

### Tasks

- [ ] T13.1 Create one figure containing a 2-state panel and a 3-state panel.
- [ ] T13.2 Plot the observed VIX level against date in both panels.
- [ ] T13.3 Color observations by the aligned discrete Markov state.
- [ ] T13.4 Include titles identifying the state count, axis labels and a state legend.
- [ ] T13.5 Create missing parent directories, save a non-empty PNG and close the figure.
- [ ] T13.6 Add synthetic/offline tests.

### Acceptance criteria

- [ ] AC13.1 (`T13.1`) the image contains both candidate state counts.
- [ ] AC13.2 (`T13.2`) the y-data are VIX levels, not VIX changes.
- [ ] AC13.3 (`T13.3`) all plotted state assignments come from the supplied state series.
- [ ] AC13.4 (`T13.4`) both panels have clear title/axes/legend.
- [ ] AC13.5 (`T13.5`) requested path exists and is non-empty after the call, and no figure remains open.
- [ ] AC13.6 (`T13.6`) tests pass offline.

---

## PR-14 — Add Step 2 HMM visualization functions

**Agent lane:** B

**Dependencies:** PR-12

**Files owned:**

```text
src/vix_regime_allocation/hmm_plots.py
tests/test_hmm_plots.py
```

### Public interfaces

```python
def plot_hmm_vix_states(
    vix: pandas.Series,
    states_2: pandas.Series,
    states_3: pandas.Series,
    output_path: pathlib.Path,
) -> None:
    ...


def plot_hmm_smoothed_probabilities(
    probabilities_2: pandas.DataFrame,
    probabilities_3: pandas.DataFrame,
    output_path: pathlib.Path,
) -> None:
    ...
```

### Tasks

- [ ] T14.1 Create a two-panel 2-state/3-state VIX-level color-coded HMM state figure.
- [ ] T14.2 Add titles, axes and state legends.
- [ ] T14.3 Create a two-panel posterior/smoothed-probability figure for both HMM candidates.
- [ ] T14.4 Show every posterior state probability and label each state.
- [ ] T14.5 Use probability y-axis bounds `[0, 1]`.
- [ ] T14.6 Save and close both figures and create missing directories.
- [ ] T14.7 Add deterministic synthetic tests.

### Acceptance criteria

- [ ] AC14.1 (`T14.1`) both HMM candidate state paths are visible against VIX level.
- [ ] AC14.2 (`T14.2`) titles/axes/legends are non-empty.
- [ ] AC14.3 (`T14.3`) posterior probabilities for both candidate state counts are shown.
- [ ] AC14.4 (`T14.4`) no posterior state column is omitted.
- [ ] AC14.5 (`T14.5`) probability axes are bounded from 0 to 1.
- [ ] AC14.6 (`T14.6`) both output images are non-empty and figures are closed.
- [ ] AC14.7 (`T14.7`) tests pass offline.

---

## PR-15 — Implement Step 3 model-comparison and preferred-model selection

**Agent lane:** A

**Dependencies:** PR-11, PR-12

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
) -> pandas.DataFrame:
    ...


def select_preferred_model(
    comparison: pandas.DataFrame,
    markov_candidates: list[dict[str, object]],
    hmm_candidates: list[dict[str, object]],
) -> dict[str, object]:
    ...
```

Comparison columns must be exactly:

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

`criterion_scope` values must be:

```text
within_markov_family
within_hmm_family
```

Selected-model dictionary keys must be exactly:

```text
family
n_states
states
selection_reason
markov_best_n_states
hmm_best_n_states
```

### Tasks

- [ ] T15.1 Build one four-row table for MC2, MC3, HMM2 and HMM3.
- [ ] T15.2 Mark information-criterion scope explicitly by family.
- [ ] T15.3 Select the minimum-BIC state count independently inside each family.
- [ ] T15.4 Apply exactly the HMM validity checks specified in this backlog.
- [ ] T15.5 Prefer valid HMM; otherwise select best-BIC Markov candidate.
- [ ] T15.6 Return the exact selected-model keys and a human-readable deterministic reason.
- [ ] T15.7 Add tests for valid-HMM selection and every HMM-fallback condition.

### Acceptance criteria

- [ ] AC15.1 (`T15.1`) comparison contains exactly four candidate rows and fixed columns.
- [ ] AC15.2 (`T15.2`) no code ranks MC AIC/BIC numerically against HMM AIC/BIC.
- [ ] AC15.3 (`T15.3`) family-specific state-count selection uses minimum BIC exactly.
- [ ] AC15.4 (`T15.4`) all six HMM validity checks are implemented.
- [ ] AC15.5 (`T15.5`) preferred family follows the exact rule.
- [ ] AC15.6 (`T15.6`) selected result has exact keys and non-empty reason.
- [ ] AC15.7 (`T15.7`) tests cover HMM success and every fallback branch.

---

## PR-16 — Implement Step 3 state-conditional ETF statistics and visualization

**Agent lane:** B

**Dependencies:** Step 1 data contract only; may be developed in parallel with PR-15

**Files owned:**

```text
src/vix_regime_allocation/state_statistics.py
tests/test_state_statistics.py
```

### Public interfaces

```python
def compute_state_asset_statistics(
    data: pandas.DataFrame,
    states: pandas.Series,
) -> pandas.DataFrame:
    ...


def plot_state_asset_statistics(
    statistics: pandas.DataFrame,
    output_path: pathlib.Path,
) -> None:
    ...
```

Statistics columns must be exactly:

```text
state
asset
mean_log_return
std_log_return
observations
```

### Tasks

- [ ] T16.1 Align states and Step 1 data by index and reject missing alignment.
- [ ] T16.2 Compute daily mean log return by state for TLT, GLD and SPY.
- [ ] T16.3 Compute sample standard deviation (`ddof=1`) by state for each ETF.
- [ ] T16.4 Count observations by state/asset.
- [ ] T16.5 Return the exact tidy schema sorted by state then fixed asset order.
- [ ] T16.6 Create one figure with separate mean-return and standard-deviation panels.
- [ ] T16.7 Add titles, axis labels and legends; save and close the figure.
- [ ] T16.8 Add hand-computable deterministic tests.

### Acceptance criteria

- [ ] AC16.1 (`T16.1`) misaligned/missing state dates fail clearly.
- [ ] AC16.2 (`T16.2`) all state/asset means equal hand calculations.
- [ ] AC16.3 (`T16.3`) standard deviations use `ddof=1` and equal hand calculations.
- [ ] AC16.4 (`T16.4`) observation counts are exact.
- [ ] AC16.5 (`T16.5`) schema/order match the contract.
- [ ] AC16.6 (`T16.6`) one figure shows both required statistics.
- [ ] AC16.7 (`T16.7`) labels are non-empty, file is non-empty and figure is closed.
- [ ] AC16.8 (`T16.8`) tests pass offline.

---

## PR-17 — Implement Step 4 deterministic state-to-allocation mapping

**Agent lane:** B

**Dependencies:** PR-16

**Files owned:**

```text
src/vix_regime_allocation/allocation.py
tests/test_allocation.py
```

### Public interface

```python
def build_state_allocation(statistics: pandas.DataFrame) -> pandas.DataFrame:
    ...
```

Output columns must be exactly:

```text
state
selected_asset
selection_mean_log_return
TLT_weight
GLD_weight
SPY_weight
```

### Tasks

- [ ] T17.1 Validate that every state contains exactly one statistics row for TLT, GLD and SPY.
- [ ] T17.2 Select the asset with maximum `mean_log_return` in each state.
- [ ] T17.3 Apply the fixed TLT -> GLD -> SPY exact-tie rule.
- [ ] T17.4 Set selected-asset weight to 1.0 and all others to 0.0.
- [ ] T17.5 Return exact schema sorted by state.
- [ ] T17.6 Add tests for each winning asset and exact ties.

### Acceptance criteria

- [ ] AC17.1 (`T17.1`) incomplete/duplicate state-asset inputs fail clearly.
- [ ] AC17.2 (`T17.2`) winner equals the largest state-conditional mean.
- [ ] AC17.3 (`T17.3`) exact ties follow TLT -> GLD -> SPY deterministically.
- [ ] AC17.4 (`T17.4`) every weight row sums exactly to 1 and contains only 0/1 weights.
- [ ] AC17.5 (`T17.5`) columns and ordering match exactly.
- [ ] AC17.6 (`T17.6`) tests pass offline.

---

## PR-18 — Build and execute the canonical notebook for complete Steps 2-4

**Agent lane:** A after implementation PRs are merged

**Dependencies:** PR-07 through PR-17 and completed Step 1 notebook/data

**Files owned:**

```text
notebooks/gwp2_vix_regime_allocation.ipynb
reports/tables/*step2*
reports/tables/*step3*
reports/tables/*step4*
reports/figures/step2_*.png
reports/figures/step3_*.png
reports/generated/steps_2_4_manifest.json
reports/generated/step3_selected_model.json
```

No other PR may edit the canonical notebook while PR-18 is open.

### Required notebook structure

```text
Step 2: Modeling VIX Regimes
  2.1 Observation definition and assumptions
  2.2 Discrete Markov chain - 2 states
  2.3 Discrete Markov chain - 3 states
  2.4 Gaussian HMM - 2 states
  2.5 Gaussian HMM - 3 states
  2.6 Regime visual comparison

Step 3: State Selection and Interpretation
  3.1 Log-likelihood, AIC and BIC equations
  3.2 Candidate comparison table
  3.3 Within-family state-count selection
  3.4 Preferred method selection and scientific rationale
  3.5 State-conditional ETF return statistics
  3.6 State-conditional return visualization
  3.7 Interpretation and limitations

Step 4: Designing the Rotation Strategy
  4.1 Decision-rule equation
  4.2 State-to-allocation table
  4.3 Economic interpretation
  4.4 Lookahead/execution-lag note for Step 5
```

### Tasks

- [ ] T18.1 Load the frozen Step 1 clean data and show shape/date range/missing-value validation.
- [ ] T18.2 Show the observation equation `X_t = VIX_change_t` and explain why change rather than level is modeled in this assignment implementation.
- [ ] T18.3 For Markov 2-state model, display quantile thresholds, transition matrix and stationary distribution.
- [ ] T18.4 For Markov 3-state model, display quantile thresholds, transition matrix and stationary distribution.
- [ ] T18.5 Show the Markov transition-probability and stationary-distribution equations with Greek-letter pronunciation where required.
- [ ] T18.6 For HMM 2-state model, display fitted mean, variance, initial probability and transition matrix.
- [ ] T18.7 For HMM 3-state model, display fitted mean, variance, initial probability and transition matrix.
- [ ] T18.8 Show the Gaussian HMM emission equation and explain EM estimation, Viterbi state decoding and posterior probabilities without exposing library implementation detail as theory.
- [ ] T18.9 Display both Step 2 state-colored VIX figures and the HMM posterior-probability figure.
- [ ] T18.10 Show log-likelihood, AIC and BIC equations and explicit parameter counts for every candidate.
- [ ] T18.11 Display the four-row model comparison table.
- [ ] T18.12 Explain explicitly why AIC/BIC are compared only within model family.
- [ ] T18.13 Display the preferred model family/state count and exact selection reason returned by project code.
- [ ] T18.14 Compute and display the preferred-state ETF mean/std/count table.
- [ ] T18.15 Display the state-conditional statistics figure.
- [ ] T18.16 Interpret each selected state from observed/fitted VIX behavior and ETF behavior; do not assign unsupported causal meaning.
- [ ] T18.17 Show the Step 4 `argmax` allocation equation and exact 100% rule.
- [ ] T18.18 Display the complete state-to-allocation mapping table.
- [ ] T18.19 Explain the economic rationale for every mapping using the displayed state-conditional means.
- [ ] T18.20 State explicitly that Step 5 must shift the selected state-driven position by one trading day; do not implement Step 5.
- [ ] T18.21 Write all canonical tables/figures to the exact paths in this backlog.
- [ ] T18.22 Write selected-model JSON and the complete Step 2-4 manifest.
- [ ] T18.23 Add scientific-paper-style assumptions/limitations, including in-sample state-conditioned mapping and model-selection caveats.
- [ ] T18.24 Include in-text citations and a bibliography section; do not invent a source not actually consulted by the team.
- [ ] T18.25 Execute every notebook cell successfully and store outputs in the committed notebook.

### Acceptance criteria

- [ ] AC18.1 (`T18.1`) notebook visibly validates Step 1 input before modeling.
- [ ] AC18.2 (`T18.2`) the modeled observation is visibly and exclusively `VIX_change`.
- [ ] AC18.3 (`T18.3`) all three required MC2 outputs are visible and saved.
- [ ] AC18.4 (`T18.4`) all three required MC3 outputs are visible and saved.
- [ ] AC18.5 (`T18.5`) Markov equations are correct and Greek letters are identified/pronounced first.
- [ ] AC18.6 (`T18.6`) all required HMM2 parameters are visible and saved.
- [ ] AC18.7 (`T18.7`) all required HMM3 parameters are visible and saved.
- [ ] AC18.8 (`T18.8`) HMM theory distinguishes emissions, EM fitting, Viterbi states and posterior probabilities precisely.
- [ ] AC18.9 (`T18.9`) all three Step 2 figures are visible in notebook and exist at canonical paths.
- [ ] AC18.10 (`T18.10`) information-criterion equations/parameter counts are visible and numerically consistent with code.
- [ ] AC18.11 (`T18.11`) comparison table contains exactly four rows and canonical columns.
- [ ] AC18.12 (`T18.12`) cross-family AIC/BIC non-comparability is explicitly stated.
- [ ] AC18.13 (`T18.13`) selected family/state count exactly matches `select_preferred_model()` output.
- [ ] AC18.14 (`T18.14`) statistics table contains every preferred state x ETF combination.
- [ ] AC18.15 (`T18.15`) statistics figure is visible and saved.
- [ ] AC18.16 (`T18.16`) state interpretation is tied to displayed evidence rather than hard-coded labels.
- [ ] AC18.17 (`T18.17`) Step 4 decision equation and 100% rule are explicit.
- [ ] AC18.18 (`T18.18`) allocation table has one row per preferred state and weights sum to 1.
- [ ] AC18.19 (`T18.19`) each allocation choice is justified by the corresponding displayed maximum mean.
- [ ] AC18.20 (`T18.20`) Step 5 lag is noted but no Step 5 backtest exists.
- [ ] AC18.21 (`T18.21`) every canonical table/figure exists and is non-empty.
- [ ] AC18.22 (`T18.22`) both generated JSON files are valid and complete.
- [ ] AC18.23 (`T18.23`) limitations include in-sample mapping and IC-scope caveats.
- [ ] AC18.24 (`T18.24`) notebook contains citations plus bibliography and no fabricated source.
- [ ] AC18.25 (`T18.25`) notebook has no failed/unexecuted Step 2-4 cells and stored outputs are present.

---

## PR-19 — Synchronize README sidecar with executed notebook outputs

**Agent lane:** B

**Dependencies:** PR-18

**Files owned:**

```text
README.md
scripts/sync_readme_analysis.py
scripts/check_readme_sidecar.py
tests/test_sync_readme_analysis.py
```

### Fixed generated README markers

```text
<!-- BEGIN NOTEBOOK ANALYSIS OUTPUT -->
<!-- END NOTEBOOK ANALYSIS OUTPUT -->
```

### Tasks

- [ ] T19.1 Add a generated analysis section between the fixed markers.
- [ ] T19.2 Populate the section from canonical Step 2-4 tables/figures and selected-model JSON; do not refit any model.
- [ ] T19.3 Show the same equations and methodological cautions as the notebook in concise form.
- [ ] T19.4 Embed/link all four canonical figures.
- [ ] T19.5 Render the model comparison, preferred-model statement, state statistics and allocation mapping from canonical files.
- [ ] T19.6 Update repository status/layout to reflect actual implemented files.
- [ ] T19.7 Extend the README sidecar checker so required notebook/report/backlog paths and generated markers cannot silently disappear.
- [ ] T19.8 Add deterministic tests for README generation from fixture artifacts.

### Acceptance criteria

- [ ] AC19.1 (`T19.1`) exactly one generated analysis block exists.
- [ ] AC19.2 (`T19.2`) synchronization performs zero model fitting and reads only canonical output files.
- [ ] AC19.3 (`T19.3`) equations/caveats are consistent with notebook definitions.
- [ ] AC19.4 (`T19.4`) README references all four canonical figures.
- [ ] AC19.5 (`T19.5`) numerical tables/model choice/allocation equal canonical files exactly.
- [ ] AC19.6 (`T19.6`) repository status/layout descriptions are factually current.
- [ ] AC19.7 (`T19.7`) sidecar checker fails when required markers/paths are removed.
- [ ] AC19.8 (`T19.8`) tests pass offline.

---

## PR-20 — Generate synchronized PDF report from the populated template

**Agent lane:** A

**Dependencies:** PR-18

**Files owned:**

```text
reports/Stochastic_Modeling_GWP2_Report.pdf
reports/Stochastic_Modeling_GWP2_Report.md
scripts/build_pdf_report.py
tests/test_build_pdf_report.py
```

### Tasks

- [ ] T20.1 Use the populated template PDF as the report cover source.
- [ ] T20.2 Copy only template page 1 into the final report; never include template page 2.
- [ ] T20.3 Build a no-code report body covering Steps 1-4, with complete Steps 2-4 equations, outputs and scientific interpretation.
- [ ] T20.4 Read numerical tables, selected model and allocation only from canonical files produced by PR-18.
- [ ] T20.5 Embed the exact same four canonical Step 2-4 figures used by notebook/README.
- [ ] T20.6 Keep the three team names already present on the cover.
- [ ] T20.7 Include citations and bibliography consistent with the notebook sources actually used.
- [ ] T20.8 Generate the final fixed-path PDF.
- [ ] T20.9 Render the final PDF to images and verify no clipped text, overlap, broken glyphs, blank figures or accidental instruction page.
- [ ] T20.10 Add tests verifying page-1 template usage, exclusion of page 2, existence of required report text and successful non-empty PDF creation.

### Acceptance criteria

- [ ] AC20.1 (`T20.1`) final first page derives from the populated project template.
- [ ] AC20.2 (`T20.2`) template instruction page is absent from the final report.
- [ ] AC20.3 (`T20.3`) report body contains no source-code listings and covers all Step 2-4 required outputs.
- [ ] AC20.4 (`T20.4`) report performs no independent estimation and numerical values equal canonical artifacts.
- [ ] AC20.5 (`T20.5`) all four canonical figures are embedded.
- [ ] AC20.6 (`T20.6`) cover visibly contains Umuhoza Denyse Graine, Opeyemi Waliyilah Oladipupo and Sergej Schweizer.
- [ ] AC20.7 (`T20.7`) citations/bibliography are present and consistent with notebook sources.
- [ ] AC20.8 (`T20.8`) `reports/Stochastic_Modeling_GWP2_Report.pdf` exists and is non-empty.
- [ ] AC20.9 (`T20.9`) render verification finds no visual defects and no template instruction page.
- [ ] AC20.10 (`T20.10`) automated tests pass offline.

---

## PR-21 — Add notebook/README/PDF sidecar parity quality gate

**Agent lane:** B after PR-19 and PR-20

**Dependencies:** PR-19, PR-20

**Files owned:**

```text
scripts/check_analysis_sidecars.py
tests/test_analysis_sidecars.py
.github/workflows/quality-gates.yml
README.md
```

### Tasks

- [ ] T21.1 Add a checker that loads `steps_2_4_manifest.json` and validates every canonical output exists.
- [ ] T21.2 Validate notebook contains references/displays for every manifest artifact.
- [ ] T21.3 Validate README generated block references every manifest figure and required table/model/allocation outputs.
- [ ] T21.4 Validate final PDF exists and its extracted text contains the selected model and all state allocation rows.
- [ ] T21.5 Add a `analysis-sidecars` CI job.
- [ ] T21.6 Make aggregate `quality-gate` depend on `analysis-sidecars`.
- [ ] T21.7 Keep lint, type, unit and integration jobs parallel and keep combined coverage threshold at 90%.
- [ ] T21.8 Document the new gate in README.
- [ ] T21.9 Add deterministic failure-mode tests for missing/stale artifacts.

### Acceptance criteria

- [ ] AC21.1 (`T21.1`) missing canonical output causes checker failure.
- [ ] AC21.2 (`T21.2`) notebook omission of a required artifact causes checker failure.
- [ ] AC21.3 (`T21.3`) README omission of a required output causes checker failure.
- [ ] AC21.4 (`T21.4`) missing/stale PDF selected-model or allocation content causes checker failure.
- [ ] AC21.5 (`T21.5`) workflow contains an independent `analysis-sidecars` job.
- [ ] AC21.6 (`T21.6`) `quality-gate` cannot pass unless `analysis-sidecars` succeeds.
- [ ] AC21.7 (`T21.7`) lint/type/unit/integration remain independent parallel jobs and coverage remains `>=90%`.
- [ ] AC21.8 (`T21.8`) README accurately documents the gate.
- [ ] AC21.9 (`T21.9`) failure-mode tests pass offline.

---

# Parallel execution schedule

The two weak agents must use this exact sequencing to avoid file conflicts.

```text
Wave 0 - sequential setup
PR-06

Wave 1 - parallel
Agent A: PR-07
Agent B: PR-09

Wave 2 - parallel
Agent A: PR-08
Agent B: PR-10

Wave 3 - parallel
Agent A: PR-11
Agent B: PR-12

Wave 4 - parallel
Agent A: PR-13
Agent B: PR-14

Wave 5 - parallel
Agent A: PR-15
Agent B: PR-16

Wave 6
Agent B: PR-17
Agent A: wait for PR-15/16/17 to merge

Wave 7 - canonical integration
Agent A: PR-18

Wave 8 - parallel sidecars
Agent B: PR-19 README
Agent A: PR-20 PDF

Wave 9 - final parity gate
Agent B: PR-21
```

No agent may start a PR before its listed dependencies are merged to `main`.

---

# Merge rules

For every implementation PR:

1. Rebase/update from current `main` before final validation.
2. Run the full repository quality suite.
3. Confirm every task checkbox and its matching acceptance criterion.
4. Confirm no files outside the PR-owned list were changed unless explicitly allowed by the sidecar policy.
5. Merge only after `quality-gate` succeeds.
6. Delete the feature branch after merge.

The final result of PR-21 must be merged to `main` before Steps 2-4 are considered complete.

---

# Steps 2-4 Definition of Done

Steps 2-4 are complete only when all conditions below are true.

- [ ] PR-06 through PR-21 are merged to `main`.
- [ ] Both 2-state and 3-state discrete Markov-chain models are implemented.
- [ ] Both 2-state and 3-state Gaussian HMMs are implemented with deterministic restart/state-ordering rules.
- [ ] Markov transition matrices and stationary distributions are displayed.
- [ ] HMM fitted parameters, decoded states and smoothed/posterior probabilities are displayed.
- [ ] VIX is plotted with color-coded Markov states and color-coded HMM states.
- [ ] HMM smoothed probabilities are plotted.
- [ ] Log-likelihood, AIC and BIC are computed for all four candidates.
- [ ] Information criteria are used only within model family and the notebook explains why.
- [ ] One preferred model family/state count is selected by the fixed deterministic rule.
- [ ] Mean and standard deviation of ETF log returns are computed by preferred state.
- [ ] State-conditional ETF statistics are visualized.
- [ ] One deterministic 100%-allocation ETF is selected for every preferred state.
- [ ] The state-to-allocation mapping is displayed and scientifically justified.
- [ ] Step 5 execution lag is explicitly deferred and not implemented.
- [ ] The canonical notebook is fully executed and contains equations, function outputs, tables, plots, interpretation, limitations, citations and bibliography.
- [ ] README displays the same canonical Step 2-4 outputs as the notebook.
- [ ] `reports/Stochastic_Modeling_GWP2_Report.pdf` is based on the populated template cover and displays the same canonical outputs without code.
- [ ] Template instruction page 2 is absent from the final report.
- [ ] Notebook/README/PDF parity is enforced by CI.
- [ ] Combined source coverage remains at least 90%.
- [ ] Lint, type, unit and integration jobs continue to run in parallel.
- [ ] Aggregate `quality-gate` passes on final `main`.
