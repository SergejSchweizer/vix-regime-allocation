from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"

nb = nbformat.read(NOTEBOOK, as_version=4)

works_index = None
for index, cell in enumerate(nb.cells):
    if cell.cell_type == "markdown" and "## Works Cited" in cell.source:
        works_index = index
        break
if works_index is None:
    raise SystemExit("Existing Works Cited section was not found.")

prefix = nb.cells[:works_index]

hmm_markdown = nbformat.v4.new_markdown_cell(
    """## Step 2 - Modeling VIX Regimes: Gaussian Hidden Markov Models

The second Step 2 specification treats the regime as an **unobserved** finite-state process and models the observed daily `VIX_change` conditional on that latent state with a Gaussian distribution. Hidden Markov models combine a Markov transition law for the latent state with state-conditional observation distributions; the standard likelihood and forward-backward framework is reviewed by Rabiner (257-286), while Baum et al. provide the classical likelihood-maximization foundation used by the EM family of procedures (Baum et al. 164-171; Rabiner 257-286).

<!-- citekey: baum1970maximization -->
<!-- citekey: rabiner1989tutorial -->

**Greek letters used in the emission equation:** μ — *mu*, pronounced “mew”; σ — *sigma*, pronounced “SIG-muh”.

For state $k$, the univariate Gaussian emission model is

$$
X_t\mid S_t=k \sim \mathcal{N}(\mu_k,\sigma_k^2),\qquad X_t=\Delta VIX_t.
$$

The implementation fits both $K=2$ and $K=3$ using five deterministic restarts with seeds 42-46, a diagonal covariance representation, at most 500 EM iterations, convergence tolerance $10^{-6}$, and minimum covariance $10^{-6}$. Only converged finite-likelihood restarts are eligible; the highest log-likelihood fit is retained, with the lower seed breaking an effectively exact likelihood tie. After fitting, every state-dependent output is relabeled by increasing fitted mean `VIX_change`, so State 0 has the lowest fitted conditional mean and the highest-numbered state has the highest fitted conditional mean.

The most-likely full-sample state path is decoded with the Viterbi algorithm, whose dynamic-programming origin is Viterbi's decoding work; smoothed posterior state probabilities are also retained for every observation (Viterbi 260-269; Rabiner 257-286).

<!-- citekey: viterbi1967decoding -->

**Source note.** Market observations and `VIX_change`: Step 1 project data derived from Yahoo Finance. HMM/EM/smoothing methodology: Baum et al. (1970) and Rabiner (1989); most-likely path decoding: Viterbi (1967). Model fitting, state relabeling, tables, diagnostics, and figures: project team."""
)

hmm_code = nbformat.v4.new_code_cell(
    """from vix_regime_allocation.hmm_evaluation import evaluate_hmm_candidate
from vix_regime_allocation.hmm_probability_plot import plot_hmm_smoothed_probabilities
from vix_regime_allocation.hmm_state_plot import plot_hmm_vix_states

hmm_output_dir = repo_root / "reports/tables"
hmm_figure_dir = repo_root / "reports/figures"
hmm_output_dir.mkdir(parents=True, exist_ok=True)
hmm_figure_dir.mkdir(parents=True, exist_ok=True)

hmm_2 = evaluate_hmm_candidate(data["VIX_change"], 2)
hmm_3 = evaluate_hmm_candidate(data["VIX_change"], 3)
hmm_candidates = {2: hmm_2, 3: hmm_3}


def _hmm_parameter_table(fit):
    counts = fit.states.value_counts().reindex(range(fit.n_states), fill_value=0).astype(int)
    posterior_means = fit.probabilities.mean(axis=0)
    return pd.DataFrame(
        {
            "state": range(fit.n_states),
            "mean_vix_change": fit.means.to_numpy(dtype=float),
            "variance_vix_change": fit.variances.to_numpy(dtype=float),
            "start_probability": list(fit.start_probabilities),
            "viterbi_observations": counts.to_numpy(dtype=int),
            "viterbi_occupancy": counts.to_numpy(dtype=float) / len(fit.states),
            "posterior_mean_probability": [
                float(posterior_means[f"state_{state}"]) for state in range(fit.n_states)
            ],
        }
    )


for k, candidate in hmm_candidates.items():
    fit = candidate["fit"]
    parameters = _hmm_parameter_table(fit)
    parameters.to_csv(hmm_output_dir / f"step2_hmm_{k}_parameters.csv", index=False)
    fit.transition_matrix.reset_index().to_csv(
        hmm_output_dir / f"step2_hmm_{k}_transition.csv", index=False
    )
    fit.states.rename_axis("Date").reset_index().to_csv(
        hmm_output_dir / f"step2_hmm_{k}_states.csv", index=False
    )

    display(Markdown(f"### Gaussian HMM candidate: K={k}"))
    display(Markdown("**Fitted and decoded state parameters**"))
    display(parameters)
    display(Markdown("**Transition matrix**"))
    display(fit.transition_matrix)
    diagnostics = pd.Series(
        {
            "selected_restart_seed": fit.seed,
            "converged": fit.converged,
            "log_likelihood": candidate["log_likelihood"],
            "observations": candidate["n_observations"],
            "free_parameters": candidate["n_parameters"],
        },
        name="value",
    )
    display(diagnostics.to_frame())

assert hmm_2["fit"].states.index.equals(data.index)
assert hmm_3["fit"].states.index.equals(data.index)
assert hmm_2["fit"].probabilities.index.equals(data.index)
assert hmm_3["fit"].probabilities.index.equals(data.index)

hmm_state_figure_path = hmm_figure_dir / "step2_hmm_vix_states.png"
hmm_probability_figure_path = hmm_figure_dir / "step2_hmm_smoothed_probabilities.png"
plot_hmm_vix_states(
    data["VIX"], hmm_2["fit"].states, hmm_3["fit"].states, hmm_state_figure_path
)
plot_hmm_smoothed_probabilities(
    hmm_2["fit"].probabilities,
    hmm_3["fit"].probabilities,
    hmm_probability_figure_path,
)

display(Image(filename=str(hmm_state_figure_path)))
display(Image(filename=str(hmm_probability_figure_path)))"""
)

hmm_interpretation = nbformat.v4.new_markdown_cell(
    """### HMM-state interpretation and limitations

The fitted HMM states are ordered by their **conditional mean daily VIX change**, not by the contemporaneous VIX level. Consequently, a state with a positive fitted mean is evidence for a regime in which VIX changes tend to be upward on average; it is not automatically a “high-VIX,” “crisis,” or “fear” state. The parameter tables therefore retain neutral numeric labels and report both the decoded-state occupancy and the average posterior probability assigned to each state.

The VIX-state figure shows the observed VIX level colored by the Viterbi path, while the probability figure displays the smoothed posterior probabilities. A sharp switch in the decoded path should be interpreted together with the posterior probabilities: diffuse probabilities indicate state uncertainty even when the Viterbi decoder must choose a single state.

The analysis is deliberately full-sample and descriptive. EM can converge to local optima, which is why deterministic restarts are used. More importantly, both the Viterbi path and the smoothed posterior probabilities condition on the full observed sequence, so they use information from observations after date $t$ when describing date $t$. They are therefore **not causal trading-time regime signals**. Step 3 will compare $K=2$ and $K=3$ within the HMM family; it will not treat raw HMM and discretized-Markov information criteria as directly comparable likelihood objects.

**Source note.** HMM interpretation and smoothing/decoding concepts: Rabiner (1989) and Viterbi (1967); fitted-sample interpretation and stated implementation limitations: project team.

<!-- citekey: rabiner1989tutorial -->
<!-- citekey: viterbi1967decoding -->"""
)

works_heading = nbformat.v4.new_markdown_cell(
    """## Works Cited

The following MLA 9 entries are rendered programmatically from the cited keys in the canonical `reports/references.bib` registry."""
)

works_code = nbformat.v4.new_code_cell(
    """def _parse_bibtex_registry(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@(\\w+)\\{([^,]+),\\s*(.*?)\\n\\}", text, flags=re.DOTALL):
        entry_type, key, body = match.groups()
        fields = {
            field: value
            for field, value in re.findall(r'(\\w+)\\s*=\\s*"([^"]*)"', body)
        }
        fields["entry_type"] = entry_type.lower()
        entries[key] = fields
    return entries


def _mla_author(author: str) -> str:
    authors = author.split(" and ")
    return authors[0] + (", et al." if len(authors) > 2 else "")


def _mla_entry(entry: dict[str, str]) -> str:
    author = _mla_author(entry["author"])
    if entry["entry_type"] == "article":
        pages = entry["pages"].replace("--", "-")
        return (
            f'{author} "{entry["title"]}." *{entry["journal"]}*, '
            f'vol. {entry["volume"]}, no. {entry["number"]}, {entry["year"]}, '
            f'pp. {pages}. doi:{entry["doi"]}.'
        )
    return (
        f'{author}. "{entry["title"]}." *{entry["publisher"]}*, '
        f'{entry["year"]}, {entry["url"]}. Accessed 19 Aug. 2026.'
    )


references = _parse_bibtex_registry(references_path.read_text(encoding="utf-8"))
cited_keys = [
    "whaley2009vix",
    "cboe2019vixfaq",
    "baum1970maximization",
    "rabiner1989tutorial",
    "viterbi1967decoding",
]
missing_keys = [key for key in cited_keys if key not in references]
assert not missing_keys, f"Unresolved citation keys: {missing_keys}"
display(Markdown(chr(10).join(f"- {_mla_entry(references[key])}" for key in cited_keys)))"""
)

nb.cells = prefix + [hmm_markdown, hmm_code, hmm_interpretation, works_heading, works_code]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
