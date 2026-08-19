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

step2_markdown = nbformat.v4.new_markdown_cell(
    """## Step 2 - Modeling VIX Regimes: Discrete Markov Chains

Step 2 first treats the observed daily change in the VIX as a discretized state variable. The input is **only** the committed `VIX_change` series from Step 1; no live data are downloaded and no VIX-level thresholds are used. For each requested state count, the empirical distribution of daily VIX changes is divided by linear quantiles. Observations exactly equal to an interior cut are assigned to the higher-numbered state. This produces an observable finite-state Markov-chain representation whose transition probabilities are estimated from consecutive state counts. The general finite-state transition framework and likelihood treatment are standard Markov-chain constructions; Baum et al. provide a classical statistical treatment of probabilistic functions of Markov chains (Baum et al. 164-171).

<!-- citekey: baum1970maximization -->

**Greek letter used below:** π — *pi*, pronounced “pie”.

For states $i,j\in\{0,\ldots,K-1\}$, the estimated transition probability is

$$
P_{ij}=\Pr(S_{t+1}=j\mid S_t=i)=\frac{N_{ij}}{\sum_j N_{ij}},
$$

with no pseudocount or smoothing. A stationary distribution satisfies

$$
\pi=\pi P,\qquad \sum_i\pi_i=1.
$$

The project estimates both $K=2$ and $K=3$. State numbers are ordered by increasingly positive `VIX_change`: State 0 contains the lower changes, while the highest-numbered state contains the larger positive changes. These labels describe **changes in implied volatility**, not low/high VIX levels themselves.

**Source note.** Market observations and `VIX_change`: Step 1 project data derived from Yahoo Finance. Markov-chain methodology: Baum et al. (1970). Quantile discretization, calculations, tables, and figure: project team."""
)

step2_code = nbformat.v4.new_code_cell(
    """from vix_regime_allocation.markov_evaluation import evaluate_markov_candidate
from vix_regime_allocation.markov_plots import plot_markov_vix_states

markov_output_dir = repo_root / "reports/tables"
markov_figure_dir = repo_root / "reports/figures"
markov_output_dir.mkdir(parents=True, exist_ok=True)
markov_figure_dir.mkdir(parents=True, exist_ok=True)

markov_2 = evaluate_markov_candidate(data["VIX_change"], 2)
markov_3 = evaluate_markov_candidate(data["VIX_change"], 3)
markov_candidates = {2: markov_2, 3: markov_3}

for k, candidate in markov_candidates.items():
    candidate["thresholds"].to_csv(
        markov_output_dir / f"step2_markov_{k}_thresholds.csv", index=False
    )
    candidate["transition"].reset_index().to_csv(
        markov_output_dir / f"step2_markov_{k}_transition.csv", index=False
    )
    candidate["stationary"].reset_index().to_csv(
        markov_output_dir / f"step2_markov_{k}_stationary.csv", index=False
    )
    candidate["states"].rename_axis("Date").reset_index().to_csv(
        markov_output_dir / f"step2_markov_{k}_states.csv", index=False
    )

markov_figure_path = markov_figure_dir / "step2_markov_vix_states.png"
plot_markov_vix_states(
    data["VIX"], markov_2["states"], markov_3["states"], markov_figure_path
)

for k, candidate in markov_candidates.items():
    display(Markdown(f"### Markov candidate: K={k}"))
    display(Markdown("**Quantile intervals**"))
    display(candidate["thresholds"])
    display(Markdown("**Transition matrix**"))
    display(candidate["transition"])
    display(Markdown("**Stationary distribution**"))
    display(candidate["stationary"].to_frame())
    diagnostics = pd.Series(
        {
            "conditional_log_likelihood": candidate["log_likelihood"],
            "transition_observations": candidate["n_observations"],
            "free_transition_parameters": candidate["n_parameters"],
        },
        name="value",
    )
    display(diagnostics.to_frame())

assert markov_2["states"].index.equals(data.index)
assert markov_3["states"].index.equals(data.index)
Image(filename=str(markov_figure_path))"""
)

step2_interpretation = nbformat.v4.new_markdown_cell(
    """### Markov-state interpretation and limitations

The two-state specification separates below-median from at-or-above-median daily VIX changes; the three-state specification provides a lower, middle, and upper quantile partition. The transition matrices therefore answer a conditional-frequency question: given today's observed VIX-change state, how often did the next trading observation fall into each state in this sample? The stationary distributions summarize the long-run probabilities implied by the fitted transition matrices, provided the estimated chain has a unique stationary distribution.

The figure deliberately plots the **VIX level** while coloring observations by states inferred from **daily VIX changes**. A high VIX level can occur in a negative-change state when volatility is falling, and a low VIX level can occur in a positive-change state when volatility is rising. For that reason, the numerical labels are retained rather than assigning unsupported names such as “calm” or “crisis.”

This construction is descriptive and full-sample. Quantile thresholds are estimated from the full Step 1 sample, the Markov property is an imposed approximation, discretization discards within-state variation, and transition probabilities are assumed time-homogeneous over the sample. Step 3 will compare the two state counts within the Markov family rather than selecting one here.

**Source note.** Interpretation of the fitted sample objects and limitations: project team; finite-state Markov methodology: Baum et al. (1970).

<!-- citekey: baum1970maximization -->"""
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
cited_keys = ["whaley2009vix", "cboe2019vixfaq", "baum1970maximization"]
missing_keys = [key for key in cited_keys if key not in references]
assert not missing_keys, f"Unresolved citation keys: {missing_keys}"
display(Markdown(chr(10).join(f"- {_mla_entry(references[key])}" for key in cited_keys)))"""
)

nb.cells = prefix + [step2_markdown, step2_code, step2_interpretation, works_heading, works_code]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
