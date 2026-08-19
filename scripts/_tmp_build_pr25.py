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

for cell in nb.cells:
    if cell.cell_type == "markdown" and "## Step 4" in cell.source:
        raise SystemExit("Step 4 already exists in the notebook.")

prefix = nb.cells[:works_index]

step4_markdown = nbformat.v4.new_markdown_cell(
    """## Step 4 — State-based rotation rule

The preferred Step 3 regime sequence is converted into a deterministic **state-to-ETF allocation map**. For each selected state, the strategy assigns 100% weight to the ETF with the largest historical mean daily log return in that state. The candidate assets are TLT, GLD, and SPY, and an exact equality is broken by the fixed priority **TLT → GLD → SPY**. The optional 60/40 variant is not used.

**Greek letters used in the following equations:** none.

For state $s$, let $\bar r_{a,s}$ denote the historical mean daily log return of asset $a$. The selected asset is

$$
a_s^*=\operatorname*{arg\,max}_{a\in\{TLT,GLD,SPY\}}\bar r_{a,s},
$$

with the fixed tie priority above, and the portfolio weights are

$$
w_{a,s}=\begin{cases}1,&a=a_s^*,\\0,&a\neq a_s^*.\end{cases}
$$

This is a project decision rule derived from the assignment rather than a claim that a maximum-mean, single-asset portfolio is generally optimal.

**Source note.** State-conditional ETF means: canonical Step 3 project artifact. Allocation rule, tie rule, and weights: project-team implementation of the assignment specification."""
)

step4_code = nbformat.v4.new_code_cell(
    """from vix_regime_allocation.allocation import ALLOCATION_COLUMNS, build_state_allocation

step3_statistics_path = repo_root / "reports/tables/step3_state_asset_statistics.csv"
step4_allocation_path = repo_root / "reports/tables/step4_allocation_mapping.csv"

step3_statistics = pd.read_csv(step3_statistics_path)
step4_allocation = build_state_allocation(step3_statistics)
assert tuple(step4_allocation.columns) == ALLOCATION_COLUMNS
assert np.allclose(
    step4_allocation[["TLT_weight", "GLD_weight", "SPY_weight"]].sum(axis=1),
    1.0,
)
for row in step4_allocation.itertuples(index=False):
    assert getattr(row, f"{row.selected_asset}_weight") == 1.0

step4_allocation.to_csv(step4_allocation_path, index=False)
display(Markdown("### Canonical state-to-allocation mapping"))
display(step4_allocation)"""
)

step4_interpretation = nbformat.v4.new_markdown_cell(
    """### Interpretation, practical use, and limitations

For the selected two-state Markov specification, **State 0 allocates to SPY** because SPY has the largest historical conditional mean daily log return in that state, while **State 1 allocates to TLT** because TLT has the largest conditional mean in State 1. GLD is not selected by the maximum-mean rule in either state. These are descriptive, sample-dependent decisions; they are not claims that SPY or TLT will remain the best asset in the corresponding regime in future data.

The rule deliberately concentrates the portfolio in one asset and uses conditional mean return as the selection criterion. Classical portfolio theory emphasizes that portfolio choice should also account for risk and diversification rather than expected return alone (Markowitz 77–91). Therefore this rotation map should be interpreted as the assignment's transparent regime rule, not as a mean-variance-efficient allocation.

<!-- citekey: markowitz1952portfolio -->

The mapping is also **in-sample**. The regime thresholds/model parameters were estimated using the full analysis sample, the selected state sequence is a full-sample artifact, and the state-conditional ETF means used here are computed on that same sample. Step 5 will apply the assignment-required one-observed-trading-row execution lag, but that lag does not retroactively make the model estimation or allocation mapping causal or out-of-sample. A stronger validation would require rolling or expanding estimation, decision-time-only state inference, and decision-time-only allocation estimates.

**Source note.** State/ETF winners and weights: project-team calculations from the canonical Step 3 statistics. Risk/diversification limitation: Markowitz (1952). In-sample qualification: direct consequence of the project's full-sample estimation design."""
)

manifest_markdown = nbformat.v4.new_markdown_cell(
    """### Deterministic Steps 2–4 artifact manifest

The manifest below records the frozen Step 1 input hash and every canonical Step 2–4 table and figure exactly once. It contains repository-relative POSIX paths only and no timestamp or environment-dependent field, so repeated generation from the same canonical analysis is deterministic.

**Source note.** Manifest contents and SHA-256 calculation: project team."""
)

manifest_code = nbformat.v4.new_code_cell(
    """import hashlib
import json

step1_path = repo_root / "data/processed/step1_data.csv"
selected_model_path = repo_root / "reports/generated/step3_selected_model.json"
manifest_path = repo_root / "reports/generated/steps_2_4_manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)

input_sha256 = hashlib.sha256(step1_path.read_bytes()).hexdigest()
selected_model = json.loads(selected_model_path.read_text(encoding="utf-8"))
assert selected_model["input_data_sha256"] == input_sha256
assert selected_model["selected_states_path"] == "reports/tables/step3_selected_states.csv"

tables = sorted(
    [
        "reports/tables/step2_hmm_2_parameters.csv",
        "reports/tables/step2_hmm_2_states.csv",
        "reports/tables/step2_hmm_2_transition.csv",
        "reports/tables/step2_hmm_3_parameters.csv",
        "reports/tables/step2_hmm_3_states.csv",
        "reports/tables/step2_hmm_3_transition.csv",
        "reports/tables/step2_markov_2_states.csv",
        "reports/tables/step2_markov_2_stationary.csv",
        "reports/tables/step2_markov_2_thresholds.csv",
        "reports/tables/step2_markov_2_transition.csv",
        "reports/tables/step2_markov_3_states.csv",
        "reports/tables/step2_markov_3_stationary.csv",
        "reports/tables/step2_markov_3_thresholds.csv",
        "reports/tables/step2_markov_3_transition.csv",
        "reports/tables/step3_model_comparison.csv",
        "reports/tables/step3_selected_states.csv",
        "reports/tables/step3_state_asset_statistics.csv",
        "reports/tables/step4_allocation_mapping.csv",
    ]
)
figures = sorted(
    [
        "reports/figures/step2_hmm_smoothed_probabilities.png",
        "reports/figures/step2_hmm_vix_states.png",
        "reports/figures/step2_markov_vix_states.png",
        "reports/figures/step3_state_asset_statistics.png",
    ]
)
for relative_path in tables + figures:
    assert (repo_root / relative_path).is_file(), relative_path

manifest = {
    "schema_version": 1,
    "input_data_path": "data/processed/step1_data.csv",
    "input_data_sha256": input_sha256,
    "notebook_path": "notebooks/gwp2_vix_regime_allocation.ipynb",
    "selected_model_path": "reports/generated/step3_selected_model.json",
    "tables": tables,
    "figures": figures,
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
display(Markdown("```json\\n" + json.dumps(manifest, indent=2, sort_keys=True) + "\\n```"))"""
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
            for field, value in re.findall(r'(\\w+)\\s*=\\s*\"([^\"]*)\"', body)
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
            f'{author}. \"{entry["title"]}.\" *{entry["journal"]}*, '
            f'vol. {entry["volume"]}, no. {entry["number"]}, {entry["year"]}, '
            f'pp. {pages}. doi:{entry["doi"]}.'
        )
    return (
        f'{author}. \"{entry["title"]}.\" *{entry["publisher"]}*, '
        f'{entry["year"]}, {entry["url"]}. Accessed 19 Aug. 2026.'
    )


references = _parse_bibtex_registry(references_path.read_text(encoding="utf-8"))
cited_keys = [
    "whaley2009vix",
    "cboe2019vixfaq",
    "baum1970maximization",
    "rabiner1989tutorial",
    "viterbi1967decoding",
    "akaike1974identification",
    "schwarz1978dimension",
    "markowitz1952portfolio",
]
missing_keys = [key for key in cited_keys if key not in references]
assert not missing_keys, f"Unresolved citation keys: {missing_keys}"
display(Markdown(chr(10).join(f"- {_mla_entry(references[key])}" for key in cited_keys)))"""
)

nb.cells = prefix + [
    step4_markdown,
    step4_code,
    step4_interpretation,
    manifest_markdown,
    manifest_code,
    works_heading,
    works_code,
]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
