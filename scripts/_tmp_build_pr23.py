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

step3_markdown = nbformat.v4.new_markdown_cell(
    """## Step 3 — Model selection and selected-state provenance

Step 3 compares the two- and three-state candidates **within each model family** using log-likelihood, Akaike's Information Criterion (AIC), and the Bayesian Information Criterion (BIC). AIC and BIC penalize likelihood improvements for model complexity; Akaike introduced the information-theoretic identification criterion, while Schwarz derived the large-sample Bayesian criterion (Akaike 716-723; Schwarz 461-464).

<!-- citekey: akaike1974identification -->
<!-- citekey: schwarz1978dimension -->

**Greek letters used in the following equations:** none.

For maximized likelihood $L$, number of free parameters $k$, and number of likelihood observations $n$,

$$
AIC = 2k - 2\log L,
$$

and

$$
BIC = k\log(n) - 2\log L.
$$

The Markov candidate likelihood is the conditional likelihood of the observed transition sequence, whereas the Gaussian HMM likelihood is the observation-sequence likelihood obtained after integrating over latent states. These are different likelihood objects and therefore the raw AIC/BIC values are **not ranked across the Markov and HMM families**. The fixed project rule first chooses the lower-BIC state count within each family, using the lower state count for an effectively exact BIC tie, and then prefers the within-family HMM winner only if its convergence, variance, probability, and decoded-state occupancy diagnostics are valid. Otherwise the selected Markov candidate is used as an explicit fallback.

The selected `Date,state` series is persisted rather than re-estimated later. This provides an auditable provenance chain from the Step 2 candidate through the Step 3 selection decision and into subsequent state-conditional ETF analysis.

**Source note.** AIC: Akaike (1974). BIC: Schwarz (1978). Candidate likelihood construction, cross-family comparability restriction, validity diagnostics, deterministic fallback, and provenance controls: project team."""
)

step3_code = nbformat.v4.new_code_cell(
    """from hashlib import sha256
import json

from vix_regime_allocation.model_selection import (
    COMPARISON_COLUMNS,
    build_model_comparison,
    select_preferred_model,
)

step3_table_dir = repo_root / "reports/tables"
step3_generated_dir = repo_root / "reports/generated"
step3_table_dir.mkdir(parents=True, exist_ok=True)
step3_generated_dir.mkdir(parents=True, exist_ok=True)

markov_candidate_list = [markov_candidates[k] for k in (2, 3)]
hmm_candidate_list = [hmm_candidates[k] for k in (2, 3)]

step3_comparison = build_model_comparison(markov_candidate_list, hmm_candidate_list)
step3_selection = select_preferred_model(
    step3_comparison, markov_candidate_list, hmm_candidate_list
)

comparison_path = step3_table_dir / "step3_model_comparison.csv"
selected_states_path = step3_table_dir / "step3_selected_states.csv"
selected_model_path = step3_generated_dir / "step3_selected_model.json"

assert tuple(step3_comparison.columns) == COMPARISON_COLUMNS
assert len(step3_comparison) == 4
step3_comparison.to_csv(comparison_path, index=False)

selected_states = step3_selection["states"].copy()
selected_states.name = "state"
assert selected_states.index.equals(data.index)
assert selected_states.index.name == "Date"
assert selected_states.notna().all()

state_source_path = repo_root / str(step3_selection["state_source"])
source_states = pd.read_csv(state_source_path, parse_dates=["Date"], index_col="Date")["state"]
source_states.index.name = "Date"
source_states.name = "state"
source_states = source_states.astype("int64")
selected_states = selected_states.astype("int64")
assert source_states.equals(selected_states)

selected_states.rename_axis("Date").reset_index().to_csv(selected_states_path, index=False)
input_sha256 = sha256(data_path.read_bytes()).hexdigest()
selected_model = {
    "family": step3_selection["family"],
    "n_states": int(step3_selection["n_states"]),
    "state_source": step3_selection["state_source"],
    "selection_reason": step3_selection["selection_reason"],
    "markov_best_n_states": int(step3_selection["markov_best_n_states"]),
    "hmm_best_n_states": int(step3_selection["hmm_best_n_states"]),
    "input_data_sha256": input_sha256,
    "selected_states_path": "reports/tables/step3_selected_states.csv",
}
assert tuple(selected_model) == (
    "family",
    "n_states",
    "state_source",
    "selection_reason",
    "markov_best_n_states",
    "hmm_best_n_states",
    "input_data_sha256",
    "selected_states_path",
)
selected_model_path.write_text(json.dumps(selected_model, indent=2) + chr(10), encoding="utf-8")

display(Markdown("### Four-candidate information-criterion table"))
display(step3_comparison)
display(Markdown("### Deterministic preferred-model decision"))
display(pd.Series(selected_model, name="value").to_frame())
display(
    Markdown(
        f"**Selected model:** {selected_model['family'].upper()} with "
        f"$K={selected_model['n_states']}$. {selected_model['selection_reason']}"
    )
)"""
)

step3_interpretation = nbformat.v4.new_markdown_cell(
    """### Interpretation, provenance, and limitations

The comparison table intentionally reports all four candidates in one place for transparency, but `criterion_scope = within_family_only` prevents the table from implying that the smallest raw AIC or BIC across unlike likelihood constructions is the winner. The two state-count decisions are therefore made separately inside the Markov and HMM families before the deterministic method-validity rule is applied.

The selected state path retains the neutral numeric labels established in Step 2: State 0 has the lowest fitted/discretized daily `VIX_change` location and higher-numbered states correspond to progressively higher daily VIX-change locations. These labels describe the direction and magnitude of **changes in VIX**, not the VIX level itself; a positive-change state must not be relabeled automatically as a “crisis” or “high-VIX” state.

The selected-state CSV is checked byte-for-byte in economic content against the canonical Step 2 state-source artifact before it is written, and the metadata JSON records the SHA-256 hash of the exact Step 1 input dataset. This makes downstream Step 3/4 analysis traceable to one immutable input sample and one selected Step 2 state sequence.

If the HMM is selected, the persisted Viterbi path remains a **full-sample descriptive classification**. It uses the full observation sequence and is not a causal, filtered trading-time signal. Likewise, model parameters and model-selection statistics are estimated on the full sample. The one-day execution lag required later in Step 5 prevents same-day execution but does not by itself make this model-selection stage out-of-sample.

**Source note.** Model-selection criterion foundations: Akaike (1974) and Schwarz (1978). Interpretation of project state ordering, provenance assertions, and in-sample qualification: project team.

<!-- citekey: akaike1974identification -->
<!-- citekey: schwarz1978dimension -->"""
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
            f'{author}. "{entry["title"]}." *{entry["journal"]}*, '
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
    "akaike1974identification",
    "schwarz1978dimension",
]
missing_keys = [key for key in cited_keys if key not in references]
assert not missing_keys, f"Unresolved citation keys: {missing_keys}"
display(Markdown(chr(10).join(f"- {_mla_entry(references[key])}" for key in cited_keys)))"""
)

nb.cells = prefix + [step3_markdown, step3_code, step3_interpretation, works_heading, works_code]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
