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

stats_markdown = nbformat.v4.new_markdown_cell(
    """## Step 3 — State-conditional ETF analysis

After the deterministic model-selection stage, the project measures how TLT, GLD, and SPY behaved **conditional on the selected state sequence**. The analysis does not refit either regime model and does not reselect the preferred candidate. It loads the canonical `reports/tables/step3_selected_states.csv`, requires exact date alignment with the frozen Step 1 dataset, and computes each ETF's daily mean log return, sample standard deviation, and observation count within each selected state.

**Greek letters used in the following equations:** none.

For asset $a$ in state $s$, with $n_s$ observations and daily log returns $r_{a,t}$, the conditional sample mean is

$$
\bar r_{a,s}=\frac{1}{n_s}\sum_{t:S_t=s} r_{a,t},
$$

and the sample standard deviation is

$$
s_{a,s}=\sqrt{\frac{1}{n_s-1}\sum_{t:S_t=s}(r_{a,t}-\bar r_{a,s})^2}.
$$

The reported quantities remain on a **daily log-return scale**; they are not annualized. The figure shows conditional mean daily log returns as bars and uses the corresponding **sample standard deviation** as the error-bar magnitude. Those error bars describe within-state return dispersion; they are not standard errors and are not confidence intervals.

**Source note.** ETF returns and selected-state assignments: canonical project artifacts. Conditional means, sample standard deviations, counts, table, and figure: project team calculations."""
)

stats_code = nbformat.v4.new_code_cell(
    """from vix_regime_allocation.state_statistics import compute_state_asset_statistics
from vix_regime_allocation.state_statistics_plot import plot_state_asset_statistics

selected_states_path = repo_root / "reports/tables/step3_selected_states.csv"
statistics_path = repo_root / "reports/tables/step3_state_asset_statistics.csv"
statistics_figure_path = repo_root / "reports/figures/step3_state_asset_statistics.png"

selected_state_frame = pd.read_csv(
    selected_states_path, parse_dates=["Date"], index_col="Date"
)
selected_state_frame.index.name = "Date"
selected_states_for_statistics = selected_state_frame["state"].astype("int64")
selected_states_for_statistics.name = "state"

assert selected_states_for_statistics.index.equals(data.index)
assert selected_states_for_statistics.notna().all()

state_asset_statistics = compute_state_asset_statistics(
    data, selected_states_for_statistics
)
state_asset_statistics.to_csv(statistics_path, index=False)
plot_state_asset_statistics(state_asset_statistics, statistics_figure_path)

expected_rows = 3 * int(selected_states_for_statistics.nunique())
assert len(state_asset_statistics) == expected_rows
assert state_asset_statistics["observations"].gt(0).all()
assert statistics_figure_path.is_file() and statistics_figure_path.stat().st_size > 0

display(Markdown("### State-conditional ETF return statistics"))
display(state_asset_statistics)
display(Image(filename=str(statistics_figure_path)))

state_leaders = (
    state_asset_statistics.sort_values(
        ["state", "mean_log_return", "asset"],
        ascending=[True, False, True],
        kind="stable",
    )
    .groupby("state", sort=True, as_index=False)
    .first()[["state", "asset", "mean_log_return", "std_log_return", "observations"]]
)
display(Markdown("### Highest conditional mean daily log return in each state"))
display(state_leaders)"""
)

stats_interpretation = nbformat.v4.new_markdown_cell(
    """### Economic interpretation and limitations

The conditional table separates two distinct quantities that should not be conflated. The mean log return describes the average daily ETF outcome observed when the selected state was active, while the sample standard deviation describes how dispersed those daily outcomes were inside the same state. A higher conditional mean therefore does not by itself imply a superior risk-adjusted investment.

The displayed `state_leaders` table identifies the ETF with the highest **historical conditional mean daily log return** in each selected state. It is diagnostic evidence for the allocation rule used in Step 4; it is not yet the Step 4 allocation mapping and it does not execute trades.

Because the selected states and the conditional ETF means are estimated from the same full historical sample, this analysis is in-sample. In particular, selecting the asset with the largest full-sample conditional mean can introduce optimistic selection bias if the same mapping is subsequently evaluated on the same dates. The assignment's later one-day execution lag addresses same-day execution timing but does not remove this parameter-estimation look-ahead. A stronger research design would estimate regimes and conditional means using only information available before each decision date.

The numeric state labels remain neutral. State 0 and higher-numbered states are ordered by the selected model's daily `VIX_change` state definition; they are not automatically labels for low/high VIX levels, calm/crisis markets, or causal market environments.

**Source note.** Interpretation of the computed conditional statistics and limitations of the full-sample project design: project team."""
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

nb.cells = prefix + [
    stats_markdown,
    stats_code,
    stats_interpretation,
    works_heading,
    works_code,
]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
