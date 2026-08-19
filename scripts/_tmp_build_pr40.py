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
    if cell.cell_type == "markdown" and "## Step 5" in cell.source:
        raise SystemExit("Step 5 already exists in the notebook.")

prefix = nb.cells[:works_index]

step5_markdown = nbformat.v4.new_markdown_cell(
    r"""## Step 5 — Backtest construction and required benchmarks

The backtest converts the ETF daily log returns to simple returns before portfolio arithmetic and applies the Step 4 allocation with an exact **one-observed-trading-row execution lag**. Therefore the state recorded on observed row $t-1$ determines the portfolio weights applied to ETF returns on observed row $t$. The first Step 1 row is excluded because no preceding observed regime decision exists.

**Greek letters used in the following equations:** none.

For an ETF log return $l_{a,t}$, the simple return is

$$
r_{a,t}=\exp(l_{a,t})-1.
$$

For the rotation portfolio,

$$
R^{\mathrm{rot}}_t=\sum_{a\in\{TLT,GLD,SPY\}} w_{a,S_{t-1}}r_{a,t}.
$$

The benchmark comparison uses exactly the same return dates as the lagged rotation. The equal-weight benchmark starts at one-third TLT, one-third GLD, and one-third SPY, resets to those weights immediately before the first comparison return and before the first observed comparison return in each new calendar month, and then allows weights to drift within the month. The second benchmark is SPY buy-and-hold. All reported backtest returns are **gross of transaction costs and taxes**, so the implemented cost assumption is zero.

The execution lag prevents using the same row's state to trade the same row's return, but it does **not** make the analysis causal or out-of-sample. Regime thresholds/model parameters and the preferred state path were derived from the full sample; if an HMM path were preferred, its Viterbi sequence would also use the full sequence; and the state-conditional allocation means are full-sample estimates. Data-snooping and repeated backtest selection can materially overstate apparent performance, so the results below are descriptive rather than evidence of live-trading profitability (White 1097–1126; Bailey and Lopez de Prado 94–107).

<!-- citekey: white2000datasnooping -->
<!-- citekey: bailey2014deflatedsharpe -->

**Source note.** Returns, state provenance, and the Step 4 allocation are canonical project artifacts. Lag/benchmark arithmetic is implemented by the project team. The non-out-of-sample qualification is supported by White (2000) and Bailey and Lopez de Prado (2014)."""
)

step5_code = nbformat.v4.new_code_cell(
    """import hashlib
import json
from pathlib import Path

from vix_regime_allocation.backtest import ROTATION_DETAIL_COLUMNS, build_rotation_returns
from vix_regime_allocation.backtest_summary import COMPARISON_COLUMNS, build_comparison
from vix_regime_allocation.benchmarks import (
    build_equal_weight_monthly_returns,
    build_spy_buy_hold_returns,
)

repo_root = Path.cwd()
step1_path = repo_root / "data/processed/step1_data.csv"
selected_model_path = repo_root / "reports/generated/step3_selected_model.json"
allocation_path = repo_root / "reports/tables/step4_allocation_mapping.csv"
daily_returns_path = repo_root / "reports/tables/step5_daily_returns.csv"

step1 = pd.read_csv(step1_path, parse_dates=["Date"]).set_index("Date")
step1.index = pd.DatetimeIndex(step1.index, name="Date")
selected_model = json.loads(selected_model_path.read_text(encoding="utf-8"))
input_sha256 = hashlib.sha256(step1_path.read_bytes()).hexdigest()
assert selected_model["input_data_sha256"] == input_sha256

selected_states_path = repo_root / selected_model["selected_states_path"]
selected_states_frame = pd.read_csv(selected_states_path, parse_dates=["Date"])
assert selected_states_frame["Date"].tolist() == step1.index.tolist()
selected_states = pd.Series(
    selected_states_frame["state"].to_numpy(dtype=int),
    index=step1.index,
    name="state",
    dtype=int,
)
allocation = pd.read_csv(allocation_path)

rotation_detail = build_rotation_returns(step1, selected_states, allocation)
assert tuple(rotation_detail.columns) == ROTATION_DETAIL_COLUMNS
comparison_index = pd.DatetimeIndex(rotation_detail.index, name="Date")
equal_weight = build_equal_weight_monthly_returns(step1, comparison_index)
spy_buy_hold = build_spy_buy_hold_returns(step1, comparison_index)
step5_daily_returns = build_comparison(rotation_detail, equal_weight, spy_buy_hold)
assert tuple(step5_daily_returns.columns) == COMPARISON_COLUMNS
assert step5_daily_returns.index.equals(comparison_index)
assert len(step5_daily_returns) == len(step1) - 1

daily_returns_path.parent.mkdir(parents=True, exist_ok=True)
step5_daily_returns.to_csv(daily_returns_path, index_label="Date")

display(Markdown("### Lagged rotation decision examples"))
display(rotation_detail.head(8))
display(Markdown("### Canonical Step 5 daily comparison returns"))
display(step5_daily_returns.head(8))
display(
    Markdown(
        f"Comparison period: **{comparison_index[0].date()}** to "
        f"**{comparison_index[-1].date()}**, observations: **{len(comparison_index)}**."
    )
)"""
)

step5_interpretation = nbformat.v4.new_markdown_cell(
    r"""### Timing and benchmark interpretation

The displayed `decision_date` is the prior observed trading row for each realized rotation return. This makes the timing convention directly auditable: a regime switch only affects the next observed trading row. Missing calendar days are not synthetically created and no state is forward-filled across a non-trading row.

The equal-weight comparator is a **monthly-reset** portfolio, not a daily-reset portfolio. After a calendar-month reset, realized asset returns mechanically change the constituent weights until the next observed comparison date in a new month. SPY is evaluated on exactly the same dates, so all three daily-return series have identical sample sizes and date support.

These comparisons remain full-sample, gross-of-cost descriptive backtests. The one-row execution delay addresses same-row execution timing only; it does not remove the full-sample regime-estimation and allocation-selection look-ahead described above. A stronger validation would estimate regimes and state-conditional allocations using only information available at each decision date and would include implementation costs."""
)

works_heading = nbformat.v4.new_markdown_cell(
    """## Works Cited

The following MLA 9 entries are rendered programmatically from the cited keys in the canonical `reports/references.bib` registry."""
)

works_code = nbformat.v4.new_code_cell(
    r'''import re

references_path = repo_root / "reports/references.bib"


def _parse_bibtex_registry(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@(\w+)\{([^,]+),\s*(.*?)\n\}", text, flags=re.DOTALL):
        entry_type, key, body = match.groups()
        fields = {
            field: value
            for field, value in re.findall(r'(\w+)\s*=\s*"([^"]*)"', body)
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
    "markowitz1952portfolio",
    "white2000datasnooping",
    "bailey2014deflatedsharpe",
]
missing_keys = [key for key in cited_keys if key not in references]
assert not missing_keys, f"Unresolved citation keys: {missing_keys}"
display(Markdown(chr(10).join(f"- {_mla_entry(references[key])}" for key in cited_keys)))'''
)

nb.cells = prefix + [
    step5_markdown,
    step5_code,
    step5_interpretation,
    works_heading,
    works_code,
]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
