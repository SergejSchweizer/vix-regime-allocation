from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
nb = nbformat.read(NOTEBOOK, as_version=4)

works_index = next(
    (i for i, cell in enumerate(nb.cells) if cell.cell_type == "markdown" and "## Works Cited" in cell.source),
    None,
)
if works_index is None:
    raise SystemExit("Works Cited section not found")

heading = "### Step 5 — Performance metrics and cumulative comparison"
existing_heading = next(
    (i for i, cell in enumerate(nb.cells) if cell.cell_type == "markdown" and heading in cell.source),
    None,
)
if existing_heading is not None:
    nb.cells = nb.cells[:existing_heading] + nb.cells[works_index:]
    works_index = existing_heading

metrics_md = nbformat.v4.new_markdown_cell(
    r"""### Step 5 — Performance metrics and cumulative comparison

Performance is evaluated on the identical lagged comparison dates established above. The metric implementation uses simple daily portfolio returns, 252 trading days per year, a zero risk-free rate, sample volatility with denominator $n-1$, and initial wealth $W_0=1$.

**Greek letters used in the following equations:** none.

Wealth evolves as

$$
W_0=1,\qquad W_t=W_{t-1}(1+R_t).
$$

Cumulative return is

$$
R_{\mathrm{cum}}=W_n-1,
$$

and annualized return is

$$
R_{\mathrm{ann}}=W_n^{252/n}-1.
$$

With sample standard deviation

$$
s=\sqrt{\frac{1}{n-1}\sum_{t=1}^{n}(R_t-\bar R)^2},
$$

the annualized volatility and zero-risk-free Sharpe ratio are

$$
V_{\mathrm{ann}}=s\sqrt{252},\qquad
S=\frac{\bar R}{s}\sqrt{252}.
$$

Drawdown is measured relative to the running wealth peak **including the initial wealth observation**:

$$
D_t=\frac{W_t}{\max_{0\le u\le t}W_u}-1,\qquad
D_{\max}=\min_t D_t.
$$

The cumulative-performance figure uses the same shared compounding function as the metric table, so the plotted terminal values and the reported cumulative returns have one numerical source of truth. These project calculations follow standard portfolio-performance conventions and remain descriptive for this full-sample exercise."""
)

metrics_code = nbformat.v4.new_code_cell(
    r'''from vix_regime_allocation.backtest_plot import plot_cumulative_performance
from vix_regime_allocation.backtest_summary import build_performance_summary

if Path.cwd().name == "notebooks":
    repo_root_step5 = Path.cwd().resolve().parent
else:
    repo_root_step5 = Path.cwd().resolve()
daily_path = repo_root_step5 / "reports/tables/step5_daily_returns.csv"
summary_path = repo_root_step5 / "reports/tables/step5_performance_summary.csv"
figure_path = repo_root_step5 / "reports/figures/step5_cumulative_performance.png"

step5_daily = pd.read_csv(daily_path, parse_dates=["Date"]).set_index("Date")
step5_daily.index = pd.DatetimeIndex(step5_daily.index, name="Date")
performance_summary = build_performance_summary(step5_daily)
summary_path.parent.mkdir(parents=True, exist_ok=True)
performance_summary.to_csv(summary_path, index=False)
plot_cumulative_performance(step5_daily, figure_path)

display(Markdown("### Required performance summary"))
display(performance_summary)
display(Markdown(f"![Cumulative performance comparison]({figure_path.as_posix()})"))

rows = performance_summary.set_index("portfolio")
lines = []
for portfolio, label in [
    ("regime_rotation", "Regime rotation"),
    ("equal_weight_monthly", "Equal-weight monthly reset"),
    ("spy_buy_hold", "SPY buy and hold"),
]:
    row = rows.loc[portfolio]
    lines.append(
        f"- **{label}:** cumulative return {row['cumulative_return']:.6f}; "
        f"annualized return {row['annualized_return']:.6f}; annualized volatility "
        f"{row['annualized_volatility']:.6f}; Sharpe {row['sharpe_ratio']:.6f}; "
        f"maximum drawdown {row['max_drawdown']:.6f}; observations {int(row['observations'])}."
    )
display(Markdown("### Numerical comparison\n" + "\n".join(lines)))'''
)

metrics_interp = nbformat.v4.new_markdown_cell(
    """### Interpretation of the performance comparison

The table and figure should be read jointly: cumulative and annualized return describe growth, annualized volatility and maximum drawdown describe different dimensions of risk, and the Sharpe ratio summarizes average return per unit of daily return dispersion under the assignment's zero-risk-free convention. The comparison covers both required benchmarks on identical dates.

The same full-sample qualification stated above remains binding for every value in this section. The one-row execution lag removes same-row trading, but it does not undo full-sample regime estimation or the use of full-sample state-conditional means in the allocation rule. Consequently, the reported risk-adjusted comparison is descriptive and in-sample rather than causal or out-of-sample evidence."""
)

nb.cells[works_index:works_index] = [metrics_md, metrics_code, metrics_interp]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
