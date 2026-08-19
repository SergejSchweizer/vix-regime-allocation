from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/gwp2_vix_regime_allocation.ipynb"
nb = nbformat.read(NOTEBOOK, as_version=4)

marker = "from vix_regime_allocation.backtest_plot import plot_cumulative_performance"
cell = next((c for c in nb.cells if c.cell_type == "code" and marker in c.source), None)
if cell is None:
    raise SystemExit("PR-41 metrics code cell not found")

cell.source = r'''from vix_regime_allocation.backtest_plot import plot_cumulative_performance
from vix_regime_allocation.backtest_summary import build_performance_summary

repo_root_step5 = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
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
display(Markdown("### Numerical comparison\\n" + "\\n".join(lines)))'''

nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
