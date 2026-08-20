from vix_regime_allocation.backtest_plot import PLOT_COLUMNS


def test_cumulative_plot_series_contract() -> None:
    assert PLOT_COLUMNS == (
        "regime_rotation",
        "equal_weight_monthly",
        "TLT",
        "GLD",
        "SPY",
    )
