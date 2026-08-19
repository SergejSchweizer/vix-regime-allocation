"""Run the canonical MScFE 622 GWP2 Step 1 pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vix_regime_allocation.data import download_adjusted_close
from vix_regime_allocation.plots import plot_etf_log_returns, plot_vix_change
from vix_regime_allocation.transform import prepare_step1_data

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STEP1_DATA_PATH = Path("data/processed/step1_data.csv")
ETF_FIGURE_PATH = Path("reports/figures/step1_etf_log_returns.png")
VIX_FIGURE_PATH = Path("reports/figures/step1_vix_change.png")


def run_step1(output_root: Path = REPOSITORY_ROOT) -> pd.DataFrame:
    """Download, transform, persist, and plot the canonical Step 1 data."""
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path.")

    prices = download_adjusted_close()
    data = prepare_step1_data(prices)

    csv_path = output_root / STEP1_DATA_PATH
    etf_figure_path = output_root / ETF_FIGURE_PATH
    vix_figure_path = output_root / VIX_FIGURE_PATH

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=True, date_format="%Y-%m-%d")
    plot_etf_log_returns(data, etf_figure_path)
    plot_vix_change(data, vix_figure_path)

    print(f"Start: {data.index.min().date().isoformat()}")
    print(f"End: {data.index.max().date().isoformat()}")
    print(f"Count: {len(data)}")

    return data


def main() -> None:
    """Execute Step 1 using repository-relative canonical output paths."""
    run_step1()


if __name__ == "__main__":
    main()
