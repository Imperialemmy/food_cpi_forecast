"""
Basic end-to-end tests for the FoodCPIForecaster pipeline.

These exercise the generic CSV-upload path using the small synthetic
``test_data.csv`` fixture (24 monthly observations, 2020-2021), so they
run quickly without needing the full NBS Excel dataset.
"""
import os
import tempfile

import numpy as np
import pandas as pd

from src.config import ForecastConfig
from src.forecaster import FoodCPIForecaster

# Resolve paths relative to the project root so tests work from any CWD.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV = os.path.join(PROJECT_ROOT, "test_data.csv")


def _make_forecaster():
    """Build a forecaster configured for the small test fixture."""
    config = ForecastConfig()
    # Override range to fit the small test dataset.
    config.series_start = "2020-01-01"
    config.series_end = "2021-12-01"
    config.forecast_start = "2022-01-01"
    # A clean CSV has a single header row, unlike the multi-row NBS Excel layout.
    config.data_row_start = 1
    # Write figures/tables to a throwaway dir so tests never overwrite the
    # committed outputs/ produced from the real NBS data.
    tmp = tempfile.mkdtemp(prefix="cpi_test_outputs_")
    config.figures_dir = os.path.join(tmp, "figures")
    config.tables_dir = os.path.join(tmp, "tables")
    return FoodCPIForecaster(config)


class _UploadStub:
    """Mimic a Streamlit UploadedFile: a file handle that carries a ``name``."""

    def __init__(self, path):
        self.name = os.path.basename(path)
        self._fh = open(path, "rb")

    def read(self, *args, **kwargs):
        return self._fh.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._fh.seek(*args, **kwargs)

    def close(self):
        self._fh.close()


def test_load_generic_csv():
    """A generic CSV upload should parse into a clean monthly series."""
    forecaster = _make_forecaster()
    upload = _UploadStub(TEST_CSV)
    try:
        series, desc_stats = forecaster.load_data(
            file_obj=upload,
            year_col=0,
            month_col=1,
            value_col=2,
        )
    finally:
        upload.close()

    # 24 monthly rows in the fixture (2020-01 .. 2021-12).
    assert len(series) == 24
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index[0] == pd.Timestamp("2020-01-01")
    assert series.index[-1] == pd.Timestamp("2021-12-01")
    assert not series.isna().any()
    # Descriptive-statistics table should be populated.
    assert desc_stats.loc["n", "Value"] == 24


def test_forecast_pipeline_runs():
    """Estimation + forecasting should yield the requested horizon with valid PIs."""
    forecaster = _make_forecaster()
    upload = _UploadStub(TEST_CSV)
    try:
        forecaster.load_data(file_obj=upload, year_col=0, month_col=1, value_col=2)
    finally:
        upload.close()

    # Use the configured manual order (avoids long auto_arima search on 24 points).
    forecaster.estimate_model(auto_optimize=False)
    assert forecaster.best_order is not None
    assert forecaster.model_results is not None

    forecast_df, _ = forecaster.generate_forecast(steps=6)
    assert len(forecast_df) == 6
    # Lower CI must not exceed Upper CI for any forecast step.
    assert (forecast_df["Lower CI"] <= forecast_df["Upper CI"]).all()
    assert not forecast_df["Forecast"].isna().any()


if __name__ == "__main__":
    test_load_generic_csv()
    test_forecast_pipeline_runs()
    print("All tests passed.")
