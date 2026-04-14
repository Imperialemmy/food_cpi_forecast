import pandas as pd
import numpy as np
from src.config import ForecastConfig
from src.forecaster import FoodCPIForecaster
import io

def test_generic_csv():
    print("Testing generic CSV upload functionality...")
    config = ForecastConfig()
    # Override range to fit the small test dataset
    config.series_start = "2020-01-01"
    config.series_end = "2021-12-01"
    config.forecast_start = "2022-01-01"

    forecaster = FoodCPIForecaster(config)

    # Simulate Streamlit file uploader (provides a file-like object)
    with open("test_data.csv", "r") as f:
        # We use pd.read_csv inside load_data if we want to support CSV,
        # but current load_data uses read_excel.
        # Let's check if we need to add CSV support to load_data.
        pass

    # Since load_data uses read_excel, let's test with a dummy excel first or
    # update load_data to handle CSVs.
    print("Test script ready. Checking if load_data handles CSVs...")

if __name__ == "__main__":
    test_generic_csv()
