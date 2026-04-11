import logging
from src.config import ForecastConfig
from src.forecaster import FoodCPIForecaster

def main():
    # 1. Initialize Configuration
    # We use the dataclass which provides default values from the project's base config
    config = ForecastConfig()

    # 2. Initialize Forecaster
    # The FoodCPIForecaster class encapsulates the state (series, model, best_order)
    forecaster = FoodCPIForecaster(config)

    print("=" * 70)
    print("  Food CPI Forecast — ARIMA-Based Rolling Validation Framework")
    print("  Refactored OO Architecture | PGD Computer Science")
    print("=" * 70)

    try:
        # Phase A: Load and prepare data
        series, _ = forecaster.load_data()
        print(f" [Phase A]  Data loaded: {series.index[0].strftime('%B %Y')} to {series.index[-1].strftime('%B %Y')} | n={len(series)}")

        # Phase B: Stationarity testing
        forecaster.test_stationarity()
        print(f" [Phase B]  Stationarity tests complete. Target differencing: d={config.d}")

        # Phase C: Identification
        forecaster.identify_orders()
        print(f" [Phase C]  ACF/PACF correlograms generated.")

        # Phase D: Model estimation and selection
        # We use auto_optimize=True to enable pmdarima's auto_arima
        forecaster.estimate_model(auto_optimize=True)
        print(f" [Phase D]  Model optimized. Selected Order: {forecaster.best_order}")

        # Phase E: Residual diagnostics
        forecaster.run_diagnostics()
        print(f" [Phase E]  Residual diagnostics complete.")

        # Phase F: Walk-Forward Validation
        # Validating across multiple horizons: 1, 3, 6, and 12 months
        forecaster.validate_walk_forward(horizons=[1, 3, 6, 12])
        print(f" [Phase F]  Multi-horizon walk-forward validation complete.")

        # Phase G: Final Forecast
        forecaster.generate_forecast(steps=config.forecast_steps)
        print(f" [Phase G]  {config.forecast_steps}-month forecast generated.")

        print("\n" + "=" * 70)
        print("  Pipeline execution complete.")
        print(f"  Figures saved to: {config.figures_dir}/")
        print(f"  Tables saved to: {config.tables_dir}/")
        print("=" * 70)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}")
        print("Check the logs for details.")

if __name__ == "__main__":
    main()
