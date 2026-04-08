# =============================================================================
# main.py
# =============================================================================
# Time Series Forecasting of Nigeria's Food Consumer Price Index:
# An ARIMA-Based Rolling Validation Framework
#
# Student  : Adeeko Oluwaseun Victor | PG/25/0274
# Supervisor: Dr. Bukola Ajayi
# Institution: Babcock University, Department of Computer Science
# Programme: Postgraduate Diploma in Computer Science, 2025/2026
#
# Entry point. All configuration is defined here and passed explicitly
# to each phase. No values are hardcoded inside the phase files.
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

# ── Phase imports ─────────────────────────────────────────────────────────────
from phases.phase_a_data         import load_and_prepare
from phases.phase_b_stationarity import run_stationarity_tests
from phases.phase_c_identification import run_identification
from phases.phase_d_estimation   import run_estimation
from phases.phase_e_diagnostics  import run_diagnostics
from phases.phase_f_validation   import run_rolling_origin
from phases.phase_g_forecast     import run_forecast

# =============================================================================
# CONFIGURATION — edit only this section to change any parameter
# =============================================================================

CONFIG = {

    # ── Paths ─────────────────────────────────────────────────────────────────
    "data_path"     : os.path.join("data", "cpi_OCT2024.xlsx"),
    "figures_dir"   : os.path.join("outputs", "figures"),
    "tables_dir"    : os.path.join("outputs", "tables"),

    # ── Data extraction parameters ────────────────────────────────────────────
    "sheet_name"    : "Table2",         # Sheet in the NBS Excel file
    "year_col"      : 0,                # Column index for year
    "month_col"     : 1,                # Column index for month
    "food_cpi_col"  : 7,                # Column index for Food & Non-Alc Bev CPI
    "data_row_start": 3,                # First data row (0-indexed, skips headers)
    "pre2024_row_end": 351,             # Last row of pre-2024 data (exclusive)
    "y2024_row_start": 351,             # First row of 2024 data
    "y2024_row_end"  : 361,             # Last row of 2024 data (exclusive)
    "series_start"  : "2010-01-01",     # First observation to include
    "series_end"    : "2024-10-01",     # Last observation to include
    "base_period"   : "November 2009 = 100",

    # ── Structural break annotation dates (YYYY-MM-DD) ────────────────────────
    "break_dates"   : [
        ("2020-04-01", "COVID-19\nShock"),
        ("2022-02-01", "Russia-Ukraine\nGrain Shock"),
        ("2023-06-01", "Naira Unification\n& Subsidy Removal"),
    ],

    # ── Stationarity test parameters ──────────────────────────────────────────
    "adf_autolag"       : "AIC",        # Lag selection method for ADF
    "kpss_regression"   : "c",          # 'c' = level stationarity; 'ct' = trend
    "kpss_nlags"        : "auto",
    "significance_level": 0.05,         # α for all hypothesis tests

    # ── ACF/PACF parameters ───────────────────────────────────────────────────
    "acf_nlags"     : 30,               # Number of lags to display
    "pacf_method"   : "ywmle",          # PACF estimation method
    "ci_level"      : 0.95,             # Confidence interval level

    # ── Candidate model grid ──────────────────────────────────────────────────
    "p_max"         : 2,                # Maximum AR order to test
    "d"             : 2,                # Differencing order (confirmed by Step 2)
    "q_max"         : 2,                # Maximum MA order to test

    # ── Selected model (set after Step 4 — do not change unless re-running) ──
    "model_order"   : (2, 2, 2),        # (p, d, q) — confirmed ARIMA(2,2,2)

    # ── Residual diagnostic parameters ────────────────────────────────────────
    "ljungbox_lags" : [10, 20],         # Lags to test in Ljung-Box

    # ── Rolling-origin evaluation parameters ──────────────────────────────────
    "eval_start"    : "2022-11-01",     # First forecast origin
    "eval_end"      : "2024-09-01",     # Last forecast origin (forecast = Oct 2024)
    "seasonal_period": 12,              # Seasonal naïve lookback (months)

    # ── Forecast parameters ───────────────────────────────────────────────────
    "forecast_steps": 12,               # Number of months to forecast ahead
    "forecast_start": "2024-11-01",     # First forecast month
    "pi_level"      : 0.95,             # Prediction interval confidence level

    # ── Figure parameters ─────────────────────────────────────────────────────
    "fig_format"    : "png",            # Output format for all figures
    "fig_dpi"       : 300,              # Resolution
    "fig_style"     : "seaborn-v0_8-whitegrid",

    # ── Month name mapping (full and abbreviated) ─────────────────────────────
    "month_map"     : {
        "Jan": 1, "Feb": 2, "Mar": 3,  "Apr": 4,  "May": 5,  "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9,  "Oct": 10, "Nov": 11, "Dec": 12,
        "January": 1,  "February": 2,  "March": 3,    "April": 4,
        "June": 6,     "July": 7,      "August": 8,   "September": 9,
        "October": 10, "November": 11, "December": 12,
    },
}

# =============================================================================
# PIPELINE — runs all seven phases in sequence
# =============================================================================

def main():
    print("=" * 70)
    print("  Food CPI Forecast — ARIMA-Based Rolling Validation Framework")
    print("  Adeeko Oluwaseun Victor | PG/25/0274 | Babcock University")
    print("=" * 70)

    # ── Phase A: Load and prepare data ───────────────────────────────────────
    print("\n[Phase A]  Loading and preparing NBS Food CPI data...")
    series, desc_stats = load_and_prepare(CONFIG)
    print(f"           Series: {series.index[0].strftime('%B %Y')} — "
          f"{series.index[-1].strftime('%B %Y')}  |  n = {len(series)}")

    # ── Phase B: Stationarity testing ────────────────────────────────────────
    print("\n[Phase B]  Running ADF and KPSS stationarity tests...")
    stationarity_results = run_stationarity_tests(series, CONFIG)
    print(f"           Confirmed differencing order: d = {CONFIG['d']}")

    # ── Phase C: ACF and PACF identification ─────────────────────────────────
    print("\n[Phase C]  Generating ACF and PACF correlograms...")
    identification_results = run_identification(series, CONFIG)
    print(f"           Candidate grid: ARIMA(p,{CONFIG['d']},q)  "
          f"p ∈ {{0..{CONFIG['p_max']}}}, q ∈ {{0..{CONFIG['q_max']}}}")

    # ── Phase D: Model estimation and selection ───────────────────────────────
    print("\n[Phase D]  Estimating all candidate ARIMA models...")
    model_results, fitted_model = run_estimation(series, CONFIG)
    p, d, q = CONFIG["model_order"]
    print(f"           Selected model: ARIMA({p},{d},{q})  "
          f"AIC={model_results.loc[model_results['Model']==f'ARIMA({p},{d},{q})','AIC'].values[0]:.4f}  "
          f"BIC={model_results.loc[model_results['Model']==f'ARIMA({p},{d},{q})','BIC'].values[0]:.4f}")

    # ── Phase E: Residual diagnostics ────────────────────────────────────────
    print("\n[Phase E]  Running residual diagnostic checks...")
    diagnostic_results = run_diagnostics(fitted_model, CONFIG)
    lb10 = diagnostic_results["ljungbox"].loc[
        diagnostic_results["ljungbox"]["Lag"] == 10, "p-value"].values[0]
    lb20 = diagnostic_results["ljungbox"].loc[
        diagnostic_results["ljungbox"]["Lag"] == 20, "p-value"].values[0]
    print(f"           Ljung-Box Q(10) p={lb10:.4f}  |  Q(20) p={lb20:.4f}")

    # ── Phase F: Rolling-origin validation ───────────────────────────────────
    print("\n[Phase F]  Running 24-month rolling-origin evaluation...")
    validation_results = run_rolling_origin(series, CONFIG)
    mape = validation_results["metrics"]["MAPE (%)"]
    u    = validation_results["metrics"]["Theil's U"]
    print(f"           MAPE = {mape:.4f}%  |  Theil's U = {u:.4f}  "
          f"({'ARIMA outperforms naïve ✓' if u < 1 else 'Naïve benchmark wins'})")

    # ── Phase G: 12-month forecast ───────────────────────────────────────────
    print("\n[Phase G]  Generating 12-month ahead forecast...")
    forecast_results = run_forecast(series, fitted_model, CONFIG)
    first = forecast_results["forecast"].iloc[0]
    last  = forecast_results["forecast"].iloc[-1]
    print(f"           {first['Month']}: {first['Forecast']:.4f}  →  "
          f"{last['Month']}: {last['Forecast']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  All phases complete. Outputs saved to outputs/")
    print(f"  Figures : {CONFIG['figures_dir']}/")
    print(f"  Tables  : {CONFIG['tables_dir']}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
