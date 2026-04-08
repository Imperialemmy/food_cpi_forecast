# =============================================================================
# phases/phase_g_forecast.py
# =============================================================================
# Phase G: 12-Month Ahead Forecast Generation
#
# Estimates the confirmed ARIMA model on the complete 178-observation
# sample and generates point forecasts with prediction intervals.
#
# Produces:
#   - Table 4.6:  Forecast table with 95% prediction intervals
#   - Figure 4.6: Historical series + 12-month forecast chart
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def run_forecast(series: pd.Series, fitted_model, cfg: dict) -> dict:
    """
    Generate 12-month ahead point forecasts with prediction intervals.

    Parameters
    ----------
    series        : pd.Series     Full Food CPI series.
    fitted_model  : ARIMAResults  Model already fitted on the full series.
    cfg           : dict          Configuration dictionary from main.py.

    Returns
    -------
    results : dict  Forecast DataFrame (Table 4.6).
    """

    forecast_steps = cfg["forecast_steps"]
    pi_level       = cfg["pi_level"]
    forecast_start = pd.Timestamp(cfg["forecast_start"])
    model_order    = cfg["model_order"]
    figures_dir    = cfg["figures_dir"]
    tables_dir     = cfg["tables_dir"]
    fig_dpi        = cfg["fig_dpi"]
    fig_format     = cfg["fig_format"]
    fig_style      = cfg["fig_style"]
    base_period    = cfg["base_period"]
    p, d, q        = model_order

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir,  exist_ok=True)

    # ── Generate forecast ──────────────────────────────────────────────────────
    # Re-fit on the complete series to ensure latest data is used
    full_model  = ARIMA(series, order=(p, d, q)).fit()
    fc_result   = full_model.get_forecast(steps=forecast_steps)
    fc_mean     = fc_result.predicted_mean
    fc_ci       = fc_result.conf_int(alpha=1 - pi_level)

    # Build forecast date index
    fc_dates = pd.date_range(
        start=forecast_start,
        periods=forecast_steps,
        freq="MS"
    )

    # ── Build Table 4.6 ────────────────────────────────────────────────────────
    forecast_df = pd.DataFrame({
        "Month"   : [d.strftime("%B %Y") for d in fc_dates],
        "Forecast": fc_mean.values.round(4),
        "Lower CI": fc_ci.iloc[:, 0].values.round(4),
        "Upper CI": fc_ci.iloc[:, 1].values.round(4),
    })
    forecast_df["PI Width"] = (
        forecast_df["Upper CI"] - forecast_df["Lower CI"]
    ).round(4)

    # ── Console output ─────────────────────────────────────────────────────────
    pi_pct = int(pi_level * 100)
    print(f"\n           Table 4.6 — 12-Month Forecast: ARIMA({p},{d},{q})")
    print(f"           {pi_pct}% Prediction Intervals")
    print(f"           {'Month':<16} {'Forecast':>10} "
          f"{'Lower {pi_pct}% PI':>12} {'Upper {pi_pct}% PI':>12} "
          f"{'PI Width':>10}")
    print(f"           {'-'*64}")
    for _, row in forecast_df.iterrows():
        print(f"           {row['Month']:<16} {row['Forecast']:>10.4f} "
              f"{row['Lower CI']:>12.4f} {row['Upper CI']:>12.4f} "
              f"{row['PI Width']:>10.4f}")

    # ── Save Table 4.6 ─────────────────────────────────────────────────────────
    forecast_df.to_csv(
        os.path.join(tables_dir, "table6_forecast.csv"), index=False
    )

    # ── Figure 4.6 — Full series + 12-month forecast ──────────────────────────
    try:
        plt.style.use(fig_style)
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical series
    ax.plot(series.index, series.values,
            color="#1F4E79", linewidth=1.5,
            label="Historical Food CPI (Jan 2010 – Oct 2024)")

    # Forecast
    ax.plot(fc_dates, forecast_df["Forecast"].values,
            color="#C00000", linewidth=2.0, linestyle="--",
            label=f"ARIMA({p},{d},{q}) Forecast (Nov 2024 – Oct 2025)")

    # Prediction interval band
    ax.fill_between(
        fc_dates,
        forecast_df["Lower CI"].values,
        forecast_df["Upper CI"].values,
        alpha=0.18, color="#C00000",
        label=f"{pi_pct}% Prediction Interval"
    )

    # Vertical line at forecast origin
    ax.axvline(
        series.index[-1], color="grey",
        linewidth=1.0, linestyle=":", alpha=0.8
    )
    ax.text(
        series.index[-1],
        series.values[-1] * 0.60,
        "Forecast\nOrigin",
        fontsize=8, color="grey", ha="right"
    )

    ax.set_title(
        f"Figure 4.6 — Nigeria Food CPI: Historical Series and "
        f"12-Month Ahead Forecast\n"
        f"ARIMA({p},{d},{q}) | Base: {base_period} | "
        f"{pi_pct}% Prediction Intervals",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Food CPI Index Value")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"fig6_forecast.{fig_format}"),
        dpi=fig_dpi, bbox_inches="tight"
    )
    plt.close(fig)

    return {"forecast": forecast_df}
