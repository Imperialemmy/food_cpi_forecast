# =============================================================================
# phases/phase_f_validation.py
# =============================================================================
# Phase F: Rolling-Origin Out-of-Sample Evaluation (Step 5 of Box-Jenkins)
#
# Implements the expanding-window rolling-origin procedure over the
# evaluation window defined in the configuration.
#
# Produces:
#   - Table 4.5:  Accuracy metrics (RMSE, MAE, MAPE, Theil's U)
#   - Figure 4.5: Rolling-origin actual vs forecast plot
#   - Appendix C: Full monthly forecast error table
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def run_rolling_origin(series: pd.Series, cfg: dict) -> dict:
    """
    Run 24-month rolling-origin expanding-window evaluation.

    Parameters
    ----------
    series : pd.Series  Full Food CPI series.
    cfg    : dict       Configuration dictionary from main.py.

    Returns
    -------
    results : dict  Accuracy metrics DataFrame, rolling-origin records.
    """

    model_order     = cfg["model_order"]
    eval_start      = pd.Timestamp(cfg["eval_start"])
    eval_end        = pd.Timestamp(cfg["eval_end"])
    seasonal_period = cfg["seasonal_period"]
    alpha           = cfg["significance_level"]
    figures_dir     = cfg["figures_dir"]
    tables_dir      = cfg["tables_dir"]
    fig_dpi         = cfg["fig_dpi"]
    fig_format      = cfg["fig_format"]
    fig_style       = cfg["fig_style"]
    p, d, q         = model_order

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir,  exist_ok=True)

    # ── Rolling-origin loop ────────────────────────────────────────────────────
    origins = series[eval_start:eval_end].index
    records = []

    print(f"\n           Running {len(origins)} forecast origins "
          f"({eval_start.strftime('%b %Y')} — "
          f"{(eval_end + pd.DateOffset(months=1)).strftime('%b %Y')}):")

    for tau in origins:
        train     = series[:tau]
        next_date = tau + pd.DateOffset(months=1)

        if next_date not in series.index:
            continue

        actual = series[next_date]

        # ── ARIMA one-step-ahead forecast ──────────────────────────────────────
        try:
            fc = (
                ARIMA(train, order=(p, d, q))
                .fit()
                .forecast(steps=1)
                .values[0]
            )
        except Exception:
            fc = np.nan

        # ── Seasonal naïve forecast: value 12 months prior to forecast date ───
        naive_date = next_date - pd.DateOffset(months=seasonal_period)
        naive_fc   = (
            series[naive_date]
            if naive_date in series.index
            else np.nan
        )

        arima_err = actual - fc
        naive_err = actual - naive_fc

        records.append({
            "Origin"     : tau,
            "Forecast_dt": next_date,
            "Train_n"    : len(train),
            "Actual"     : actual,
            "ARIMA_Fc"   : fc,
            "ARIMA_Err"  : arima_err,
            "Naive_Fc"   : naive_fc,
            "Naive_Err"  : naive_err,
        })

        pct = arima_err / actual * 100
        print(f"             {next_date.strftime('%b %Y'):>8}  "
              f"Actual={actual:>9.4f}  "
              f"Forecast={fc:>9.4f}  "
              f"Error={arima_err:>8.4f} ({pct:+.2f}%)")

    df_eval = pd.DataFrame(records)
    T       = len(df_eval)

    # ── Compute accuracy metrics ───────────────────────────────────────────────
    ae  = df_eval["ARIMA_Err"].values
    ne  = df_eval["Naive_Err"].values
    av  = df_eval["Actual"].values

    rmse_a  = np.sqrt(np.mean(ae ** 2))
    mae_a   = np.mean(np.abs(ae))
    mape_a  = np.mean(np.abs(ae / av)) * 100

    rmse_n  = np.sqrt(np.mean(ne ** 2))
    mae_n   = np.mean(np.abs(ne))
    mape_n  = np.mean(np.abs(ne / av)) * 100

    theils_u = rmse_a / rmse_n

    metrics = {
        "T (origins)"  : T,
        "RMSE"         : round(rmse_a,  4),
        "MAE"          : round(mae_a,   4),
        "MAPE (%)"     : round(mape_a,  4),
        "Theil's U"    : round(theils_u, 4),
        "RMSE (Naïve)" : round(rmse_n,  4),
        "MAE (Naïve)"  : round(mae_n,   4),
        "MAPE (Naïve)%": round(mape_n,  4),
    }

    # ── Sub-period split analysis ──────────────────────────────────────────────
    break_dt = pd.Timestamp("2023-06-01")
    pre_df   = df_eval[df_eval["Forecast_dt"] <  break_dt]
    post_df  = df_eval[df_eval["Forecast_dt"] >= break_dt]

    sub_period = {}
    for label, sub in [("Pre-June 2023",  pre_df),
                        ("Post-June 2023", post_df)]:
        if len(sub) == 0:
            continue
        se = sub["ARIMA_Err"].values
        sv = sub["Actual"].values
        sub_period[label] = {
            "n"       : len(sub),
            "RMSE"    : round(np.sqrt(np.mean(se ** 2)), 4),
            "MAE"     : round(np.mean(np.abs(se)),       4),
            "MAPE (%)" : round(np.mean(np.abs(se / sv)) * 100, 4),
        }

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n           T = {T} forecast origins")
    print(f"           {'Metric':<14} {'ARIMA':>10} {'Naïve':>10}  Verdict")
    print(f"           {'-'*52}")
    for m_label, arima_v, naive_v in [
        ("RMSE",     rmse_a, rmse_n),
        ("MAE",      mae_a,  mae_n),
        ("MAPE (%)", mape_a, mape_n),
    ]:
        verdict = "ARIMA better ✓" if arima_v < naive_v else "Naïve better"
        print(f"           {m_label:<14} {arima_v:>10.4f} {naive_v:>10.4f}  {verdict}")
    verdict_u = "ARIMA outperforms ✓" if theils_u < 1 else "Naïve wins ✗"
    theils_label = "Theil's U"
    print(f"           {theils_label:<14} {theils_u:>10.4f} {'1.0000':>10}  {verdict_u}")
    for lbl, sp in sub_period.items():
        print(f"           {lbl} (n={sp['n']}): "
              f"RMSE={sp['RMSE']:.4f}  MAE={sp['MAE']:.4f}  "
              f"MAPE={sp['MAPE (%)']:.4f}%")

    # ── Save Table 4.5 ─────────────────────────────────────────────────────────
    metrics_rows = [
        {"Metric": "RMSE",      "ARIMA(p,d,q)": rmse_a, "Seasonal Naïve": rmse_n,
         "Verdict": "ARIMA better" if rmse_a < rmse_n else "Naïve better"},
        {"Metric": "MAE",       "ARIMA(p,d,q)": mae_a,  "Seasonal Naïve": mae_n,
         "Verdict": "ARIMA better" if mae_a  < mae_n  else "Naïve better"},
        {"Metric": "MAPE (%)",  "ARIMA(p,d,q)": mape_a, "Seasonal Naïve": mape_n,
         "Verdict": "ARIMA better" if mape_a < mape_n else "Naïve better"},
        {"Metric": "Theil's U", "ARIMA(p,d,q)": theils_u, "Seasonal Naïve": 1.0,
         "Verdict": "ARIMA outperforms" if theils_u < 1 else "Naïve wins"},
    ]
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(tables_dir, "table5_accuracy_metrics.csv"), index=False
    )

    # ── Save Appendix C ────────────────────────────────────────────────────────
    app_c = df_eval.copy()
    app_c["Month"]        = app_c["Forecast_dt"].dt.strftime("%B %Y")
    app_c["ARIMA % Error"] = (
        app_c["ARIMA_Err"] / app_c["Actual"] * 100
    ).round(4)
    app_c_out = app_c[[
        "Month", "Actual", "ARIMA_Fc", "ARIMA_Err",
        "Naive_Fc", "Naive_Err", "ARIMA % Error"
    ]].copy()
    app_c_out.columns = [
        "Month", "Actual Food CPI",
        f"ARIMA({p},{d},{q}) Forecast", "ARIMA Error",
        "Naïve Forecast", "Naïve Error", "ARIMA % Error"
    ]
    app_c_out = app_c_out.round(4)
    app_c_out.to_csv(
        os.path.join(tables_dir, "appendix_c_rolling_origin.csv"), index=False
    )

    # ── Figure 4.5 — Rolling-origin actual vs forecast plot ───────────────────
    try:
        plt.style.use(fig_style)
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_eval["Forecast_dt"], df_eval["Actual"],
            color="#1F4E79", linewidth=1.8, label="Actual Food CPI", zorder=3)
    ax.plot(df_eval["Forecast_dt"], df_eval["ARIMA_Fc"],
            color="#C00000", linewidth=1.4, linestyle="--",
            label=f"ARIMA({p},{d},{q}) One-Step Forecast", zorder=3)
    ax.fill_between(
        df_eval["Forecast_dt"],
        df_eval["Actual"],
        df_eval["ARIMA_Fc"],
        alpha=0.12, color="#C00000", label="Forecast Error"
    )
    ax.axvline(pd.Timestamp("2023-06-01"), color="#7030A0",
               linewidth=1.0, linestyle=":", alpha=0.8,
               label="Naira Unification (Jun 2023)")
    ax.set_title(
        f"Figure 4.5 — Rolling-Origin Evaluation: ARIMA({p},{d},{q})\n"
        f"November 2022 – October 2024  |  MAPE = {mape_a:.4f}%  |  "
        f"Theil's U = {theils_u:.4f}",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Date"); ax.set_ylabel("Food CPI Index Value")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"fig5_rolling_origin.{fig_format}"),
        dpi=fig_dpi, bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "records"    : df_eval,
        "metrics"    : metrics,
        "sub_period" : sub_period,
    }
