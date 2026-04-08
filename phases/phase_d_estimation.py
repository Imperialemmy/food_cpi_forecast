# =============================================================================
# phases/phase_d_estimation.py
# =============================================================================
# Phase D: Model Estimation and Selection (Step 3 of Box-Jenkins)
#
# Estimates all candidate ARIMA(p, d, q) models in the grid defined by
# p_max, d, q_max in the configuration. Ranks by AIC and BIC.
# Fits and returns the confirmed model specified in model_order.
# Saves Table 4.3.
# =============================================================================

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def run_estimation(series: pd.Series, cfg: dict):
    """
    Estimate all candidate ARIMA models, rank by AIC and BIC,
    and fit the selected model.

    Parameters
    ----------
    series : pd.Series  Food CPI level series.
    cfg    : dict       Configuration dictionary from main.py.

    Returns
    -------
    model_results : pd.DataFrame  AIC/BIC comparison table (Table 4.3).
    fitted_model  : ARIMAResults  Fitted instance of the selected model.
    """

    p_max       = cfg["p_max"]
    d           = cfg["d"]
    q_max       = cfg["q_max"]
    model_order = cfg["model_order"]
    tables_dir  = cfg["tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)

    # ── Estimate all candidate models ─────────────────────────────────────────
    records = []
    print(f"\n           Estimating ARIMA(p,{d},q) for "
          f"p ∈ {{0..{p_max}}}, q ∈ {{0..{q_max}}}:")

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                res = ARIMA(series, order=(p, d, q)).fit()
                k   = p + q + 1          # AR params + MA params + sigma²
                records.append({
                    "Model"        : f"ARIMA({p},{d},{q})",
                    "p"            : p,
                    "d"            : d,
                    "q"            : q,
                    "k (params)"   : k,
                    "Log-Likelihood": round(res.llf, 4),
                    "AIC"          : round(res.aic, 4),
                    "BIC"          : round(res.bic, 4),
                    "Converged"    : True,
                })
                print(f"             ARIMA({p},{d},{q})  "
                      f"LogLik={res.llf:>10.4f}  "
                      f"AIC={res.aic:>10.4f}  "
                      f"BIC={res.bic:>10.4f}  [OK]")
            except Exception as err:
                records.append({
                    "Model": f"ARIMA({p},{d},{q})",
                    "p": p, "d": d, "q": q,
                    "k (params)": p + q + 1,
                    "Log-Likelihood": None, "AIC": None, "BIC": None,
                    "Converged": False,
                })
                print(f"             ARIMA({p},{d},{q})  [FAILED: {err}]")

    # ── Build comparison DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame(records)
    df_conv = df[df["Converged"]].copy()
    df_conv["AIC Rank"] = df_conv["AIC"].rank().astype(int)
    df_conv["BIC Rank"] = df_conv["BIC"].rank().astype(int)
    df_conv = df_conv.sort_values("AIC").reset_index(drop=True)

    # ── Identify best models ───────────────────────────────────────────────────
    best_aic_model = df_conv.loc[df_conv["AIC Rank"] == 1, "Model"].values[0]
    best_bic_model = df_conv.loc[df_conv["BIC Rank"] == 1, "Model"].values[0]
    confirmed      = (
        f"ARIMA({model_order[0]},{model_order[1]},{model_order[2]})"
    )

    print(f"\n           Best AIC: {best_aic_model}")
    print(f"           Best BIC: {best_bic_model}")
    if best_aic_model == best_bic_model == confirmed:
        print(f"           ✓ Both AIC and BIC agree on: {confirmed}")
    else:
        print(f"           ℹ Confirmed model (from cfg): {confirmed}")

    # ── Save Table 4.3 ─────────────────────────────────────────────────────────
    output_cols = [
        "Model", "k (params)", "Log-Likelihood",
        "AIC", "BIC", "AIC Rank", "BIC Rank"
    ]
    df_conv[output_cols].to_csv(
        os.path.join(tables_dir, "table3_model_comparison.csv"), index=False
    )

    # ── Print clean Table 4.3 ─────────────────────────────────────────────────
    print(f"\n           Table 4.3 — Candidate Model Comparison:")
    print(f"           {'Model':<16} {'k':>4} {'LogLik':>12} "
          f"{'AIC':>10} {'BIC':>10} {'AIC Rank':>9} {'BIC Rank':>9}")
    print(f"           {'-'*78}")
    for _, row in df_conv.iterrows():
        marker = " ← SELECTED" if row["Model"] == confirmed else ""
        print(
            f"           {row['Model']:<16} {row['k (params)']:>4} "
            f"{row['Log-Likelihood']:>12.4f} {row['AIC']:>10.4f} "
            f"{row['BIC']:>10.4f} {row['AIC Rank']:>9} "
            f"{row['BIC Rank']:>9}{marker}"
        )

    # ── Fit and return the confirmed model ─────────────────────────────────────
    p_sel, d_sel, q_sel = model_order
    fitted_model = ARIMA(series, order=(p_sel, d_sel, q_sel)).fit()

    # ── Print parameter estimates ─────────────────────────────────────────────
    print(f"\n           Parameter Estimates — ARIMA({p_sel},{d_sel},{q_sel}):")
    print(f"           {'Parameter':<20} {'Coeff':>10} {'StdErr':>10} "
          f"{'z-stat':>10} {'p-value':>10} {'Significant?'}")
    print(f"           {'-'*78}")
    for name in fitted_model.params.index:
        coef = fitted_model.params[name]
        se   = fitted_model.bse[name]
        zval = fitted_model.tvalues[name]
        pval = fitted_model.pvalues[name]
        sig  = (
            "Yes ***" if pval < 0.01 else
            "Yes **"  if pval < 0.05 else
            "Yes *"   if pval < 0.10 else
            "No"
        )
        print(f"           {name:<20} {coef:>10.4f} {se:>10.4f} "
              f"{zval:>10.4f} {pval:>10.4f}  {sig}")

    return df_conv, fitted_model
