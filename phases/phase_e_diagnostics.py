# =============================================================================
# phases/phase_e_diagnostics.py
# =============================================================================
# Phase E: Residual Diagnostic Checking (Step 4 of Box-Jenkins)
#
# Produces:
#   - Figure 4.4: Four-panel residual diagnostic chart
#   - Table 4.4:  Ljung-Box portmanteau test results
#   - Jarque-Bera normality test
#   - Top residual observations table
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats as scipy_stats
from scipy.stats import norm


def run_diagnostics(fitted_model, cfg: dict) -> dict:
    """
    Run residual diagnostics on the fitted ARIMA model.

    Parameters
    ----------
    fitted_model : ARIMAResults  Fitted ARIMA model from Phase D.
    cfg          : dict          Configuration dictionary from main.py.

    Returns
    -------
    results : dict  Ljung-Box table, JB result, residual statistics.
    """

    lb_lags    = cfg["ljungbox_lags"]
    alpha      = cfg["significance_level"]
    p, d, q    = cfg["model_order"]
    figures_dir = cfg["figures_dir"]
    tables_dir  = cfg["tables_dir"]
    fig_dpi     = cfg["fig_dpi"]
    fig_format  = cfg["fig_format"]
    fig_style   = cfg["fig_style"]

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir,  exist_ok=True)

    # ── Extract residuals ──────────────────────────────────────────────────────
    resid     = fitted_model.resid.dropna()
    std_resid = (resid - resid.mean()) / resid.std()
    n         = len(resid)
    ci_bound  = 1.96 / np.sqrt(n)

    # ── Residual summary statistics ────────────────────────────────────────────
    resid_stats = {
        "Mean"          : round(resid.mean(),              6),
        "Std Dev"       : round(resid.std(),               4),
        "Min"           : round(resid.min(),               4),
        "Max"           : round(resid.max(),               4),
        "Skewness"      : round(scipy_stats.skew(resid),  4),
        "Excess Kurtosis": round(scipy_stats.kurtosis(resid), 4),
    }

    # ── ACF of residuals ───────────────────────────────────────────────────────
    acf_resid = acf(resid, nlags=20, fft=True)
    sig_lags  = [k for k in range(1, 21) if abs(acf_resid[k]) > ci_bound]

    # ── Ljung-Box test ─────────────────────────────────────────────────────────
    # degrees of freedom = lag - p - q
    lb_records = []
    for lag in lb_lags:
        lb  = acorr_ljungbox(resid, lags=[lag], return_df=True)
        q_s = round(lb["lb_stat"].values[0],   4)
        p_v = round(lb["lb_pvalue"].values[0], 4)
        df  = lag - p - q
        dec = (
            "Fail to reject H₀ — white noise ✓"
            if p_v > alpha
            else "Reject H₀ — autocorrelation present ✗"
        )
        lb_records.append({
            "Lag"        : lag,
            "Q-statistic": q_s,
            "df"         : df,
            "p-value"    : p_v,
            "Decision"   : dec,
        })

    lb_df = pd.DataFrame(lb_records)
    lb_df.to_csv(
        os.path.join(tables_dir, "table4_ljungbox.csv"), index=False
    )

    # ── Jarque-Bera normality test ─────────────────────────────────────────────
    jb_stat, jb_pval = scipy_stats.jarque_bera(resid)
    jb_result = {
        "JB Statistic": round(jb_stat, 4),
        "p-value"     : round(jb_pval, 6),
        "Decision"    : (
            "Reject H₀ — non-normal residuals (attributed to structural shocks)"
            if jb_pval < alpha
            else "Fail to reject H₀ — residuals approximately normal"
        ),
    }

    # ── Top residuals ──────────────────────────────────────────────────────────
    top_resid = (
        resid.abs()
        .nlargest(10)
        .reset_index()
        .rename(columns={"index": "Date", 0: "Abs Residual"})
    )
    top_resid["Actual Residual"] = resid[top_resid["Date"]].values

    # ── Console output ─────────────────────────────────────────────────────────
    print(f"\n           Residual mean  = {resid_stats['Mean']:.6f}")
    print(f"           Residual SD    = {resid_stats['Std Dev']:.4f}")
    print(f"           Skewness       = {resid_stats['Skewness']:.4f}")
    print(f"           Kurtosis       = {resid_stats['Excess Kurtosis']:.4f}")
    print(f"           Sig ACF lags   = "
          f"{sig_lags if sig_lags else 'None — white noise confirmed ✓'}")
    for rec in lb_records:
        print(f"           Ljung-Box Q({rec['Lag']:>2}) : "
              f"stat={rec['Q-statistic']:.4f}  "
              f"p={rec['p-value']:.4f}  {rec['Decision']}")
    print(f"           Jarque-Bera    : "
          f"stat={jb_result['JB Statistic']:.4f}  "
          f"p={jb_result['p-value']:.6f}  {jb_result['Decision'][:40]}")
    print(f"           Largest residual: "
          f"{resid.abs().max():.4f}  "
          f"on {resid.abs().idxmax().strftime('%B %Y')}")

    # ── Figure 4.4 — Four-panel residual diagnostic chart ─────────────────────
    try:
        plt.style.use(fig_style)
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Figure 4.4 — Residual Diagnostics: ARIMA({p},{d},{q})",
        fontsize=12, fontweight="bold", y=1.01
    )

    # Panel 1: Residual time series
    ax1 = axes[0, 0]
    ax1.plot(resid.index, resid.values, color="#1F4E79", linewidth=0.9)
    ax1.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax1.set_title("(i) Residuals over Time", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Residual")

    # Panel 2: ACF of residuals
    ax2 = axes[0, 1]
    lags_range = np.arange(1, 21)
    acf_vals_plot = acf_resid[1:21]
    ax2.bar(lags_range, acf_vals_plot, color="#2E5496", width=0.6)
    ax2.axhline( ci_bound, color="red", linewidth=0.9, linestyle="--",
                label=f"95% CI (±{ci_bound:.3f})")
    ax2.axhline(-ci_bound, color="red", linewidth=0.9, linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("(ii) ACF of Residuals", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Lag"); ax2.set_ylabel("ACF")
    ax2.legend(fontsize=8)

    # Panel 3: Histogram with normal overlay
    ax3 = axes[1, 0]
    ax3.hist(std_resid.values, bins=40, density=True,
             color="#2E5496", alpha=0.7, edgecolor="white",
             label="Standardised residuals")
    x_range = np.linspace(std_resid.min(), std_resid.max(), 200)
    ax3.plot(x_range, norm.pdf(x_range, 0, 1), color="red",
             linewidth=1.5, label="N(0,1) density")
    ax3.set_title("(iii) Histogram with Normal Overlay",
                  fontsize=10, fontweight="bold")
    ax3.set_xlabel("Standardised Residual"); ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)
    ax3.set_xlim(-5, min(std_resid.max() + 1, 15))

    # Panel 4: Q-Q plot
    ax4 = axes[1, 1]
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(
        resid.values, dist="norm"
    )
    ax4.scatter(osm, osr, color="#2E5496", s=10, alpha=0.7,
                label="Residual quantiles")
    x_line = np.array([osm.min(), osm.max()])
    ax4.plot(x_line, slope * x_line + intercept,
             color="red", linewidth=1.5, label="Normal reference line")
    ax4.set_title("(iv) Normal Q-Q Plot", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Theoretical Quantiles")
    ax4.set_ylabel("Sample Quantiles")
    ax4.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"fig4_residual_diagnostics.{fig_format}"),
        dpi=fig_dpi, bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "resid"        : resid,
        "resid_stats"  : resid_stats,
        "sig_acf_lags" : sig_lags,
        "ljungbox"     : lb_df,
        "jarque_bera"  : jb_result,
        "top_residuals": top_resid,
    }
