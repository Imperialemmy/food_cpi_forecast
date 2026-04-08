# =============================================================================
# phases/phase_c_identification.py
# =============================================================================
# Phase C: Model Identification — ACF and PACF (Step 2 of Box-Jenkins)
#
# Computes and plots the ACF and PACF of:
#   - The level series (Figure 4.2)
#   - The d-times differenced stationary series (Figure 4.3)
#
# Identifies candidate AR order p and MA order q from spike patterns
# and checks seasonal lags 12 and 24.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def run_identification(series: pd.Series, cfg: dict) -> dict:
    """
    Compute and plot ACF/PACF for identification of candidate p and q.

    Parameters
    ----------
    series : pd.Series  Food CPI level series.
    cfg    : dict       Configuration dictionary from main.py.

    Returns
    -------
    results : dict  Significant lags, candidate grid, CI bounds.
    """

    d           = cfg["d"]
    nlags       = cfg["acf_nlags"]
    pacf_method = cfg["pacf_method"]
    ci_level    = cfg["ci_level"]
    p_max       = cfg["p_max"]
    q_max       = cfg["q_max"]
    figures_dir = cfg["figures_dir"]
    fig_dpi     = cfg["fig_dpi"]
    fig_format  = cfg["fig_format"]
    fig_style   = cfg["fig_style"]

    os.makedirs(figures_dir, exist_ok=True)

    # ── Compute differenced series ─────────────────────────────────────────────
    diff_series = series.copy()
    for _ in range(d):
        diff_series = diff_series.diff()
    diff_series = diff_series.dropna()

    n_diff = len(diff_series)
    z_score = abs(np.percentile(
        np.random.standard_normal(100000),
        (1 - ci_level) / 2 * 100
    ))
    ci_bound = z_score / np.sqrt(n_diff)

    # ── Compute ACF and PACF values for differenced series ────────────────────
    acf_vals  = acf(diff_series,  nlags=nlags, fft=True)
    pacf_vals = pacf(diff_series, nlags=nlags, method=pacf_method)

    # ── Identify significant non-seasonal lags (1 to 11) ─────────────────────
    sig_acf_lags  = [k for k in range(1, 12) if abs(acf_vals[k])  > ci_bound]
    sig_pacf_lags = [k for k in range(1, 12) if abs(pacf_vals[k]) > ci_bound]

    # ── Check seasonal lags ───────────────────────────────────────────────────
    seasonal_check = {}
    for s_lag in [12, 24]:
        if s_lag <= nlags:
            seasonal_check[s_lag] = {
                "ACF" : round(acf_vals[s_lag],  4),
                "PACF": round(pacf_vals[s_lag], 4),
                "ACF_sig" : abs(acf_vals[s_lag])  > ci_bound,
                "PACF_sig": abs(pacf_vals[s_lag]) > ci_bound,
            }

    # ── Print identification summary ──────────────────────────────────────────
    print(f"\n           n (after {d} differences) = {n_diff}")
    print(f"           95% CI bounds = ±{ci_bound:.4f}")
    print(f"           Significant ACF  lags (1–11): {sig_acf_lags}")
    print(f"           Significant PACF lags (1–11): {sig_pacf_lags}")
    for sl, sv in seasonal_check.items():
        print(f"           Seasonal lag {sl}: "
              f"ACF={sv['ACF']:.4f} ({'sig' if sv['ACF_sig'] else 'not sig'})  "
              f"PACF={sv['PACF']:.4f} ({'sig' if sv['PACF_sig'] else 'not sig'})")

    # ── Try to use requested style ────────────────────────────────────────────
    try:
        plt.style.use(fig_style)
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    # ── Figure 4.2 — ACF and PACF of the LEVEL series ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf( series, lags=nlags, ax=axes[0], color="#1F4E79", zero=False)
    plot_pacf(series, lags=nlags, ax=axes[1], color="#1F4E79",
              method=pacf_method, zero=False)
    axes[0].set_title("ACF — Level Series", fontsize=10, fontweight="bold")
    axes[1].set_title("PACF — Level Series", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Lag"); axes[1].set_xlabel("Lag")
    fig.suptitle(
        "Figure 4.2 — ACF and PACF of Food CPI Level Series\n"
        "(Non-stationarity evident from slow ACF decay)",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"fig2_acf_pacf_level.{fig_format}"),
        dpi=fig_dpi, bbox_inches="tight"
    )
    plt.close(fig)

    # ── Figure 4.3 — ACF and PACF of DIFFERENCED series ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf( diff_series, lags=nlags, ax=axes[0], color="#375623", zero=False)
    plot_pacf(diff_series, lags=nlags, ax=axes[1], color="#375623",
              method=pacf_method, zero=False)
    axes[0].set_title(
        f"ACF — {d}-times Differenced Series", fontsize=10, fontweight="bold"
    )
    axes[1].set_title(
        f"PACF — {d}-times Differenced Series", fontsize=10, fontweight="bold"
    )
    axes[0].set_xlabel("Lag"); axes[1].set_xlabel("Lag")
    fig.suptitle(
        f"Figure 4.3 — ACF and PACF of Food CPI after {d}-fold Differencing\n"
        f"(Candidate orders: p ∈ {{0..{p_max}}}, q ∈ {{0..{q_max}}})",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"fig3_acf_pacf_diff.{fig_format}"),
        dpi=fig_dpi, bbox_inches="tight"
    )
    plt.close(fig)

    results = {
        "n_diff"          : n_diff,
        "ci_bound"        : ci_bound,
        "acf_values"      : acf_vals,
        "pacf_values"     : pacf_vals,
        "sig_acf_lags"    : sig_acf_lags,
        "sig_pacf_lags"   : sig_pacf_lags,
        "seasonal_check"  : seasonal_check,
        "diff_series"     : diff_series,
    }
    return results
