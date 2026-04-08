# =============================================================================
# phases/phase_b_stationarity.py
# =============================================================================
# Phase B: Stationarity Testing (Step 1 of Box-Jenkins Procedure)
#
# Applies ADF and KPSS unit root tests sequentially to:
#   - Level series
#   - First-differenced series  (d=1)
#   - Second-differenced series (d=2)
#
# Returns a DataFrame of results for Table 4.2.
# =============================================================================

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def run_stationarity_tests(series: pd.Series, cfg: dict) -> pd.DataFrame:
    """
    Run ADF and KPSS tests on the level, first-differenced, and
    second-differenced Food CPI series.

    Parameters
    ----------
    series : pd.Series  Food CPI level series.
    cfg    : dict       Configuration dictionary from main.py.

    Returns
    -------
    results_df : pd.DataFrame  Full results table for Table 4.2.
    """

    adf_autolag  = cfg["adf_autolag"]
    kpss_reg     = cfg["kpss_regression"]
    kpss_nlags   = cfg["kpss_nlags"]
    alpha        = cfg["significance_level"]
    tables_dir   = cfg["tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)

    # ── Helper: run one ADF test ───────────────────────────────────────────────
    def _adf(s):
        res = adfuller(s, autolag=adf_autolag)
        stat      = round(res[0], 4)
        pval      = round(res[1], 4)
        lags_used = res[2]
        cv1       = round(res[4]["1%"],  4)
        cv5       = round(res[4]["5%"],  4)
        cv10      = round(res[4]["10%"], 4)
        decision  = (
            "Stationary (reject H₀)"
            if pval < alpha
            else "Non-stationary (fail to reject H₀)"
        )
        return {
            "ADF Statistic" : stat,
            "ADF p-value"   : pval,
            "ADF CV 1%"     : cv1,
            "ADF CV 5%"     : cv5,
            "ADF CV 10%"    : cv10,
            "ADF Lags"      : lags_used,
            "ADF Decision"  : decision,
        }

    # ── Helper: run one KPSS test ──────────────────────────────────────────────
    def _kpss(s):
        stat, pval, lags_used, cv = kpss(
            s, regression=kpss_reg, nlags=kpss_nlags
        )
        stat = round(stat, 4)
        pval = round(pval, 4)
        cv1  = round(cv["1%"],  4)
        cv5  = round(cv["5%"],  4)
        cv10 = round(cv["10%"], 4)
        decision = (
            "Non-stationary (reject H₀)"
            if pval < alpha
            else "Stationary (fail to reject H₀)"
        )
        return {
            "KPSS Statistic": stat,
            "KPSS p-value"  : pval,
            "KPSS CV 1%"    : cv1,
            "KPSS CV 5%"    : cv5,
            "KPSS CV 10%"   : cv10,
            "KPSS Lags"     : lags_used,
            "KPSS Decision" : decision,
        }

    # ── Run tests on all three series ─────────────────────────────────────────
    test_series = {
        "Level"               : series,
        "First Difference (d=1)" : series.diff().dropna(),
        "Second Difference (d=2)": series.diff().diff().dropna(),
    }

    rows = []
    for label, s in test_series.items():
        adf_res  = _adf(s)
        kpss_res = _kpss(s)

        # Joint conclusion using decision matrix (Table 3.1)
        adf_stat  = "stationary"  in adf_res["ADF Decision"].lower()
        kpss_stat = "stationary"  in kpss_res["KPSS Decision"].lower() and \
                    "non"         not in kpss_res["KPSS Decision"].lower()

        if adf_stat and kpss_stat:
            joint = "Stationary ✓"
        elif not adf_stat and not kpss_stat:
            joint = "Non-stationary"
        else:
            joint = "Borderline — conservative: treat as non-stationary"

        row = {"Series": label}
        row.update(adf_res)
        row.update(kpss_res)
        row["Joint Conclusion"] = joint
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # ── Save Table 4.2 ─────────────────────────────────────────────────────────
    results_df.to_csv(
        os.path.join(tables_dir, "table2_stationarity_tests.csv"), index=False
    )

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n           Table 4.2 — Stationarity Test Results:")
    for _, row in results_df.iterrows():
        print(
            f"           {row['Series']:<28}  "
            f"ADF={row['ADF Statistic']:>8.4f} p={row['ADF p-value']:.4f} "
            f"[{row['ADF Decision'][:14]}]  |  "
            f"KPSS={row['KPSS Statistic']:.4f} p={row['KPSS p-value']:.4f} "
            f"[{row['KPSS Decision'][:14]}]  →  {row['Joint Conclusion']}"
        )

    return results_df
