# =============================================================================
# phases/phase_a_data.py
# =============================================================================
# Phase A: Data Loading and Preparation
#
# Loads the NBS Food CPI series from the Excel file, handles the mixed
# abbreviated/full month name format, correctly assigns 2024 year labels,
# and computes descriptive statistics.
#
# Returns
# -------
# series      : pd.Series  — Food CPI indexed by monthly DatetimeIndex
# desc_stats  : dict       — descriptive statistics for Table 4.1
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


def load_and_prepare(cfg: dict):
    """
    Load and prepare the NBS Food CPI series.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary from main.py.

    Returns
    -------
    series     : pd.Series   Monthly Food CPI, DatetimeIndex.
    desc_stats : dict        Descriptive statistics for Table 4.1.
    """

    # ── 1. Read raw Excel file ────────────────────────────────────────────────
    df_raw = pd.read_excel(
        cfg["data_path"],
        sheet_name=cfg["sheet_name"],
        header=None
    )

    year_col     = cfg["year_col"]
    month_col    = cfg["month_col"]
    food_col     = cfg["food_cpi_col"]
    month_map    = cfg["month_map"]
    row_start    = cfg["data_row_start"]
    pre_end      = cfg["pre2024_row_end"]
    y24_start    = cfg["y2024_row_start"]
    y24_end      = cfg["y2024_row_end"]
    series_start = pd.Timestamp(cfg["series_start"])

    # ── 2. Extract pre-2024 rows ───────────────────────────────────────────────
    # Rows from data_row_start up to (not including) pre2024_row_end
    pre = df_raw.iloc[row_start:pre_end, [year_col, month_col, food_col]].copy()
    pre.columns = ["Year", "Month", "FoodCPI"]

    # Forward-fill the year column (only populated on January rows)
    pre["Year"] = pre["Year"].ffill()
    pre["Year"] = pd.to_numeric(pre["Year"], errors="coerce")

    # Keep only valid month rows and years >= series_start year
    pre = pre[pre["Month"].isin(month_map.keys())]
    pre = pre.dropna(subset=["Year", "FoodCPI"])
    pre["Year"] = pre["Year"].astype(int)
    pre["FoodCPI"] = pd.to_numeric(pre["FoodCPI"], errors="coerce")
    pre["MonthNum"] = pre["Month"].map(month_map)
    pre["Date"] = pd.to_datetime(
        dict(year=pre["Year"], month=pre["MonthNum"], day=1)
    )
    pre = pre[pre["Date"] >= series_start]

    # ── 3. Extract 2024 rows ───────────────────────────────────────────────────
    # These rows have NaN in the year column (not forward-filled to avoid
    # contaminating pre-2024 data). Year is set explicitly to 2024.
    y24 = df_raw.iloc[y24_start:y24_end, [year_col, month_col, food_col]].copy()
    y24.columns = ["Year", "Month", "FoodCPI"]
    y24["Year"] = 2024
    y24["FoodCPI"] = pd.to_numeric(y24["FoodCPI"], errors="coerce")
    y24 = y24[y24["Month"].isin(month_map.keys())]
    y24 = y24.dropna(subset=["FoodCPI"])
    y24["MonthNum"] = y24["Month"].map(month_map)
    y24["Date"] = pd.to_datetime(
        dict(year=y24["Year"], month=y24["MonthNum"], day=1)
    )

    # ── 4. Combine and build final series ─────────────────────────────────────
    combined = pd.concat(
        [pre[["Date", "FoodCPI"]], y24[["Date", "FoodCPI"]]]
    )
    combined = (
        combined
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )
    combined.index.freq = pd.tseries.frequencies.to_offset("MS")

    series_end = pd.Timestamp(cfg["series_end"])
    series = combined["FoodCPI"].dropna()
    series = series[series.index <= series_end]

    # ── 5. Descriptive statistics — Table 4.1 ─────────────────────────────────
    desc_stats = {
        "n"                         : len(series),
        "Mean"                      : round(series.mean(),   4),
        "Median"                    : round(series.median(), 4),
        "Standard Deviation"        : round(series.std(),    4),
        "Minimum"                   : round(series.min(),    4),
        "Maximum"                   : round(series.max(),    4),
        "Skewness"                  : round(series.skew(),   4),
        "Excess Kurtosis"           : round(series.kurt(),   4),
        "Coefficient of Variation (%)": round(
            series.std() / series.mean() * 100, 2
        ),
    }

    # ── 6. Save Table 4.1 to CSV ───────────────────────────────────────────────
    tables_dir = cfg["tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)
    desc_df = pd.DataFrame(
        list(desc_stats.items()), columns=["Statistic", "Value"]
    )
    desc_df.to_csv(
        os.path.join(tables_dir, "table1_descriptive_stats.csv"), index=False
    )

    # ── 7. Figure 4.1 — Time series level plot ────────────────────────────────
    figures_dir = cfg["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    fig_style   = cfg["fig_style"]
    fig_dpi     = cfg["fig_dpi"]
    fig_format  = cfg["fig_format"]
    break_dates = cfg["break_dates"]

    try:
        plt.style.use(fig_style)
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(series.index, series.values, color="#1F4E79", linewidth=1.5,
            label="Food CPI (Food & Non-Alcoholic Beverages)")

    colours = ["#C00000", "#FF8C00", "#7030A0"]
    for (bd, label), colour in zip(break_dates, colours):
        ts = pd.Timestamp(bd)
        if ts in series.index or ts > series.index[0]:
            ax.axvline(ts, color=colour, linewidth=1.2, linestyle="--", alpha=0.8)
            ax.text(ts, series.max() * 0.72, label,
                    fontsize=7.5, color=colour, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_title(
        "Figure 4.1 — Nigeria Monthly Food CPI: January 2010 – October 2024\n"
        f"(Base: {cfg['base_period']})",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Food CPI Index Value", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, f"fig1_level_series.{fig_format}"),
        dpi=fig_dpi, bbox_inches="tight"
    )
    plt.close(fig)

    return series, desc_stats
