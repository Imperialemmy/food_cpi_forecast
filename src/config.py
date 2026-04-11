from dataclasses import dataclass, field
import os
from typing import List, Tuple, Dict

@dataclass
class ForecastConfig:
    # -- Paths -----------------------------------------------------------------
    data_path: str = os.path.join("data", "cpi_OCT2024.xlsx")
    figures_dir: str = os.path.join("outputs", "figures")
    tables_dir: str = os.path.join("outputs", "tables")

    # -- Data extraction parameters --------------------------------------------
    sheet_name: str = "Table2"
    year_col: int = 0
    month_col: int = 1
    food_cpi_col: int = 7
    data_row_start: int = 3
    pre2024_row_end: int = 351
    y2024_row_start: int = 351
    y2024_row_end: int = 361
    series_start: str = "2010-01-01"
    series_end: str = "2024-10-01"
    base_period: str = "November 2009 = 100"

    # -- Structural break annotation dates (YYYY-MM-DD) ------------------------
    break_dates: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("2020-04-01", "COVID-19\nShock"),
        ("2022-02-01", "Russia-Ukraine\nGrain Shock"),
        ("2023-06-01", "Naira Unification\n& Subsidy Removal"),
    ])

    # -- Stationarity test parameters ------------------------------------------
    adf_autolag: str = "AIC"
    kpss_regression: str = "c"
    kpss_nlags: str = "auto"
    significance_level: float = 0.05

    # -- ACF/PACF parameters ---------------------------------------------------
    acf_nlags: int = 30
    pacf_method: str = "ywmle"
    ci_level: float = 0.95

    # -- Candidate model grid ---------------------------------------------------
    p_max: int = 2
    d: int = 2
    q_max: int = 2

    # -- Selected model ---------------------------------------------------------
    model_order: Tuple[int, int, int] = (2, 2, 2)

    # -- Residual diagnostic parameters ------------------------------------------
    ljungbox_lags: List[int] = field(default_factory=lambda: [10, 20])

    # -- Rolling-origin evaluation parameters ------------------------------------
    eval_start: str = "2022-11-01"
    eval_end: str = "2024-09-01"
    seasonal_period: int = 12

    # -- Forecast parameters ----------------------------------------------------
    forecast_steps: int = 12
    forecast_start: str = "2024-11-01"
    pi_level: float = 0.95

    # -- Figure parameters -----------------------------------------------------
    fig_format: str = "png"
    fig_dpi: int = 300
    fig_style: str = "seaborn-v0_8-whitegrid"

    # -- Month name mapping -----------------------------------------------------
    month_map: Dict[str, int] = field(default_factory=lambda: {
        "Jan": 1, "Feb": 2, "Mar": 3,  "Apr": 4,  "May": 5,  "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9,  "Oct": 10, "Nov": 11, "Dec": 12,
        "January": 1,  "February": 2,  "March": 3,    "April": 4,
        "June": 6,     "July": 7,      "August": 8,   "September": 9,
        "October": 10, "November": 11, "December": 12,
    })
