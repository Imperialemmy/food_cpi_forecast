import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import pmdarima as pm

from src.config import ForecastConfig

warnings.filterwarnings("ignore")

class FoodCPIForecaster:
    """
    An Object-Oriented implementation of the Box-Jenkins ARIMA forecasting pipeline
    for Nigeria's Food Consumer Price Index.
    """
    def __init__(self, config: ForecastConfig):
        self.cfg = config
        self.series: Optional[pd.Series] = None
        self.model_results: Optional[Any] = None
        self.best_order: Optional[Tuple[int, int, int]] = None

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------

    def _save_table(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.cfg.tables_dir, filename)
        df.to_csv(path, index=True)
        self.logger.debug(f"Table saved to {path}")

    def _save_figure(self, fig: plt.Figure, filename: str):
        # Ensure figure and axes backgrounds are fully transparent for Streamlit Dark/Light mode
        fig.patch.set_alpha(0.0)

        # Correct way to set figure facecolor for the Agg backend
        fig.set_facecolor('none')

        for ax in fig.get_axes():
            ax.set_facecolor('none')
            ax.xaxis.label.set_color('#444444') # Neutral dark grey that works on both
            ax.yaxis.label.set_color('#444444')
            ax.tick_params(colors='#444444')
            ax.title.set_color('#444444')

        path = os.path.join(self.cfg.figures_dir, filename.replace('.png', f'.{self.cfg.fig_format}'))
        fig.savefig(path, dpi=self.cfg.fig_dpi, bbox_inches='tight', transparent=True)
        plt.close(fig)
        self.logger.debug(f"Figure saved to {path}")

    # --------------------------------------------------------------------------
    # Phase A: Data Loading & Preparation
    # --------------------------------------------------------------------------

    def load_data(
        self,
        file_obj: Optional[Any] = None,
        year_col: Optional[int] = None,
        month_col: Optional[int] = None,
        value_col: Optional[int] = None,
        sheet_name: Optional[str] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        self.logger.info("Phase A: Loading and preparing Food CPI data...")

        # 1. Handle Input Source
        if file_obj is not None:
            # Uploaded file from Streamlit
            # Determine if it's a CSV or Excel based on file name/type
            file_name = getattr(file_obj, 'name', 'data.xlsx')
            if file_name.endswith('.csv'):
                df_raw = pd.read_csv(file_obj, header=None)
            else:
                df_raw = pd.read_excel(file_obj, sheet_name=sheet_name or self.cfg.sheet_name, header=None)
        else:
            # Fallback to local path
            df_raw = pd.read_excel(self.cfg.data_path, sheet_name=sheet_name or self.cfg.sheet_name, header=None)

        # 2. Resolve Column Indices
        y_col = year_col if year_col is not None else self.cfg.default_year_col
        m_col = month_col if month_col is not None else self.cfg.default_month_col
        v_col = value_col if value_col is not None else self.cfg.default_food_cpi_col

        # 3. Extract Data
        # To support both NBS format and simple CSVs, we use a broad slice and filter
        # instead of hardcoded row ranges.
        data_rows = df_raw.iloc[self.cfg.data_row_start:]

        years = data_rows.iloc[:, y_col].astype(str)
        months = data_rows.iloc[:, m_col].astype(str)
        values = data_rows.iloc[:, v_col]

        # Forward-fill years if sparse (NBS format)
        # We convert to numeric first to identify gaps
        years_numeric = pd.to_numeric(years, errors='coerce')
        if years_numeric.isna().any():
            years_numeric = years_numeric.ffill()

        years = years_numeric.fillna(0).astype(int)
        values = pd.to_numeric(values, errors='coerce')

        # 4. Date Construction
        dates = []
        for y, m in zip(years, months):
            m_clean = str(m).strip()
            m_num = self.cfg.month_map.get(m_clean, None)
            if m_num and y > 0:
                dates.append(f"{y}-{m_num:02d}-01")
            else:
                dates.append(np.nan)

        # Filter out rows with invalid dates or values
        valid_mask = pd.notna(dates) & pd.notna(values)
        filtered_values = values[valid_mask]
        filtered_dates = pd.to_datetime([d for d in dates if pd.notna(d)])

        self.series = pd.Series(filtered_values.values, index=filtered_dates, name="Food_CPI")

        # 5. Range Slicing
        series_start_ts = pd.Timestamp(self.cfg.series_start)
        series_end_ts = pd.Timestamp(self.cfg.series_end)

        # Sort index to ensure it is monotonic before any slicing
        self.series = self.series.sort_index()

        # Use boolean masking instead of .loc slicing to avoid KeyError when exact labels are missing
        self.series = self.series[(self.series.index >= series_start_ts) & (self.series.index <= series_end_ts)]

        # 6. Descriptive stats
        desc_stats_dict = {
            "n": len(self.series),
            "Mean": round(self.series.mean(), 4),
            "Median": round(self.series.median(), 4),
            "Standard Deviation": round(self.series.std(), 4),
            "Minimum": round(self.series.min(), 4),
            "Maximum": round(self.series.max(), 4),
            "Skewness": round(self.series.skew(), 4),
            "Excess Kurtosis": round(self.series.kurt(), 4),
            "Coefficient of Variation (%)": round(self.series.std() / self.series.mean() * 100, 2),
        }
        desc_stats = pd.DataFrame(list(desc_stats_dict.items()), columns=["Statistic", "Value"]).set_index("Statistic")
        self._save_table(desc_stats, "table1_descriptive_stats.csv")

        # Figure 4.1 — Time series level plot
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.style.use(self.cfg.fig_style)
        ax.plot(self.series.index, self.series.values, color="#1F4E79", linewidth=1.5,
                label="Food CPI (Food & Non-Alcoholic Beverages)")

        colours = ["#C00000", "#FF8C00", "#7030A0"]
        for (bd, label), colour in zip(self.cfg.break_dates, colours):
            ts = pd.Timestamp(bd)
            if ts in self.series.index or ts > self.series.index[0]:
                ax.axvline(ts, color=colour, linewidth=1.2, linestyle="--", alpha=0.8)
                ax.text(ts, self.series.max() * 0.72, label,
                        fontsize=7.5, color=colour, ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_title(
            f"Figure 4.1 — Nigeria Monthly Food CPI: {self.series.index[0].strftime('%B %Y')} – {self.series.index[-1].strftime('%B %Y')}\n(Base: {self.cfg.base_period})",
            fontsize=11, fontweight="bold", pad=10
        )
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Food CPI Index Value", fontsize=10)
        ax.legend(fontsize=9)
        fig.tight_layout()
        self._save_figure(fig, "fig1_level_series.png")

        return self.series, desc_stats

    # --------------------------------------------------------------------------
    # Phase B: Stationarity Testing
    # --------------------------------------------------------------------------

    def test_stationarity(self) -> pd.DataFrame:
        self.logger.info("Phase B: Running ADF and KPSS stationarity tests...")

        results = []
        # Test levels, 1st diff, 2nd diff
        for d in [0, 1, 2]:
            # Handle differencing correctly:
            # d=0 -> Use original level series
            # d=1 -> Use first difference
            # d=2 -> Use second difference (diff of the diff)
            if d == 0:
                diff_series = self.series
            else:
                diff_series = self.series.diff(d)

            diff_series = diff_series.dropna()

            # Handle edge case where diff_series might be constant (causes adfuller to fail)
            if diff_series.nunique() <= 1:
                self.logger.warning(f"Series at d={d} is constant. Skipping tests.")
                results.append({
                    "Order": f"d={d}",
                    "ADF Stat": np.nan, "ADF p-value": np.nan,
                    "KPSS Stat": np.nan, "KPSS p-value": np.nan,
                    "Stationary": "No (Constant)"
                })
                continue

            try:
                # ADF
                adf_res = adfuller(diff_series, autolag=self.cfg.adf_autolag)
                # KPSS
                kpss_res = kpss(diff_series, regression=self.cfg.kpss_regression, nlags=self.cfg.kpss_nlags)

                results.append({
                    "Order": f"d={d}",
                    "ADF Stat": adf_res[0],
                    "ADF p-value": adf_res[1],
                    "KPSS Stat": kpss_res[0],
                    "KPSS p-value": kpss_res[1],
                    "Stationary": "Yes" if (adf_res[1] < self.cfg.significance_level and kpss_res[1] > self.cfg.significance_level) else "No"
                })
            except Exception as e:
                self.logger.error(f"Error testing d={d}: {e}")
                results.append({
                    "Order": f"d={d}",
                    "ADF Stat": np.nan, "ADF p-value": np.nan,
                    "KPSS Stat": np.nan, "KPSS p-value": np.nan,
                    "Stationary": "Error"
                })

        res_df = pd.DataFrame(results).set_index("Order")
        self._save_table(res_df, "table2_stationarity_tests.csv")
        return res_df

    # --------------------------------------------------------------------------
    # Phase C: Identification (ACF/PACF)
    # --------------------------------------------------------------------------

    def identify_orders(self) -> Dict[str, Any]:
        self.logger.info("Phase C: Generating ACF and PACF correlograms...")

        # Prepare differenced series
        diff_series = self.series.diff(self.cfg.d).dropna()

        # Compute ACF/PACF
        acf_vals = acf(diff_series, nlags=self.cfg.acf_nlags)
        pacf_vals = pacf(diff_series, nlags=self.cfg.acf_nlags, method=self.cfg.pacf_method)

        # Plotting
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plt.style.use(self.cfg.fig_style)

        # ACF Plot
        plot_acf(diff_series, lags=self.cfg.acf_nlags, ax=axes[0], alpha=self.cfg.ci_level)
        axes[0].set_title(f"ACF of Food CPI (d={self.cfg.d})")

        # PACF Plot
        plot_pacf(diff_series, lags=self.cfg.acf_nlags, ax=axes[1], alpha=self.cfg.ci_level, method=self.cfg.pacf_method)
        axes[1].set_title(f"PACF of Food CPI (d={self.cfg.d})")

        self._save_figure(fig, "fig3_acf_pacf_diff.png")

        return {"acf": acf_vals, "pacf": pacf_vals}

    # --------------------------------------------------------------------------
    # Phase D: Model Estimation and Selection
    # --------------------------------------------------------------------------

    def estimate_model(self, auto_optimize: bool = True) -> pd.DataFrame:
        self.logger.info("Phase D: Estimating ARIMA models...")

        # 1. Grid Search (Manual Baseline)
        grid_results = []
        for p in range(self.cfg.p_max + 1):
            for q in range(self.cfg.q_max + 1):
                try:
                    model = ARIMA(self.series, order=(p, self.cfg.d, q))
                    res = model.fit()
                    grid_results.append({
                        "Model": f"ARIMA({p},{self.cfg.d},{q})",
                        "AIC": res.aic,
                        "BIC": res.bic,
                        "RMSE": np.sqrt(np.mean(res.resid**2))
                    })
                except:
                    continue

        grid_df = pd.DataFrame(grid_results).set_index("Model")

        # 2. Auto-ARIMA Optimization
        if auto_optimize:
            self.logger.info("Applying Auto-ARIMA optimization...")
            auto_model = pm.auto_arima(
                self.series,
                start_p=0, start_q=0,
                max_p=self.cfg.p_max, max_q=self.cfg.q_max,
                d=self.cfg.d,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            self.best_order = auto_model.order
            self.logger.info(f"Auto-ARIMA selected order: {self.best_order}")
        else:
            # Fallback to manually configured order
            self.best_order = self.cfg.model_order

        # Fit the final selected model
        final_model = ARIMA(self.series, order=self.best_order)
        self.model_results = final_model.fit()

        # Save Comparison Table
        self._save_table(grid_df, "table3_model_comparison.csv")

        return grid_df

    # --------------------------------------------------------------------------
    # Phase E: Residual Diagnostics
    # --------------------------------------------------------------------------

    def run_diagnostics(self) -> Dict[str, Any]:
        self.logger.info("Phase E: Running residual diagnostic checks...")

        if self.model_results is None:
            self.logger.error("No model has been fitted. Please run estimate_model() first.")
            raise RuntimeError("Model results are missing. You must call estimate_model() before run_diagnostics().")

        residuals = self.model_results.resid

        # Ljung-Box
        lb_results = []
        for lag in self.cfg.ljungbox_lags:
            lb = acorr_ljungbox(residuals, lags=[lag], return_df=True)
            lb_results.append({
                "Lag": lag,
                "Q-Stat": lb.iloc[0, 0],
                "p-value": lb.iloc[0, 1]
            })

        lb_df = pd.DataFrame(lb_results).set_index("Lag")
        self._save_table(lb_df, "table4_ljungbox.csv")

        # Normality test (Jarque-Bera)
        jb_stat, jb_p = stats.jarque_bera(residuals)

        # Figure: Residuals analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.style.use(self.cfg.fig_style)

        # 1. Residuals Plot
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title("Residuals Over Time")

        # 2. Histogram
        axes[0, 1].hist(residuals, bins=30, edgecolor='black')
        axes[0, 1].set_title("Residuals Distribution")

        # 3. ACF of Residuals
        plot_acf(residuals, ax=axes[1, 0])
        axes[1, 0].set_title("Residuals ACF")

        # 4. QQ Plot
        stats.probplot(residuals, plot=axes[1, 1])
        axes[1, 1].set_title("Normal Q-Q Plot")

        self._save_figure(fig, "fig4_residual_diagnostics.png")

        return {
            "ljungbox": lb_df,
            "jb_p": jb_p,
            "residuals": residuals
        }

    # --------------------------------------------------------------------------
    # Phase F: Walk-Forward Validation
    # --------------------------------------------------------------------------

    def validate_walk_forward(self, horizons: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        self.logger.info("Phase F: Running multi-horizon walk-forward validation...")

        # Evaluation window bounds
        start_date = pd.to_datetime(self.cfg.eval_start)
        end_date = pd.to_datetime(self.cfg.eval_end)

        # Get indices for rolling window
        eval_indices = self.series.index[(self.series.index >= start_date) & (self.series.index <= end_date)]

        horizon_results = []
        for h in horizons:
            h_actuals = []
            h_preds = []

            for origin in eval_indices:
                train = self.series[:origin]
                try:
                    # Forecast h-steps ahead
                    m = ARIMA(train, order=self.best_order).fit()
                    f = m.forecast(steps=h)

                    # Explicitly cast to float to ensure we have a scalar, not a sequence
                    y_pred = float(f.iloc[-1])

                    actual_idx = origin + pd.DateOffset(months=h)
                    if actual_idx in self.series.index:
                        y_actual = float(self.series.loc[actual_idx])
                        h_actuals.append(y_actual)
                        h_preds.append(y_pred)
                except Exception as e:
                    self.logger.debug(f"Forecast failed at origin {origin} for horizon {h}: {e}")
                    continue

            if not h_actuals:
                self.logger.warning(f"No valid forecasts found for horizon {h}")
                continue

            h_actuals = np.array(h_actuals)
            h_preds = np.array(h_preds)

            mape = np.mean(np.abs((h_actuals - h_preds) / h_actuals)) * 100
            rmse = np.sqrt(np.mean((h_actuals - h_preds)**2))
            mae = np.mean(np.abs(h_actuals - h_preds))

            horizon_results.append({
                "Horizon": f"{h} Month(s)",
                "MAPE (%)": mape,
                "RMSE": rmse,
                "MAE": mae
            })

        if not horizon_results:
            self.logger.error("No results generated for any horizon.")
            return pd.DataFrame()

        res_df = pd.DataFrame(horizon_results).set_index("Horizon")
        self._save_table(res_df, "table5_accuracy_metrics.csv")

        # Plot Accuracy Decay
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.style.use(self.cfg.fig_style)
        res_df["MAPE (%)"].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Forecast Accuracy Decay by Horizon")
        ax.set_ylabel("MAPE (%)")
        self._save_figure(fig, "fig5_rolling_origin.png")

        return res_df

    # --------------------------------------------------------------------------
    # Phase G: Final Forecast
    # --------------------------------------------------------------------------

    def generate_forecast(self, steps: int = 12) -> Tuple[pd.DataFrame, Any]:
        self.logger.info(f"Phase G: Generating {steps}-month ahead forecast...")

        # Re-fit on full data
        model = ARIMA(self.series, order=self.best_order)
        res = model.fit()

        forecast_obj = res.get_forecast(steps=steps)
        forecast_values = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=1 - self.cfg.pi_level)

        # Create DataFrame
        forecast_dates = pd.date_range(
            start=pd.to_datetime(self.cfg.forecast_start),
            periods=steps,
            freq='MS'
        )

        forecast_df = pd.DataFrame({
            "Month": forecast_dates,
            "Forecast": forecast_values.values,
            "Lower CI": conf_int.iloc[:, 0].values,
            "Upper CI": conf_int.iloc[:, 1].values
        })

        self._save_table(forecast_df, "table6_forecast.csv")

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.style.use(self.cfg.fig_style)

        # Plot historical
        ax.plot(self.series, label="Historical Food CPI", color='#2c3e50', linewidth=1.5)

        # Plot forecast
        ax.plot(forecast_dates, forecast_values, label="ARIMA Forecast", color='#007bff', linewidth=2)
        ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='#007bff', alpha=0.2, label="95% Confidence Interval")

        ax.set_title(f"Nigeria Food CPI Forecast: ARIMA{self.best_order}")
        ax.set_ylabel("Index Value")
        ax.legend()

        self._save_figure(fig, "fig6_forecast.png")

        return forecast_df, res
