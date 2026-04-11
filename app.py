import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from src.config import ForecastConfig
from src.forecaster import FoodCPIForecaster

# Set page configuration
st.set_page_config(
    page_title="Nigeria Food CPI Forecaster",
    page_icon="📈",
    layout="wide"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
    <style>
    .main {
        background-color: transparent;
    }
    /* Remove the hardcoded white background from metrics to let Streamlit's theme handle it */
    div[data-testid="stMetric"] {
        background-color: transparent !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'forecaster' not in st.session_state:
    st.session_state.config = ForecastConfig()
    st.session_state.forecaster = FoodCPIForecaster(st.session_state.config)
    # Pre-load data and initial state
    st.session_state.forecaster.load_data()

    # AUTO-LOAD FEATURE:
    # If a saved model order exists in a table, we can use it to immediately fit the model
    # Otherwise, we use the default from config.
    try:
        # We check if the model comparison table exists to see if we have a 'best' order from a previous run
        table_path = os.path.join(st.session_state.config.tables_dir, "table3_model_comparison.csv")
        if os.path.exists(table_path):
            # Load the results and pick the best one (lowest AIC)
            df_conv = pd.read_csv(table_path, index_col=0)
            best_model_name = df_conv['AIC'].idxmin() # e.g., "ARIMA(2,2,2)"
            order_str = best_model_name.replace("ARIMA(", "").replace(")", "").split(",")
            best_order = tuple(map(int, order_str))
            st.session_state.config.model_order = best_order
            st.session_state.forecaster.best_order = best_order
    except Exception as e:
        st.session_state.forecaster.best_order = st.session_state.config.model_order

    # Fit the model immediately on startup so the charts aren't empty
    st.session_state.forecaster.estimate_model(auto_optimize=False)
    st.session_state.forecaster.generate_forecast(steps=st.session_state.config.forecast_steps)

forecaster = st.session_state.forecaster
config = st.session_state.config

# --- Sidebar Configuration ---
st.sidebar.header("🛠️ Model Configuration")
st.sidebar.markdown("Adjust parameters to update the forecast in real-time.")

st.sidebar.subheader("ARIMA Order (p, d, q)")
p = st.sidebar.slider("AR Order (p)", 0, config.p_max, 2)
d = st.sidebar.number_input("Integration Order (d)", value=config.d, step=1)
q = st.sidebar.slider("MA Order (q)", 0, config.q_max, 2)

st.sidebar.subheader("Forecast Settings")
steps = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, config.forecast_steps)
auto_opt = st.sidebar.checkbox("Use Auto-ARIMA Optimization", value=True)

if st.sidebar.button("🚀 Update Model"):
    # Update config values
    config.model_order = (p, d, q)
    config.forecast_steps = steps

    # Re-run estimation and forecast
    forecaster.estimate_model(auto_optimize=auto_opt)
    forecaster.generate_forecast(steps=steps)
    st.sidebar.success("Model Updated!")

# --- Main Dashboard UI ---
st.title("📈 Nigeria Food Consumer Price Index Forecast")
st.markdown("An ARIMA-Based Rolling Validation Framework for Food Price Analysis")

# 1. Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

# Get the best order from the forecaster
best_order = forecaster.best_order if forecaster.best_order else config.model_order

with col1:
    st.metric("Selected Model", f"ARIMA{best_order}")
with col2:
    # We can't easily get MAPE without running validation, but we can show the latest value
    latest_val = forecaster.series.iloc[-1] if forecaster.series is not None else 0
    st.metric("Latest CPI Value", f"{latest_val:.2f}")
with col3:
    st.metric("Data Span", f"{forecaster.series.index[0].year} - {forecaster.series.index[-1].year}")
with col4:
    st.metric("Observations (n)", len(forecaster.series))

# 2. Visualizations
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🔍 Diagnostics", "📊 Validation"])

with tab1:
    st.subheader("Food CPI Level Series & Forecast")

    # We'll generate the plot using the existing logic but display it in Streamlit
    # Since generate_forecast saves a file, we can read that file or just re-plot
    # For better performance and interactivity, we'll re-plot here
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use(config.fig_style)

    ax.plot(forecaster.series, label="Historical Food CPI", color='black')

    # Generate forecast on the fly for the plot
    if forecaster.model_results:
        forecast_obj = forecaster.model_results.get_forecast(steps=steps)
        forecast_values = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=1 - config.pi_level)

        forecast_dates = pd.date_range(
            start=pd.to_datetime(config.forecast_start),
            periods=steps,
            freq='MS'
        )

        ax.plot(forecast_dates, forecast_values, label="ARIMA Forecast", color='blue', linewidth=2)
        ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='blue', alpha=0.2, label="95% Confidence Interval")

    ax.set_title(f"Nigeria Food CPI Forecast: ARIMA{best_order}")
    ax.set_ylabel("Index Value")
    ax.legend()
    st.pyplot(fig)

    # Forecast Table
    if forecaster.model_results:
        st.subheader("Forecasted Values")
        # Re-generate the forecast dataframe
        forecast_obj = forecaster.model_results.get_forecast(steps=steps)
        forecast_df = pd.DataFrame({
            "Month": pd.date_range(start=pd.to_datetime(config.forecast_start), periods=steps, freq='MS'),
            "Forecast": forecast_obj.predicted_mean.values,
            "Lower CI": forecast_obj.conf_int(alpha=1-config.pi_level).iloc[:, 0].values,
            "Upper CI": forecast_obj.conf_int(alpha=1-config.pi_level).iloc[:, 1].values
        })
        st.table(forecast_df.set_index("Month"))

with tab2:
    st.subheader("Model Residuals & Diagnostics")
    if st.button("Run Diagnostics"):
        diag_results = forecaster.run_diagnostics()
        # The run_diagnostics method saves fig4_residual_diagnostics.png
        # We can load and display it
        fig_path = f"{config.figures_dir}/fig4_residual_diagnostics.png"
        if os.path.exists(fig_path):
            st.image(fig_path, caption="Residual Analysis (Time Series, Histogram, ACF, Q-Q Plot)")

        st.write("Ljung-Box Test Results:")
        st.dataframe(diag_results["ljungbox"])

with tab3:
    st.subheader("Walk-Forward Validation (Multi-Horizon)")
    if st.button("Run Validation"):
        with st.spinner("Computing errors across horizons..."):
            val_df = forecaster.validate_walk_forward(horizons=[1, 3, 6, 12])
            st.write("Accuracy Metrics per Horizon:")
            st.dataframe(val_df)

            fig_path = f"{config.figures_dir}/fig5_rolling_origin.png"
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Forecast Accuracy Decay (MAPE)")

# Footer
st.markdown("---")
st.markdown(f"**Developed by Adeeko Oluwaseun Victor** | PGD Computer Science, Babcock University")
