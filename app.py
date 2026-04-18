import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    /* Use Streamlit's theme variables for dynamic dark/light mode support */
    .stApp {
        background-color: var(--secondary-background-color);
    }

    h1, h2, h3 {
        color: var(--text-color) !important;
        font-family: 'Inter', sans-serif;
    }

    /* Professional Metric Cards - Theme Aware */
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #C5B358;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
    }

    .metric-label {
        color: var(--secondary-text-color);
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .metric-value {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Welcome Screen Styling */
    .welcome-container {
        text-align: center;
        padding: 3rem;
        background-color: #002147;
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }

    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--secondary-text-color);
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }

    /* Button Overrides */
    .stButton>button {
        background-color: #002147;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #C5B358;
        color: #002147;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'forecaster' not in st.session_state:
    st.session_state.config = ForecastConfig()
    st.session_state.forecaster = FoodCPIForecaster(st.session_state.config)
    st.session_state.data_loaded = False

forecaster = st.session_state.forecaster
config = st.session_state.config

# --- Helper for Metric Cards ---
def render_metric_card(label, value):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header("🛠️ Model Configuration")

# DATA UPLOAD SECTION
st.sidebar.subheader("📂 Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload CPI Data (Excel/CSV)", type=["xlsx", "csv"])

if uploaded_file is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Column Mapping**")
    u_year = st.sidebar.number_input("Year Column Index", value=config.default_year_col, min_value=0)
    u_month = st.sidebar.number_input("Month Column Index", value=config.default_month_col, min_value=0)
    u_val = st.sidebar.number_input("CPI Value Column Index", value=config.default_food_cpi_col, min_value=0)
    u_sheet = st.sidebar.text_input("Sheet Name (Excel only)", value=config.sheet_name)

    if st.sidebar.button("🔄 Process Uploaded File"):
        with st.spinner("Loading and preparing data..."):
            try:
                forecaster.load_data(
                    file_obj=uploaded_file,
                    year_col=u_year,
                    month_col=u_month,
                    value_col=u_val,
                    sheet_name=u_sheet
                )
                last_date = forecaster.series.index[-1]
                config.forecast_start = (last_date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')

                forecaster.estimate_model(auto_optimize=True)
                forecaster.generate_forecast(steps=config.forecast_steps)
                st.session_state.data_loaded = True
                st.sidebar.success("New data processed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")

st.sidebar.markdown("---")
if st.session_state.get('data_loaded', False):
    st.sidebar.subheader("ARIMA Order (p, d, q)")
    p = st.sidebar.slider("AR Order (p)", 0, config.p_max, 2)
    d = st.sidebar.number_input("Integration Order (d)", value=config.d, step=1)
    q = st.sidebar.slider("MA Order (q)", 0, config.q_max, 2)

    st.sidebar.subheader("Forecast Settings")
    steps = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, config.forecast_steps)
    auto_opt = st.sidebar.checkbox("Use Auto-ARIMA Optimization", value=True)

    if st.sidebar.button("🚀 Update Model"):
        config.model_order = (p, d, q)
        config.forecast_steps = steps
        forecaster.estimate_model(auto_optimize=auto_opt)
        forecaster.generate_forecast(steps=steps)
        st.sidebar.success("Model Updated!")
else:
    st.sidebar.warning("Please upload and process data to unlock model settings.")

# --- Main Dashboard UI ---
if not st.session_state.get('data_loaded', False):
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">📈 Nigeria Food CPI Forecaster</div>
            <p>An ARIMA-Based Rolling Validation Framework for Professional Price Analysis</p>
            <p>Please use the <b>Data Ingestion</b> panel in the sidebar to upload your dataset to begin.</p>
        </div>
    """, unsafe_allow_html=True)
    st.info("💡 Tip: If you are using the default NBS dataset, simply upload it and click 'Process'.")
    st.stop()

st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><h1>📈 Nigeria Food Consumer Price Index Forecast</h1><p>Professional ARIMA Rolling Validation Framework</p></div>', unsafe_allow_html=True)

# 1. Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
best_order = forecaster.best_order if forecaster.best_order else config.model_order
latest_val = forecaster.series.iloc[-1] if forecaster.series is not None else 0

with col1:
    render_metric_card("Selected Model", f"ARIMA{best_order}")
with col2:
    render_metric_card("Latest CPI Value", f"{latest_val:.2f}")
with col3:
    render_metric_card("Data Span", f"{forecaster.series.index[0].year} - {forecaster.series.index[-1].year}")
with col4:
    render_metric_card("Observations (n)", str(len(forecaster.series)))

# 2. Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🔍 Identification", "⚙️ Diagnostics", "📊 Validation"])

with tab1:
    # --- Figure 4.1: Historical Level Series ---
    st.subheader("Historical Analysis")
    st.markdown("**Figure 4.1: Nigeria Monthly Food CPI Level Series**")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=forecaster.series.index,
        y=forecaster.series.values,
        mode='lines',
        name='Food CPI',
        line=dict(color='#002147', width=2)
    ))
    fig_hist.update_layout(
        xaxis_title="Date",
        yaxis_title="Index Value",
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # --- Figure 4.6: Forecast ---
    st.subheader("Forecasting Results")
    st.markdown("**Figure 4.6: Nigeria Food CPI: Historical Series and 12-Month Ahead Forecast**")

    if forecaster.model_results:
        # Prepare Plotly Figure
        fig = go.Figure()

        # Historical Data
        fig.add_trace(go.Scatter(
            x=forecaster.series.index,
            y=forecaster.series.values,
            mode='lines',
            name='Historical CPI',
            line=dict(color='#002147', width=2)
        ))

        # Forecast Data
        forecast_df, _ = forecaster.generate_forecast(steps=steps)
        f_dates = pd.to_datetime(forecast_df['Month'])

        fig.add_trace(go.Scatter(
            x=f_dates,
            y=forecast_df['Forecast'],
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='#C5B358', width=3)
        ))

        # Confidence Interval
        fig.add_trace(go.Scatter(
            x=pd.concat([f_dates, f_dates[::-1]]),
            y=pd.concat([forecast_df['Upper CI'], forecast_df['Lower CI'][::-1]]),
            fill='toself',
            fillcolor='rgba(197, 179, 88, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Interval'
        ))

        fig.update_layout(
            title=f"ARIMA{best_order} Forecast",
            xaxis_title="Date",
            yaxis_title="Index Value",
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecasted Values")
        st.table(forecast_df.set_index("Month"))

with tab2:
    st.subheader("Model Identification (ACF & PACF)")

    # Figure 4.2: Level Series Identification
    st.markdown("**Figure 4.2: ACF and PACF of Food CPI Level Series**")

    fig_id_level = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))

    from statsmodels.tsa.stattools import acf, pacf
    l_acf = acf(forecaster.series, nlags=20)
    l_pacf = pacf(forecaster.series, nlags=20)

    fig_id_level.add_trace(go.Bar(x=list(range(len(l_acf))), y=l_acf, marker_color='#002147', name="ACF"), row=1, col=1)
    fig_id_level.add_trace(go.Bar(x=list(range(len(l_pacf))), y=l_pacf, marker_color='#C5B358', name="PACF"), row=2, col=1)

    fig_id_level.update_layout(
        height=600,
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        showlegend=False,
        yaxis1_title="Autocorrelation",
        yaxis2_title="Partial Autocorrelation"
    )
    st.plotly_chart(fig_id_level, use_container_width=True)

    st.markdown("---")

    # Figure 4.3: Second-Differenced Identification
    st.markdown("**Figure 4.3: ACF and PACF of Second-Differenced Food CPI Series**")

    diff_series = forecaster.series.diff(1).diff(1).dropna()
    d_acf = acf(diff_series, nlags=20)
    d_pacf = pacf(diff_series, nlags=20)

    fig_id_diff = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))
    fig_id_diff.add_trace(go.Bar(x=list(range(len(d_acf))), y=d_acf, marker_color='#002147', name="ACF"), row=1, col=1)
    fig_id_diff.add_trace(go.Bar(x=list(range(len(d_pacf))), y=d_pacf, marker_color='#C5B358', name="PACF"), row=2, col=1)

    fig_id_diff.update_layout(
        height=600,
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        showlegend=False,
        yaxis1_title="Autocorrelation",
        yaxis2_title="Partial Autocorrelation"
    )
    st.plotly_chart(fig_id_diff, use_container_width=True)

with tab3:
    st.subheader("Residual Diagnostics")
    st.markdown("**Figure 4.4: ARIMA Residual Analysis**")

    if st.button("🚀 Run Diagnostic Checks"):
        with st.spinner("Analyzing residuals..."):
            if forecaster.model_results:
                # We call run_diagnostics to ensure figures are saved and data is prepared
                diag_results = forecaster.run_diagnostics()

                # Create a 2x2 Plotly Grid for Diagnostics
                fig_diag = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Residuals", "Distribution", "ACF of Residuals", "Q-Q Plot"),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )

                # 1. Residuals Time Series
                fig_diag.add_trace(go.Scatter(x=diag_results['residuals'].index, y=diag_results['residuals'].values,
                                             mode='lines', name='Residuals', line=dict(color='#002147')), row=1, col=1)

                # 2. Histogram/KDE
                fig_diag.add_trace(go.Histogram(x=diag_results['residuals'].values,
                                              name='Distribution', marker_color='#C5B358', nbinsx=30), row=1, col=2)

                # 3. ACF of Residuals
                from statsmodels.tsa.stattools import acf
                res_acf = acf(diag_results['residuals'], nlags=20)
                fig_diag.add_trace(go.Bar(x=list(range(len(res_acf))), y=res_acf,
                                         marker_color='#002147', name="Res ACF"), row=2, col=1)

                # 4. Q-Q Plot (Approximate with Scatter)
                import scipy.stats as stats
                qq = stats.probplot(diag_results['residuals'], dist="norm")
                fig_diag.add_trace(go.Scatter(x=qq[0], y=qq[1], mode='markers',
                                             marker=dict(color='#C5B358'), name="Q-Q Plot"), row=2, col=2)

                fig_diag.update_layout(
                    height=800,
                    template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
                    showlegend=False,
                    title_text="Model Diagnostic Checks"
                )
                st.plotly_chart(fig_diag, use_container_width=True)

                st.markdown("---")
                st.subheader("Ljung-Box Test Results")
                if 'ljungbox' in diag_results:
                    st.table(diag_results['ljungbox'])
                else:
                    st.error("Ljung-Box results not found in diagnostic output.")
            else:
                st.warning("Please update the model first to see diagnostic results.")

with tab4:
    st.subheader("Walk-Forward Validation (Multi-Horizon)")
    st.markdown("**Figure 4.5: Forecast Accuracy Decay by Horizon**")
    if st.button("Run Validation"):
        with st.spinner("Computing errors across horizons..."):
            val_df = forecaster.validate_walk_forward(horizons=[1, 3, 6, 12])

            # Plotly Bar Chart for MAPE
            fig = go.Figure(go.Bar(
                x=val_df.index,
                y=val_df["MAPE (%)"],
                marker_color='#002147'
            ))
            fig.update_layout(
                title="Forecast Accuracy Decay (MAPE %)",
                xaxis_title="Horizon",
                yaxis_title="MAPE (%)",
                template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("Accuracy Metrics per Horizon:")
            st.dataframe(val_df)

st.markdown(f"""
    <div class="footer">
        <b>Developed by Adeeko Oluwaseun Victor</b> | PGD Computer Science, Babcock University<br>
        © 2026 ARIMA-Based Rolling Validation Framework
    </div>
    """, unsafe_allow_html=True)
