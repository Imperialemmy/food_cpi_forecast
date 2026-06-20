# food_cpi_forecast

## Time Series Forecasting of Nigeria's Food Consumer Price Index
### An ARIMA-Based Rolling Validation Framework

**Student:** Adeeko Oluwaseun Victor | PG/25/0274  
**Supervisor:** Dr. Bukola Ajayi  
**Institution:** Babcock University, Department of Computer Science  
**Programme:** Postgraduate Diploma in Computer Science, 2025/2026  

---

## Project Structure

```
food_cpi_forecast/
├── main.py                        ← Batch entry point; runs Phases A–G end to end
├── app.py                         ← Streamlit interactive dashboard
├── requirements.txt               ← Python dependencies
├── README.md                      ← This file
├── data/
│   └── cpi_OCT2024.xlsx           ← NBS source file (place here before running)
├── src/
│   ├── config.py                  ← ForecastConfig dataclass (all parameters)
│   └── forecaster.py              ← FoodCPIForecaster class (Phases A–G as methods)
├── phases/                        ← Legacy functional pipeline (superseded by src/)
│   ├── phase_a_data.py            ← Phase A: Data loading and preparation
│   ├── phase_b_stationarity.py    ← Phase B: ADF and KPSS stationarity tests
│   ├── phase_c_identification.py  ← Phase C: ACF and PACF model identification
│   ├── phase_d_estimation.py      ← Phase D: Candidate model estimation (AIC/BIC)
│   ├── phase_e_diagnostics.py     ← Phase E: Residual diagnostic checking
│   ├── phase_f_validation.py      ← Phase F: Rolling-origin out-of-sample evaluation
│   └── phase_g_forecast.py        ← Phase G: 12-month ahead forecast generation
├── tests/
│   └── test_generic.py            ← End-to-end tests for the generic CSV-upload path
└── outputs/                       ← Created automatically on first run
    ├── figures/                   ← All PNG plots saved here
    └── tables/                    ← All CSV tables saved here
```

> **Architecture note.** The active pipeline lives in `src/` as the
> object-oriented `FoodCPIForecaster` class, used by both `main.py` and
> `app.py`. The `phases/` package is an earlier functional implementation of
> the same seven-phase methodology, retained for reference but no longer
> imported.

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the NBS data file
Copy the NBS CPI Excel file into the `data/` folder and update the
`DATA_FILENAME` variable in `main.py` if the filename differs.

### 3. Run the project

**Batch pipeline** (runs all phases and writes figures/tables to `outputs/`):
```bash
python main.py
```

**Interactive dashboard** (upload data, tune the model, view live charts):
```bash
streamlit run app.py
```

The `outputs/figures/` and `outputs/tables/` directories are created
automatically on first run.

### 4. Run the tests
```bash
python tests/test_generic.py
```

---

## Configuration
All configurable parameters are defined at the top of `main.py`.
No values are hardcoded inside the phase files — every parameter is
passed explicitly from `main.py` through to each phase function.

---

## Outputs

| File | Description |
|---|---|
| `outputs/figures/fig1_level_series.png` | Food CPI time series with structural break annotations |
| `outputs/figures/fig3_acf_pacf_diff.png` | ACF and PACF of the stationary differenced series |
| `outputs/figures/fig4_residual_diagnostics.png` | Four-panel residual diagnostic chart |
| `outputs/figures/fig5_rolling_origin.png` | Walk-forward forecast accuracy decay (MAPE by horizon) |
| `outputs/figures/fig6_forecast.png` | 12-month ahead forecast with prediction intervals |
| `outputs/tables/table1_descriptive_stats.csv` | Descriptive statistics |
| `outputs/tables/table2_stationarity_tests.csv` | ADF and KPSS test results |
| `outputs/tables/table3_model_comparison.csv` | Candidate model AIC/BIC comparison |
| `outputs/tables/table4_ljungbox.csv` | Ljung-Box test results |
| `outputs/tables/table5_accuracy_metrics.csv` | Walk-forward accuracy metrics (MAPE/RMSE/MAE by horizon) |
| `outputs/tables/table6_forecast.csv` | 12-month point forecasts and prediction intervals |

> The interactive dashboard (`app.py`) additionally renders the ACF/PACF of the
> **level** series (Figure 4.2) on its Identification tab; the batch pipeline
> only saves the differenced correlogram (`fig3`).
