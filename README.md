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
├── main.py                        ← Entry point; all configuration lives here
├── requirements.txt               ← Python dependencies
├── README.md                      ← This file
├── data/
│   └── cpi_OCT2024.xlsx           ← NBS source file (place here before running)
├── phases/
│   ├── __init__.py
│   ├── phase_a_data.py            ← Phase A: Data loading and preparation
│   ├── phase_b_stationarity.py    ← Phase B: ADF and KPSS stationarity tests
│   ├── phase_c_identification.py  ← Phase C: ACF and PACF model identification
│   ├── phase_d_estimation.py      ← Phase D: Candidate model estimation (AIC/BIC)
│   ├── phase_e_diagnostics.py     ← Phase E: Residual diagnostic checking
│   ├── phase_f_validation.py      ← Phase F: Rolling-origin out-of-sample evaluation
│   └── phase_g_forecast.py        ← Phase G: 12-month ahead forecast generation
└── outputs/
    ├── figures/                   ← All PNG plots saved here
    └── tables/                    ← All CSV tables saved here
```

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
```bash
python main.py
```

All outputs (figures and tables) are saved to the `outputs/` directory.

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
| `outputs/figures/fig2_acf_pacf_level.png` | ACF and PACF of the level series |
| `outputs/figures/fig3_acf_pacf_diff.png` | ACF and PACF of the stationary differenced series |
| `outputs/figures/fig4_residual_diagnostics.png` | Four-panel residual diagnostic chart |
| `outputs/figures/fig5_rolling_origin.png` | Rolling-origin actual vs forecast plot |
| `outputs/figures/fig6_forecast.png` | 12-month ahead forecast with prediction intervals |
| `outputs/tables/table1_descriptive_stats.csv` | Descriptive statistics |
| `outputs/tables/table2_stationarity_tests.csv` | ADF and KPSS test results |
| `outputs/tables/table3_model_comparison.csv` | Candidate model AIC/BIC comparison |
| `outputs/tables/table4_ljungbox.csv` | Ljung-Box test results |
| `outputs/tables/table5_accuracy_metrics.csv` | Rolling-origin accuracy metrics |
| `outputs/tables/table6_forecast.csv` | 12-month point forecasts and prediction intervals |
| `outputs/tables/appendix_c_rolling_origin.csv` | Full rolling-origin error table (Appendix C) |
