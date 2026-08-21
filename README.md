# Price-Prediction

## Motivation

Previous studies have shown that combining fundamental electricity-market
models with econometric or machine-learning approaches can improve electricity
price forecasting accuracy. In these hybrid approaches, information
derived from a fundamental dispatch model, such as the estimated market-clearing
price, is used as an additional input to a data-driven forecasting model [1-3].

Motivated by these results, this project investigates whether a price signal
derived from the EOLES dispatch model can improve day-ahead electricity price
forecasts for the French market.

## First step: LEAR Benchmark Model

As a first step, the LEAR (Lasso Estimated AutoRegressive) model is implemented
as an econometric benchmark for day-ahead electricity price forecasting[4].

The model produces the 24 hourly electricity prices for day `d` using
information that is available before delivery.


### Input features

The current implementation uses:

- Historical day-ahead electricity prices
- Day-ahead load forecast
- Day-ahead generation forecast
- Day-of-week information
For each forecast day `d`, historical electricity prices from:

- `d-1`
- `d-2`
- `d-3`
- `d-7`

are considered. However for the exogenous variables only lag of one day and one week are considered.

Training period: 2020-2024
Testing period: 2024- 2025

#Result of implementation of the lear and its Limitations

Full-year 2025 backtest, EPEX France day-ahead. **MAE 13.2 €/MWh**, **rMAE 0.43** vs naive weekly persistence — the model beats the benchmark in every month, but:

**Near-zero and negative prices are poorly calibrated.** France's PV penetration produces frequent midday price collapses. The model tracks their shape but overshoots their depth: 884 negative forecasts, half of them on hours that were actually positive (~11% of total error). Clipping at zero would cut MAE by 11.6% — deliberately not applied, since negative prices are the signal that matters for storage dispatch.

**sMAPE is unreliable here.** It ranges 7.5%–98.5% across months while MAE stays within 8–17 €/MWh. Near-zero denominators inflate it regardless of absolute error. Read MAE and rMAE; sMAPE is reported only for comparability with the EPF literature.

**Error is structural, not event-driven.** Removing the 30 worst days (8% of the sample) improves MAE by only 12%. Data cleaning offers little upside.

**Worst hours are 19:00, 18:00 and 08:00** — the demand peaks, i.e. the hours that matter most for trading. Headline MAE understates operationally relevant error.

**Linear specification cannot separate regimes.** LEAR fits one set of coefficients across nuclear-marginal nights, solar-marginal middays and gas-marginal peaks — the likely root cause of the two points above.

**Missing fundamentals:** nuclear availability (dominant French price driver), TTF gas, EU ETS carbon.

**Calibration window untuned:** single 3-year window; the literature recommends ensembling short (8–12 week) and long (3–4 year) windows.

## Planned

- Couple with Pyomo dispatch-model shadow prices (hybrid fundamental + ML)
- Benchmark LightGBM — non-linear learner should handle regime separation
- Add nuclear availability, TTF, EU ETS
- Test calibration-window ensembling




[1] S. Ben Amor, T. Möbius, F. Ziel, and F. Müsgens,
"Bridging an Energy System Model with an Ensemble Deep-Learning Approach
for Electricity Price Forecasting," Preprint, 2026.
[2] R. A. de Marcos, A. Bello, and J. Reneses,
"Electricity price forecasting in the short term hybridising fundamental
and econometric modelling,"
Electric Power Systems Research, vol. 167, pp. 240–251, 2019.
doi: 10.1016/j.epsr.2018.10.034.

[3] P. Beran, A. Vogler, and C. Weber,
"Multi-Day-Ahead Electricity Price Forecasting: A Comparison of Fundamental,
Econometric and Hybrid Models," 2021.

[4] Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). 
Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, 
best practices and an open-access benchmark. Applied Energy, 293, 116983.
