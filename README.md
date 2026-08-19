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
