from eoles_dispatch.run.rolling import run_rolling_backtest

prices = run_rolling_backtest("runs/smoke_test_jan2")
prices.to_csv("runs/smoke_test_jan2/outputs/rolling_prices.csv", index=False)
print(prices.head())
print(len(prices))
