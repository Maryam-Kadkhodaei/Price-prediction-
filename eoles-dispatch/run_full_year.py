import time
from eoles_dispatch.run.rolling import run_rolling_backtest

start = time.time()
prices = run_rolling_backtest("runs/my_run")
elapsed = time.time() - start

prices.to_csv("runs/my_run/outputs/rolling_prices_2019.csv", index=False)
print(f"\ndone in {elapsed/60:.1f} minutes")
print(prices.head())
print(len(prices))
