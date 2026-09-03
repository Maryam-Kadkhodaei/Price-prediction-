"""
Ablation: same January 2019 backtest as smoke_test_jan2, but with a 5-day
rolling window (buffer_days=2: two days of context before the committed
day, two days of look-ahead after) instead of the original 3-day window
(buffer_days=1).

Reuses the inputs/ already built for smoke_test_jan2 -- no need to rebuild
them, run_rolling_backtest just re-solves the windows.
"""
import time
from eoles_dispatch.run.rolling import run_rolling_backtest

RUN_DIR = "runs/smoke_test_jan2"

start = time.time()
prices = run_rolling_backtest(RUN_DIR, buffer_days=2)
elapsed = time.time() - start

out_path = f"{RUN_DIR}/outputs/rolling_prices_5day.csv"
prices.to_csv(out_path, index=False)

print(f"\ndone in {elapsed/60:.1f} minutes")
print(f"saved to {out_path}")
print(prices.head())
print(len(prices))
