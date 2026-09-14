import time
from pathlib import Path

from eoles_dispatch.run.rolling import run_rolling_backtest

RUN_DIR = "runs/test_2025"

start = time.time()
prices = run_rolling_backtest(RUN_DIR, buffer_days=1)
elapsed = time.time() - start

out_path = Path(RUN_DIR) / "outputs" / "rolling_prices_2025.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
prices.to_csv(out_path, index=False)

print(f"\ndone in {elapsed/60:.1f} minutes")
print(f"saved to {out_path}")
print(prices.head())
print(len(prices))
