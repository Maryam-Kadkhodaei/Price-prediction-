import time
from pathlib import Path

from eoles_dispatch.run.rolling import run_rolling_backtest

# Reuses runs/test_2025_realgas's already-built inputs/ (create_run doesn't
# need to run again -- the water-value fix only changes how rolling.py
# slices the lake_phs budget across days, not the formatted input CSVs).
# Writes to a new output file so rolling_prices_2025_realgas.csv (the
# flat-proration baseline) is untouched, letting you diff the two directly
# to isolate the hydro-weighting effect alone (gas price held constant).
RUN_DIR = "runs/test_2025_realgas"

start = time.time()
prices = run_rolling_backtest(RUN_DIR, buffer_days=1)
elapsed = time.time() - start

out_path = Path(RUN_DIR) / "outputs" / "rolling_prices_2025_realgas_watervalue.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
prices.to_csv(out_path, index=False)

print(f"\ndone in {elapsed/60:.1f} minutes")
print(f"saved to {out_path}")
print(prices.head())
print(len(prices))
