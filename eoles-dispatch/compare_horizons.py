"""
Compare the rolling-horizon (look-ahead-free) price series against a
full-foresight single solve of the same period.

The rolling series was already computed by test_rolling.py and saved to
runs/smoke_test_jan2/outputs/rolling_prices.csv.

This script solves the SAME run_dir the old way: one single solve that sees
the entire month at once (no committed_hours / window_hours restriction),
which is exactly the property ("sees the future") we removed with the
rolling rewrite. Comparing the two isolates that one variable.
"""
import pandas as pd
import pyomo.environ  # noqa: F401
from pyomo.opt import SolverFactory

from eoles_dispatch.models.default import build_model

run_dir = "runs/smoke_test_jan2"

print("Solving full-foresight (whole month, single shot)...")
model = build_model(run_dir)  # all defaults -> sees all 744 hours at once

opt = SolverFactory("appsi_highs")
opt.highs_options["solver"] = "ipm"
opt.highs_options["run_crossover"] = "on"
results = opt.solve(model, tee=False)
print("termination condition:", results.solver.termination_condition)

dual_dict = dict(model.dual)
rows = []
for a in model.a:
    for h in model.h:
        price = dual_dict.get(model.adequacy_constraint[a, h], 0.0)
        rows.append({"hour": h, "area": a, "price": price})

full_foresight = pd.DataFrame(rows).sort_values(["hour", "area"]).reset_index(drop=True)
full_foresight.to_csv("runs/smoke_test_jan2/outputs/full_foresight_prices.csv", index=False)

rolling = pd.read_csv("runs/smoke_test_jan2/outputs/rolling_prices.csv")

merged = rolling.merge(
    full_foresight, on=["hour", "area"], suffixes=("_rolling", "_foresight")
)
merged["diff"] = merged["price_rolling"] - merged["price_foresight"]
merged.to_csv("runs/smoke_test_jan2/outputs/horizon_comparison.csv", index=False)

print()
print("=== overall diff stats (rolling - full_foresight) ===")
print("mean abs diff:", merged["diff"].abs().mean())
print("max abs diff: ", merged["diff"].abs().max())
print("correlation:  ", merged["price_rolling"].corr(merged["price_foresight"]))
print()
print("=== per-area mean/median diff ===")
print(merged.groupby("area")["diff"].agg(["mean", "median", "std", lambda s: s.abs().mean()]))
print()
print("=== % of hours identical (diff < 0.01) ===")
print((merged["diff"].abs() < 0.01).mean() * 100, "%")
