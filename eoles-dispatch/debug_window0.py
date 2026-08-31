import pandas as pd
from eoles_dispatch.run.rolling import (
    make_rolling_windows, compute_day_to_month, load_budget_inputs,
    initial_budgets, build_ceilings,
)
from eoles_dispatch.models.default import build_model
from pyomo.opt import SolverFactory
import pyomo.environ

run_dir = "runs/smoke_test_jan2"
input_dir = f"{run_dir}/inputs"

hours = sorted(pd.read_csv(f"{input_dir}/hours.csv", header=None).squeeze(axis=1).tolist())
hour_month_df = pd.read_csv(f"{input_dir}/hour_month.csv", header=None, names=["hour", "month"])
hours_months = hour_month_df.set_index("hour")["month"].to_dict()

windows = make_rolling_windows(hours)
days = [w["committed_hours"] for w in windows]
day_to_month = compute_day_to_month(days, hours_months)
n_days_total = len(windows)

budget_inputs = load_budget_inputs(run_dir)
remaining_lake, remaining_thermal = initial_budgets(budget_inputs, n_hours=len(hours))

window0 = windows[0]
lake_ceiling, thermal_ceiling = build_ceilings(0, day_to_month, remaining_lake, remaining_thermal, n_days_total)
print("lake_ceiling:", lake_ceiling)
print("thermal_ceiling:", thermal_ceiling)
print("window_hours:", window0["window_hours"][0], "-", window0["window_hours"][-1], "count=", len(window0["window_hours"]))
print("committed_hours:", window0["committed_hours"][0], "-", window0["committed_hours"][-1])

model = build_model(
    run_dir,
    committed_hours=window0["committed_hours"],
    lake_ceiling=lake_ceiling,
    thermal_ceiling=thermal_ceiling,
    window_hours=window0["window_hours"],
)
opt = SolverFactory("appsi_highs")
opt.highs_options["solver"] = "ipm"
opt.highs_options["run_crossover"] = "on"
results = opt.solve(model, tee=True, load_solutions=False)
print("termination condition:", results.solver.termination_condition)

CANDIDATES = [
    "initial_soc_constraint",
    "initial_on_constraint",
    "yearly_maxON_constraint",
    "lake_res_constraint",
]


def quiet_solve(label):
    opt2 = SolverFactory("appsi_highs")
    opt2.highs_options["solver"] = "ipm"
    opt2.highs_options["run_crossover"] = "on"
    opt2.highs_options["output_flag"] = False
    result = opt2.solve(model, tee=False, load_solutions=False)
    print(f"[{label}] {result.solver.termination_condition}")
    return result.solver.termination_condition


print("=== baseline (everything active) ===")
quiet_solve("baseline")

for name in CANDIDATES:
    comp = getattr(model, name)
    comp.deactivate()
    quiet_solve(f"WITHOUT {name}")
    comp.activate()
