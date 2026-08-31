"""
Check whether the disappearance of deep low-price troughs in the rolling
version (vs. the full-foresight version) is driven by the lake/hydro
proration ceiling actually being binding at those hours.

Finds the hours where full-foresight price was very low but the rolling
price came out much higher, then re-solves the rolling backtest up through
those days, printing whether each area's lake_res_constraint was maxed out
(used ~= ceiling) and its shadow price (dual value) at that point.
A nonzero shadow price / used-at-ceiling means the constraint we added was
the active limiting factor -- confirming the proration hypothesis.
"""
import pandas as pd
import pyomo.environ  # noqa: F401
from pyomo.opt import SolverFactory, TerminationCondition

from eoles_dispatch.run.rolling import (
    make_rolling_windows, compute_day_to_month, load_budget_inputs,
    initial_budgets, build_ceilings, extract_committed_state, state_source_day_index,
)
from eoles_dispatch.models.default import build_model

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

comparison = pd.read_csv(f"{run_dir}/outputs/horizon_comparison.csv")
low_cutoff = comparison["price_foresight"].quantile(0.10)
suspects = comparison[comparison["price_foresight"] < low_cutoff].sort_values("diff", ascending=False).head(15)
print("=== biggest 'missing low price' cases (full-foresight was very low, rolling wasn't) ===")
print(suspects[["area", "hour", "price_rolling", "price_foresight", "diff"]].to_string(index=False))

target_days = set()
for _, row in suspects.iterrows():
    for w in windows:
        if row["hour"] in w["committed_hours"]:
            target_days.add(w["day_index"])
            break
max_day = max(target_days)
print(f"\nre-solving days 0..{max_day} to inspect hydro ceilings on the flagged days: {sorted(target_days)}")

committed_results = {}
for window in windows:
    day_index = window["day_index"]
    if day_index > max_day:
        break
    source = state_source_day_index(day_index)
    state = committed_results.get(source) if source is not None else None

    lake_ceiling, thermal_ceiling = build_ceilings(
        day_index, day_to_month, remaining_lake, remaining_thermal, n_days_total
    )
    initial_soc = state["initial_soc"] if state is not None else None
    initial_on = state["initial_on"] if state is not None else None

    model = build_model(
        run_dir,
        initial_soc=initial_soc,
        initial_on=initial_on,
        committed_hours=window["committed_hours"],
        lake_ceiling=lake_ceiling,
        thermal_ceiling=thermal_ceiling,
        window_hours=window["window_hours"],
    )
    opt = SolverFactory("appsi_highs")
    opt.highs_options["solver"] = "ipm"
    opt.highs_options["run_crossover"] = "on"
    results = opt.solve(model, tee=False)
    tc = results.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        raise RuntimeError(f"day {day_index} infeasible: {tc}")

    extracted = extract_committed_state(model, window["committed_hours"], hours_months)

    if day_index in target_days:
        month = day_to_month[day_index]
        dual_dict = dict(model.dual)
        print(f"\n=== day_index {day_index} (hours {window['committed_hours'][0]}-{window['committed_hours'][-1]}, month {month}) ===")
        for a in model.a:
            key = (a, month)
            ceiling = lake_ceiling.get(key)
            used = extracted["lake_used"].get(key, 0.0)
            try:
                constraint = model.lake_res_constraint[a, month]
                shadow = dual_dict.get(constraint, None)
            except KeyError:
                shadow = None
            pct = (used / ceiling * 100) if ceiling else None
            print(f"  area={a}: ceiling={ceiling} used={used} ({pct}% of ceiling) shadow_price={shadow}")

    for key, used in extracted["on_used"].items():
        remaining_thermal[key] = remaining_thermal.get(key, 0.0) - used
    for key, used in extracted["lake_used"].items():
        remaining_lake[key] = remaining_lake.get(key, 0.0) - used
    committed_results[day_index] = extracted
    del model
