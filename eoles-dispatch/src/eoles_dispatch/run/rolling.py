"""Rolling-horizon driver for EOLES-Dispatch.

Builds and solves a sequence of overlapping 3-day windows instead of one
whole-year perfect-foresight model, to remove look-ahead bias from the MCP
series used as a LEAR feature.

Each window covers [day-1, day, day+1] but commits only the middle day:
the day before acts as a warm-up buffer that absorbs boundary-condition
artifacts (unknown pre-window on/off state, SOC, ...), and the day after
acts as a look-ahead buffer that avoids end-of-horizon effects (e.g. a
storage unit draining itself right before the window ends because it gets
no credit for leftover energy). Only the middle day's results are trusted
and stitched into the final output series.
"""


def make_rolling_windows(hours):
    """Split a sorted, consecutive hour range into overlapping 3-day windows.

    Args:
        hours: Sorted list/array of consecutive POSIX hours (int) spanning
            the whole backtest period, with no gaps, and a length that is a
            whole number of 24-hour calendar days (POSIX/UTC hours have no
            DST gaps, so this always holds for a period produced by
            compute_hour_mappings).

    Returns:
        List of dicts, one per calendar day, in chronological order of the
        *committed* day:
            {
                "day_index": int,           # 0-based index into calendar days
                "window_hours": [int, ...], # hours to build the model over
                "committed_hours": [int, ...],  # this window's middle day
            }
        The first and last windows have no previous/next day to draw on and
        fall back to a 2-day window (missing that side's buffer) -- they are
        the only ones with no real prior state / no look-ahead.
    """
    hours = sorted(hours)
    if len(hours) % 24 != 0:
        raise ValueError(
            f"expected a whole number of 24h calendar days, got {len(hours)} hours"
        )

    days = [hours[i : i + 24] for i in range(0, len(hours), 24)]
    n_days = len(days)

    windows = []
    for d in range(n_days):
        window_hours = []
        if d > 0:
            window_hours += days[d - 1]
        window_hours += days[d]
        if d < n_days - 1:
            window_hours += days[d + 1]

        windows.append(
            {
                "day_index": d,
                "window_hours": window_hours,
                "committed_hours": list(days[d]),
            }
        )

    return windows


def state_source_day_index(day_index):
    """Which committed day's end-state a window's initial conditions come from.

    Window `day_index` covers [day_index - 1, day_index, day_index + 1] and
    commits day_index. Its first modeled hour is the start of day
    (day_index - 1), so it needs the real state at the end of day
    (day_index - 2) -- which was committed by the window at day_index - 2,
    not by the immediately preceding window (whose committed day,
    day_index - 1, gets *re-solved* here as this window's own warm-up
    buffer, not reused directly).

    Returns None when no such day exists yet (the first two windows of the
    whole backtest), meaning the cold-start defaults in build_model apply.
    """
    source = day_index - 2
    return source if source >= 0 else None


def extract_committed_state(model, committed_hours, hours_months):
    """Pull the real state and resource usage out of a solved window.

    Only the committed (middle) day's results are trusted, so this reads
    the state at the hour right after it ends (the first hour of the
    window's own look-ahead buffer day) -- not the window's own last hour,
    which belongs to a day that gets re-solved as a *different* window's
    buffer and isn't itself trustworthy.

    Args:
        model: a solved Pyomo ConcreteModel, as returned by build_model.
        committed_hours: this window's middle-day hours (sorted list of int).
        hours_months: {hour: month} lookup, same convention as build_model's
            internal dict (derived from hour_month.csv).

    Returns:
        dict with:
            "next_hour": hour right after the committed day, or None if it
                isn't part of this window (only happens for the very last
                window of the whole backtest -- nothing needs it then).
            "initial_soc": {(area, sto): stored value at next_hour} -- feed
                straight into the next relevant window's initial_soc.
            "initial_on": {(area, thr): on value at next_hour} -- feed into
                initial_on.
            "on_used": {(area, thr): sum of on[h] over committed_hours} --
                subtract from the running yearly EAF budget.
            "lake_used": {(area, month): net hydro output over committed_hours
                in that month} -- subtract from the running monthly hydro
                budget.
    """
    import pyomo.environ as pyo

    from ..config import ETA_IN, ETA_OUT

    last_hour = committed_hours[-1]
    next_hour = last_hour + 1
    have_next = next_hour in model.h

    initial_soc = {}
    initial_on = {}
    if have_next:
        for a in model.a:
            for sto in model.sto:
                initial_soc[(a, sto)] = pyo.value(model.stored[a, sto, next_hour])
            for thr in model.thr:
                initial_on[(a, thr)] = pyo.value(model.on[a, thr, next_hour])

    on_used = {}
    for a in model.a:
        for thr in model.thr:
            on_used[(a, thr)] = sum(pyo.value(model.on[a, thr, h]) for h in committed_hours)

    eta_lake = ETA_IN["lake_phs"] * ETA_OUT["lake_phs"]
    lake_used = {}
    for a in model.a:
        for h in committed_hours:
            month = hours_months[h]
            net = pyo.value(model.gene[a, "lake_phs", h]) - pyo.value(
                model.storage[a, "lake_phs", h]
            ) * eta_lake
            lake_used[(a, month)] = lake_used.get((a, month), 0.0) + net

    return {
        "next_hour": next_hour if have_next else None,
        "initial_soc": initial_soc,
        "initial_on": initial_on,
        "on_used": on_used,
        "lake_used": lake_used,
    }


def load_budget_inputs(run_dir):
    """Read the full-period totals needed to seed the running budgets.

    Reads directly from the run's precomputed inputs/ (built once by
    create_run for the whole backtest), using the same column conventions
    as models/default.py.

    Args:
        run_dir: path to the run directory (containing inputs/).

    Returns:
        dict with:
            "lake_inflows": {(area, month): GWh} -- full month totals.
            "capa": {(area, thr): GW}
            "eaf": {(area, thr): fraction}
    """
    from pathlib import Path

    import pandas as pd

    input_dir = Path(run_dir) / "inputs"

    lake_inflows_df = pd.read_csv(
        input_dir / "lake_inflows.csv", header=None, names=["a", "month", "value"]
    )
    lake_inflows = {
        (row.a, row.month): row.value * 1000  # TWh -> GWh, matches default.py
        for row in lake_inflows_df.itertuples()
    }

    capa_df = pd.read_csv(input_dir / "capa.csv", header=None, names=["a", "tec", "value"])
    capa = {(row.a, row.tec): row.value for row in capa_df.itertuples()}

    eaf_df = pd.read_csv(input_dir / "yEAF.csv", header=None, names=["a", "thr", "value"])
    eaf = {(row.a, row.thr): row.value for row in eaf_df.itertuples()}

    return {"lake_inflows": lake_inflows, "capa": capa, "eaf": eaf}


def compute_day_to_month(days, hours_months):
    """Map each calendar day index to the month its hours belong to.

    Args:
        days: list of per-day hour lists (days[d] = day d's 24 hours), as
            chunked internally by make_rolling_windows.
        hours_months: {hour: month} lookup.

    Returns:
        list of month values, one per day index (uses the day's first hour;
        a calendar day is assumed to belong to a single month).
    """
    return [hours_months[day_hours[0]] for day_hours in days]


def initial_budgets(budget_inputs, n_hours):
    """Compute the starting remaining-budget trackers for a whole backtest.

    Args:
        budget_inputs: the dict returned by load_budget_inputs.
        n_hours: total hours in the whole backtest period (the thermal
            budget spans the whole period, not just one month).

    Returns:
        (remaining_lake, remaining_thermal) dicts:
            remaining_lake: {(area, month): GWh}
            remaining_thermal: {(area, thr): GWh-equivalent on-hours}
    """
    remaining_lake = dict(budget_inputs["lake_inflows"])
    capa = budget_inputs["capa"]
    eaf = budget_inputs["eaf"]
    remaining_thermal = {
        key: capa[key] * eaf[key] * n_hours for key in capa if key in eaf
    }
    return remaining_lake, remaining_thermal


def prorated_ceiling(remaining_budget, periods_left):
    """One committed day's fair share of a remaining budget.

    `periods_left` counts today plus every remaining day of the enclosing
    period (month, for hydro; whole backtest, for thermal EAF) -- so the
    very last day of a month gets periods_left == 1, i.e. its full
    remaining share, avoiding leftover water/quota stranded unused.

    Returns 0 for a non-positive periods_left (bookkeeping edge case)
    instead of raising, so it degrades to "nothing left" rather than
    crashing a whole backtest.
    """
    if periods_left <= 0:
        return 0.0
    return remaining_budget / periods_left


def build_ceilings(day_index, day_to_month, remaining_lake, remaining_thermal, n_days_total):
    """Compute this window's prorated budget ceilings.

    Args:
        day_index: this window's committed day.
        day_to_month: list mapping day_index -> month (compute_day_to_month).
        remaining_lake: running {(area, month): GWh} budget tracker.
        remaining_thermal: running {(area, thr): GWh-equivalent} tracker.
        n_days_total: total days in the whole backtest.

    Returns:
        (lake_ceiling, thermal_ceiling) dicts, ready to pass straight into
        build_model. lake_ceiling only has entries for the committed day's
        own month -- lake_res_rule skips every other month anyway once
        committed_hours doesn't touch it, so there's nothing to prorate
        there.
    """
    month = day_to_month[day_index]
    periods_left_lake = sum(1 for m in day_to_month[day_index:] if m == month)
    periods_left_thermal = n_days_total - day_index

    lake_ceiling = {
        (a, m): prorated_ceiling(budget, periods_left_lake)
        for (a, m), budget in remaining_lake.items()
        if m == month
    }
    thermal_ceiling = {
        key: prorated_ceiling(budget, periods_left_thermal)
        for key, budget in remaining_thermal.items()
    }
    return lake_ceiling, thermal_ceiling


def solve_window(
    run_dir,
    window,
    state,
    remaining_lake,
    remaining_thermal,
    day_to_month,
    n_days_total,
    hours_months,
    solver="highs",
):
    """Build, solve, and extract results for one rolling window.

    Args:
        run_dir: run directory (inputs/ already built for the whole backtest
            by create_run -- this does NOT reformat inputs per window).
        window: one entry from make_rolling_windows.
        state: the extract_committed_state() result from the window at
            state_source_day_index(window["day_index"]), or None for a cold
            start (the first two windows of the whole backtest).
        remaining_lake, remaining_thermal, day_to_month, n_days_total: as
            produced by initial_budgets / compute_day_to_month -- read here,
            not modified (the caller applies extracted_state's on_used /
            lake_used to decrement them, keeping this function a pure
            "solve one window" step).
        hours_months: {hour: month} lookup, passed through to
            extract_committed_state.
        solver: solver name (default "highs", matching solve_run's default).

    Returns:
        (model, extracted_state) tuple.

    Raises:
        RuntimeError if the solver doesn't reach an optimal/feasible
        solution -- this is not caught here so a broken window stops the
        whole backtest loudly instead of silently poisoning later windows'
        state.
    """
    import pyomo.environ  # noqa: F401 -- registers solver plugins
    from pyomo.opt import SolverFactory, TerminationCondition

    from ..models.default import build_model

    day_index = window["day_index"]
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

    solver_name = "appsi_highs" if solver == "highs" else solver
    opt = SolverFactory(solver_name)
    if solver == "highs":
        opt.highs_options["solver"] = "ipm"
        opt.highs_options["run_crossover"] = "on"
    results = opt.solve(model, tee=False)

    tc = results.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        raise RuntimeError(
            f"day_index={day_index}: solver did not find an optimal solution "
            f"(termination condition: {tc})"
        )

    extracted = extract_committed_state(model, window["committed_hours"], hours_months)
    return model, extracted


def run_rolling_backtest(run_dir, solver="highs", verbose=True):
    """Run a full rolling-horizon backtest over an existing run's period.

    Requires the run to already exist (created via create_run, which builds
    inputs/ once for the whole period) -- this reuses those same inputs for
    every window instead of reformatting data per window.

    Args:
        run_dir: path to the run directory.
        solver: solver name, passed through to solve_window.
        verbose: print a one-line progress message per committed day.

    Returns:
        pandas.DataFrame with columns ['hour', 'area', 'price'] -- one row
        per (area, hour) for every committed hour of the whole backtest, in
        chronological order. This is the look-ahead-free MCP series meant
        to feed LEAR, as opposed to the perfect-foresight prices.csv a
        plain `solve_run` produces.
    """
    from pathlib import Path

    import pandas as pd

    run_dir = Path(run_dir)
    input_dir = run_dir / "inputs"

    hours = sorted(pd.read_csv(input_dir / "hours.csv", header=None).squeeze(axis=1).tolist())
    hour_month_df = pd.read_csv(
        input_dir / "hour_month.csv", header=None, names=["hour", "month"]
    )
    hours_months = hour_month_df.set_index("hour")["month"].to_dict()

    windows = make_rolling_windows(hours)
    days = [w["committed_hours"] for w in windows]
    day_to_month = compute_day_to_month(days, hours_months)
    n_days_total = len(windows)

    budget_inputs = load_budget_inputs(run_dir)
    remaining_lake, remaining_thermal = initial_budgets(budget_inputs, n_hours=len(hours))

    committed_results = {}  # day_index -> extract_committed_state() result
    price_rows = []

    for window in windows:
        day_index = window["day_index"]
        source = state_source_day_index(day_index)
        state = committed_results.get(source) if source is not None else None

        if verbose:
            ch = window["committed_hours"]
            print(
                f"[rolling] day {day_index + 1}/{n_days_total} "
                f"(hours {ch[0]}-{ch[-1]})..."
            )

        model, extracted = solve_window(
            run_dir,
            window,
            state,
            remaining_lake,
            remaining_thermal,
            day_to_month,
            n_days_total,
            hours_months,
            solver=solver,
        )

        # Decrement the running budgets by what this committed day actually used.
        for key, used in extracted["on_used"].items():
            remaining_thermal[key] = remaining_thermal.get(key, 0.0) - used
        for key, used in extracted["lake_used"].items():
            remaining_lake[key] = remaining_lake.get(key, 0.0) - used

        committed_results[day_index] = extracted

        # Pull this day's committed prices (dual of the adequacy constraint).
        dual_dict = dict(model.dual)
        for a in model.a:
            for h in window["committed_hours"]:
                price = dual_dict.get(model.adequacy_constraint[a, h], 0.0)
                price_rows.append({"hour": h, "area": a, "price": price})

        del model  # don't hold ~365 solved LPs in memory over a full backtest

    return pd.DataFrame(price_rows).sort_values(["hour", "area"]).reset_index(drop=True)
