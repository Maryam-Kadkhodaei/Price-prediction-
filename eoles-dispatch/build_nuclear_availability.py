"""
Build an hourly nuclear availability factor for one area/year from a raw
ENTSO-E "unavailability of generation units" export, and write it to
data/<year>/nuclear_availability_<area>.csv in the format expected by
compute.py's _load_nuclear_availability_override (and therefore
compute_nuclear_max_af / format_inputs.py / the model's nuc_maxON_rule).

Why this exists: EOLES's nuclear cap was previously derived from *realized*
historical nuclear production (compute_nuclear_max_af's weekly-max-AF
proxy) -- which leaks future information in the rolling-horizon backtest,
since a window's committed/buffer days see production data from days it
wouldn't actually know yet. ENTSO-E's outage/unavailability records are
announced in advance (planned maintenance weeks ahead, unplanned outages as
soon as they occur) and are therefore forecast-safe to use as a nuclear
availability input, unlike realized production.

Logic:
- Keep only Nuclear, Active rows (docstatus is NaN -- Cancelled/Withdrawn
  revisions of the same outage are dropped; ENTSO-E leaves only one Active
  version per outage at a time).
- Each active row = one flat segment [start, end) where the unit's
  *available* capacity is avail_qty (out of nominal_power). curvetype is
  A03 (flat) and pstn is basically always 1 in this export, so there is no
  sub-segment curve to unpack within a row.
- Per unit: build an hourly series initialised at full nominal_power (no
  outage), then for every active row belonging to that unit, take the
  elementwise MIN of the existing value and avail_qty over the overlapping
  hours -- this handles overlapping/duplicate outage records for the same
  unit correctly (worst case wins) instead of double-subtracting them, and
  correctly leaves units with zero outage rows in the period at full
  capacity (rather than silently dropping them, a bug in an earlier
  version of this script).
- Sum available MW across all units, divide by installed capacity to get
  the hourly availability factor, clipped to [0, 1].
- The hour grid is built to exactly match data/<year>/demand_<area>.csv's
  convention: 8760 (or 8784 in a leap year) naive-UTC hourly timestamps
  starting at Dec 31 23:00 of the previous year -- NOT the raw calendar
  Jan 1 00:00 - Dec 31 23:00 range, which is off by one hour from the
  model's own hour grid.

Usage:
    python build_nuclear_availability.py <raw_outages_csv> <area> <year> \
        [--installed-capacity-mw MW] [--data-dir data]

Example (the FR/2019 case this was built for):
    python build_nuclear_availability.py FR_outages_new.csv FR 2019 \
        --installed-capacity-mw 63130
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_hour_grid(data_dir, area, year):
    """Return the exact naive-UTC hourly DatetimeIndex used by this model run.

    Reuses data/<year>/demand_<area>.csv's own 'hour' column as the source
    of truth, rather than reconstructing the year boundary by hand, so this
    always matches whatever convention the rest of the pipeline uses.
    """
    demand_path = Path(data_dir) / str(year) / f"demand_{area}.csv"
    if not demand_path.exists():
        raise FileNotFoundError(
            f"{demand_path} not found -- run "
            f"'eoles-dispatch collect --start {year} --end {int(year) + 1}' first, "
            f"or pass a different --data-dir."
        )
    demand = pd.read_csv(demand_path)
    hours_naive = pd.to_datetime(demand["hour"])
    return hours_naive


def compute_hourly_availability(raw_outages_csv, area, hours_naive, installed_capacity_mw):
    """Compute the hourly nuclear availability factor for one area.

    Args:
        raw_outages_csv: path to a raw ENTSO-E query_unavailability_of_generation_units
            export (one row per outage record/revision; see module docstring
            for the expected columns).
        area: ENTSO-E biddingzone_domain code to filter to (e.g. 'FR').
        hours_naive: naive-UTC DatetimeIndex/Series, the target hour grid
            (from build_hour_grid).
        installed_capacity_mw: total installed nuclear capacity for this
            area/year, in MW (e.g. from data/<year>/installed_capacity_<area>.csv).

    Returns:
        pd.DataFrame with columns ['hour', 'availability_factor'], 'hour'
        formatted as 'YYYY-MM-DD HH:MM:SS' strings matching the other
        data/<year>/*.csv inputs.
    """
    hours_utc = pd.DatetimeIndex(hours_naive).tz_localize("UTC")
    n_hours = len(hours_utc)
    range_start = hours_utc[0]
    range_end = hours_utc[-1] + pd.Timedelta(hours=1)

    df = pd.read_csv(raw_outages_csv)
    df["start"] = pd.to_datetime(df["start"], utc=True, format="mixed")
    df["end"] = pd.to_datetime(df["end"], utc=True, format="mixed")
    if "biddingzone_domain" in df.columns:
        df = df[df["biddingzone_domain"] == area]

    nuc_all = df[df["plant_type"] == "Nuclear"].copy()
    nuc = nuc_all[nuc_all["docstatus"].isna()].copy()  # Active only
    nuc = nuc[(nuc["end"] > range_start) & (nuc["start"] < range_end)].copy()

    # Stable nominal capacity per unit: the mode across all its records
    # (reactors don't change nominal capacity often; this is robust to the
    # occasional data-entry glitch).
    nominal_by_unit = nuc_all.groupby("production_resource_id")["nominal_power"].agg(
        lambda s: s.mode().iloc[0]
    )

    hour_index = pd.Series(np.arange(n_hours), index=hours_utc)
    nuc_by_unit = {uid: g for uid, g in nuc.groupby("production_resource_id")}

    available_mw = np.zeros(n_hours)
    for unit_id, nominal in nominal_by_unit.items():
        unit_avail = np.full(n_hours, nominal, dtype=float)  # default: fully available
        unit_rows = nuc_by_unit.get(unit_id)
        if unit_rows is not None:
            for _, row in unit_rows.iterrows():
                seg_start = max(row["start"], range_start)
                seg_end = min(row["end"], range_end)
                if seg_end <= seg_start:
                    continue
                start_idx = max(hour_index.index.searchsorted(seg_start, side="right") - 1, 0)
                end_idx = min(hour_index.index.searchsorted(seg_end, side="left"), n_hours)
                if end_idx <= start_idx:
                    continue
                avail_qty = max(row["avail_qty"], 0.0)
                unit_avail[start_idx:end_idx] = np.minimum(unit_avail[start_idx:end_idx], avail_qty)
        available_mw += unit_avail

    availability_factor = np.clip(available_mw / installed_capacity_mw, 0.0, 1.0)

    return pd.DataFrame(
        {
            "hour": pd.DatetimeIndex(hours_naive).strftime("%Y-%m-%d %H:%M:%S"),
            "availability_factor": availability_factor,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_outages_csv", help="Raw ENTSO-E unavailability-of-generation-units export")
    parser.add_argument("area", help="Area code, e.g. FR")
    parser.add_argument("year", type=int, help="Simulation year, e.g. 2019")
    parser.add_argument(
        "--installed-capacity-mw",
        type=float,
        default=None,
        help="Installed nuclear capacity in MW. If omitted, read from "
        "data/<year>/installed_capacity_<area>.csv.",
    )
    parser.add_argument("--data-dir", default="data", help="Path to the data/ directory (default: data)")
    args = parser.parse_args()

    hours_naive = build_hour_grid(args.data_dir, args.area, args.year)

    if args.installed_capacity_mw is not None:
        installed_capacity_mw = args.installed_capacity_mw
    else:
        capa_path = Path(args.data_dir) / str(args.year) / f"installed_capacity_{args.area}.csv"
        capa_df = pd.read_csv(capa_path)
        installed_capacity_mw = float(capa_df.set_index("tec").loc["nuclear", "value"]) * 1000  # GW -> MW

    print(f"installed nuclear capacity for {args.area}/{args.year}: {installed_capacity_mw:.1f} MW")

    out = compute_hourly_availability(args.raw_outages_csv, args.area, hours_naive, installed_capacity_mw)

    out_path = Path(args.data_dir) / str(args.year) / f"nuclear_availability_{args.area}.csv"
    out.to_csv(out_path, index=False)

    print(f"saved {out_path}")
    print(out["availability_factor"].describe())


if __name__ == "__main__":
    main()
