"""
Rebuild nuclear_outage_hourly.csv (referenced by "Econometric model.py" as
Exogenous 6, but missing on disk / never committed) from the raw FR_outages.csv
ENTSO-E export.

Same per-unit min-clip logic as eoles-dispatch/build_nuclear_availability.py,
but reports total MW of nuclear capacity OUT (nominal - available) per hour,
summed across units, rather than a 0-1 availability factor -- matching what
the original Econometric model.py column name ('nuclear_outage_mw') implies.

Index is written as the exact same strings as FR_model_data.csv's own index
(hourly, Europe/Paris offset, 2020-01-01 00:00:00+01:00 .. 2025-12-31
23:00:00+01:00) so that `data['nuclear_outage'] = outage['nuclear_outage_mw']`
in Econometric model.py aligns correctly (that assignment happens before
data.index is parsed to datetime, so it's a plain string-label join).
"""
import numpy as np
import pandas as pd

MODEL_DATA_CSV = "FR_model_data.csv"
RAW_OUTAGES_CSV = "FR_outages.csv"
OUT_CSV = "nuclear_outage_hourly.csv"
AREA = "FR"

# 1. hour grid: literally FR_model_data.csv's own index strings
model_data = pd.read_csv(MODEL_DATA_CSV, index_col=0)
index_strings = model_data.index.to_numpy()
hours_utc = pd.to_datetime(index_strings, utc=True)
n_hours = len(hours_utc)
range_start = hours_utc[0]
range_end = hours_utc[-1] + pd.Timedelta(hours=1)
print(f"hour grid: {range_start} .. {range_end} ({n_hours} hours)")

# 2. raw outages
df = pd.read_csv(RAW_OUTAGES_CSV)
df["start"] = pd.to_datetime(df["start"], utc=True, format="mixed")
df["end"] = pd.to_datetime(df["end"], utc=True, format="mixed")
if "biddingzone_domain" in df.columns:
    df = df[df["biddingzone_domain"] == AREA]

nuc_all = df[df["plant_type"] == "Nuclear"].copy()
nuc = nuc_all[nuc_all["docstatus"].isna()].copy()  # Active only
nuc = nuc[(nuc["end"] > range_start) & (nuc["start"] < range_end)].copy()
print(f"active nuclear outage rows in range: {len(nuc)}, units: {nuc['production_resource_id'].nunique()}")

nominal_by_unit = nuc_all.groupby("production_resource_id")["nominal_power"].agg(
    lambda s: s.mode().iloc[0]
)

hour_index = pd.Series(np.arange(n_hours), index=hours_utc)
nuc_by_unit = {uid: g for uid, g in nuc.groupby("production_resource_id")}

total_outage_mw = np.zeros(n_hours)
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
    total_outage_mw += (nominal - unit_avail)

out = pd.DataFrame(
    {"nuclear_outage_mw": total_outage_mw},
    index=index_strings,
)
out.index.name = model_data.index.name or "Date"
out.to_csv(OUT_CSV)

print(f"saved {OUT_CSV}")
print(out["nuclear_outage_mw"].describe())
