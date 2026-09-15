"""
Build a historical (non-leaky) day-of-week weight table for hydro (lake_phs)
proration in the rolling backtest.

Motivation: rolling.py's prorated_ceiling() currently splits each month's
lake_phs budget FLAT across the remaining days of the month (remaining_budget
/ periods_left). Real hydro operators concentrate water use on high-value
days instead (see README "Known limitation: hydro/thermal budget
proration"). This script computes, from years OTHER than the test year
(2019-2024, never 2025), the average historical share of each month's net
hydro output attributable to each day-of-week -- a legitimate seasonal
prior, not a leak of the test year's own realized dispatch.

net_hydro = lake + phs + eta_phs * phs_in   (same formula as
compute_lake_inflows in run/compute.py, so the weights are consistent with
how the monthly TOTAL budget itself was derived).

Output: scenarios/baseline/hydro_daily_weights.csv, columns
[area, month(1-12), day_of_week(0=Mon..6=Sun), weight]
weight is a *relative* share (mean net hydro MWh for that
area/month/day-of-week cell across 2019-2024), floored at 5% of that
area/month's own max day-of-week weight so no single weekday collapses to
a near-zero ceiling.
"""
import pandas as pd

AREAS = ["FR", "BE", "DE", "CH", "IT", "ES", "UK"]
CALIBRATION_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]  # 2025 (test year) excluded
ETA_PHS = 0.9 * 0.95  # matches ETA_IN["lake_phs"] * ETA_OUT["lake_phs"] in compute.py

rows = []
for area in AREAS:
    frames = []
    for year in CALIBRATION_YEARS:
        path = f"data/{year}/production_{area}.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if "lake" not in df.columns:
            continue
        ts = pd.to_datetime(df["hour"], utc=True, errors="coerce").dt.tz_convert("Europe/Paris")
        lake = df.get("lake", 0.0).fillna(0.0)
        phs = df.get("phs", 0.0).fillna(0.0)
        phs_in = df.get("phs_in", 0.0).fillna(0.0)
        net = lake + phs + ETA_PHS * phs_in
        frames.append(pd.DataFrame({"ts": ts, "net": net}))
    if not frames:
        continue
    hourly = pd.concat(frames, ignore_index=True).dropna(subset=["ts"])
    hourly["date"] = hourly["ts"].dt.date
    hourly["month"] = hourly["ts"].dt.month
    hourly["dow"] = hourly["ts"].dt.dayofweek
    daily = hourly.groupby(["date", "month", "dow"], as_index=False)["net"].sum()
    cell = daily.groupby(["month", "dow"], as_index=False)["net"].mean()
    cell["net"] = cell["net"].clip(lower=0)
    for m in cell["month"].unique():
        sub = cell[cell["month"] == m]
        max_w = sub["net"].max()
        floor = 0.05 * max_w if max_w > 0 else 0.0
        for _, r in sub.iterrows():
            rows.append(
                {"area": area, "month": int(m), "day_of_week": int(r["dow"]),
                 "weight": max(r["net"], floor)}
            )

out = pd.DataFrame(rows).sort_values(["area", "month", "day_of_week"])
out.to_csv("scenarios/baseline/hydro_daily_weights.csv", index=False)
print(out.shape)
print(out.head(20))
print()
# sanity: FR January weekday shape
fr_jan = out[(out.area == "FR") & (out.month == 1)]
print("FR January weights by day-of-week (0=Mon...6=Sun):")
print(fr_jan[["day_of_week", "weight"]].to_string(index=False))
