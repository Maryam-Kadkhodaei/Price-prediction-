import pandas as pd

df = pd.read_csv("runs/smoke_test_jan2/outputs/rolling_prices.csv")

print("=== shape / coverage ===")
print("rows:", len(df))
print("areas:", sorted(df["area"].unique()))
print("hours:", df["hour"].nunique(), "min", df["hour"].min(), "max", df["hour"].max())
print("any NaN:", df.isna().any().any())

expected_hours = df["hour"].max() - df["hour"].min() + 1
print("expected distinct hours:", expected_hours, "-- matches:", expected_hours == df["hour"].nunique())

print()
print("=== price range per area ===")
print(df.groupby("area")["price"].agg(["min", "max", "mean", "std"]))

print()
print("=== hour-to-hour jumps, flagged at day-boundary hours ===")
df = df.sort_values(["area", "hour"]).reset_index(drop=True)
df["day"] = (df["hour"] - df["hour"].min()) // 24
df["hour_in_day"] = (df["hour"] - df["hour"].min()) % 24

results = []
for area, g in df.groupby("area"):
    g = g.sort_values("hour").reset_index(drop=True)
    g["prev_price"] = g["price"].shift(1)
    g["jump"] = (g["price"] - g["prev_price"]).abs()
    g["is_boundary"] = g["hour_in_day"] == 0  # first hour of each committed day = window seam
    results.append(g)
all_g = pd.concat(results)

boundary_jumps = all_g[all_g["is_boundary"]]["jump"].dropna()
interior_jumps = all_g[~all_g["is_boundary"]]["jump"].dropna()

print("boundary (day-seam) hour-to-hour jump: mean=%.2f  median=%.2f  max=%.2f" % (
    boundary_jumps.mean(), boundary_jumps.median(), boundary_jumps.max()))
print("interior hour-to-hour jump:            mean=%.2f  median=%.2f  max=%.2f" % (
    interior_jumps.mean(), interior_jumps.median(), interior_jumps.max()))

print()
print("=== top 10 largest jumps overall (area, hour, jump, is_boundary) ===")
top = all_g.dropna(subset=["jump"]).sort_values("jump", ascending=False).head(10)
print(top[["area", "hour", "hour_in_day", "price", "prev_price", "jump", "is_boundary"]].to_string(index=False))
