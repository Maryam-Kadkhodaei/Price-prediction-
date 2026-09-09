"""
Step 1: fetch raw nuclear outage/unavailability records for FR, 2019,
and print their structure so we can build the hourly availability series
correctly (the outage dataset has a more complex per-unit, per-period shape
than the other ENTSO-E endpoints we've used so far).

Run this from your eoles-dispatch (or Day-a-head-price-prediction) venv
where ENTSOE_API_KEY is set, e.g.:

    (.venv_epf) $ python fetch_nuclear_outages_2019.py
"""
import os
import pandas as pd
from entsoe import EntsoePandasClient

API_KEY = os.getenv("ENTSOE_API_KEY")
if not API_KEY:
    raise SystemExit("ENTSOE_API_KEY not set in environment (.env not loaded?)")

client = EntsoePandasClient(api_key=API_KEY)

start = pd.Timestamp("2019-01-01", tz="Europe/Paris")
end = pd.Timestamp("2020-01-01", tz="Europe/Paris")

print("Fetching unavailability of generation units for FR, 2019 ...")
outages = client.query_unavailability_of_generation_units(
    country_code="FR",
    start=start,
    end=end,
)

print("\n=== type ===")
print(type(outages))

print("\n=== shape ===")
print(outages.shape)

print("\n=== columns ===")
print(outages.columns.tolist())

print("\n=== dtypes ===")
print(outages.dtypes)

print("\n=== first 10 rows ===")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(outages.head(10))

# Try to spot the plant-type / technology column and see what values it has,
# so we know how to filter down to nuclear only.
candidate_cols = [c for c in outages.columns if "plant" in c.lower() or "type" in c.lower() or "psr" in c.lower()]
print("\n=== candidate technology columns ===")
print(candidate_cols)
for c in candidate_cols:
    print(f"\n-- unique values in '{c}' --")
    print(outages[c].value_counts().head(20))

# Save the raw thing untouched so we don't have to re-fetch from ENTSO-E again
outages.to_csv("FR_outages_2019.csv")
print("\nSaved raw output to FR_outages_2019.csv")
print("\nDone. Please share the printed output above (columns / dtypes / head / unique technology values).")
