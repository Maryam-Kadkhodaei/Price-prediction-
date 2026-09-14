"""
Same LEAR setup as "Econometric model.py" (calibration_window=3*364,
test period = 2025, same 6 exogenous features incl. rebuilt nuclear_outage),
but:
  - fixed the YT2/2024-start mismatch in the original script's result-reading
    section (years_test=1 here, so epftoolbox names the output file
    ..._YT1_CW1092.csv, and the test period is 2025 only, not 2024-2025)
  - no plt.show() (headless box) -- saves a PNG instead
  - actually prints MAE / RMSE / rMAE vs a naive weekly-persistence benchmark,
    since the original script never computed these numbers
  - saves the forecast+actual as a CSV for the follow-up residual-vs-MCP test
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from epftoolbox.models import evaluate_lear_in_test_dataset

print("EPF Toolbox works")
data = pd.read_csv("FR_model_data.csv", index_col=0)
outage = pd.read_csv("nuclear_outage_hourly.csv", index_col=0)

data["residual gen"] = (
    data["Actual Aggregated"] - data["Wind Onshore"] - data["Wind Offshore"] - data["Solar"]
).copy()
data["nuclear_outage"] = outage["nuclear_outage_mw"].copy()
data = data[
    ["price", "Forecasted Load", "residual gen", "Solar", "Wind Onshore", "Wind Offshore", "nuclear_outage"]
].copy()

data.index = pd.to_datetime(data.index, utc=True).tz_convert("Europe/Paris")

print(data.shape)
print("NaNs after merge:\n", data.isna().sum())

data = data.rename(
    columns={
        "price": "Price",
        "Forecasted Load": "Exogenous 1",
        "residual gen": "Exogenous 2",
        "Solar": "Exogenous 3",
        "Wind Onshore": "Exogenous 4",
        "Wind Offshore": "Exogenous 5",
        "nuclear_outage": "Exogenous 6",
    }
)

dataset = "France"
years_test = 1
calibration_window = 3 * 364
begin_test_date = "01/01/2025 00:00"
end_test_date = "31/12/2025 23:00"

path_datasets_folder = os.path.join(".", "datasets")
path_recalibration_folder = os.path.join(".", "experimental_files")
os.makedirs(path_datasets_folder, exist_ok=True)
os.makedirs(path_recalibration_folder, exist_ok=True)

lear_data = data.copy()
lear_data.index = pd.date_range(start="2020-01-01 00:00:00", periods=len(lear_data), freq="h")
lear_data.index.name = "Date"

np.random.seed(42)
for col in ["Exogenous 3", "Exogenous 5"]:
    lear_data[col] = lear_data[col].astype(float) + np.random.normal(loc=0, scale=1e-6, size=len(lear_data))

lear_data.to_csv(os.path.join(path_datasets_folder, "France.csv"), index_label="Date")
print("France.csv written:", os.path.exists(os.path.join(path_datasets_folder, "France.csv")))

evaluate_lear_in_test_dataset(
    path_recalibration_folder=path_recalibration_folder,
    path_datasets_folder=path_datasets_folder,
    dataset=dataset,
    years_test=years_test,
    calibration_window=calibration_window,
    begin_test_date=begin_test_date,
    end_test_date=end_test_date,
)

print(os.listdir("experimental_files"))

# years_test=1 -> epftoolbox names the file YT1, not YT2 (the original
# script's hardcoded YT2/2024-start was left over from an earlier config).
forecast_path = "experimental_files/LEAR_forecast_datFrance_YT1_CW1092.csv"
forecast = pd.read_csv(forecast_path, index_col=0)
forecast.index = pd.to_datetime(forecast.index)
print(forecast.head())
print(forecast.shape)

forecast_hourly = forecast.stack()
forecast_hourly.index = pd.date_range(start="2025-01-01 00:00:00", periods=len(forecast_hourly), freq="h")
forecast_hourly.name = "LEAR Forecast"

actual = lear_data.loc["2025-01-01 00:00:00":"2025-12-31 23:00:00", "Price"].copy()
actual.name = "Actual Price"

results = pd.concat([actual, forecast_hourly], axis=1).dropna()
print(results.head())
print(len(results))

mae = mean_absolute_error(results["Actual Price"], results["LEAR Forecast"])
rmse = np.sqrt(mean_squared_error(results["Actual Price"], results["LEAR Forecast"]))

naive = results["Actual Price"].shift(168)
valid = results.join(naive.rename("naive")).dropna()
naive_mae = mean_absolute_error(valid["Actual Price"], valid["naive"])
rmae = mae / naive_mae

print(f"\n=== LEAR 2025 full-year (FR) ===")
print(f"MAE  = {mae:.3f} EUR/MWh")
print(f"RMSE = {rmse:.3f} EUR/MWh")
print(f"naive weekly-persistence MAE = {naive_mae:.3f} EUR/MWh")
print(f"rMAE (LEAR / naive) = {rmae:.3f}")

results.to_csv("lear_forecast_2025_with_actual.csv")
print("saved lear_forecast_2025_with_actual.csv")

results.loc["2025-01-01":"2025-01-31"].plot(figsize=(15, 6))
plt.ylabel("Price [EUR/MWh]")
plt.xlabel("Date")
plt.title("Actual vs LEAR Forecast - January 2025")
plt.tight_layout()
plt.savefig("lear_forecast_jan2025.png", dpi=110)
print("saved lear_forecast_jan2025.png")
