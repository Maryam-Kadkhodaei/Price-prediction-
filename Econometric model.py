import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from epftoolbox.models import LEAR
from epftoolbox.models import evaluate_lear_in_test_dataset

print("EPF Toolbox works")
data = pd.read_csv(
    "FR_model_data.csv",
    index_col=0
)

#data = data[['price', 'Forecasted Load', 'Solar',"Wind Onshore", "Wind Offshore"]]

data.index = pd.to_datetime(
    data.index,
    utc=True
).tz_convert("Europe/Paris")

print(data.head())
print(data.columns)
print(data.shape)


data = data.rename(columns={
    "price": "Price",
    "Forecasted Load": "Exogenous 1",
    'Actual Aggregated': "Exogenous 2",# genertion forecast
     "Solar": "Exogenous 3",# forecasted solar
     "Wind Onshore": "Exogenous 4",# forecasted wind onshore
     "Wind Offshore": "Exogenous 5"# forecasted wind offshore
})


print(data.columns)

print(data.isna().sum())
print(data.index.min())
print(data.index.max())


dataset = "France"
years_test = 1
calibration_window = 3 * 364

begin_test_date = "01/01/2025 00:00"
end_test_date   = "31/12/2025 23:00"

path_datasets_folder = os.path.join(".", "datasets")
path_recalibration_folder = os.path.join(".", "experimental_files")

os.makedirs(path_datasets_folder, exist_ok=True)
os.makedirs(path_recalibration_folder, exist_ok=True)

lear_data = data.copy()

# Create a continuous hourly index for EPF Toolbox
lear_data.index = pd.date_range(
    start="2020-01-01 00:00:00",
    periods=len(lear_data),
    freq="h"
)

lear_data.index.name = "Date"
np.random.seed(42)

exogenous_cols = [
    "Exogenous 1",
    "Exogenous 2",
    "Exogenous 3",
    "Exogenous 4",
    "Exogenous 5"

]

for col in exogenous_cols:
    lear_data[col] = (
        lear_data[col].astype(float)
        + np.random.normal(
            loc=0,
            scale=1e-6,
            size=len(lear_data)
        )
    )

lear_data.to_csv(
    os.path.join(path_datasets_folder, "France.csv"),
    index_label="Date"
)

print("Dataset path:", os.path.abspath(path_datasets_folder))
print(
    "France.csv exists:",
    os.path.exists(os.path.join(path_datasets_folder, "France.csv"))
)

evaluate_lear_in_test_dataset(
    path_recalibration_folder=path_recalibration_folder,
    path_datasets_folder=path_datasets_folder,
    dataset=dataset,
    years_test=years_test,
    calibration_window=calibration_window,
    begin_test_date=begin_test_date,
    end_test_date=end_test_date
)

print(os.listdir("experimental_files"))

forecast = pd.read_csv(
    "experimental_files/LEAR_forecast_datFrance_YT2_CW1092.csv",
    index_col=0
)

forecast.index = pd.to_datetime(forecast.index)

print(forecast.head())
print(forecast.shape)

forecast_hourly = forecast.stack()

forecast_hourly.index = pd.date_range(
    start="2024-01-01 00:00:00",
    periods=len(forecast_hourly),
    freq="h"
)

forecast_hourly.name = "LEAR Forecast"

actual = lear_data.loc[
    "2024-01-01 00:00:00":"2025-12-31 23:00:00",
    "Price"
].copy()

actual.name = "Actual Price"

results = pd.concat(
    [actual, forecast_hourly],
    axis=1
)

print(results.head())

results.loc[
    "2024-01-01":"2024-01-07"
].plot(figsize=(14, 6))

plt.ylabel("Price [€/MWh]")
plt.xlabel("Date")
plt.title("LEAR Day-Ahead Price Forecast - France")
plt.grid(True)
plt.tight_layout()
plt.show()

results.loc[
    "2024-01-01":"2024-01-31"
].plot(figsize=(15, 6))

plt.ylabel("Price [€/MWh]")
plt.xlabel("Date")
plt.title("Actual vs LEAR Forecast - January 2024")
plt.tight_layout()
plt.show()


mae = mean_absolute_error(
    results["Actual Price"],
    results["LEAR Forecast"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["Actual Price"],
        results["LEAR Forecast"]
    )
)

print("Overall MAE:", mae)
print("Overall RMSE:", rmse)