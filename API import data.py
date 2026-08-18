import pandas as pd
from entsoe import EntsoePandasClient

import os

# Better: store your token as an environment variable
API_KEY = "cd8ec880-a866-4d8f-9868-52f6695c24bd"
client = EntsoePandasClient(api_key=API_KEY)

country = "FR"

start = pd.Timestamp("2020-01-01", tz="Europe/Paris"g
end   = pd.Timestamp("2026-01-01", tz="Europe/Paris")
print('start')
prices_day_ahead = client.query_day_ahead_prices(
    country,
    start=start,
    end=end
)

# France
country = "FR"
# ============================================================
# 8. SAVE RAW DATA SEPARATELY FIRST
# ============================================================

gen_forecast = client.query_generation_forecast(country, start=start, end=end)
gen_forecast.to_csv("FR_generation_forecast.csv")

# ============================================================
# 2. OBSERVED DAY-AHEAD SPOT PRICE
# ============================================================

print("Downloading day-ahead prices...")

price = client.query_day_ahead_prices(
    country,
    start=start,
    end=end
)

print(price.head())
print("Price finished.")


# Give the series a clear name
price = price.rename(
    "price"
)

price.to_csv(
    "FR_day_ahead_price.csv"
)

# ============================================================
# 3. DAY-AHEAD LOAD FORECAST
# ============================================================

print("\nDownloading load forecast...")

load_forecast = client.query_load_forecast(
    country,
    start=start,
    end=end
)

print(load_forecast.head())
print(load_forecast.columns)
print("Load forecast finished.")

load_forecast.to_csv(
    "FR_load_forecast.csv"
)

# ============================================================
# 4. DAY-AHEAD WIND + SOLAR FORECAST
# ============================================================

print("\nDownloading wind and solar forecast...")

renewable_forecast = client.query_wind_and_solar_forecast(
    country,
    start=start,
    end=end
)

print(renewable_forecast.head())
print(renewable_forecast.columns)
print("Renewable forecast finished.")

renewable_forecast.to_csv(
    "FR_wind_solar_forecast.csv"
)

# ============================================================
# 5. OBSERVED GENERATION
# ============================================================

print("\nDownloading observed generation...")

generation = client.query_generation(
    country,
    start=start,
    end=end
)

print(generation.head())
print(generation.columns)
print("Observed generation finished.")
generation.to_csv(
    "FR_generation_observed.csv"
)
# ============================================================
# 6. OBSERVED LOAD
# ============================================================
print("\nDownloading observed load...")

load = client.query_load(country, start=start, end=end)
load.to_csv(
    "FR_load.csv"
)
# ============================================================
# 7. SAVE RAW DATA SEPARATELY FIRST
# ============================================================






print("\nAll files saved.")