import pandas as pd
import matplotlib.pyplot as plt
generation_forecast = pd.read_csv(
    "FR_generation_forecast.csv",
    index_col=0
)

generation_forecast.index = (
    pd.to_datetime(generation_forecast.index, utc=True)
    .tz_convert("Europe/Paris")
)

generation_forecast_resampled = (
    generation_forecast
    .resample("h")
    .mean()
    .copy()
)

# Rows where ALL columns are missing
missing = generation_forecast_resampled.isna().all(axis=1)

# Give each consecutive block a group number
group = (missing != missing.shift()).cumsum()

gaps = []

for _, x in generation_forecast_resampled.index.to_series()[missing].groupby(
    group[missing]
):
    gaps.append({
        "start": x.iloc[0],
        "end": x.iloc[-1],
        "hours": len(x)
    })

gaps = pd.DataFrame(gaps)

print(gaps)

for _, row in gaps.iterrows():

    start = row["start"]
    end = row["end"]
    hours = row["hours"]

    missing_index = generation_forecast_resampled.loc[start:end].index

    # 1-hour gap -> average previous and next hour
    if hours == 1:
        t = start

        generation_forecast_resampled.loc[t] = (
            generation_forecast_resampled.loc[t - pd.Timedelta(hours=1)]
            + generation_forecast_resampled.loc[t + pd.Timedelta(hours=1)]
        ) / 2

    # 24 hours or more, if multiple of 24
    elif hours >= 24 and hours % 24 == 0:

        days_back = hours // 24

        source_index = missing_index - pd.Timedelta(days=days_back)

        generation_forecast_resampled.loc[missing_index] = (
            generation_forecast_resampled.loc[source_index].values
        )
print(generation_forecast_resampled.isna().sum())
print(
    generation_forecast_resampled[
        generation_forecast_resampled.isna().any(axis=1)
    ]
)
#----------------------------------------------------------------------------------
price = pd.read_csv("FR_day_ahead_price.csv", index_col=0)
price.index = pd.to_datetime(price.index, utc=True)
price.index = price.index.tz_convert("Europe/Paris")
price = price[price.index.year<2026].copy()
price_resampled = price.resample('h').mean().copy()
price_resampled["price"] = price_resampled["price"].interpolate(method="time").copy()


load_forecast = pd.read_csv("FR_load_forecast.csv", index_col=0)
load_forecast.index = pd.to_datetime(load_forecast.index, utc=True).tz_convert("Europe/Paris")
load_forecast_resampled = load_forecast.resample('h').mean()
load_forecast_resampled[load_forecast_resampled['Forecasted Load'].isna()]
load_forecast_resampled["Forecasted Load"] = load_forecast_resampled["Forecasted Load"].interpolate(method="time").copy()

wind_pv_forecast = pd.read_csv("FR_wind_solar_forecast.csv", index_col=0)
wind_pv_forecast.index = (
    pd.to_datetime(wind_pv_forecast.index, utc=True)
    .tz_convert("Europe/Paris")
)

#### handeling missing data in PV #######
wind_pv_forecast.loc["2020-10-24", "Solar"] = (
    wind_pv_forecast.loc["2020-10-23", "Solar"].values)

wind_pv_forecast.loc["2021-10-30", "Solar"] = \
    wind_pv_forecast.loc["2021-10-29", "Solar"].values

wind_pv_forecast.loc["2022-04-07", "Solar"] = \
    wind_pv_forecast.loc["2022-04-06", "Solar"].values

night = (wind_pv_forecast.index.hour >= 20) | (wind_pv_forecast.index.hour <= 6)

# NaN during night -> 0
wind_pv_forecast.loc[wind_pv_forecast['Solar'].isna() & night,'Solar'] = 0

######## Handeling missing data in onshore wind #############
dates = ["2020-03-18", "2021-02-20", "2021-03-04", '2022-11-12']

for date in dates:
    mask = (
        (wind_pv_forecast.index.date == pd.Timestamp(date).date()) &
        (wind_pv_forecast["Wind Onshore"].isna()))

    for t in wind_pv_forecast.index[mask]:
        wind_pv_forecast.loc[t, "Wind Onshore"] = \
            wind_pv_forecast.loc[t - pd.Timedelta(days=1), "Wind Onshore"]
wind_pv_forecast["Wind Onshore"] = (
    wind_pv_forecast["Wind Onshore"].interpolate(method="linear"))


## handeling Nan in offshore wind ###
first_valid = wind_pv_forecast["Wind Offshore"].first_valid_index()

wind_pv_forecast.loc[
    wind_pv_forecast.index < first_valid,
    "Wind Offshore"
] = 0

wind_pv_forecast_hourly = wind_pv_forecast.resample('h').mean().copy()
missing = wind_pv_forecast_hourly.isna().all(axis=1)

group = (missing != missing.shift()).cumsum()

gaps = []

for _, x in wind_pv_forecast_hourly.index.to_series()[missing].groupby(group[missing]):
    gaps.append({
        "start": x.iloc[0],
        "end": x.iloc[-1],
        "hours": len(x)
    })

gaps = pd.DataFrame(gaps)

for _, row in gaps.iterrows():

    if row["hours"] > 6:

        start = row["start"]
        end = row["end"]

        missing_index = wind_pv_forecast_hourly.loc[start:end].index

        # 48-hour gap -> use data from two days before
        if row["hours"] == 48:
            source_index = missing_index - pd.Timedelta(days=2)

        # Other long gaps -> use previous day
        else:
            source_index = missing_index - pd.Timedelta(days=1)

        wind_pv_forecast_hourly.loc[missing_index] = \
            wind_pv_forecast_hourly.loc[source_index].values

for _, row in gaps.iterrows():

    if row["hours"] == 1:
        t = row["start"]

        wind_pv_forecast_hourly.loc[t] = (
            wind_pv_forecast_hourly.loc[t - pd.Timedelta(hours=1)]
            + wind_pv_forecast_hourly.loc[t + pd.Timedelta(hours=1)]
        ) / 2

print(wind_pv_forecast_hourly.isna().sum())

print(len(price_resampled), len(load_forecast_resampled), len(wind_pv_forecast_hourly))

###### Creat the data for the #############

data = pd.concat(
    [
        price_resampled,
        load_forecast_resampled,
        wind_pv_forecast_hourly,
        generation_forecast_resampled
    ],
    axis=1,
    join="inner"
)

print(data.columns)
print(data.shape)

print("\nMissing values:")
print(data.isna().sum())

print("\nPeriod:")
print(data.index.min())
print(data.index.max())

print("\nDuplicates:")
print(data.index.duplicated().sum())
#print(data.head())
#rint(data.shape)
#print(data.columns)

data.to_csv("FR_model_data.csv")
