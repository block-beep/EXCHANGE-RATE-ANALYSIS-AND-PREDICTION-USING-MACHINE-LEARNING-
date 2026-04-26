import requests
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# ── Config — change PERIOD to "1month", "1year", or "5year" ──
PERIOD = "1year"
TARGET_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CNY"]

# ── Date range ────────────────────────────────
yesterday = date.today() - timedelta(days=1)
if PERIOD == "1month":
    start_date = yesterday - relativedelta(months=1)
elif PERIOD == "1year":
    start_date = yesterday - relativedelta(years=1)
elif PERIOD == "5year":
    start_date = yesterday - relativedelta(years=5)
end_date = yesterday

# ── API Call ──────────────────────────────────
url = f"https://api.frankfurter.dev/v1/{start_date}..{end_date}"
params = {
    "base":    "INR",
    "symbols": ",".join(TARGET_CURRENCIES),
}

response = requests.get(url, params=params)
data = response.json()

# ── Parse into long format ────────────────────
rows = []
for date_str, rates in data.get("rates", {}).items():
    for currency in TARGET_CURRENCIES:
        rate = rates.get(currency)
        rows.append({
            "Date":           date_str,
            "Currency":       currency,
            "INR_per_1_Unit": round(1 / rate, 4) if rate else None,
        })

# ── Build DataFrame ───────────────────────────
dataset = pd.DataFrame(rows)
dataset["Date"]      = pd.to_datetime(dataset["Date"])
dataset["Year"]      = dataset["Date"].dt.year
dataset["Month"]     = dataset["Date"].dt.month
dataset["MonthName"] = dataset["Date"].dt.strftime("%B")
dataset["Quarter"]   = dataset["Date"].dt.quarter.apply(lambda q: f"Q{q}")
dataset["YearMonth"] = dataset["Date"].dt.strftime("%Y-%m")
dataset["DayOfWeek"] = dataset["Date"].dt.day_name()

# Day-over-day % change per currency
dataset = dataset.sort_values(["Currency", "Date"]).reset_index(drop=True)
dataset["DayChange_pct"] = (
    dataset.groupby("Currency")["INR_per_1_Unit"]
    .pct_change() * 100
).round(4)

# ── Power BI reads the variable named 'dataset' ──
dataset

dataset.to_csv("forex_data.csv", index=False)


# ----PREDICTIONS-----

import pandas as pd
from sklearn.linear_model import LinearRegression

# Ensure date format
dataset['Date'] = pd.to_datetime(dataset['Date']) 

# Get all unique currencies
currencies = dataset['Currency'].unique()

all_forecasts = []

# -------- LOOP THROUGH EACH CURRENCY --------
for curr in currencies:

    print(f"Processing {curr}...")

    df = dataset[dataset['Currency'] == curr].copy()
    df = df.sort_values('Date')

    # -------- Feature Engineering --------
    df['lag1'] = df['INR_per_1_Unit'].shift(1)
    df['lag2'] = df['INR_per_1_Unit'].shift(2)
    df['ma7'] = df['INR_per_1_Unit'].rolling(7).mean()

    df = df.dropna()

    # Skip if not enough data
    if len(df) < 10:
        continue

    # -------- Train Model --------
    X = df[['lag1', 'lag2', 'ma7']]
    y = df['INR_per_1_Unit']

    model = LinearRegression()
    model.fit(X, y)

    # -------- FUTURE PREDICTION --------
    future_days = 150
    last_known = df.copy()

    future_predictions = []

    for i in range(future_days):
        last_row = last_known.iloc[-1]

        lag1 = last_row['INR_per_1_Unit']
        lag2 = last_known.iloc[-2]['INR_per_1_Unit']
        ma7 = last_known['INR_per_1_Unit'].tail(7).mean()

        X_new = pd.DataFrame([[lag1, lag2, ma7]], columns=['lag1', 'lag2', 'ma7'])

        next_value = model.predict(X_new)[0]

        next_date = last_row['Date'] + pd.Timedelta(days=1)

        new_row = {
            'Date': next_date,
            'INR_per_1_Unit': next_value
        }

        last_known = pd.concat([last_known, pd.DataFrame([new_row])], ignore_index=True)

        future_predictions.append({
            'Date': next_date,
            'Currency': curr,
            'Predicted_Value': next_value
        })

    # Convert to DataFrame
    temp_df = pd.DataFrame(future_predictions)

    all_forecasts.append(temp_df)

# -------- Combine ALL currencies --------
final_forecast = pd.concat(all_forecasts, ignore_index=True)

# -------- Save --------
final_forecast.to_csv("ALL_CURRENCY_5months_forecast.csv", index=False)

print(final_forecast.head())

# -------- Combine ALL currencies properly --------
final_forecast = pd.concat(all_forecasts, ignore_index=True)

# Ensure correct column order
final_forecast = final_forecast[['Date', 'Currency', 'Predicted_Value']]

# Sort nicely (important for clean output)
final_forecast = final_forecast.sort_values(by=['Date', 'Currency']) 

#ADDITION OF DAYS 
final_forecast['Day'] = final_forecast['Date'].dt.strftime('%a')

# Reset index (VERY IMPORTANT)
final_forecast = final_forecast.reset_index(drop=True)

# -------- Save --------
final_forecast.to_excel("ALL_CURRENCY_5months_LONG.xlsx", index=False)

print(final_forecast.head(10))

#-----TRAINING MODEL AND TEST SPLIT ON THE DATASET ------

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Predictions on training data (or test split if you have)
y_pred = model.predict(X)

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----MAE (Mean Absolute Error) Average error in actual units (₹)---- 
mae = mean_absolute_error(y_test, y_pred)

#---RMSE (Root Mean Squared Error) Penalizes large mistakes more---
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

#---MAPE (Mean Absolute Percentage Error) (gives %). This is what you want for “accuracy %”------
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

#---Accuracy (derived)---
accuracy = 100 - mape

#---Print all---
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE (%):", mape)
print("Model Accuracy (%):", accuracy)


#-----DATA STRUCTURING----
print(final_forecast.head())

wide_forecast = final_forecast.pivot_table(
    index='Date',
    columns='Currency',
    values='Predicted_Value',
    aggfunc='first'   # avoids duplicate issues
)

wide_forecast.columns.name = None
wide_forecast = wide_forecast.reset_index()

wide_forecast = wide_forecast.sort_values('Date')

#----Downloading dataset in excel form---
from google.colab import files
files.download("FINAL_WIDE_FORECAST.xlsx")




