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
