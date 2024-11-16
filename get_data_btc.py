import requests
import pandas as pd

url_current = "https://api.binance.com/api/v3/ticker/price"
params_current = {"symbol": "BTCUSDT"}


# Solicitar precios históricos de Bitcoin en USD (últimos 30 días)
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
response = requests.get(url, params=params)
data = response.json()

# Convertir los datos a un DataFrame
prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])[:-1]
prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
prices['timestamp'] +=  pd.to_timedelta(1, unit='h')

response = requests.get(url_current, params=params_current)
current_price_row = {
    "timestamp": pd.Timestamp.now() + pd.Timedelta(days=1),
    "price": float(response.json()["price"])
}
prices = prices._append(current_price_row, ignore_index = True)
prices["price"] = prices["price"].round(2)
print(prices.tail())

# Guardamos datos
prices.to_csv("data/prices.csv", index=False, header=True)