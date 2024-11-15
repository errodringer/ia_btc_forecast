import pandas as pd
from sklearn.preprocessing import MinMaxScaler


stock_prices_path = "data/stock.csv"
btc_prices_path = "data/prices.csv"
output_path = "data/data.csv"

btc_prices = pd.read_csv(btc_prices_path)
btc_prices["Date"] = pd.to_datetime(btc_prices["timestamp"].astype("datetime64[ns]").dt.date)

stock_prices = pd.read_csv(stock_prices_path)
stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])


result = btc_prices\
    .merge(stock_prices, on=["Date"], how="left")\
    .drop(columns=["timestamp"])\
    .fillna(0)

X_cols = [col for col in result.columns if col not in btc_prices.columns]

scaler = MinMaxScaler(feature_range=(0, 1))
result[X_cols] = scaler.fit_transform(result[X_cols])

print(result)

result.to_csv(output_path, index=False, header=True)

