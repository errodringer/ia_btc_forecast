import pandas as pd
import yfinance as yf

output_path = "data/stock.csv"
btc_prices_path = "data/prices.csv"

data = yf.download(
    ["SPY", "QQQ", "DIA", "ONEQ", "AMZN", "AAPL", "PINK", "MSFT", "GOOGL", "NVDA", "MSFT", "META", "TSLA", "AVGO"],
    period="1y"
)
data = data["Adj Close"].reset_index()
data["Date"] = data["Date"].dt.date
data["Date"] = pd.to_datetime(data["Date"])
data.to_csv(output_path, index=False, header=True)
