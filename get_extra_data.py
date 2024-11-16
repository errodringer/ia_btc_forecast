import pandas as pd
import snscrape.modules.twitter as sntwitter


def fetch_tweets_with_snscrape(query="Bitcoin", start_date="2023-01-01", end_date="2023-12-31"):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(f'{query} since:{start_date} until:{end_date}').get_items():
        tweets.append({"date": tweet.date.date(), "text": tweet.content})
    return pd.DataFrame(tweets)

tweets_data = fetch_tweets_with_snscrape("Bitcoin", "2023-01-01", "2023-01-02")

print(tweets_data)


from pytrends.request import TrendReq

def fetch_google_trends(keyword="Bitcoin", timeframe="today 12-m"):
    pytrends = TrendReq()
    pytrends.build_payload([keyword], timeframe=timeframe)
    trends_data = pytrends.interest_over_time()
    trends_data.reset_index(inplace=True)
    trends_data = trends_data[["date", keyword]]
    trends_data.rename(columns={keyword: "google_trends"}, inplace=True)
    return trends_data

google_trends = fetch_google_trends()


def fetch_fear_greed_index(days=365):
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    response = requests.get(url)
    data = response.json()["data"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    df.set_index("timestamp", inplace=True)
    df.rename(columns={"value": "fear_greed_index"}, inplace=True)
    df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"])
    return df[["fear_greed_index"]]

fear_greed_index = fetch_fear_greed_index()

import pandas_ta as ta

def calculate_technical_indicators(prices):
    prices["SMA_30"] = prices["price"].ta.sma(length=30)
    prices["RSI"] = prices["price"].ta.rsi()
    return prices

bitcoin_prices = calculate_technical_indicators(bitcoin_prices)

