import requests
import pandas as pd


class BitcoinPriceFetcher:
    """
    Clase para obtener precios históricos y actuales de Bitcoin y guardarlos en un archivo CSV.

    Métodos:
        fetch_historical_prices: Obtiene los precios históricos de Bitcoin en USD.
        fetch_current_price: Obtiene el precio actual de Bitcoin en USD.
        save_to_csv: Guarda los precios obtenidos en un archivo CSV.
    """

    def __init__(self, historical_days=365, interval="daily"):
        """
        Inicializa el objeto BitcoinPriceFetcher con parámetros para la consulta histórica.

        Args:
            historical_days (int): Número de días de datos históricos a obtener.
            interval (str): Intervalo de datos históricos (e.g., 'daily').
        """
        self.historical_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        self.current_url = "https://api.binance.com/api/v3/ticker/price"
        self.symbol = "BTCUSDT"
        self.historical_days = historical_days
        self.interval = interval
        self.prices = pd.DataFrame()

    def fetch_historical_prices(self):
        """
        Obtiene los precios históricos de Bitcoin en USD desde CoinGecko.

        Returns:
            pd.DataFrame: DataFrame con las columnas 'timestamp' y 'price'.
        """
        params = {
            "vs_currency": "usd",
            "days": str(self.historical_days),
            "interval": self.interval
        }
        response = requests.get(self.historical_url, params=params)
        response.raise_for_status()  # Lanza una excepción si la solicitud falla

        data = response.json()
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])[:-1]
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
        prices["timestamp"] += pd.to_timedelta(1, unit="h")  # Ajuste UTC+1
        self.prices = prices
        return prices

    def fetch_current_price(self):
        """
        Obtiene el precio actual de Bitcoin en USD desde Binance.

        Returns:
            dict: Diccionario con las claves 'timestamp' y 'price'.
        """
        params = {"symbol": self.symbol}
        response = requests.get(self.current_url, params=params)
        response.raise_for_status()  # Lanza una excepción si la solicitud falla

        data = response.json()
        current_price = {
            "timestamp": pd.Timestamp.now(),
            "price": float(data["price"])
        }
        return current_price

    def save_to_csv(self, filepath="data/prices.csv"):
        """
        Guarda los precios históricos y actuales en un archivo CSV.

        Args:
            filepath (str): Ruta del archivo donde se guardarán los datos.
        """
        if self.prices.empty:
            raise ValueError("Debe obtener primero los precios históricos con fetch_historical_prices().")

        # Obtener el precio actual y agregarlo al DataFrame
        current_price_row = self.fetch_current_price()
        self.prices = self.prices._append(current_price_row, ignore_index=True)
        self.prices["price"] = self.prices["price"].round(2)

        # Guardar en el archivo CSV
        self.prices.to_csv(filepath, index=False, header=True)
        print(f"Datos guardados correctamente en: {filepath}")


# Ejemplo de uso
if __name__ == "__main__":
    fetcher = BitcoinPriceFetcher(historical_days=365, interval="daily")
    fetcher.fetch_historical_prices()
    fetcher.save_to_csv("data/prices.csv")
