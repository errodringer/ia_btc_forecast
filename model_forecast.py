import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import requests


class BitcoinPricePredictor:
    """
    Clase para predecir el precio del Bitcoin utilizando una red neuronal LSTM.
    Proporciona funcionalidades para entrenar dinámicamente el modelo, visualizar los resultados
    y predecir precios actuales basados en datos históricos.
    """

    def __init__(self, seq_length=30, epochs=1, batch_size=1):
        """
        Inicializa el modelo y las configuraciones.

        Args:
            seq_length (int): Longitud de las secuencias de entrada para el modelo.
            epochs (int): Número de épocas para entrenar el modelo.
            batch_size (int): Tamaño del batch durante el entrenamiento.
        """
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._create_lstm_model()
        self.df = None

    def _create_lstm_model(self):
        """
        Crea y compila el modelo LSTM.

        Returns:
            keras.Model: Modelo LSTM compilado.
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.seq_length, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def load_data(self, file_path):
        """
        Carga los datos desde un archivo CSV y los escala.

        Args:
            file_path (str): Ruta al archivo CSV con los datos.

        Returns:
            pd.DataFrame: DataFrame original con los datos cargados.
            np.array: Datos escalados.
        """
        df = pd.read_csv(file_path)
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index)
        self.df = df
        return df["price"].values.reshape(-1, 1)

    def create_sequences(self, data):
        """
        Crea secuencias de datos para entrenar el modelo.

        Args:
            data (np.array): Datos escalados.

        Returns:
            tuple: Secuencias de entrada (X) y etiquetas de salida (y).
        """
        [X, y] = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length])
        return np.array(X), np.array(y)

    def train_dynamically(self, X, y):
        """
        Entrena el modelo dinámicamente y visualiza los resultados.

        Args:
            X (np.array): Secuencias de entrada.
            y (np.array): Etiquetas de salida.
        """
        full_y_val_descaled = np.array([]).reshape(-1, 1)
        full_predictions = np.array([]).reshape(-1, 1)

        plt.ion()
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle("Análisis de Predicción del Precio de Bitcoin", fontsize=16)

        n_train = 5
        n_start = len(X) - n_train - 2

        for n, i in enumerate(range(1 + n_start, len(X) - 1)):
            print(f"Processing record: {i}")

            # Crear conjuntos de entrenamiento, prueba y validación
            X_train = self.scaler.fit_transform(X[:i].reshape(-1, 1)).reshape(i, self.seq_length, 1)
            X_test = self.scaler.transform(X[i:i + 1].reshape(-1, 1)).reshape(1, self.seq_length, 1)
            X_val = self.scaler.transform(X[i + 1:i + 2].reshape(-1, 1)).reshape(1, self.seq_length, 1)

            y_train = self.scaler.transform(y[:i])
            y_test = self.scaler.transform(y[i:i + 1])
            y_val = self.scaler.transform(y[i + 1:i + 2])

            # Entrenar el modelo
            self.model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_test, y_test),
                verbose=2
            )

            # Predecir y desescalar
            predictions = self.scaler.inverse_transform(self.model.predict(X_val))
            y_val_descaled = self.scaler.inverse_transform(y_val)

            # Guardar valores para la grafica
            full_y_val_descaled = np.concatenate([full_y_val_descaled, y_val_descaled])
            full_predictions = np.concatenate([full_predictions, predictions])

            # Actualizar gráficos
            if n != 0:
                save_path = "images/predictions.png" if i == len(X) - 2 else None
                self._plot_predictions(fig, ax, self.df, full_y_val_descaled, full_predictions, save_path)

        plt.ioff()
        plt.show()

    @staticmethod
    def _plot_predictions(fig, ax, df, real_prices, predictions, save_path=None):
        """
        Genera los gráficos de precios reales y predicciones, destacando el último valor.

        Args:
            fig (plt.fig): figura de la grafica.
            ax (list): Lista de ejes para la grafica.
            df (pd.DataFrame): DataFrame original.
            real_prices (np.array): Precios reales desescalados.
            predictions (np.array): Predicciones desescaladas.
            save_path (str, optional): Ruta para guardar la figura. Si es None, no se guarda.
        """
        for axe in ax:
            axe.clear()

        # Gráfico 1: Precio real vs Predicción
        ax[0].plot(df.index[-len(real_prices):], real_prices, color='blue', label='Precio Real', linewidth=2)
        ax[0].plot(df.index[-len(predictions):], predictions, color='red', label='Predicción', linestyle='--',
                   linewidth=2)

        # Destacar el último valor con un marcador y etiquetas
        last_date = df.index[-1]
        last_real_price = real_prices[-1]
        last_prediction = predictions[-1]

        ax[0].scatter(last_date, last_real_price, color='blue', edgecolor='black', s=100, zorder=5,
                      label='Último Precio Real')
        ax[0].scatter(last_date, last_prediction, color='red', edgecolor='black', s=100, zorder=5,
                      label='Última Predicción')

        ax[0].text(
            last_date, last_real_price, f"{last_real_price[0]:.2f} USD",
            color='blue', fontsize=10, ha='left', va='bottom'
        )
        ax[0].text(
            last_date, last_prediction, f"{last_prediction[0]:.2f} USD (Predicción)",
            color='red', fontsize=10, ha='left', va='bottom'
        )

        ax[0].set_title('Predicción del Precio de Bitcoin', fontsize=14)
        ax[0].set_ylabel('Precio en USD', fontsize=12)
        ax[0].legend(loc='upper left', fontsize=10)
        ax[0].grid(alpha=0.3)

        # Gráfico 2: Diferencia entre precio real y predicción
        difference = real_prices - predictions
        ax[1].plot(df.index[-len(real_prices):], difference, color='purple', label='Error (Precio Real - Predicción)',
                   linewidth=2)
        ax[1].set_title('Error de Predicción', fontsize=14)
        ax[1].set_ylabel('Error en USD', fontsize=12)
        ax[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax[1].grid(alpha=0.3)

        # Formatear fechas
        ax[1].set_xlabel('Fecha', fontsize=12)
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Guardar la figura si se especifica una ruta
        if save_path:
            plt.savefig(save_path, format='png', dpi=300)
            print(f"Figura guardada en: {save_path}")

    def save_model(self, path):
        """
        Guarda el modelo una vez esta entrenado.

        Args:
            path (str): ruta donde va a guardarse el modelo.
        """
        self.model.save(path)

    def predict_current_price(self, X_val):
        """
        Predice el precio actual de Bitcoin usando datos en tiempo real.

        Args:
            X_val (np.array): Última secuencia para la predicción.

        Returns:
            tuple: Precio actual, precio predicho y diferencia porcentual.
        """
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": "BTCUSDT"}
        response = requests.get(url, params=params)
        current_price = float(response.json()["price"])
        predicted_price = self.scaler.inverse_transform(
            self.model.predict(self.scaler.transform(X_val.reshape(-1, 1)).reshape(1, self.seq_length, 1))
        )[0][0]
        diff_percentage = abs(current_price - predicted_price) / current_price * 100
        print(f"Precio actual: {current_price}, Predicción: {predicted_price}, Diferencia: {diff_percentage:.2f}%")
        return current_price, predicted_price, diff_percentage


# Ejecución principal
if __name__ == "__main__":
    predictor = BitcoinPricePredictor(seq_length=30, epochs=1, batch_size=1)
    data = predictor.load_data("data/prices.csv")
    X, y = predictor.create_sequences(data)
    predictor.train_dynamically(X, y)
    predictor.save_model("model/btc_forecast_model.keras")
    predictor.predict_current_price(X[-1:])
