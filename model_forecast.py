import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM
import requests

prices = pd.read_csv("data/prices.csv")
prices.set_index("timestamp", inplace=True)

# Paso 2: Escalar los precios
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices["price"].values.reshape(-1, 1))

# Paso 3: Crear secuencias de datos para el modelo
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Usaremos los últimos 30 días para predecir el siguiente día
X, y = create_sequences(scaled_prices, seq_length)

# Paso 4: Dividir los datos en entrenamiento y prueba
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Paso 5: Definir la red neuronal LSTM
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(100),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=3, batch_size=1, validation_data=(X_test, y_test))

# Paso 6: Hacer predicciones y desescalar
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_descaled = scaler.inverse_transform(y_test)

# Paso 7: Graficar los resultados
plt.figure(figsize=(14, 7))
plt.plot(prices.index[-len(y_test):], y_test_descaled, color='blue', label='Precio Real')
plt.plot(prices.index[-len(y_test):], predictions, color='red', label='Predicción')
plt.title('Predicción del Precio de Bitcoin')
plt.xlabel('Fecha')
plt.ylabel('Precio en USD')
plt.legend()
plt.show()
