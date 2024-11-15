import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM
from keras._tf_keras.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler


prices = pd.read_csv("data/prices.csv")
prices.set_index("timestamp", inplace=True)

# prices = prices["price"]

# Paso 2: Generar señales de compra/venta
def generate_signals(prices, threshold=0.02):
    signals = []
    for i in range(1, len(prices)):
        change = (prices[i] - prices[i - 1]) / prices[i - 1]
        if change > threshold:
            signals.append(1)  # Señal de compra
        elif change < -threshold:
            signals.append(-1)  # Señal de venta
        else:
            signals.append(0)  # Mantener
    return signals

signals = generate_signals(prices["price"].values)

# Paso 3: Preprocesar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices["price"].values.reshape(-1, 1))

def create_sequences(data, signals, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(signals[i + seq_length - 1])  # La señal en el último punto de la secuencia
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(scaled_prices, signals, seq_length)

# Paso 4: Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# Paso 5: Definir y entrenar la red neuronal
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamos el modelo
model.fit(X_train, y_train_cat, epochs=5, batch_size=1, validation_data=(X_test, y_test_cat))

# Paso 6: Hacer predicciones y simular estrategia de trading
predictions = model.predict(X_test)
predicted_signals = np.argmax(predictions, axis=1)

print("Predicciones", predicted_signals[:20], y_test[:20])

# Simulación de trading
initial_balance = 1000
balance = initial_balance
holding = 0

for i in range(len(predicted_signals)):
    current_price = scaler.inverse_transform(X_test[i][-1].reshape(-1, 1))[0][0]
    if predicted_signals[i] == 1 and balance > 0:  # Comprar
        holding = balance / current_price
        balance = 0
        print(f"Comprado a {current_price:.2f}")
    elif predicted_signals[i] == -1 and holding > 0:  # Vender
        balance = holding * current_price
        holding = 0
        print(f"Vendido a {current_price:.2f}")

# Beneficio neto
net_profit = balance - initial_balance
print(f"Beneficio neto: {net_profit:.2f} USD")

# Paso 7: Graficar las señales en el precio
plt.figure(figsize=(14, 7))
plt.plot(prices.index[-len(y_test):], scaler.inverse_transform(X_test[:, -1, 0].reshape(-1, 1)), label='Precio de Cierre')

# Marcar señales de compra y venta
buy_signals = [i for i in range(len(predicted_signals)) if predicted_signals[i] == 1]
sell_signals = [i for i in range(len(predicted_signals)) if predicted_signals[i] == -1]
try:
    plt.scatter(prices.index[-len(y_test):].to_numpy()[buy_signals], scaler.inverse_transform(X_test[buy_signals, -1, 0].reshape(-1, 1)), marker='^', color='green', label='Comprar', alpha=1)
except:
    pass
try:
    plt.scatter(prices.index[-len(y_test):].to_numpy()[sell_signals], scaler.inverse_transform(X_test[sell_signals, -1, 0].reshape(-1, 1)), marker='v', color='red', label='Vender', alpha=1)
except:
    pass

plt.title('Estrategia de Compra y Venta de Bitcoin')
plt.xlabel('Fecha')
plt.ylabel('Precio en USD')
plt.legend()
plt.show()

