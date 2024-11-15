import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential, load_model, Model
from keras._tf_keras.keras.layers import Dense, LSTM, concatenate, Input

import requests

n_records_train = 300
n_records_test = 20
n_days_predict = 1
# seq_length = 30

df = pd.read_csv("data/data.csv")
# TODO: quitar:
df = df[-n_records_train-n_records_test:]
df.set_index("Date", inplace=True)
df["y_price"] = df["price"].shift(-1)
# prices = prices.dropna()

scaler = MinMaxScaler(feature_range=(0, 1))

model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=(df.shape[0], 1)),
        LSTM(50),
        Dense(1)
    ]
)

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.compile(optimizer='adam', loss='mean_squared_error')

full_y_val_descaled = np.array([]).reshape(-1, 1)
full_predictions = np.array([]).reshape(-1, 1)

X_data = df["price"].values.reshape(-1, 1)
y_data = df["y_price"].values.reshape(-1, 1)

# plt.ion()
for i in range(n_records_train, (df.shape[0] - n_days_predict * 2) + 1):
    print(f"Processing record from: {i} to {i + n_days_predict}")
    X_train = X_data[0:i]
    y_train = y_data[0:i]
    X_test = X_data[i:i + n_days_predict]
    y_test = y_data[i:i + n_days_predict]
    X_val = X_data[i + n_days_predict:i + n_days_predict * 2]
    y_val = y_data[i + n_days_predict:i + n_days_predict * 2]

    X_train = scaler.fit_transform(X_train)
    y_train = scaler.transform(y_train)
    X_test = scaler.transform(X_test)
    y_test = scaler.transform(y_test)
    X_val = scaler.transform(X_val)
    y_val = scaler.transform(y_val)

    model.fit(X_train, y_train, epochs=1, batch_size=1, validation_data=(X_test, y_test), verbose=2)

    predictions = model.predict(X_val)
    predictions = scaler.inverse_transform(predictions)
    y_val_descaled = scaler.inverse_transform(y_val)

    full_y_val_descaled = np.concatenate([full_y_val_descaled, y_val_descaled])
    full_predictions = np.concatenate([full_predictions, predictions])

    # plt.figure(figsize=(14, 7))
    fig, ax = plt.subplots(2)
    ax[0].plot(df.index[-len(full_y_val_descaled):], full_y_val_descaled, color='blue', label='Precio Real')
    ax[0].plot(df.index[-len(full_predictions):], full_predictions, color='red', label='Predicción')
    ax[0].set_title('Predicción del Precio de Bitcoin')
    ax[1].plot(df.index[-len(full_y_val_descaled):], full_y_val_descaled-full_predictions)
    plt.xlabel('Fecha')
    plt.xticks(rotation=45)
    plt.ylabel('Precio en USD')
    ax[0].legend()
    # plt.draw()
    # plt.pause(0.01)
    # plt.clf()
    plt.show()

model.save('model/btc_forecast_model.keras')

url_current = "https://api.binance.com/api/v3/ticker/price"
params_current = {"symbol": "BTCUSDT"}
current_price_iterations = 5

response = requests.get(url_current, params=params_current)
current_price = np.array(
    [
        scaler.inverse_transform(X_val)[-1][0],
        response.json()["price"]
    ]
).astype(np.float32)
print(f"Current price: {current_price[1]}")
print(f"Predicted: {full_predictions[-1][0]}")
diff = abs(full_predictions[-1][0] - current_price[1])
print(f"Difference: {diff}, {round(float(diff/current_price[1]*100), 3)} %")

fig, ax = plt.subplots(2)
ax[0].plot(df.index[-len(full_y_val_descaled):], full_y_val_descaled, color='blue', label='Precio Real')
ax[0].plot(df.index[-len(full_predictions):], full_predictions, color='red', label='Predicción')
ax[0].plot(df.index[-len(current_price):], current_price, color='orange', label='Precio Actual')
ax[1].plot(df.index[-len(full_y_val_descaled):], full_y_val_descaled - full_predictions)
plt.title('Predicción del Precio de Bitcoin')
plt.xlabel('Fecha')
plt.xticks(rotation=45)
plt.ylabel('Precio en USD')
ax[0].legend()
plt.show()
