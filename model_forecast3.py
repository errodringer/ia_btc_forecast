import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential, load_model, Model
from keras._tf_keras.keras.layers import Dense, LSTM, concatenate, Input, Concatenate

import requests

n_records_train = 300
n_records_test = 10
n_days_predict = 1
seq_length = 30

df = pd.read_csv("data/data.csv")
# TODO: quitar:
df = df[-n_records_train-n_records_test-seq_length:]
X_stock_columns = [col for col in df.columns if col not in ["Date", "price"]]
df.set_index("Date", inplace=True)
df["y_price"] = df["price"].shift(-1)
# prices = prices.dropna()

linear_scaler = MinMaxScaler(feature_range=(0, 1))
seq_scaler = MinMaxScaler(feature_range=(0, 1))

linear_model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=(df.shape[0], 1)),
        LSTM(50),
        Dense(1)
    ]
)

seq_model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ]
)

stock_model = Sequential(
    [
        Dense(16, input_shape=(len(X_stock_columns),)),
        Dense(4),
        Dense(1)
    ]
)

merged_layer = Concatenate()([linear_model.outputs[0], seq_model.outputs[0], stock_model.outputs[0]])
model = Model([linear_model.inputs[0], seq_model.inputs[0], stock_model.inputs[0]], merged_layer)

model.compile(optimizer='adam', loss='mean_squared_error')

full_y_val_descaled = np.array([]).reshape(-1, 1)
full_predictions = np.array([]).reshape(-1, 1)

X_linear_data = df[seq_length:]["price"].values.reshape(-1, 1)
y_data = df[seq_length:]["y_price"].values.reshape(-1, 1)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X_seq_data, y_seq_data = create_sequences(df["y_price"].values.reshape(-1, 1), seq_length)

X_stock_data = df[X_stock_columns].values

# plt.ion()
for i in range(n_records_train, (df.shape[0] - seq_length - n_days_predict * 2) + 1):
    print(f"Processing record from: {i} to {i + n_days_predict}")
    X_linear_train = X_linear_data[0:i]
    X_linear_test = X_linear_data[i:i + n_days_predict]
    X_linear_val = X_linear_data[i + n_days_predict:i + n_days_predict * 2]

    X_seq_train = X_seq_data[0:i]
    X_seq_test = X_seq_data[i:i + n_days_predict]
    X_seq_val = X_seq_data[i + n_days_predict:i + n_days_predict * 2]

    X_stock_train = X_stock_data[0:i]
    X_stock_test = X_stock_data[i:i + n_days_predict]
    X_stock_val = X_stock_data[i + n_days_predict:i + n_days_predict * 2]

    y_train = y_data[0:i]
    y_test = y_data[i:i + n_days_predict]
    y_val = y_data[i + n_days_predict:i + n_days_predict * 2]

    X_linear_train = linear_scaler.fit_transform(X_linear_train)
    X_linear_test = linear_scaler.transform(X_linear_test)
    X_linear_val = linear_scaler.transform(X_linear_val)

    X_seq_train = seq_scaler.fit_transform(X_seq_train.reshape(-1, 1)).reshape(i, seq_length, 1)
    X_seq_test = seq_scaler.transform(X_seq_test.reshape(-1, 1)).reshape(n_days_predict, seq_length, 1)
    X_seq_val = seq_scaler.transform(X_seq_val.reshape(-1, 1)).reshape(n_days_predict, seq_length, 1)

    y_train = linear_scaler.transform(y_train)
    y_test = linear_scaler.transform(y_test)
    y_val = linear_scaler.transform(y_val)

    model.fit(
        [X_linear_train, X_seq_train, X_stock_train],
        y_train,
        epochs=1,
        batch_size=1,
        validation_data=([X_linear_test, X_seq_test, X_stock_test], y_test),
        verbose=0
    )

    # Paso 6: Hacer predicciones y desescalar
    predictions = model.predict([X_linear_val, X_seq_val, X_stock_val])
    predictions = linear_scaler.inverse_transform(np.array([predictions.mean()]).reshape(-1, 1))
    y_val_descaled = linear_scaler.inverse_transform(y_val)

    full_y_val_descaled = np.concatenate([full_y_val_descaled, y_val_descaled])
    full_predictions = np.concatenate([full_predictions, predictions])

    fig, ax = plt.subplots(2)
    ax[0].plot(df.index[-len(full_y_val_descaled):], full_y_val_descaled, color='blue', label='Precio Real')
    ax[0].plot(df.index[-len(full_predictions):], full_predictions, color='red', label='Predicción')
    ax[0].set_title('Predicción del Precio de Bitcoin')
    ax[1].plot(df.index[-len(full_y_val_descaled):], full_y_val_descaled-full_predictions)
    plt.xlabel('Fecha')
    plt.xticks(rotation=45)
    plt.ylabel('Precio en USD')
    plt.show()

model.save('model/btc_forecast_model.keras')

url_current = "https://api.binance.com/api/v3/ticker/price"
params_current = {"symbol": "BTCUSDT"}
current_price_iterations = 5

response = requests.get(url_current, params=params_current)
current_price = np.array(
    [
        linear_scaler.inverse_transform(X_linear_val)[-1][0],
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
plt.show()
