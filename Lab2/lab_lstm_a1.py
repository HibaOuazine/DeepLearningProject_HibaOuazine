import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Prepare data for LSTM model
def prepare_data(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

# Load and preprocess data
ticker = "AAPL"  # Replace with your stock ticker
start_date = "2015-01-01"
end_date = "2024-11-21"
time_step = 60

# Fetch data
stock_data = fetch_stock_data(ticker, start_date, end_date)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))

# Prepare training and testing data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

X_train, y_train = prepare_data(train_data, time_step)
X_test, y_test = prepare_data(test_data, time_step)

# Reshape data for LSTM (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

#save model 
model.save('lstm_model.h5')

# Predict and inverse transform
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# Plot results
plt.figure(figsize=(14, 5))
plt.plot(stock_data.index, stock_data.values, label='Actual Price', color='blue')
train_predict_plot = np.empty_like(stock_data.values)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step] = train_predict

test_predict_plot = np.empty_like(stock_data.values)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(stock_data) - 1] = test_predict

plt.plot(stock_data.index, train_predict_plot, label='Train Prediction', color='green')
plt.plot(stock_data.index, test_predict_plot, label='Test Prediction', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.show()
