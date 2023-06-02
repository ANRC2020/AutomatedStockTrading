import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers, Sequential
import random

try:
    os.system("cls")
except:
    pass

with open('stock_data.pkl', 'rb') as f:
    stock_data = pickle.load(f)

# Surpress Scientific Notation
np.set_printoptions(suppress=True)

df = pd.DataFrame(stock_data)

# Generate Data Metrics



# Create training, validation, and testing sets

close_prices = np.array(df[[3]])

values = close_prices
training_data_len = math.ceil(len(values)* 0.7)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

r = 60

for i in range(r, len(train_data)):
    x_train.append(train_data[i-r:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[training_data_len-r: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(r, len(test_data)):
  x_test.append(test_data[i-r:i, 0])

x_test = np.array(x_test)
print(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test)

print(x_test.shape)
print(y_test.shape)

model = Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 1, epochs=3)

# The predictions must be used as input for subsequent predictions
predictions = []
input_prices = []

for i in range(len(y_test)):

    if i == 0:
        input_prices = list(x_test[0])
        input_prices = np.array(input_prices)
        input_prices = np.reshape(input_prices, (1, x_test.shape[1], 1))

    else:
        input_prices = input_prices[0]
        input_prices = list(input_prices)
        input_prices = input_prices[1::]
        input_prices.append(predictions[-1])
        input_prices = np.array(input_prices)
        input_prices = np.reshape(input_prices, (1, x_test.shape[1], 1))

    pred = model.predict(input_prices)
    predictions.append([pred[0][0]])

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

data = stock_data[:,3]
train = data[:training_data_len]
validation = data[training_data_len:]
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(validation)
plt.plot(predictions)
plt.legend(['Val', 'Predictions'], loc='lower right')
plt.show()