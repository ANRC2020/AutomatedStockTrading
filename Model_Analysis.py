import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle as pk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

try:
    os.system('cls')
except:
    pass

df = pd.read_pickle('Dataset')  # load df from Extract_Metrics.py

# Normalize data
# x = df.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)

# Load the best saved model
model = pk.load(open('ML_MODEL.pickle', 'rb'))

# Split data into training and testing sets
percentage = 0.7
labels_column = "labels"  # 30
X_train, X_test, y_train, y_test = df.loc[:, df.columns != labels_column][0:int(percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(
    percentage*df.shape[0]):df.shape[0]], df[labels_column][0:int(percentage*df.shape[0])], df[labels_column][int(percentage*df.shape[0]):df.shape[0]]

print(model.score(X_train, y_train), model.score(X_test, y_test))

# Get predictions from the saved model

preds = model.predict(X_test).tolist()

# Calculate the net profit from trading

# History
trades = []

# Personal Info
budget = 1000

# Calculation Variables
prev_pred = 2
enter_price = 0
exit_price = 0
num_stocks_held = 0

for i, pred in enumerate(preds):
    
    if pred == 1 and prev_pred == 2:
        prev_pred = pred

        num_stocks_held = int(budget/X_test['close'].iloc[i])

        # print(budget, X_test['close'].iloc[i], num_stocks_held)

        budget -= num_stocks_held * X_test['close'].iloc[i] 

        enter_price = X_test['close'].iloc[i]

    elif pred == 2 and prev_pred == 1:
        prev_pred = pred

        budget += num_stocks_held * X_test['close'].iloc[i]

        exit_price = X_test['close'].iloc[i]

        trades.append(exit_price*num_stocks_held  - enter_price*num_stocks_held)
        num_stocks_held = 0

print(trades)
print(budget)

# Visualize the model's predictions on the true data

fig = plt.figure()
plt.plot([i for i in range(len(X_test['close']))],
         X_test['close'])  # Plot Closing Prices

for i, pred in enumerate(preds):

    if pred == 1:  # "Good time to enter"
        # plt.plot(i, X_test['close'].iloc[i] + 0.01 *
        #          (X_test['close'].iloc[i]), ".g")

        # plot 0.01 * (X_test['close'].iloc[i]) below the closing price as a line
        plt.plot([i, i], [X_test['close'].iloc[i], X_test['close'].iloc[i] -
                          0.01 * (X_test['close'].iloc[i])], "g")

    elif pred == 2:  # "Good time to exit"
        # plt.plot(i, X_test['close'].iloc[i] + 0.01 *
        #          (X_test['close'].iloc[i]), ".r")

        # plot 0.01 * (X_test['close'].iloc[i]) below the closing price as a line
        plt.plot([i, i], [X_test['close'].iloc[i], X_test['close'].iloc[i] -
                          0.01 * (X_test['close'].iloc[i])], "r")

plt.show()

# print(preds, y_test)

arr = []

for i, feat_im in enumerate(model.feature_importances_):
    arr.append([df.columns[i], feat_im])

print(arr)

sns.heatmap(confusion_matrix(y_test, preds),annot=True)
plt.show()