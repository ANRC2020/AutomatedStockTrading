import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pickle as pk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from tensorflow import keras
from RunDetails import ticker

try:
    os.system('cls')
except:
    pass

df = pd.read_pickle('Council_Dataset')  # load df from Extract_Metrics.py

model = pk.load(open("ML_COUNCIL_MODEL.pickle", 'rb'))

train_percentage = 0.7
valid_percentage = 0.1
labels_column = "labels"
X_train, X_valid, X_test, y_train, y_valid, y_test = df.loc[:, df.columns != labels_column][0:int(train_percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df.loc[:, df.columns != labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]], df[labels_column][0:int(train_percentage*df.shape[0])], df[labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df[labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]]

preds = model.predict(X_test)

print(preds)

# Repeat to get Non-Normalized Data 
df = pd.read_pickle('Dataset')  # load df from Extract_Metrics.py

labels_column = "labels"
X_train, X_valid, X_test, y_train, y_valid, y_test = df.loc[:, df.columns != labels_column][0:int(train_percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df.loc[:, df.columns != labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]], df[labels_column][0:int(train_percentage*df.shape[0])], df[labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df[labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]]

# Calculate the net profit from trading
mode = "close"

# History
trades = []

# Personal Info
budget = 1000

# Calculation Variables
prev_pred = 1
enter_price = 0
exit_price = 0
num_stocks_held = 0

losing_trades = 0
net_loss = 0
winning_trades = 0
net_winnings = 0

for i, pred in enumerate(preds):
    
    if pred == 0 and prev_pred == 1:
        prev_pred = pred

        num_stocks_held = int(budget/X_test[mode].iloc[i])

        # print(budget, X_test['close'].iloc[i], num_stocks_held)

        budget -= num_stocks_held * X_test[mode].iloc[i] 

        enter_price = X_test[mode].iloc[i]

        print(f"Bought {num_stocks_held} shares at ${enter_price} each for a total of {num_stocks_held * X_test[mode].iloc[i]} at iteration {i}")

    elif pred == 1 and prev_pred == 0:
        prev_pred = pred

        budget += num_stocks_held * X_test[mode].iloc[i]

        exit_price = X_test[mode].iloc[i]

        print(f"Sold {num_stocks_held} shares at ${exit_price} each for a total of {num_stocks_held * X_test[mode].iloc[i]} at iteration {i}")
        print(f"Profit for this trade was {exit_price*num_stocks_held  - enter_price*num_stocks_held}\n")

        trades.append(exit_price*num_stocks_held  - enter_price*num_stocks_held)
        num_stocks_held = 0

        enter_price = 0
        exit_price = 0

        if trades[-1] < 0:
            losing_trades += 1
            net_loss += trades[-1]
        elif trades[-1] > 0:
            winning_trades += 1
            net_winnings += trades[-1]

if enter_price > 0:
    budget += num_stocks_held * X_test[mode].iloc[i]

    exit_price = X_test[mode].iloc[i]

    print(f"Sold {num_stocks_held} shares at ${exit_price} each for a total of {num_stocks_held * X_test[mode].iloc[i]} at iteration {i}")
    print(f"Profit for this trade was {exit_price*num_stocks_held  - enter_price*num_stocks_held}\n")

    trades.append(exit_price*num_stocks_held  - enter_price*num_stocks_held)
    num_stocks_held = 0

    enter_price = 0
    exit_price = 0

    if trades[-1] < 0:
        losing_trades += 1
        net_loss += trades[-1]
    elif trades[-1] > 0:
        winning_trades += 1
        net_winnings += trades[-1]


print(trades)
print(f"\nBudget: {budget}\n")

print(f"\nLoss Count: {losing_trades}\tWin Count: {winning_trades}\n")
print(f"Net Loss: {net_loss}\tNet Winnings: {net_winnings}\n")

# Visualize the model's predictions on the true data

fig = plt.figure()
plt.plot([i for i in range(len(X_test[mode]))],X_test[mode])  # Plot Closing Prices

for i, pred in enumerate(preds):

    if pred == 0:  # "Good time to enter"
        # plt.plot(i, X_test['close'].iloc[i] + 0.01 *
        #          (X_test['close'].iloc[i]), ".g")

        # plot 0.01 * (X_test['close'].iloc[i]) below the closing price as a line
        plt.plot([i, i], [X_test[mode].iloc[i], X_test[mode].iloc[i] - 0.01 * (X_test[mode].iloc[i])], "g")

    elif pred == 1:  # "Good time to exit"
        # plt.plot(i, X_test['close'].iloc[i] + 0.01 *
        #          (X_test['close'].iloc[i]), ".r")

        # plot 0.01 * (X_test['close'].iloc[i]) below the closing price as a line
        plt.plot([i, i], [X_test[mode].iloc[i], X_test[mode].iloc[i] - 0.01 * (X_test[mode].iloc[i])], "r")

plt.show()

sns.heatmap(confusion_matrix(y_test, preds),annot=True)
plt.show()