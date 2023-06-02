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
import time

def Model_Analysis(graphs, ticker):

    # print(ticker)
    # time.sleep(5)

    try:
        os.system('cls')
    except:
        pass

    df = pd.read_pickle('Dataset')  # load df from Extract_Metrics.py

    df = df.drop(['close'], axis=1)

    # Normalize data
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    # Load the best saved model
    # model = pk.load(open('ML_MODEL.pickle', 'rb'))
    n = 10
    models = [0] * n

    for i in range(n):
        print(i)
        models[i] = pk.load(open(f'ML_MODEL_{ticker}_{i}.pickle', 'rb'))

    print(f"Loaded {n} models!\n")

    # model = keras.models.load_model('ML_MODEL')
    # model = keras.models.load_model('ML_MODEL_RNN')

    # # Split data into training and testing sets
    # percentage = 0.1
    # labels_column = "labels"
    # # labels_column = 36
    # X_train, X_test, y_train, y_test = df.loc[:, df.columns != labels_column][0:int(percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(
    #     percentage*df.shape[0]):df.shape[0]], df[labels_column][0:int(percentage*df.shape[0])], df[labels_column][int(percentage*df.shape[0]):df.shape[0]]

    train_percentage = 0.7
    valid_percentage = 0.1
    # labels_column = "labels"
    labels_column = 33

    X_train, X_valid, X_test, y_train, y_valid, y_test = df.loc[:, df.columns != labels_column][0:int(train_percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df.loc[:, df.columns != labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]], df[labels_column][0:int(train_percentage*df.shape[0])], df[labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df[labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]]

    # # Use for RNN
    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)

    # X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    # X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # print(model.score(X_train, y_train), model.score(X_test, y_test))
    # print(accuracy_score(np.array(y_test), np.array(model.predict(X_test))))

    # Get predictions from the saved model
    # preds = []
    # for entry in model.predict(X_test).tolist():
    #     if entry[0] <= 0.5:
    #         preds.append(0)
    #     elif entry[0] >= 0.5:
    #         preds.append(1)

    # print(preds)

    # preds = model.predict(X_test)
    # print(preds)

    preds = models[0].predict(X_test)

    for i in range(1, n):
        temp = models[i].predict(X_test)

        for j in range(len(preds)):
            preds[j] += temp[j]
            
    for i in range(len(preds)):
        if preds[i]/n > 0.5:
            preds[i] = 1

            if X_test[9].iloc[i] > 0.7:
                preds[i] = 0

        else:
            preds[i] = 0

    print(preds)

    # Repeat to get Non-Normalized Data 
    df = pd.read_pickle('Dataset')  # load df from Extract_Metrics.py

    # Split data into training and testing sets
    # labels_column = "labels"
    # X_train, X_test, y_train, y_test = df.loc[:, df.columns != labels_column][0:int(percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(
    #     percentage*df.shape[0]):df.shape[0]], df[labels_column][0:int(percentage*df.shape[0])], df[labels_column][int(percentage*df.shape[0]):df.shape[0]]

    labels_column = "labels"
    X_train, X_valid, X_test, y_train, y_valid, y_test = df.loc[:, df.columns != labels_column][0:int(train_percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df.loc[:, df.columns != labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]], df[labels_column][0:int(train_percentage*df.shape[0])], df[labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df[labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]]


    # Calculate the net profit from trading

    mode = "close"
    # mode = 3

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
    print(f"Win to Loss Ratio: {abs(net_winnings/net_loss)}")

    data = f"\nTicker: {ticker} \nLoss Count: {losing_trades}\tWin Count: {winning_trades}\n Net Loss: {net_loss}\tNet Winnings: {net_winnings}\n Win to Loss Ratio: {abs(net_winnings/net_loss)}"

    # Visualize the model's predictions on the true data
    if graphs == True:
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

        # print(preds, y_test)

        # arr = []

        # for i, feat_im in enumerate(model.feature_importances_):
        #     arr.append([df.columns[i], feat_im])

        # print(arr)

    if graphs == True:
        sns.heatmap(confusion_matrix(y_test, preds),annot=True)
        plt.show()

    return data

if __name__ == "__main__":
    Model_Analysis(False, ticker)