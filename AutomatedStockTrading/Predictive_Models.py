import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle as pk
from RunDetails import ticker

def train_models(ticker):
    try:
        os.system('cls')
    except:
        pass

    df = pd.read_pickle('Dataset') # load df from Extract_Metrics.py

    print(df.describe())

    df = df.drop(['close'], axis=1)

    # Normalize data
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    # print(df.describe())

    # print(df.loc[:, df.columns != 'labels'][0:int(.7*df.shape[0])])

    # Split data into training and testing sets
    # percentage = 0.5
    # labels_column = "labels"
    # labels_column = 36

    # X_train, X_test, y_train, y_test = df.loc[:, df.columns != labels_column][0:int(percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(percentage*df.shape[0]):df.shape[0]], df[labels_column][0:int(percentage*df.shape[0])], df[labels_column][int(percentage*df.shape[0]):df.shape[0]]

    train_percentage = .7
    valid_percentage = .1

    # labels_column = "labels"
    labels_column = 33

    X_train, X_valid, X_test, y_train, y_valid, y_test = df.loc[:, df.columns != labels_column][0:int(train_percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df.loc[:, df.columns != labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]], df[labels_column][0:int(train_percentage*df.shape[0])], df[labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df[labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]]

    # # Fit a model to the training data
    # # Lets try a decision tree to start off
    # from sklearn import tree
    # from sklearn.tree import DecisionTreeClassifier

    # dtree = DecisionTreeClassifier()
    # dtree = dtree.fit(X_train, y_train)

    # tree.plot_tree(dtree, feature_names=X_train.columns)

    # print(dtree.score(X_train, y_train))
    # print(dtree.score(X_test, y_test))
    # print(tree.plot_tree(dtree, feature_names=X_train.columns))

    # Try using a Random Forest Classifier

    from sklearn.ensemble import RandomForestClassifier

    best_score = 0
    best_config = {0, 0, 0}

    n = 10

    compiled_models = []

    for n_estimators in [i for i in range(1,10)]: # 5
        for max_depth in [i for i in range(1, 50)]: # 50
            for min_samples_split in [i for i in range(2, 10)]: # 8
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split = min_samples_split).fit(X_train, y_train)

                # if best_score < model.score(X_train, y_train):
                #     print(model.score(X_train, y_train), model.score(X_test, y_test))
                #     best_score = model.score(X_test, y_test)
                #     best_config = {n_estimators, max_depth, 2}
                #     print(model)

                #     filename = "ML_MODEL.pickle"
                #     pk.dump(model, open(filename, 'wb'))

                compiled_models.append([float(model.score(X_valid,y_valid)), model])

    compiled_models.sort(key=lambda row: (row[0]))
    compiled_models = compiled_models[::-1]
    compiled_models = compiled_models[0:n]

    for i, model in enumerate(compiled_models):
        print(model[0], model[1])
        filename = f"ML_MODEL_{ticker}_{i}.pickle"
        pk.dump(model[1], open(filename, 'wb'))


    # Try using XGBoost
    # from xgboost import XGBClassifier

    # # Use classifier or regressor according to your problem

    # best_score = []
    # best_model = None

    # for colsample_bytree in [i/100 for i in range(100)]:
    #     for learning_rate in [i/100 for i in range(100)]:
    #         for max_depth in [i for i in range(15)]:
    #             for alpha in [i for i in range(10)]:
    #                 for n_estimators in [1 for i in range(15)]:
    #                     model = XGBClassifier(colsample_bytree = colsample_bytree, learning_rate = learning_rate, max_depth = max_depth, alpha = 10, n_estimators = n_estimators)
    #                     model.fit(X_train, y_train)

    #                     try:
    #                         os.system('cls')
    #                     except:
    #                         pass

    #                     if best_score < model.score(X_test, y_test):
    #                         print(model.score(X_train, y_train), model.score(X_test, y_test))
    #                         best_score = [model.score(X_train, y_train), model.score(X_test, y_test)]
    #                         print(model)

    #                         filename = "ML_MODEL.pickle"
    #                         pk.dump(model, open(filename, 'wb'))

    # print(best_score)
    # print(best_model)

    # # Neural Nets (Mark 1)

    # import tensorflow as tf
    # from tensorflow import keras
    # from keras import Sequential, layers

    # model = Sequential()

    # # Input Layer
    # model.add(layers.Dense(36, activation='relu', input_dim=36))

    # # Hidden Layer
    # model.add(layers.Dense(60, activation='relu', kernel_regularizer='l2'))
    # model.add(layers.BatchNormalization())

    # model.add(layers.Dense(60, activation='relu'))
    # model.add(layers.BatchNormalization())

    # # Output Layer
    # model.add(layers.Dense(1, activation ='sigmoid'))

    # model.summary()

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.fit(np.array(X_train), np.array(y_train), epochs=30, batch_size=26, verbose=1, validation_data = (np.array(X_test), np.array(y_test)))

    # score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)

    # print(score[1])

    # model.save('ML_MODEL')

    # print(model.predict(X_test).tolist())

    # pred = [1 if entry[0] > 0.5 else 2 for entry in model.predict(X_test).tolist()]

    # print(pred)

    # # Neural Nets (Mark 2)
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)

    # X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    # X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

    # print(X_train.shape[1:])

    # import tensorflow as tf
    # from tensorflow import keras
    # from keras import layers, Sequential
    # from keras.layers import LSTM

    # #Initializing the classifier Network
    # model = Sequential()

    # #Adding the input LSTM network layer
    # model.add(LSTM(36, activation='relu', kernel_regularizer='l2', input_shape=(36, 1))) #  return_sequences=True,
    # # model.add(layers.Dropout(0.1))

    # #Hidden layer

    # model.add(layers.Dense(36, activation='relu'))
    # model.add(layers.Dense(36, activation='relu'))

    # # Output Layer
    # model.add(layers.Dense(1, activation ='sigmoid'))

    # #Compiling the network
    # model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    # #Fitting the data to the model
    # model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

    # score = model.evaluate(X_test, y_test, verbose=0)

    # print(score[1])

    # model.save('ML_MODEL_RNN')

    # # print(model.predict(X_test).tolist())

    # pred = [1 if entry[0] > 0.5 else 2 for entry in model.predict(X_test).tolist()]

    # print(pred)

    # from sklearn import svm

    # clf = svm.SVC(kernel='rbf')
    # clf.fit(X_train, y_train)

    # print(clf.score(X_valid, y_valid))

    # # Catboost
    # import numpy as np

    # from catboost import CatBoostClassifier, Pool

    # # initialize data
    # # test_data = catboost_pool = Pool(X_train, y_train)

    # model = CatBoostClassifier(iterations=2,
    #                            depth=3,
    #                            learning_rate=0.01,
    #                            loss_function='CrossEntropy',
    #                            verbose=True)
    # # train the model
    # model.fit(X_train, y_train)
    # # make the prediction using the resulting model
    # y_pred = model.predict(X_valid)
    # # preds_proba = model.predict_proba(test_data)

    # num_correct = 0
    # y_valid = list(y_valid)

    # for i, entry in enumerate(y_pred):
    #     if entry == y_valid[i]:
    #         num_correct += 1

    # print(f"Accuracy: {(num_correct/len(y_valid))}")

if __name__ == "__main__":
    train_models(ticker)