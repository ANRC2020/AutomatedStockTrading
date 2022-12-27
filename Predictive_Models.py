import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle as pk

try:
    os.system('cls')
except:
    pass

df = pd.read_pickle('Dataset') # load df from Extract_Metrics.py

# Normalize data
# x = df.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)

# print(df.loc[:, df.columns != 'labels'][0:int(.7*df.shape[0])])

# Split data into training and testing sets
percentage = 0.7
labels_column = "labels"
# labels_column = 30
X_train, X_test, y_train, y_test = df.loc[:, df.columns != labels_column][0:int(percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(percentage*df.shape[0]):df.shape[0]], df[labels_column][0:int(percentage*df.shape[0])], df[labels_column][int(percentage*df.shape[0]):df.shape[0]]

# Fit a model to the training data
# Lets try a decision tree to start off
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

tree.plot_tree(dtree, feature_names=X_train.columns)

print(dtree.score(X_train, y_train))
print(dtree.score(X_test, y_test))
# print(tree.plot_tree(dtree, feature_names=X_train.columns))

# Try using a Random Forest Classifier

# from sklearn.ensemble import RandomForestClassifier

# best_score = 0
# best_config = {0, 0, 0}

# for n_estimators in [i for i in range(1,10)]: # 5
#     for max_depth in [i for i in range(1, 50)]: # 50
#         for min_samples_split in [i for i in range(2, 10)]: # 8
#             model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split = min_samples_split).fit(X_train, y_train)

#             if best_score < model.score(X_test, y_test):
#                 print(model.score(X_train, y_train), model.score(X_test, y_test))
#                 best_score = model.score(X_test, y_test)
#                 best_config = {n_estimators, max_depth, 2}
#                 print(model)

#                 filename = "ML_MODEL.pickle"
#                 pk.dump(model, open(filename, 'wb'))

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

# Neural Nets

import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers

model = Sequential()

# Input Layer
model.add(layers.Dense(30, activation='relu', input_dim=30))

# Hidden Layer
model.add(layers.Dense(60, activation='relu', kernel_regularizer='l1'))

model.add(layers.Dense(60, activation='relu', kernel_regularizer='l1'))

model.add(layers.Dense(60, activation='relu', kernel_regularizer='l1'))

# Output Layer
model.add(layers.Dense(1, activation ='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=10, verbose=1, validation_data = (np.array(X_test), np.array(y_test)))

score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)

print(score[1])
