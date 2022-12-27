import pandas as pd
import numpy
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
# print(df.head())

# print(df.head)
# print(df.shape)

# print(df.loc[:, df.columns != 'labels'][0:int(.7*df.shape[0])])

# Split data into training and testing sets
percentage = 0.9
labels_column  =  "labels" # 30
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

from sklearn.ensemble import RandomForestClassifier

best_score = 0
best_config = {0, 0, 0}

for n_estimators in [i for i in range(1,10)]: # 5
    for max_depth in [i for i in range(1, 50)]: # 50
        for min_samples_split in [i for i in range(2, 10)]: # 8
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split = min_samples_split).fit(X_train, y_train)

            if best_score < model.score(X_test, y_test):
                print(model.score(X_train, y_train), model.score(X_test, y_test))
                best_score = model.score(X_test, y_test)
                best_config = {n_estimators, max_depth, min_samples_split}
                print(model)

                filename = "ML_MODEL.pickle"
                pk.dump(model, open(filename, 'wb'))
