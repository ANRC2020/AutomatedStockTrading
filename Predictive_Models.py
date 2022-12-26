import pandas as pd
import numpy
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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

# print(df.head)
# print(df.shape)

print(df.loc[:, df.columns != 'labels'][0:int(.7*df.shape[0])])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = df.loc[:, df.columns != 'labels'][0:int(.7*df.shape[0])], df.loc[:, df.columns != 'labels'][int(.7*df.shape[0]):df.shape[0]], df['labels'][0:int(.7*df.shape[0])], df['labels'][int(.7*df.shape[0]):df.shape[0]]

# print(X_train, X_test, y_train, y_test)

# Fit a model to the training data
# Lets try a decision tree to start off
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

tree.plot_tree(dtree, feature_names=X_train.columns)

print(dtree.score(X_train, y_train))
print(dtree.score(X_test, y_test))
print(tree.plot_tree(dtree, feature_names=X_train.columns))

from sklearn.ensemble import RandomForestClassifier

best_score = 0
best_config = {0, 0}

for n_estimators in [1, 5, 10, 50, 100, 200, 400, 800, 1600, 3200]:
    for max_depth in [1, 5, 10, 50, 100]:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth).fit(X_train, y_train)

        if best_score < model.score(X_test, y_test):
            print(model.score(X_test, y_test))
            best_score = model.score(X_test, y_test)
            best_config = {n_estimators, max_depth}

print(best_score, best_config)