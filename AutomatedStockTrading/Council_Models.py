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

df = pd.read_pickle('Council_Dataset') # load df from Extract_Metrics.py

print(df.shape)

train_percentage = 0.7
valid_percentage = 0.1
labels_column = "labels"
X_train, X_valid, X_test, y_train, y_valid, y_test = df.loc[:, df.columns != labels_column][0:int(train_percentage*df.shape[0])], df.loc[:, df.columns != labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df.loc[:, df.columns != labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]], df[labels_column][0:int(train_percentage*df.shape[0])], df[labels_column][int(train_percentage*df.shape[0]):int((train_percentage + valid_percentage)*df.shape[0])], df[labels_column][int((train_percentage + valid_percentage)*df.shape[0]):df.shape[0]]

# Try using a Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

best_score = 0
best_config = {0, 0, 0}

for n_estimators in [i for i in range(1,10)]: # 5
    for max_depth in [i for i in range(1, 50)]: # 50
        for min_samples_split in [i for i in range(2, 10)]: # 8
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split = min_samples_split).fit(X_train, y_train)

            if best_score < model.score(X_valid, y_valid):
                print(model.score(X_train, y_train), model.score(X_valid, y_valid))
                print(model)

                filename = "ML_COUNCIL_MODEL.pickle"
                pk.dump(model, open(filename, 'wb'))
                best_score = model.score(X_valid, y_valid)


print(best_score)
