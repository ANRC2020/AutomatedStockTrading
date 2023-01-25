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

df = pd.read_pickle('Dataset')  # load df from Extract_Metrics.py

# Normalize data
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

tickers = ['AAPL','BABA', 'GOOG', 'META', 'TSLA', 'MSFT', 'NFLX', 'NVDA', "AMD", 'MHK','CUEN','ACN','INTC','UBER','EURUSD','AUDUSD','GBPUSD','HKD','EURCAD','EURJPY','EURSEK',"EURINR","ORCL","AMAT",'ALLY']
n = 10 * len(tickers)
models = [0] * n

j = 0
for ticker in tickers:
    for i in range(10):
        models[j] = pk.load(open(f'ML_MODEL_{ticker}_{i}.pickle', 'rb'))
        j += 1

print(f"Loaded {n} models!\n")

# Create a Dataset of model predictions

# labels_column = "labels"
labels_column = 37
data, labels = df.loc[:, df.columns != labels_column][0:df.shape[0]], df.loc[:, df.columns == labels_column][0:df.shape[0]]

# Get predictions of each model and make a dataframe of them

df = pd.DataFrame()

for i, model in enumerate(models):

    temp = []

    for j in range(data.shape[0]):
        temp.append(model.predict(data.iloc[[j]])[0])

    df.insert(i, f'{i}', temp)

temp = []

for i in range(labels.shape[0]):    
    temp.append(labels[labels_column].iloc[i])

df.insert(len(models), 'labels', temp)


df.to_pickle('Council_Dataset')