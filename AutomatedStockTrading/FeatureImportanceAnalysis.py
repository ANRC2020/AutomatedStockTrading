import time
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt

df = pd.read_pickle('Dataset')  # load df from Extract_Metrics.py

feature_names = df.columns.values.tolist()

print(feature_names)

feature_names.pop(len(feature_names) - 1)
feature_names.pop(0)

print(feature_names)

print(len(feature_names))

ticker = 'AUD_CAD'

models = []

for i in range(10):
    models.append(pk.load(open(f'ML_MODEL_{ticker}_{i}.pickle', 'rb')))

for forest in models:

    start_time = time.time()
    importances = forest.feature_importances_

    print(len(importances))

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    plt.show()