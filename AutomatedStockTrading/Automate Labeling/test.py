import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

try:
    os.system("cls")
except:
    pass

global enable
enable = True

with open('stock_data.pkl', 'rb') as f:
    candle_sticks = pickle.load(f)

# Surpress Scientific Notation
np.set_printoptions(suppress=True)

# Set labels for this data
labels = [-1]*len(candle_sticks[:, 3])

k = 5

sell = [0]*(k+1)        # initialize max profit after kth sell as 0. k+1 to make it convenient for buy1 boundary condition. 
buy = [-math.inf]*(k+1) # initialize max profit after kth buy as -inf
sell_index = [0]*(k + 1)
buy_index = [0]*(k + 1)

prev_sell = sell

for j, p in enumerate(candle_sticks[:, 3]):
    for i in range(1,k+1):
        # sell[i] = max(sell[i], buy[i]+p) # it's important to compute sell first without updating buy i to today.
        # buy[i] = max(buy[i],  sell[i-1]-p)

        if sell[i] < buy[i]+p:
            sell[i] = buy[i]+p
            sell_index[i] = j + i

        if buy[i] < sell[i-1]-p:
            buy[i] = sell[i-1]-p
            buy_index[i] = i - 1 + j

print(buy_index, sell_index)
print(sell)

fig = plt.figure()
plt.plot([i for i in range(len(candle_sticks[:,3]))], candle_sticks[:,3]) # Plot Closing Prices
plt.xlabel("Interval",fontsize='13')
plt.ylabel("Prices",fontsize='13')
plt.legend('Closing Prices',loc='best')
# plt.savefig('Y_X.png')
plt.show()