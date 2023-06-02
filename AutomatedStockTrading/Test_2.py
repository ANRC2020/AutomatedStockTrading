import pickle
import matplotlib.pyplot as plt
import os

try:
    os.system('cls')
except:
    pass

with open('stock_data.pkl', 'rb') as f:
    candle_sticks = pickle.load(f)

# Calculate Average True Range (ATR)

ATR = []

for i, price in enumerate(candle_sticks[:,3]):

    if i < 14:
        n_sum = sum(candle_sticks[:,3][0:i])
        ATR.append(n_sum/(i + 1))

        continue

    ATR.append((ATR[-1] + max(candle_sticks[:,1][i] - candle_sticks[:,2][i], abs(candle_sticks[:,1][i] - candle_sticks[:,3][i - 1]), abs(candle_sticks[:,2][i] - candle_sticks[:,3][i - 1])))/14)

fig = plt.figure()
plt.plot([i for i in range(len(candle_sticks[:,3]))], candle_sticks[:,3]) # Plot Closing Prices
plt.plot([i for i in range(len(candle_sticks[:,3]))], ATR)

plt.xlabel("Interval",fontsize='13')
plt.ylabel("Prices",fontsize='13')
plt.legend('Closing Prices',loc='best')
plt.show()
