# Alphavantage
# API Key: YER7LLFWFGTLGLE9

import requests
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    os.system("cls")
except:
    pass

key = "YER7LLFWFGTLGLE9"
ticker = "GOOG"

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=5min&apikey={key}'
r = requests.get(url)
data = r.json()

time_interval = ""
print("Meta Data:")
for i, key in enumerate(data['Meta Data'].keys()):
    print(data['Meta Data'][key])

    if i == 3:
        time_interval = str(data['Meta Data'][key])

print("\n")

candle_sticks = []

print("Time Series Data: ")
for time_stamp in data[f'Time Series ({time_interval})']:
    print(data[f'Time Series ({time_interval})'][time_stamp])

    temp = []
    for i, info in enumerate(data[f'Time Series ({time_interval})'][time_stamp].keys()):
        temp.append(float(data[f'Time Series ({time_interval})'][time_stamp][info]))

    candle_sticks.append(temp)

print("\n")

candle_sticks = np.array(candle_sticks)

# Save data to a pickle file
with open('stock_data.pkl','wb') as f:
    pickle.dump(candle_sticks, f)

fig = plt.figure()
plt.title(f"{ticker}, {time_interval}", fontsize='16')
plt.plot([i for i in range(len(candle_sticks[:,3]))], candle_sticks[:,3]) # Plot Closing Prices
plt.xlabel("Interval",fontsize='13')
plt.ylabel("Prices",fontsize='13')
plt.legend('Closing Prices',loc='best')
# plt.savefig('Y_X.png')
plt.show()