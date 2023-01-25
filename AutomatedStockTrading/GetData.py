# Alphavantage
# API Key: YER7LLFWFGTLGLE9

import requests
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from RunDetails import ticker as ticker

from oandapyV20 import API
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import json

try:
    os.system("cls")
except:
    pass

mode = 'OANDA'
# mode = 'AlphaVantage'
count = 1000
time_interval = "H1"

candle_sticks = []

if mode == 'AlphaVantage':
    key = "YER7LLFWFGTLGLE9"
    # ticker = 'AAPL'
    interval = "60min"

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&outputsize=full&apikey={key}' # &outputsize=full
    r = requests.get(url)
    data = r.json()

    time_interval = ""
    print("Meta Data:")
    for i, key in enumerate(data['Meta Data'].keys()):
        print(data['Meta Data'][key])

        if i == 3:
            time_interval = str(data['Meta Data'][key])

    print("\n")

    print("Time Series Data: ")
    for time_stamp in data[f'Time Series ({time_interval})']:
        print(data[f'Time Series ({time_interval})'][time_stamp])
        temp = []
        for i, info in enumerate(data[f'Time Series ({time_interval})'][time_stamp].keys()):
            temp.append(float(data[f'Time Series ({time_interval})'][time_stamp][info]))

        candle_sticks.append(temp)

    print("\n")

    candle_sticks = candle_sticks[::-1]
    candle_sticks = np.array(candle_sticks)

elif mode == 'OANDA':

    client = API(access_token="f87d95b30fc0ee18ccd0987c07e402a7-2f2e3c9e794ff9334c7f6c52975432f6")
    accountID = "101-001-19777424-001"

    params = {"count": count,  "granularity": time_interval}

    r = instruments.InstrumentsCandles(instrument=ticker,params=params)

    client.request(r)

    for line in r.response['candles']:
        print(line)        
        temp = [float(line['mid']['o']), float(line['mid']['h']), float(line['mid']['l']), float(line['mid']['c']), int(line['volume'])]
        
        candle_sticks.append(temp)

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