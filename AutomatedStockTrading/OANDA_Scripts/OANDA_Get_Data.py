# get a list of trades
from oandapyV20 import API
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import json

try:
    import os
    os.system('cls')
except:
    pass

ticker = 'EUR_USD'
mode = 'OANDA'
# mode = 'AlphaVantage'
count = 1000
time_interval = "H1"

client = API(access_token="f87d95b30fc0ee18ccd0987c07e402a7-2f2e3c9e794ff9334c7f6c52975432f6")
accountID = "101-001-19777424-001"

params = {"count": count,  "granularity": time_interval}

r = instruments.InstrumentsCandles(instrument=ticker,params=params)

client.request(r)

for line in r.response['candles']:
    print(line['time'], line['mid'])        
