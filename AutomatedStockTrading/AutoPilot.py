# The purpose of this script is to run automated tests en masse to obtain
# insights about the affect of changes in the algorithm in general

# Import functions and variables from all necesary files

from RunDetails import Ticker_List
from GetData import retrieve_data # Pass True/False to enable or disable graphs
from Extract_Metrics import Extract_Metrics
from Predictive_Models import train_models
from Model_Analysis import Model_Analysis
import os

data = ""

for T in Ticker_List:
    try:
        retrieve_data(False, T)
        Extract_Metrics()
        train_models(T)
        results = Model_Analysis(False, T)

        data += results
    except:
        data += f"\nSkipped {T}\n"

try:
    os.system("cls")
except:
    pass

file = open("Results_9.txt","w")

file.write(data)
 
file.close() 

print("\nDone!\n\nResults are now posted in 'Results.txt'\n")