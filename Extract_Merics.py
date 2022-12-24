import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    os.system("cls")
except:
    pass


with open('stock_data.pkl','rb') as f:
    candle_sticks = pickle.load(f)

# Surpress Scientific Notation
np.set_printoptions(suppress=True)

# Print out summary of the data
print(f"Summary:\n Data Shape {candle_sticks.shape}")
print(candle_sticks)

print("\nopen, high, low, close, volume\n")

# PSAR Value Calculations

calc_arr = np.zeros((2,6))
PSAR_arr = []

# Initialize the calc_arr
calc_arr[0][5] = 0 # Set Trend to be falling (does not matter)
calc_arr[0][0] = candle_sticks[0][2] # Set EP to be current low price
calc_arr[0][4] = candle_sticks[0][1] # Set PSAR to be opposite to the EP (set to be current high)
calc_arr[0][1] = 0.02 # Initial Acc value is 0.02
calc_arr[0][2] = (calc_arr[0][4] - calc_arr[0][0]) * calc_arr[0][1]

PSAR_arr.append(calc_arr[0][5])

# Now use the first line of calculations to find the next and repeat:

i = 1

while i < len(candle_sticks): 

    # Calculate the Initial PSAR
    
    if(calc_arr[0][5] == 0):
        calc_arr[1][3] = max(calc_arr[0][4] - calc_arr[0][2], candle_sticks[i - 1][1])

        try:
            calc_arr[1][3] = max(calc_arr[1][3], candle_sticks[i - 2][1])
        except:
            pass
    
    elif(calc_arr[0][5] == 1):
        calc_arr[1][3] = min(calc_arr[0][4] - calc_arr[0][2], candle_sticks[i - 1][2])

        try:
            calc_arr[1][3] = min(calc_arr[1][3], candle_sticks[i - 2][2])
        except:
            pass
        
    # Calculate the current PSAR
    
    if calc_arr[0][5] == 0 and candle_sticks[i][1] < calc_arr[1][3]:
        calc_arr[1][4] = calc_arr[1][3]

    elif calc_arr[0][5] == 1 and candle_sticks[i][2] > calc_arr[1][3]:
        calc_arr[1][4] = calc_arr[1][3]

    elif calc_arr[0][5] == 0 and candle_sticks[i][1] >= calc_arr[1][3]:
        calc_arr[1][4] = calc_arr[0][0]

    elif calc_arr[0][5] == 1 and candle_sticks[i][2] <= calc_arr[1][3]:
        calc_arr[1][4] = calc_arr[0][0]

    # Update the current Trend

    if calc_arr[1][4] > candle_sticks[i][3]:
        calc_arr[1][5] = 0

    else:
        calc_arr[1][5] = 1

    # Update the current EP

    if calc_arr[1][5] == 0:
        calc_arr[1][0] = min(calc_arr[0][0], candle_sticks[i][2])

    elif calc_arr[1][5] == 1:
        calc_arr[1][0] = max(calc_arr[0][0], candle_sticks[i][1])

    # Update the current Acc

    if calc_arr[1][5] == calc_arr[0][5] and calc_arr[1][0] != calc_arr[0][0] and calc_arr[0][1] < 0.2:
        calc_arr[1][1] = calc_arr[0][1] + 0.02
    
    elif calc_arr[1][5] == calc_arr[0][5] and calc_arr[1][0] == calc_arr[0][0]:
        calc_arr[1][1] = calc_arr[0][1]

    elif calc_arr[1][5] != calc_arr[0][5]:
        calc_arr[1][1] = 0.02
    
    else:
        calc_arr[1][1] = 0.2

    # Recalculate Delta PSAR
    calc_arr[1][2] = (calc_arr[1][4] - calc_arr[1][0]) * calc_arr[1][1]

    # Shift the Rows of the Cal_arr up by one:
    calc_arr[0] = calc_arr[1]

    PSAR_arr.append(calc_arr[0][5])

    i += 1

fig = plt.figure()
plt.plot([i for i in range(len(candle_sticks[:,3]))], candle_sticks[:,3]) # Plot Closing Prices

# Get Plottable PSAR values
for i, val in enumerate(PSAR_arr):
    if val == 0:
        plt.plot(i, candle_sticks[i][3] + (candle_sticks[i][3] * 0.001), "r.")
    else:
        plt.plot(i, candle_sticks[i][3] - (candle_sticks[i][3] * 0.001), "g.")    

plt.xlabel("Interval",fontsize='13')
plt.ylabel("Prices",fontsize='13')
plt.show()
