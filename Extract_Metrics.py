import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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

# Print out summary of the data
# print(f"Summary:\n Data Shape {candle_sticks.shape}")
# print(candle_sticks)

# print("\nopen, high, low, close, volume\n")

# PSAR Value Calculations

calc_arr = np.zeros((2, 6))
PSAR_arr = []

# Initialize the calc_arr
calc_arr[0][5] = 0  # Set Trend to be falling (does not matter)
calc_arr[0][0] = candle_sticks[0][2]  # Set EP to be current low price
calc_arr[0][4] = candle_sticks[0][1]  # Set PSAR to be opposite to the EP (set to be current high)
calc_arr[0][1] = 0.02  # Initial Acc value is 0.02
calc_arr[0][2] = (calc_arr[0][4] - calc_arr[0][0]) * calc_arr[0][1]

PSAR_arr.append(calc_arr[0][5])

# Now use the first line of calculations to find the next and repeat:

i = 1

while i < len(candle_sticks):

    # Calculate the Initial PSAR

    if (calc_arr[0][5] == 0):
        calc_arr[1][3] = max(calc_arr[0][4] - calc_arr[0]
                             [2], candle_sticks[i - 1][1])

        try:
            calc_arr[1][3] = max(calc_arr[1][3], candle_sticks[i - 2][1])
        except:
            pass

    elif (calc_arr[0][5] == 1):
        calc_arr[1][3] = min(calc_arr[0][4] - calc_arr[0]
                             [2], candle_sticks[i - 1][2])

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

# Simple Moving Average (3 points)

SMA_3 = []

for i in range(len(candle_sticks[:, 3])):
    temp = 0
    num_vals = 0

    j = 0

    while i - j >= 0 and j < 3:
        temp += candle_sticks[:, 3][i - j]
        num_vals += 1
        j += 1

    SMA_3.append(temp/num_vals)

# Simple Moving Average (5 points)

SMA_5 = []

for i in range(len(candle_sticks[:, 3])):
    temp = 0
    num_vals = 0

    j = 0

    while i - j >= 0 and j < 5:
        temp += candle_sticks[:, 3][i - j]
        num_vals += 1
        j += 1

    SMA_5.append(temp/num_vals)

# Relative Strength Index (9 points)

RSI_9 = []

for i in range(len(candle_sticks[:, 3])):
    prev = 0
    curr = 0
    num_up = 0
    num_down = 0

    j = 9

    while j > 0:

        if i - j < 0:
            j -= 1
            continue

        prev = curr
        curr = candle_sticks[:, 3][i - j]

        if prev < curr:
            num_up += 1

        elif prev > curr:
            num_down += 1

        j -= 1

    if num_down == 0:
        RSI_9.append(100)
    else:
        RSI_9.append(100 - (100/(1 + (num_up/num_down))))

# Relative Strength Index (13 points)

RSI_13 = []

for i in range(len(candle_sticks[:, 3])):
    prev = 0
    curr = 0
    num_up = 0
    num_down = 0

    j = 13

    while j > 0:

        if i - j < 0:
            j -= 1
            continue

        prev = curr
        curr = candle_sticks[:, 3][i - j]

        if prev < curr:
            num_up += 1

        elif prev > curr:
            num_down += 1

        j -= 1

    if num_down == 0:
        RSI_13.append(100)
    else:
        RSI_13.append(100 - (100/(1 + (num_up/num_down))))

# Generate Labels we want to predict

# labels = []

# for i in range(1, len(candle_sticks[:, 3])):
#     if candle_sticks[:, 3][i - 1] < candle_sticks[:, 3][i]:
#         labels.append(2)  # up
#     elif candle_sticks[:, 3][i - 1] == candle_sticks[:, 3][i]:
#         labels.append(1)  # same
#     else:
#         labels.append(0)  # down

# labels.append(-1)  # Append to match length (disregard last entry)

# print(labels, len(labels))

# Experiment 1: Supply the previous iteration's info

copy_candle_sticks = candle_sticks
copy_candle_sticks = np.insert(
    copy_candle_sticks, 0, np.array([0, 0, 0, 0, 0]), axis=0)
copy_candle_sticks = np.delete(copy_candle_sticks, 100, axis=0)

copy_PSAR_arr = PSAR_arr
copy_PSAR_arr.insert(0, 0)
copy_PSAR_arr = copy_PSAR_arr[0:len(copy_PSAR_arr)-1]

PSAR_arr = PSAR_arr[1::]

copy_SMA_3 = SMA_3
copy_SMA_3.insert(0, 0)
copy_SMA_3 = copy_SMA_3[0:len(copy_SMA_3)-1]

SMA_3 = SMA_3[1::]

copy_SMA_5 = SMA_5
copy_SMA_5.insert(0, 0)
copy_SMA_5 = copy_SMA_5[0:len(copy_SMA_5)-1]

SMA_5 = SMA_5[1::]

copy_RSI_9 = RSI_9
copy_RSI_9.insert(0, 0)
copy_RSI_9 = copy_RSI_9[0:len(copy_RSI_9)-1]

RSI_9 = RSI_9[1::]

copy_RSI_13 = RSI_13
copy_RSI_13.insert(0, 0)
copy_RSI_13 = copy_RSI_13[0:len(copy_RSI_13)-1]

RSI_13 = RSI_13[1::]

# Experiment 2: Supply info from 2 iterations ago

copy_candle_sticks_2 = candle_sticks
copy_candle_sticks_2 = np.insert(
    copy_candle_sticks_2, 0, np.array([0, 0, 0, 0, 0]), axis=0)
copy_candle_sticks_2 = np.insert(
    copy_candle_sticks_2, 0, np.array([0, 0, 0, 0, 0]), axis=0)
copy_candle_sticks_2 = np.delete(copy_candle_sticks_2, 101, axis=0)
copy_candle_sticks_2 = np.delete(copy_candle_sticks_2, 100, axis=0)

copy_PSAR_arr_2 = PSAR_arr
copy_PSAR_arr_2.insert(0, 0)
copy_PSAR_arr_2.insert(0, 0)
copy_PSAR_arr_2 = copy_PSAR_arr_2[0:len(copy_PSAR_arr_2)-2]

PSAR_arr = PSAR_arr[2::]

copy_SMA_3_2 = SMA_3
copy_SMA_3_2.insert(0, 0)
copy_SMA_3_2.insert(0, 0)
copy_SMA_3_2 = copy_SMA_3_2[0:len(copy_SMA_3_2)-2]

SMA_3 = SMA_3[2::]

copy_SMA_5_2 = SMA_5
copy_SMA_5_2.insert(0, 0)
copy_SMA_5_2.insert(0, 0)
copy_SMA_5_2 = copy_SMA_5_2[0:len(copy_SMA_5_2)-2]

SMA_5 = SMA_5[2::]

copy_RSI_9_2 = RSI_9
copy_RSI_9_2.insert(0, 0)
copy_RSI_9_2.insert(0, 0)
copy_RSI_9_2 = copy_RSI_9_2[0:len(copy_RSI_9_2)-2]

RSI_9 = RSI_9[2::]

copy_RSI_13_2 = RSI_13
copy_RSI_13_2.insert(0, 0)
copy_RSI_13_2.insert(0, 0)
copy_RSI_13_2 = copy_RSI_13_2[0:len(copy_RSI_13_2)-2]

RSI_13 = RSI_13[2::]

# Experiment 3: Include Variables Measuring the Values of Indicators to each other

# SMA_3 relation to SMA_5
SMA_3v5 = []

for i in range(len(SMA_3)):
    if SMA_3[i] < SMA_5[i]:
        SMA_3v5.append(0)
    elif SMA_3[i] > SMA_5[i]:
        SMA_3v5.append(1)
    else:
        SMA_3v5.append(2)

copy_SMA_3v5 = SMA_3v5
copy_SMA_3v5.insert(0, 0)
copy_SMA_3v5 = copy_SMA_3v5[0:len(copy_SMA_3v5)-1]

SMA_3v5 = SMA_3v5[1::]

copy_SMA_3v5_2 = SMA_3v5
copy_SMA_3v5_2.insert(0, 0)
copy_SMA_3v5_2.insert(0, 0)
copy_SMA_3v5_2 = copy_SMA_3v5_2[0:len(copy_SMA_3v5_2)-2]

SMA_3v5 = SMA_3v5[2::]


# Set labels for this data
labels = [0]*len(candle_sticks[:, 3])
points = []

fig, (ax1) = plt.subplots(1, 1)
ax1.plot([i for i in range(len(candle_sticks[:, 3]))], candle_sticks[:, 3])  # Plot Closing Prices

# Get Plottable PSAR values
for i, val in enumerate(PSAR_arr):
    if val == 0:
        ax1.plot(i, candle_sticks[i][3] + (candle_sticks[i][3] * 0.001), "r.")
    else:
        ax1.plot(i, candle_sticks[i][3] - (candle_sticks[i][3] * 0.001), "g.")

# legend
ax1.plot(0, candle_sticks[0][3] +
         (candle_sticks[0][3] * 0.001), "r.", label="Sell")
ax1.plot(0, candle_sticks[0][3] -
         (candle_sticks[0][3] * 0.001), "g.", label="Buy")
ax1.legend(loc='best', numpoints=1)

# adding seperate legend for the SMA values
ax1.plot([i for i in range(len(candle_sticks[:, 3]))],
         SMA_3, "-", color="orange", label="SMA_3")
ax1.plot([i for i in range(len(candle_sticks[:, 3]))],
         SMA_5, "g-", label="SMA_5")
ax1.legend(loc='best', numpoints=1)

# # adding seperate legend for the RSI values
# ax2.plot([i for i in range(len(candle_sticks[:, 3]))],
#          RSI_9, "b-", label="RSI_9")
# ax2.plot([i for i in range(len(candle_sticks[:, 3]))],
#          RSI_13, "-", color="orange", label="RSI_13")
# ax2.legend(loc='best', numpoints=1)

# plot points coordinates using mouse click and seperate left and right click

print("\nLeft Click to Enter Trade and Right Click to Exit Trade\n")


def onclick(event):

    global enable

    if event.button == 1 and enable == True:
        print('Enter Trade: ', int(event.xdata), event.ydata)
        labels[int(event.xdata)] = 1
        points.append(int(event.xdata))

        # create a vertical line at the point clicked and display the line
        ax1.axvline(x=event.xdata, color='m', linestyle='--', linewidth=1)

        # display the line
        plt.draw()

    elif event.button == 3 and enable == True:
        print('Exit Trade: ', int(event.xdata), event.ydata)
        labels[int(event.xdata)] = 2
        points.append(int(event.xdata))

        # create a vertical line at the point clicked
        ax1.axvline(x=event.xdata, color='c', linestyle='--', linewidth=1)

        # display the line
        plt.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)


def on_key_press(event):

    global enable

    # if button 'b' is pressed, then remove the last line
    if event.key == 'b':
        ax1.lines.pop()
        plt.draw()

        labels[points[-1]] = 0
        points.pop()

    elif event.key == 'l':
        if enable == True:
            enable = False
        else:
            enable = True

cid = fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()

prev_label = 2
for i, label in enumerate(labels):
    if label == 0:
        labels[i] = prev_label    
    elif label == 1:
        prev_label = label
    elif label == 2:
        prev_label = label
    
print(labels)

# Construct a dataframe of the stock prices and indicators
df = pd.DataFrame({'open': candle_sticks[:, 0], 'high': candle_sticks[:, 1], 'low': candle_sticks[:, 2], 'close': candle_sticks[:, 3], 'volume': candle_sticks[:, 4], 'PSAR': PSAR_arr, 'SMA_3': SMA_3, 'SMA_5': SMA_5, 'RSI_9': RSI_9, 'RSI_13': RSI_13,
    'prev_open': copy_candle_sticks[:, 0], 'prev_high': copy_candle_sticks[:, 1], 'prev_low': copy_candle_sticks[:, 2], 'prev_close': copy_candle_sticks[:, 3], 'prev_volume': copy_candle_sticks[:, 4], 'prev_PSAR': copy_PSAR_arr, 'prev_SMA_3': copy_SMA_3, 'prev_SMA_5': copy_SMA_5, 'prev_RSI_9': copy_RSI_9, 'prev_RSI_13': copy_RSI_13,
    'prev_open_2': copy_candle_sticks_2[:, 0], 'prev_high_2': copy_candle_sticks_2[:, 1], 'prev_low_2': copy_candle_sticks_2[:, 2], 'prev_close_2': copy_candle_sticks_2[:, 3], 'prev_volume_2': copy_candle_sticks_2[:, 4], 'prev_PSAR_2': copy_PSAR_arr_2, 'prev_SMA_3_2': copy_SMA_3_2, 'prev_SMA_5_2': copy_SMA_5_2, 'prev_RSI_9_2': copy_RSI_9_2, 'prev_RSI_13_2': copy_RSI_13_2,
    'labels': labels})

df = df.iloc[2:]
df.drop(df.tail(1).index, inplace=True)  # drop last n rows

print(df.head())
# print(df.shape)

df.to_pickle('Dataset')
