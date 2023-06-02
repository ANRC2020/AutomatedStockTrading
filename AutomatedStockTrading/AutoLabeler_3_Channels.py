import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

try:
    os.system("cls")
except:
    pass

def AutoLabeler(graph):

    with open('stock_data.pkl', 'rb') as f:
        candle_sticks = pickle.load(f)

    # Surpress Scientific Notation
    np.set_printoptions(suppress=True)

    # Calculate the first derivative equivalent of the closing prices

    first_der = [0]

    for i in range(len(candle_sticks) - 1):
        first_der.append((candle_sticks[:,3][i + 1] - candle_sticks[:,3][i - 1])/2)

    # Calculate the second derivative equivalent of the closing prices

    second_der = [0]

    for i in range(1, len(first_der) - 1):
        second_der.append((first_der[i + 1] - first_der[i - 1])/2)

    # Determine the upper and lower thresholds
    upper_sum = 0
    lower_sum = 0

    upper_counter = 0
    lower_counter = 0

    for i in range(len(first_der) - 1):
        if first_der[i]  > 0:
            upper_sum += first_der[i]
            upper_counter += 1
        elif first_der[i] < 0:
            lower_sum += first_der[i]
            lower_counter += 1

    upper_thres = upper_sum / upper_counter
    lower_thres = lower_sum / lower_counter
    
    upper_sum = 0
    lower_sum = 0

    upper_counter = 0
    lower_counter = 0

    for i in range(len(first_der) - 1):
        if first_der[i] > lower_thres and first_der[i] < 0:
            lower_sum += first_der[i]
            lower_counter += 1

    # upper_thres = upper_sum / upper_counter
    lower_thres = lower_sum / lower_counter
    # lower_thres = 0

    # Construct a list of all entries and exits 
    arr = []

    for i in range(len(first_der) - 1):
        if first_der[i]  > upper_thres:
            arr.append([i,'g'])
        elif first_der[i] < lower_thres:
            arr.append([i,'r'])
        else:
            arr.append([i, 'y'])

    

    if graph == True:

        fig, (ax1) = plt.subplots(1, 1)
        ax1.plot([i for i in range(len(candle_sticks[:, 3]))], candle_sticks[:, 3])  # Plot Closing Prices

        for i in range(len(arr)):
            if arr[i][1] == 'g' :
                ax1.vlines(x = arr[i][0], ymin=min(candle_sticks[:,3]), ymax= max(candle_sticks[:,3]), colors='green')
            elif arr[i][1] == 'r':
                ax1.vlines(x = arr[i][0], ymin=min(candle_sticks[:,3]), ymax= max(candle_sticks[:,3]), colors='red')
            elif arr[i][1] == 'y':
                ax1.vlines(x = arr[i][0], ymin=min(candle_sticks[:,3]), ymax= max(candle_sticks[:,3]), colors='yellow')
            

        # ax2.plot([i for i in range(len(first_der))], first_der)
        # ax3.plot([i for i in range(len(second_der))], second_der) 

        plt.show()

        labels = labels[1::]
        labels.append(labels[-1])

    return labels

if __name__ == "__main__":
    AutoLabeler(True)