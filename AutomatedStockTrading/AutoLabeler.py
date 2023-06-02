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
        first_der.append((candle_sticks[:,3][i] - candle_sticks[:,3][i - 1])/2)

    # Calculate the second derivative equivalent of the closing prices

    second_der = [0]

    for i in range(1, len(first_der) - 1):
        second_der.append((first_der[i] - first_der[i - 1])/2)

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
        elif first_der[i] < upper_thres and first_der[i] > 0:
            upper_sum += first_der[i]
            upper_counter += 1

    upper_thres = upper_sum / upper_counter
    lower_thres = lower_sum / lower_counter
    # lower_thres = 0

    # Construct a list of all entries and exits 
    arr = []

    for i in range(len(first_der) - 1):
        if first_der[i]  > upper_thres:
            arr.append([i,'g'])
        elif first_der[i] < lower_thres:
            arr.append([i,'r'])

    # Remove all neigboring exits that are boardered by other exits
    i = 0
    while i < (len(arr) - 1):
        if arr[i][1] == 'r' and arr[i + 1][1] == 'r' and arr[i - 1][1] == 'r':
            del arr[i]
        else:
            i += 1

    # Start to fill in sections between the entry points

    updated_arr = []

    for i in range(len(first_der)):
        updated_arr.append([i,''])

    for entry in arr:
        updated_arr[entry[0]] = [entry[0], entry[1]]

    prev_label = "r"
    next_label = "r"

    prev_labels = []
    next_labels = []

    prev_labels.append(prev_label)
    next_labels.append(next_label)

    for i, entry in enumerate(updated_arr):
        
        # Update the Prev Label
        if entry[1] != "":
            prev_label = entry[1]
        prev_labels.append(prev_label)

        # Update the Next Label
        try:
            j = i + 1
            while updated_arr[j][1] == "":
                j += 1

            next_label = updated_arr[j][1]
            next_labels.append(next_label)
        except:
            next_label = prev_label
            next_labels.append(next_label)

    for i in range(len(prev_labels) - 1):
        if prev_labels[i] == next_labels[i] and prev_labels[i] == "g":
            updated_arr[i][1] = "g"
        elif prev_labels[i] == next_labels[i] and prev_labels[i] == "r":
            updated_arr[i][1] = "r"
        elif prev_labels[i] != next_labels[i] and prev_labels[i] == "r":
            updated_arr[i][1] = "g"
        elif prev_labels[i] != next_labels[i] and prev_labels[i] == "g":
            updated_arr[i][1] = "g"

    labels = []

    for entry in updated_arr:
        if entry[1] == "g":
            labels.append(0)
        elif entry[1] == "r":
            labels.append(1)

    if graph == True:

        fig, (ax1) = plt.subplots(1, 1)
        ax1.plot([i for i in range(len(candle_sticks[:, 3]))], candle_sticks[:, 3])  # Plot Closing Prices

        for i in range(len(updated_arr)):
            if updated_arr[i][1] == 'g' :
                ax1.vlines(x = updated_arr[i][0], ymin=min(candle_sticks[:,3]), ymax= max(candle_sticks[:,3]), colors='green')
            elif updated_arr[i][1] == 'r':
                ax1.vlines(x = updated_arr[i][0], ymin=min(candle_sticks[:,3]), ymax= max(candle_sticks[:,3]), colors='red')
            

        # ax2.plot([i for i in range(len(first_der))], first_der)
        # ax3.plot([i for i in range(len(second_der))], second_der) 

        plt.show()

        labels = labels[1::]
        labels.append(labels[-1])

    return labels

if __name__ == "__main__":
    AutoLabeler(True)