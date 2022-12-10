# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Calculates RSI and returns the sliced data and RSI

# Calcualte RSI
def rsi_func(data, graphing):
    range_rsi = 14
    move_u = [0]
    move_d = [0]
    ave_u = []
    ave_d = []
    rs = []
    rsi = []

    # Convert to numpy array
    data = data.to_numpy()

    
    # Make daily changes
    for x in range(1, len(data)):
        if data[x] - data[x-1] > 0:
            move_u.append(data[x] - data[x-1])
            move_d.append(0)
        
        else:
            move_u.append(0)
            move_d.append(abs(data[x] - data[x-1]))

    # Moving average of up and down
    for x in range(range_rsi, len(data)):
        buffer_sum_u = sum(move_u[x-range_rsi:x])
        buffer_sum_d = sum(move_d[x-range_rsi:x])
        ave_u.append(buffer_sum_u)
        ave_d.append(buffer_sum_d)   

    # Calc RS
    for x in range(0,len(ave_u)):
        rs.append(ave_u[x]/ave_d[x])

    # Cacl RSI
    for x in range(0,len(rs)):
        rsi.append(100-(100)/(1 + rs[x]))

    # Slice data points for RSI
    slice_data_points = data[range_rsi:]

    if graphing == True:
        plt.plot(rsi)
        plt.legend(['RSI'])
        plt.show()

    return rsi, slice_data_points