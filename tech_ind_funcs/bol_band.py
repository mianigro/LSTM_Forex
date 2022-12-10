# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Calculates bollinger bands and returns the sliced data, moving average and upper/lower bol bands

# Calculate bollinger bands
def bol_bands(data, graphing):

    # Setup the variables to track
    moving_ave = []
    std_n = []
    range_ave = 20


    # Convert to numpy for faster processing
    data = data.to_numpy()


    # Loop through the NP array to do the calculations for bol bands
    for x in range(range_ave, len(data)):
        buffer_sum = sum(data[x-range_ave:x])
        std_n.append(np.std(data[x-range_ave:x]))
        buffer_ave = buffer_sum/range_ave
        moving_ave.append(buffer_ave)


    # Generate upper and lower bol bands
    bol_u = [x + y for x,y in zip(moving_ave, std_n)]
    bol_d = [x - y for x,y in zip(moving_ave, std_n)]

    # Cut out the initial 20 bands as they wont have a bol band
    data_out_sliced = data[20:]

    if graphing == True:
        plt.plot(moving_ave)
        plt.plot(data_out_sliced)
        plt.plot(bol_u)
        plt.plot(bol_d)
        plt.legend(['Moving Ave', 'Data', "Bol 1", "Bol 2"])
        plt.show()

    # Return the results as a list
    return moving_ave, bol_u, bol_d, data_out_sliced