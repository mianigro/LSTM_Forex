# Third party imports
import numpy as np


# Making the lag features for the data. Returns the new dataframe with a list of lag feature names
def make_lag_features(n, data):

    # Setup list for lag feature names
    lag_list = []

    # Loop through the amount of lag features needed to generate lags
    for x in range(1,n+1):

        data[f"high_{x}"] = data["high"].shift(x)
        lag_list.append(f"high_{x}")

    # Lag features creates NaN rows at the start of the data, this slices after those NaN rows    
    data = data[n::]

    # Returns new dataset dataframe and lag feature list
    return data, lag_list