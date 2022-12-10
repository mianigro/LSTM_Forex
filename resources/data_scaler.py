# Third party imports
import pandas as pd
import numpy as np


# Scales the data
def scale_data_min_max(data, scaler, cols=[]):

    data = scaler.transform(data)

    if cols != []:
        data = pd.DataFrame(data, columns=cols)

    return data


# Unscales the data
def inv_data_min_max(normalized, scaler, cols=[]):

    data = scaler.inverse_transform(normalized)
    
    if cols != []:
        data = pd.DataFrame(data, columns=cols)

    return data