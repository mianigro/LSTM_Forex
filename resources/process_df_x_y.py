# Third party imports
import pandas as pd
import numpy as np

# Import local modules
from resources.helper_funcs import fix_input_data
from resources.data_scaler import scale_data_min_max



def make_df_xy(data, scaler_train_x):
# Generate a list of the feature columns and the amount of them
    data_cols = [col for col in data.columns]
    data_cols.remove("high")
    feat_count = len(data_cols)

    # Split dataframe in x,y f or train/valid/test
    y_preds = data["high"].copy()
    x_preds = data.drop("high", axis=1).copy()

    # Fit standard scaler and scale the data
    scaled_x_preds = scale_data_min_max(x_preds, scaler_train_x, data_cols)

    # Convert the dataframe into the right format for LSTM (batch_size, time_steps, seq_len)
    scaled_x_preds = fix_input_data(scaled_x_preds, feat_count)

    return scaled_x_preds