# Third party imports
import pandas as pd
import numpy as np

# Local imports
from tech_ind_funcs.bol_band import bol_bands
from tech_ind_funcs.rsi_calc import rsi_func
from tech_ind_funcs.lag_feats import make_lag_features

def gen_feat(data, lag_features, graphing=False):

    data_orig = data.drop("high", axis=1).copy()
    data_orig = data_orig[20:].reset_index()
    data_high = data["high"].copy()

    # Make bollinger bands
    moving_ave, bol_u, bol_d, data_out_bol = bol_bands(data_high, graphing)


    # Make RSI indicator
    rsi, data_out_rsi = rsi_func(data_high, graphing)


    # Shortest dataset will be bol bands dataset, so slice rsi to match length
    len_bol = len(data_out_bol)
    len_rsi = len(rsi)
    rsi = rsi[len_rsi-len_bol:]


    # Columns to make data dataframe and constructing it from high and bollinger bands
    cols = ["ma", "bol_u", "bol_d", "rsi", "high"]
    df_list = list(zip(moving_ave, bol_u, bol_d, rsi, data_out_bol))
    data_feats = pd.DataFrame(df_list, columns=cols)

    data = pd.concat([data_feats, data_orig], axis=1)
    data.index.name = "time"
    data = data.drop("time", axis=1)

    # Make lag features
    data, lag_list = make_lag_features(lag_features, data)

    return data