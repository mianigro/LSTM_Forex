# Third party imports
import pandas as pd
import numpy as np

# Import local modules
from tech_ind_funcs.gen_features import gen_feat
from resources.process_df_x_y import make_df_xy
from resources.data_scaler import inv_data_min_max

def rec_pred(model, data, forecasts, lag_features, scaler_train_x, scaler_train_y, graphing):

    pred_out = [data["high"].iloc[-1]]

    # Loop through for how many forecasts needed
    for x in range(0, forecasts):

        # Process data - Generate the data features - Bollinger Bands, RSI and Lags 
        backtest_processed_data = gen_feat(data, lag_features, graphing)

        # Turn data_backtest into data for model
        scaled_x_preds = make_df_xy(backtest_processed_data, scaler_train_x)

        # Make a prediction based of last entry on the data
        pred_scaled = model(scaled_x_preds[-1].reshape(1, 24, 1))
        pred_y = inv_data_min_max(pred_scaled, scaler_train_y)[0][0]

        pred_out.append(pred_y)

        # Apppend the new y value onto the raw data to restart prediction
        data = np.append(data["high"].to_numpy(), pred_y)
        data = pd.DataFrame(data, columns = ["high"])
        data.index.name='time'
    
    return data, pred_out
