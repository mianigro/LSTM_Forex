# Import Python modules
import os
from datetime import datetime

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Import local modules
from ml_funcs.tf_lstm import run_model
from ml_funcs.predict_data import predict_from_model
from resources.helper_funcs import fix_input_data, make_x_y_data
from resources.data_scaler import scale_data_min_max, inv_data_min_max
from tech_ind_funcs.gen_features import gen_feat


# Process data and train the model
def process_train(graphing, data):

    # Showing the label in a graph
    if graphing == True:
        plt.plot(data["high"])
        plt.show()
    

    # Generate a list of the feature columns and the amount of them
    data_cols = [col for col in data.columns]
    data_cols.remove("high")
    feat_count = len(data_cols)

    if graphing == True:
        print(data[:5])


    # Split dataframe in x,y f or train/valid/test
    train_x, train_y, valid_x, valid_y = make_x_y_data(data, 0.8, 0.2)


    # Fit standard scaler and scale the data
    scaler_train_x = MinMaxScaler()
    scaler_train_x.fit(train_x)

    train_x = scale_data_min_max(train_x, scaler_train_x, data_cols)
    valid_x = scale_data_min_max(valid_x, scaler_train_x, data_cols)

    train_y = np.asarray(train_y).astype('float32').reshape(-1, 1)
    scaler_train_y = MinMaxScaler()
    scaler_train_y.fit(train_y)
    train_y = scale_data_min_max(train_y, scaler_train_y)
    
    valid_y = np.asarray(valid_y).astype('float32').reshape(-1, 1)
    valid_y = scale_data_min_max(valid_y, scaler_train_y)



    # Convert the dataframe into the right format for LSTM (batch_size, time_steps, seq_len)
    x_train = fix_input_data(train_x, feat_count)
    x_valid = fix_input_data(valid_x, feat_count)


    # Setup and fit the model
    lr = 0.000005
    print("Train")
    model = run_model(x_train, train_y, x_valid, valid_y, lr, graphing, feat_count)
    print("Finished training")


    # Show results of training
    if graphing == True:
        # Create training predictions to review accuracy
        valid_pred = predict_from_model(model, x_valid)
        valid_pred = inv_data_min_max(valid_pred, scaler_train_y)

        valid_y = inv_data_min_max(valid_y, scaler_train_y)

        # Plot the data
        plt.plot(valid_pred)
        plt.plot(valid_y)
        plt.legend(["V Pred", "V Real"])
        plt.show()

    return model, scaler_train_y, scaler_train_x
