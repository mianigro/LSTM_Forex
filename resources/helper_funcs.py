# Third party imports
import numpy as np

# These are a list of functions to help support the scripts for the data management

# Convert data into the batchs and sizes for LSTM
# Returns a numpy array in the correct size and dimensions in batches
def fix_input_data(df, df_rows):

    # Setup blank list
    df_out = []

    # Iterrate through the dataframe to create batches based on the lag feature list amount
    for index, d_row in df.iterrows():
        lag_list_buffer = []
        
        
        # Generates batches of rows
        for lags in range(0, df_rows):
            lag_list_buffer.append(d_row[lags])

        df_out.append(lag_list_buffer)

    # Reshapes the list into a numpy array of the correct dimensions
    df_out = np.reshape(np.array(df_out), (np.array(df_out).shape[0], np.array(df_out).shape[1], 1))

    # Returns the numpy array correctly formatted
    return df_out


# Process data frame into x and y for training, validation and test data
# Time series data needs it not random and organised correctly
def make_x_y_data(data, train_amount, valid_amount):
    # Setup x and y dataframes
    y_data = data["high"].copy()
    x_data = data.drop("high", axis=1)

    # Setup train/valid/test data points
    train_r = int(len(x_data)*train_amount)
    valid_r = int(len(x_data)*valid_amount)

    # Slice the arrays into the right length
    train_x, train_y = x_data[:train_r], y_data[:train_r]
    valid_x, valid_y = x_data[train_r:], y_data[train_r:]

    # Return the train/valid/test data
    return train_x, train_y, valid_x, valid_y