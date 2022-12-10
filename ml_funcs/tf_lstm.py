# Third party imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt

# This is the NN model, training and graphing of the results

# Build and train the model
def run_model(x_train, y_train, x_valid, y_valid, lr, graphing, feats):

    # Set optimiser and learning rate
    opt = Adam(learning_rate=lr)

    # Build the model
    print("Build")
    model = Sequential([layers.LSTM(64, activation="relu", input_shape=(feats, 1), return_sequences=True),
                    layers.Dropout(0.2),
                    layers.LSTM(64, activation="relu", return_sequences=True),
                    layers.Flatten(),
                    layers.Dropout(0.2),
                    layers.Dense(32, activation="relu"),
                    layers.Dropout(0.2),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1)])

    # Compile model
    print("Compile")
    model.compile(loss="mse", optimizer=opt, metrics=["mean_absolute_error"])

    # Training the model
    print("Training model.")
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=60, batch_size=32)

    if graphing == True:
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Validation MAE'])
        plt.show()

    # Return the trained model
    return model
