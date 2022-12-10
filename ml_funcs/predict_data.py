# Third party import
import numpy as np

# This predicts from the model

# Prediction function
def predict_from_model(model, X):

    # Make prediction
    preds = model.predict(X)

    # Return results
    return preds