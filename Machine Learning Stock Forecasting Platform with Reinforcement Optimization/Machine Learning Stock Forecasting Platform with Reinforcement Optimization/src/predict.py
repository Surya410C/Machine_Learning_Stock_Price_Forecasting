import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_data


def predict():

    data, scaler = preprocess_data()

    model = load_model("../models/lstm_model.h5")

    last_60_days = data[-60:]

    X_test = np.reshape(last_60_days, (1,60,1))

    prediction = model.predict(X_test)

    prediction = scaler.inverse_transform(prediction)

    print("Next day predicted price:", prediction[0][0])

    return prediction