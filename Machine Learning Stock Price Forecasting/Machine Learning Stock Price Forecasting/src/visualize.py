import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def plot_prediction():

    # Load dataset
    data = pd.read_csv("data/stock_data.csv")

    # Keep only Close column
    data = data[['Close']]

    # Convert to numeric (remove AAPL or text rows)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    # Remove invalid rows
    data = data.dropna()

    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    # Load trained model
    model = load_model("models/lstm_model.h5")

    # Prepare last 60 days
    last_60_days = scaled_data[-60:]

    X_test = np.reshape(last_60_days, (1,60,1))

    # Predict
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    print("Next day predicted price:", predicted_price[0][0])

    # Plot graph
    plt.figure(figsize=(10,6))
    plt.plot(data['Close'], label="Actual Price")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_prediction()