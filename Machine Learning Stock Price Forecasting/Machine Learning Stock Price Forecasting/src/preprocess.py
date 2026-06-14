import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data():
    # Load dataset
    data = pd.read_csv("data/stock_data.csv")

    # Select Close column only
    data = data[['Close']]

    # Convert Close to numeric (remove strings like AAPL)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    # Remove rows that became NaN
    data = data.dropna()

    return data


def preprocess_data():
    # Load cleaned data
    data = load_data()

    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale values
    scaled_data = scaler.fit_transform(data.values)

    return scaled_data, scaler