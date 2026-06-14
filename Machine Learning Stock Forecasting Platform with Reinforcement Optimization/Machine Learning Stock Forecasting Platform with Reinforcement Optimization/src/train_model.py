import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from preprocess import preprocess_data


def create_dataset(dataset, time_step=60):

    X = []
    y = []

    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])

    return np.array(X), np.array(y)


def train():

    data, scaler = preprocess_data()

    X, y = create_dataset(data)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    model.fit(
        X,
        y,
        batch_size=32,
        epochs=10,
    )

    # Save model
    model.save("models/lstm_model.h5")

    print("Model trained and saved successfully")


if __name__ == "__main__":
    train()