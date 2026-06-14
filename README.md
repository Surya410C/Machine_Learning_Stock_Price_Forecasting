# 📈 Machine Learning Stock Price Forecasting 

A deep learning-based stock price prediction platform that uses an **LSTM (Long Short-Term Memory)** neural network to forecast next-day stock closing prices. Historical data is fetched via Yahoo Finance, preprocessed with MinMax scaling, and the trained model is visualized against actual prices.

---

## 🚀 Features

- 📥 Automatic stock data download using `yfinance`
- 🧹 Data preprocessing with MinMax normalization
- 🧠 LSTM model with Dropout regularization for time-series forecasting
- 💾 Model persistence — trained model saved as `.h5`
- 📊 Visualization of actual vs. predicted stock prices
- 🔁 End-to-end pipeline runnable from a single `main.py`

---

## 🗂️ Project Structure

```
stock-price-prediction/
│
├── dataset/
│   └── download_data.py       # Downloads historical stock data via yfinance
│
├── data/
│   └── stock_data.csv         # Downloaded stock data (auto-generated)
│
├── models/
│   └── lstm_model.h5          # Saved trained LSTM model (auto-generated)
│
├── src/
│   ├── preprocess.py          # Loads and scales the stock data
│   ├── train_model.py         # Builds and trains the LSTM model
│   ├── predict.py             # Runs inference on the last 60 days
│   └── visualize.py           # Plots actual price vs. predicted price
│
├── requirements.txt           # Python dependencies
└── main.py                    # Full pipeline entry point
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | LSTM model training |
| scikit-learn | MinMax data normalization |
| yfinance | Stock data download |
| pandas / numpy | Data manipulation |
| matplotlib | Visualization |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A — Run the full pipeline at once

```bash
python main.py
```

This will automatically:
1. Download stock data
2. Train the LSTM model
3. Generate predictions and plot the chart

---

### Option B — Run step by step

**Step 1 — Download stock data**
```bash
python dataset/download_data.py
```

**Step 2 — Train the model**
```bash
python src/train_model.py
```

**Step 3 — Predict & visualize**
```bash
python src/visualize.py
```

---

## 🧠 Model Architecture

```
Input (60 timesteps)
    ↓
LSTM (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM (50 units)
    ↓
Dropout (0.2)
    ↓
Dense (25 units)
    ↓
Dense (1 unit) → Predicted Price
```

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** 32
- **Epochs:** 10
- **Time Step / Lookback Window:** 60 days

---

## 📉 Sample Output

The visualization plots the full historical closing price alongside a red dot marking the predicted next-day price.

```
Next day predicted price: $189.42
```

---

## 🔧 Configuration

To change the stock ticker or date range, edit `dataset/download_data.py`:

```python
def download_stock_data(ticker="AAPL"):   # ← change ticker here
    data = yf.download(
        ticker,
        start="2015-01-01",              # ← change start date
        end="2024-01-01"                 # ← change end date
    )
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
scikit-learn
tensorflow
yfinance
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- The `data/` and `models/` directories are auto-created on first run. Make sure they exist or the scripts will raise a file path error.
- The model is trained on closing prices only (`Close` column).
- Predictions are for educational purposes — not financial advice.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [Yahoo Finance API (yfinance)](https://pypi.org/project/yfinance/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
