import yfinance as yf
import pandas as pd

def download_stock_data(ticker="AAPL"):
    
    data = yf.download(
        ticker,
        start="2015-01-01",
        end="2026-01-01"
    )

    data.to_csv("data/stock_data.csv")
    print("Data downloaded successfully")

if __name__ == "__main__":
    download_stock_data()

   