import subprocess

print("Downloading stock data...")
subprocess.run(["python", "dataset/download_data.py"])

print("Training model...")
subprocess.run(["python", "src/train_model.py"])

print("Predicting stock price...")
subprocess.run(["python", "src/visualize.py"])