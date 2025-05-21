# scripts/predict_volume_lstm.pypyt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# === CONFIG ===
DATA_PATH = '../data/traffic_cleaned.csv'
MODEL_PATH = '../models/lstm_traffic_model.h5'
OUTPUT_PATH = '../predictions/predicted_volume_LSTM.csv'
SEQ_LEN = 10  # Days of data per input sequence

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate daily volume
daily_volume = df.groupby('Date')['Volume'].sum().reset_index()

# Scale volume
scaler = MinMaxScaler()
daily_volume['Volume_scaled'] = scaler.fit_transform(daily_volume[['Volume']])

# Create sequences
def create_sequences(data, seq_len):
    xs, dates = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        date = daily_volume['Date'].iloc[i+seq_len]  # predict date
        xs.append(x)
        dates.append(date)
    return np.array(xs), dates

volume_series = daily_volume['Volume_scaled'].values
X, predict_dates = create_sequences(volume_series, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Load LSTM model
model = load_model(MODEL_PATH)

# Predict
pred_scaled = model.predict(X)
pred_volume = scaler.inverse_transform(pred_scaled)

# Output DataFrame
results_df = pd.DataFrame({
    'Date': predict_dates,
    'PredictedVolume': pred_volume.flatten().round(2)
})

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f" Saved predictions to {OUTPUT_PATH}")
# scripts/predict_volume_lstm.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# === CONFIG ===
DATA_PATH = '../data/traffic_cleaned.csv'
MODEL_PATH = '../models/lstm_traffic_model.h5'
OUTPUT_PATH = '../predictions/predicted_volume_LSTM.csv'
SEQ_LEN = 10  # Days of data per input sequence

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate daily volume
daily_volume = df.groupby('Date')['Volume'].sum().reset_index()

# Scale volume
scaler = MinMaxScaler()
daily_volume['Volume_scaled'] = scaler.fit_transform(daily_volume[['Volume']])

# Create sequences
def create_sequences(data, seq_len):
    xs, dates = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        date = daily_volume['Date'].iloc[i+seq_len]  # predict date
        xs.append(x)
        dates.append(date)
    return np.array(xs), dates

volume_series = daily_volume['Volume_scaled'].values
X, predict_dates = create_sequences(volume_series, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Load LSTM model
model = load_model(MODEL_PATH)

# Predict
pred_scaled = model.predict(X)
pred_volume = scaler.inverse_transform(pred_scaled)

# Output DataFrame
results_df = pd.DataFrame({
    'Date': predict_dates,
    'PredictedVolume': pred_volume.flatten().round(2)
})

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f" Saved predictions to {OUTPUT_PATH}")
