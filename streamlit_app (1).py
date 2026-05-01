import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# ======================
# MODEL CLASS
# ======================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.fc = nn.Linear(64,1)

    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

# ======================
# DEBUG (optional)
# ======================



import os

# ======================
# LOAD MODEL + SCALER
# ======================
model = LSTMModel()

model_path = os.path.join("src", "best_model.pth")
scaler_path = os.path.join("src", "scaler.save")

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

scaler = joblib.load(scaler_path)

# ======================
# UI
# ======================
st.title(" Stock Price Prediction App")
st.write("Enter latest stock values to predict next day price")

close = st.number_input("Close Price")
open_ = st.number_input("Open Price")
high = st.number_input("High Price")
low = st.number_input("Low Price")
volume = st.number_input("Volume")

# ======================
# PREDICTION
# ======================
if st.button("Predict"):
    input_data = np.array([[close, open_, high, low, volume]])

    scaled = scaler.transform(input_data)

    sequence = np.repeat(scaled, 100, axis=0)
    sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

    pred_scaled = model(sequence).detach().numpy()

    dummy = np.zeros((1, 5))
    dummy[:, 0] = pred_scaled.flatten()

    pred = scaler.inverse_transform(dummy)[:, 0]

    st.success(f"Predicted Next Day Price: {pred[0]:.2f}")