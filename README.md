# 📈 Stock Price Prediction System

A deep learning-based stock price prediction system that forecasts the **next day's closing price** using historical market data. Three models were implemented and compared — LSTM, Transformer, and a Hybrid (LSTM + Transformer) — with the best-performing model deployed as an interactive **Streamlit web application**.

---

## 🧠 Models Used

| Model | Description |
|-------|-------------|
| **LSTM** | Long Short-Term Memory — best performer (lowest RMSE, highest R²) |
| **Transformer** | Self-attention based sequence model |
| **Hybrid** | LSTM + Transformer combined with dropout regularization |

---

## 🗂️ Project Structure

```
stock-price-prediction/
│
├── data/                   # Raw and processed data
├── notebooks/              # EDA and experimentation notebooks
├── models/
│   ├── lstm_model.py
│   ├── transformer_model.py
│   └── hybrid_model.py
├── best_model.pth          # Saved best model weights
├── scaler.save             # Saved MinMaxScaler
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Requirements

Make sure you have **Python 3.8+** installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

**`requirements.txt`** includes:
```
streamlit
torch
numpy
scikit-learn
joblib
yfinance
pandas
matplotlib
seaborn
```

---

## 🚀 How to Deploy

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional — skip if using saved model)

```bash
python train.py
```

This will:
- Fetch stock data from Yahoo Finance
- Preprocess and scale the data
- Train LSTM, Transformer, and Hybrid models
- Save the best model as `best_model.pth` and the scaler as `scaler.save`

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at:
```
http://localhost:8501
```

### ☁️ Deploy on Streamlit Cloud (Free Hosting)

1. Push your project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New app"**
4. Connect your GitHub repo and select `app.py` as the entry point
5. Click **Deploy** — your app will be live at a public URL!

---

## 🖥️ How to Access

| Method | URL |
|--------|-----|
| **Local** | `http://localhost:8501` (after running `streamlit run app.py`) |
| **Streamlit Cloud** | `https://your-app-name.streamlit.app` (after cloud deployment) |

---

## 📊 How to Use the App

1. **Open the app** in your browser (local or hosted URL)
2. **Enter the following stock data** for the current day:
   - `Close Price` — today's closing price
   - `Open Price` — today's opening price
   - `High Price` — today's highest price
   - `Low Price` — today's lowest price
   - `Volume` — number of shares traded
3. **Click the "Predict" button**
4. The app will display the **Next Day Predicted Closing Price**

---

## 📉 Data & Preprocessing

- **Data Source:** Yahoo Finance (`yfinance`)
- **Features used:** Open, Close, High, Low, Volume
- **Scaling:** MinMaxScaler (normalized to [0, 1])
- **Sequence length:** 100 time steps
- **Train/Test split:** 80% / 20%

**Moving Averages & EDA:**
```python
df['MA50']   = df['Close'].rolling(50).mean()
df['MA200']  = df['Close'].rolling(200).mean()
df['Return'] = df['Close'].pct_change()
```

---

## 🏋️ Training Details

- **Loss Function:** Mean Squared Error (MSELoss)
- **Optimizer:** Adam (`lr=0.0005`)
- **Evaluation Metrics:** RMSE, MAE, R² Score
- **Best Model:** LSTM (saved as `best_model.pth`)

---

## 🔮 Future Improvements

- Incorporate news sentiment analysis as additional features
- Experiment with more advanced architectures (e.g., Temporal Fusion Transformer)
- Deploy on cloud platforms (AWS, GCP, Azure) for production-grade scalability
- Add real-time stock data fetching in the app

---

## 📄 License

This project is for educational and research purposes only. Stock predictions should **not** be used as financial advice.

---

## 🙋 Author

> Built as part of a deep learning project.  
> Feel free to fork, star ⭐, and contribute!
