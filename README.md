# Apple Stock Predictor

Minimal Streamlit app for Apple stock ML models (Random Forest, XGBoost, LSTM).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Connect this repo
3. Main file: `app.py`

## Models

| File | Description |
|------|-------------|
| `rf_model.pkl` | Random Forest |
| `xgb_model.pkl` | XGBoost |
| `lstm_model.keras` | LSTM (30 timesteps × 12 features) |
| `scaler.pkl` | Feature scaler |
| `features.pkl` | Feature column names |

Upload a CSV with the 12 feature columns, or use manual input in the app.
