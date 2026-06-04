# Apple Stock Predictor

Minimal **website-style** UI on **Streamlit Cloud**.

## Deploy on Streamlit (recommended for you)

1. Push this repo to GitHub: [github.com/Adithkumarrai/Apple.com](https://github.com/Adithkumarrai/Apple.com)
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub
3. **Create app** → repository: `Adithkumarrai/Apple.com`
4. **Main file path:** `app.py`
5. **Deploy**

Your public link will look like: `https://apple-com-xxxx.streamlit.app`

First deploy can take **5–10 minutes** (TensorFlow install). Leave **Include LSTM** unchecked for faster predictions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Also available (optional)

- `server.py` + `website/` — standalone HTML site (Render / local uvicorn)

## Models

`features.pkl`, `scaler.pkl`, `rf_model.pkl`, `xgb_model.pkl`, `lstm_model.keras`
