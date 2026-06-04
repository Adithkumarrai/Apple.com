# Apple Stock Predictor — Website only

This project is a **website** (HTML + CSS + JavaScript), not a Streamlit app.

## Run locally

```bash
pip install -r requirements.txt
uvicorn server:app --reload
```

Open **http://127.0.0.1:8000**

## Deploy online (Render)

1. Connect [github.com/Adithkumarrai/Apple.com](https://github.com/Adithkumarrai/Apple.com)
2. Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`

## Files

| File / folder | Purpose |
|---------------|---------|
| `server.py` | Backend API |
| `website/` | Frontend (the website you see in the browser) |
| `model_utils.py` | ML models |
| `*.pkl`, `lstm_model.keras` | Saved models |
