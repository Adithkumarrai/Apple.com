# Apple Stock Predictor (Website)

A simple **website** (HTML + FastAPI) instead of Streamlit.

## Run locally

```bash
pip install -r requirements.txt
uvicorn server:app --reload
```

Open **http://127.0.0.1:8000**

## Deploy as a website (free)

### Render (recommended)

1. Push this folder to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo.
3. **Start command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Deploy — you get a URL like `https://apple-predictor.onrender.com`

### Railway

Same repo; start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`

## Project layout

| Path | Role |
|------|------|
| `server.py` | API + serves the website |
| `website/` | HTML, CSS, JavaScript |
| `model_utils.py` | ML models |
| `*.pkl`, `lstm_model.keras` | Saved models |

## Old Streamlit app

`app.py` is kept for reference. Use `server.py` for the website.
