"""Web server: static site + prediction API."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from feature_engineering import prepare_csv_features
from model_utils import SEQUENCE_LENGTH, ModelBundle, get_feature_names

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "website"

app = FastAPI(title="Apple Stock Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=2)
def get_models(use_lstm: bool) -> ModelBundle:
    bundle = ModelBundle()
    bundle._ensure_tabular()
    if use_lstm:
        bundle._ensure_lstm()
    return bundle


class PredictRequest(BaseModel):
    features: dict[str, float]
    use_lstm: bool = False
    rows: list[dict[str, float]] | None = None


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/features")
def features():
    return {"features": get_feature_names(), "sequence_length": SEQUENCE_LENGTH}


@app.post("/api/predict")
def predict_json(body: PredictRequest):
    names = get_feature_names()
    missing = [n for n in names if n not in body.features]
    if missing:
        raise HTTPException(400, f"Missing features: {', '.join(missing)}")

    row_df = pd.DataFrame([{n: float(body.features[n]) for n in names}])
    seq_df = None
    lstm_ok = False

    if body.use_lstm and body.rows:
        seq_df = pd.DataFrame(body.rows)[names].astype(float)
        lstm_ok = len(seq_df) >= SEQUENCE_LENGTH
    elif body.use_lstm:
        raise HTTPException(400, f"LSTM needs {SEQUENCE_LENGTH}+ rows in 'rows'.")

    try:
        bundle = get_models(lstm_ok)
        rf = bundle.predict_tabular(row_df, "Random Forest")
        xgb = bundle.predict_tabular(row_df, "XGBoost")
        out = {
            "random_forest": rf,
            "xgboost": xgb,
            "lstm": None,
            "ensemble": (rf + xgb) / 2,
        }
        if lstm_ok:
            lstm = bundle.predict_lstm(seq_df)
            out["lstm"] = lstm
            out["ensemble"] = (rf + xgb + lstm) / 3
        return out
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    use_lstm: bool = Form(False),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file.")

    raw = pd.read_csv(io.BytesIO(await file.read()))
    try:
        seq_df, msg = prepare_csv_features(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    row_df = seq_df.tail(1)
    lstm_ok = use_lstm and len(seq_df) >= SEQUENCE_LENGTH

    if use_lstm and not lstm_ok:
        raise HTTPException(
            400, f"LSTM needs at least {SEQUENCE_LENGTH} rows (got {len(seq_df)})."
        )

    try:
        bundle = get_models(lstm_ok)
        rf = bundle.predict_tabular(row_df, "Random Forest")
        xgb = bundle.predict_tabular(row_df, "XGBoost")
        out = {
            "random_forest": rf,
            "xgboost": xgb,
            "lstm": None,
            "ensemble": (rf + xgb) / 2,
            "rows_loaded": len(seq_df),
            "message": msg,
        }
        if lstm_ok:
            lstm = bundle.predict_lstm(seq_df)
            out["lstm"] = lstm
            out["ensemble"] = (rf + xgb + lstm) / 3
        return out
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


app.mount("/static", StaticFiles(directory=WEB), name="static")
