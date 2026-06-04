"""Load saved models and run predictions."""

from __future__ import annotations

import pickle
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEQUENCE_LENGTH = 30

FEATURE_LABELS = {
    "momentum_3": "3-day price momentum",
    "momentum_5": "5-day price momentum",
    "momentum_10": "10-day price momentum",
    "volatility_5": "5-day rolling volatility",
    "volatility_10": "10-day rolling volatility",
    "hl_range": "High − low range (normalized)",
    "oc_range": "Open − close range (normalized)",
    "rolling_mean_5": "5-day rolling mean",
    "rolling_std_5": "5-day rolling standard deviation",
    "lag_1": "1-day lag feature",
    "lag_2": "2-day lag feature",
    "lag_3": "3-day lag feature",
}


def _load_features() -> list[str]:
    with open(ROOT / "features.pkl", "rb") as f:
        return pickle.load(f)


def _load_scaler():
    return joblib.load(ROOT / "scaler.pkl")


def _load_rf():
    return joblib.load(ROOT / "rf_model.pkl")


def _load_xgb():
    return joblib.load(ROOT / "xgb_model.pkl")


def _strip_conflicting_python_homes() -> None:
    """Remove Python install dirs that contain a practice script named python.py."""
    import sys

    for entry in list(sys.path):
        python_py = Path(entry) / "python.py"
        if not python_py.is_file():
            continue
        try:
            head = python_py.read_text(encoding="utf-8", errors="ignore")[:300]
        except OSError:
            continue
        if "input(" in head and "list" in head.lower():
            sys.path.remove(entry)


def _build_lstm():
    """Rebuild LSTM from known architecture and load weights from .keras archive."""
    _strip_conflicting_python_homes()
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, len(_load_features()))),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    keras_path = ROOT / "lstm_model.keras"
    with zipfile.ZipFile(keras_path, "r") as archive:
        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "model.weights.h5"
            with open(weights_path, "wb") as out:
                out.write(archive.read("model.weights.h5"))
            model.load_weights(weights_path)
    return model


class ModelBundle:
    def __init__(self) -> None:
        self.feature_names = _load_features()
        self.scaler = _load_scaler()
        self.rf = _load_rf()
        self.xgb = _load_xgb()
        self._lstm = None

    @property
    def lstm(self):
        if self._lstm is None:
            self._lstm = _build_lstm()
        return self._lstm

    def _scale_tabular(self, frame: pd.DataFrame) -> np.ndarray:
        ordered = frame[self.feature_names].astype(float)
        return self.scaler.transform(ordered)

    def predict_tabular(self, frame: pd.DataFrame, model: str) -> float:
        """Predict from a single row of features (RF / XGBoost)."""
        scaled = self._scale_tabular(frame)
        if model == "Random Forest":
            return float(self.rf.predict(scaled)[0])
        if model == "XGBoost":
            return float(self.xgb.predict(scaled)[0])
        raise ValueError(f"Unsupported tabular model: {model}")

    def predict_lstm(self, sequence: pd.DataFrame) -> float:
        """Predict from the last 30 rows of scaled features."""
        if len(sequence) < SEQUENCE_LENGTH:
            raise ValueError(
                f"LSTM needs at least {SEQUENCE_LENGTH} rows; got {len(sequence)}."
            )
        window = sequence.tail(SEQUENCE_LENGTH)
        scaled = self._scale_tabular(window)
        batch = scaled.reshape(1, SEQUENCE_LENGTH, len(self.feature_names))
        return float(self.lstm.predict(batch, verbose=0)[0, 0])

    def predict_ensemble(
        self, row: pd.DataFrame, sequence: pd.DataFrame | None
    ) -> dict[str, float]:
        preds = {
            "Random Forest": self.predict_tabular(row, "Random Forest"),
            "XGBoost": self.predict_tabular(row, "XGBoost"),
        }
        if sequence is not None and len(sequence) >= SEQUENCE_LENGTH:
            preds["LSTM"] = self.predict_lstm(sequence)
        values = list(preds.values())
        preds["Ensemble (average)"] = float(np.mean(values))
        return preds
