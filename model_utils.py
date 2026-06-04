"""Load saved models and run predictions (lazy loading for speed)."""

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


def get_feature_names() -> list[str]:
    with open(ROOT / "features.pkl", "rb") as f:
        return pickle.load(f)


def _strip_conflicting_python_homes() -> None:
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


def _build_lstm(feature_count: int):
    _strip_conflicting_python_homes()
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, feature_count)),
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
    with zipfile.ZipFile(ROOT / "lstm_model.keras", "r") as archive:
        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "model.weights.h5"
            weights_path.write_bytes(archive.read("model.weights.h5"))
            model.load_weights(weights_path)
    return model


class ModelBundle:
    """Loads RF/XGB quickly; LSTM only when requested."""

    def __init__(self, load_lstm: bool = False) -> None:
        self.feature_names = get_feature_names()
        self._scaler = None
        self._rf = None
        self._xgb = None
        self._lstm = None
        self._lstm_requested = load_lstm

    def _ensure_tabular(self) -> None:
        if self._rf is not None:
            return
        self._scaler = joblib.load(ROOT / "scaler.pkl")
        self._rf = joblib.load(ROOT / "rf_model.pkl")
        self._xgb = joblib.load(ROOT / "xgb_model.pkl")

    def _ensure_lstm(self) -> None:
        if self._lstm is not None:
            return
        self._lstm = _build_lstm(len(self.feature_names))

    def _scale(self, frame: pd.DataFrame) -> np.ndarray:
        self._ensure_tabular()
        return self._scaler.transform(frame[self.feature_names].astype(float))

    def predict_tabular(self, frame: pd.DataFrame, model: str) -> float:
        scaled = self._scale(frame)
        if model == "Random Forest":
            return float(self._rf.predict(scaled)[0])
        if model == "XGBoost":
            return float(self._xgb.predict(scaled)[0])
        raise ValueError(model)

    def predict_lstm(self, sequence: pd.DataFrame) -> float:
        if len(sequence) < SEQUENCE_LENGTH:
            raise ValueError(f"LSTM needs {SEQUENCE_LENGTH}+ rows.")
        self._ensure_lstm()
        window = sequence.tail(SEQUENCE_LENGTH)
        batch = self._scale(window).reshape(1, SEQUENCE_LENGTH, len(self.feature_names))
        return float(self._lstm.predict(batch, verbose=0)[0, 0])
