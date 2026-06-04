"""Build model features from any uploaded CSV — no fixed column names required."""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_utils import get_feature_names


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(x).strip() for x in col if str(x) != "").lower()
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip().lower() for c in out.columns]
    drop = [c for c in out.columns if c.startswith("unnamed") or c in ("", "index")]
    return out.drop(columns=drop, errors="ignore")


def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep every column that is mostly numeric (any header names OK)."""
    out = {}
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= max(3, len(series) // 10):
            out[str(col)] = series
    if not out:
        for col in df.columns:
            series = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            if series.notna().any():
                out[str(col)] = series
    if not out:
        raise ValueError("This file has no usable numeric data.")
    return pd.DataFrame(out)


def _find_col(columns: list[str], patterns: tuple[str, ...]) -> str | None:
    for col in columns:
        for p in patterns:
            if p in col or col == p:
                return col
    return None


def _infer_prices(numeric: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    cols = list(numeric.columns)

    close_key = _find_col(
        cols,
        ("adj close", "adjclose", "adj_close", "close", "closing", "price", "last"),
    )
    open_key = _find_col(cols, ("open", "opening", "o"))
    high_key = _find_col(cols, ("high", "hi", "h"))
    low_key = _find_col(cols, ("low", "lo", "l"))

    if close_key is None:
        # Use the numeric column with the largest typical values (usually price)
        means = {c: numeric[c].mean(skipna=True) for c in cols}
        close_key = max(means, key=lambda k: means[k] if pd.notna(means[k]) else -1)

    close = numeric[close_key].astype(float)

    if open_key and open_key in numeric:
        open_ = numeric[open_key].astype(float)
    else:
        open_ = close.shift(1).fillna(close)

    if high_key and high_key in numeric:
        high = numeric[high_key].astype(float)
    else:
        high = pd.concat([open_, close], axis=1).max(axis=1)

    if low_key and low_key in numeric:
        low = numeric[low_key].astype(float)
    else:
        low = pd.concat([open_, close], axis=1).min(axis=1)

    return open_, high, low, close


def build_features_from_any(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    numeric = _to_numeric_frame(df)
    open_, high, low, close = _infer_prices(numeric)

    ret = close.pct_change()
    safe = close.replace(0, np.nan)

    features = pd.DataFrame(
        {
            "momentum_3": close.pct_change(3),
            "momentum_5": close.pct_change(5),
            "momentum_10": close.pct_change(10),
            "volatility_5": ret.rolling(5, min_periods=2).std(),
            "volatility_10": ret.rolling(10, min_periods=3).std(),
            "hl_range": (high - low) / safe,
            "oc_range": (open_ - close) / safe,
            "rolling_mean_5": close.rolling(5, min_periods=1).mean(),
            "rolling_std_5": close.rolling(5, min_periods=2).std(),
            "lag_1": ret.shift(1),
            "lag_2": ret.shift(2),
            "lag_3": ret.shift(3),
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)
    return features


def prepare_csv_features(
    raw: pd.DataFrame, feature_names: list[str] | None = None
) -> tuple[pd.DataFrame, str]:
    names = feature_names or get_feature_names()
    raw = _normalize_columns(raw)

    if all(n in raw.columns for n in names):
        out = raw[names].apply(pd.to_numeric, errors="coerce").dropna()
        if len(out) > 0:
            return out, f"Ready — {len(out)} rows."

    out = build_features_from_any(raw)
    if len(out) < 1:
        raise ValueError("Not enough rows in this file to predict.")
    return out, f"Ready — {len(out)} rows from your file."
