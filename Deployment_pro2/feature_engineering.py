"""Build model features from raw stock CSV or use existing feature columns."""

from __future__ import annotations

import pandas as pd

from model_utils import get_feature_names

# Common header aliases (after lowercasing)
_OHLCV_ALIASES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "adjclose": "close",
    "price": "close",
    "volume": "volume",
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(x).strip() for x in col if str(x) != "").lower()
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip().lower() for c in out.columns]

    renamed = {}
    for col in out.columns:
        key = col.replace("_", " ").strip()
        if key in _OHLCV_ALIASES:
            renamed[col] = _OHLCV_ALIASES[key]
        elif col in _OHLCV_ALIASES:
            renamed[col] = _OHLCV_ALIASES[col]
    if renamed:
        out = out.rename(columns=renamed)
    return out


def _pick_price_column(df: pd.DataFrame) -> str:
    for name in ("close", "open", "high", "low"):
        if name in df.columns:
            return name
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric price columns found.")
    return numeric.columns[-1]


def build_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12 training features from OHLCV-style data."""
    df = _normalize_columns(df)

    if "close" not in df.columns:
        price_col = _pick_price_column(df)
        close = pd.to_numeric(df[price_col], errors="coerce")
    else:
        close = pd.to_numeric(df["close"], errors="coerce")

    open_ = pd.to_numeric(df["open"], errors="coerce") if "open" in df.columns else close
    high = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else close

    ret = close.pct_change()
    safe_close = close.replace(0, pd.NA)

    features = pd.DataFrame(
        {
            "momentum_3": close.pct_change(3),
            "momentum_5": close.pct_change(5),
            "momentum_10": close.pct_change(10),
            "volatility_5": ret.rolling(5).std(),
            "volatility_10": ret.rolling(10).std(),
            "hl_range": (high - low) / safe_close,
            "oc_range": (open_ - close) / safe_close,
            "rolling_mean_5": close.rolling(5).mean(),
            "rolling_std_5": close.rolling(5).std(),
            "lag_1": ret.shift(1),
            "lag_2": ret.shift(2),
            "lag_3": ret.shift(3),
        }
    )
    return features.dropna()


def prepare_csv_features(raw: pd.DataFrame, feature_names: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """
    Accept either:
    - CSV that already has the 12 model features, or
    - Any stock CSV with Open/High/Low/Close (names flexible).

    Returns (feature dataframe, message for UI).
    """
    names = feature_names or get_feature_names()
    raw = _normalize_columns(raw)

    if all(n in raw.columns for n in names):
        out = raw[names].astype(float).dropna()
        return out, f"Using {len(out)} rows (features already in file)."

    out = build_features_from_ohlcv(raw)
    if len(out) == 0:
        raise ValueError(
            "Could not build features. Use a CSV with columns like "
            "Date, Open, High, Low, Close — or the 12 feature columns."
        )
    return out, f"Built features from your data ({len(out)} rows)."
