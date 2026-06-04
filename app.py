"""
Apple predictor — Streamlit Cloud deploy (website-style UI).
Deploy at share.streamlit.io with main file: app.py
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import streamlit as st

from feature_engineering import prepare_csv_features
from model_utils import SEQUENCE_LENGTH, ModelBundle, get_feature_names

st.set_page_config(
    page_title="Apple Predictor",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 42rem;}
    h1 {font-weight: 500; font-size: 1.75rem; letter-spacing: -0.02em;}
    .sub {color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;}
    div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #e5e7eb;
        border-radius: 8px; padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models…")
def get_models(use_lstm: bool) -> ModelBundle:
    bundle = ModelBundle()
    bundle._ensure_tabular()
    if use_lstm:
        bundle._ensure_lstm()
    return bundle


def show_results(rf: float, xgb: float, lstm: float | None) -> None:
    st.markdown("### Results")
    c1, c2 = st.columns(2)
    c1.metric("Random Forest", f"{rf:.6f}")
    c2.metric("XGBoost", f"{xgb:.6f}")
    if lstm is not None:
        st.metric("LSTM", f"{lstm:.6f}")
        st.metric("Ensemble", f"{(rf + xgb + lstm) / 3:.6f}")
    else:
        st.metric("Ensemble", f"{(rf + xgb) / 2:.6f}")


def main() -> None:
    st.markdown("# Apple predictor")
    st.markdown(
        '<p class="sub">Random Forest · XGBoost · LSTM — hosted on Streamlit</p>',
        unsafe_allow_html=True,
    )

    names = get_feature_names()
    use_lstm = st.checkbox("Include LSTM (needs CSV with 30+ rows)", value=False)

    tab_csv, tab_manual = st.tabs(["CSV upload", "Manual input"])

    row_df: pd.DataFrame | None = None
    seq_df: pd.DataFrame | None = None

    with tab_csv:
        st.caption(
            "Upload any stock CSV (Open, High, Low, Close) or a file that already "
            "has the 12 feature columns."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                raw = pd.read_csv(uploaded)
                seq_df, msg = prepare_csv_features(raw, names)
                row_df = seq_df.tail(1)
                st.success(msg)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

    with tab_manual:
        st.caption("Enter values in training scale (default 0).")
        cols = st.columns(3)
        values: dict[str, float] = {}
        for i, name in enumerate(names):
            with cols[i % 3]:
                values[name] = st.number_input(
                    name.replace("_", " "),
                    value=0.0,
                    format="%.4f",
                    key=f"in_{name}",
                )
        row_df = pd.DataFrame([values])

    if st.button("Predict", type="primary", use_container_width=True):
        if row_df is None:
            st.warning("Add data first.")
            st.stop()

        lstm_ok = use_lstm and seq_df is not None and len(seq_df) >= SEQUENCE_LENGTH
        if use_lstm and not lstm_ok:
            st.warning(f"LSTM needs a CSV with at least {SEQUENCE_LENGTH} rows.")

        try:
            msg = "Running…" + (" (first time may take 1–2 min)" if lstm_ok else "")
            with st.spinner(msg):
                bundle = get_models(lstm_ok)
                rf = bundle.predict_tabular(row_df, "Random Forest")
                xgb = bundle.predict_tabular(row_df, "XGBoost")
                lstm = bundle.predict_lstm(seq_df) if lstm_ok else None
            show_results(rf, xgb, lstm)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
