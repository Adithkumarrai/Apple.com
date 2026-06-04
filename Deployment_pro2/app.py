"""Apple stock predictor — fast minimal UI."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import streamlit as st

from model_utils import SEQUENCE_LENGTH, ModelBundle, get_feature_names

st.set_page_config(
    page_title="Apple Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 42rem;}
    h1 {font-weight: 500; font-size: 1.75rem;}
    .subtitle {color: #6b7280; font-size: 0.9rem; margin-bottom: 1.5rem;}
    .result-box {border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.4rem 0; background: #fff;}
    .result-label {color: #6b7280; font-size: 0.75rem; text-transform: uppercase;}
    .result-value {font-size: 1.35rem; font-weight: 500;}
    .hero-value {font-size: 2rem; font-weight: 500;}
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


def render_prediction(label: str, value: float, hero: bool = False) -> None:
    css = "hero-value" if hero else "result-value"
    st.markdown(
        f'<div class="result-box"><div class="result-label">{label}</div>'
        f'<div class="{css}">{value:.6f}</div></div>',
        unsafe_allow_html=True,
    )


def manual_form(names: list[str]) -> pd.DataFrame:
    cols = st.columns(3)
    values = {}
    for i, name in enumerate(names):
        with cols[i % 3]:
            values[name] = st.number_input(name.replace("_", " "), 0.0, format="%.4f", key=f"f_{name}")
    return pd.DataFrame([values])


def main() -> None:
    st.markdown("# Apple predictor")
    st.markdown('<p class="subtitle">RF · XGBoost · optional LSTM</p>', unsafe_allow_html=True)

    names = get_feature_names()
    use_lstm = st.checkbox("Include LSTM (slower, needs 30+ CSV rows)", value=False)

    tab_csv, tab_manual = st.tabs(["CSV", "Manual"])
    row_df, seq_df = None, None

    with tab_csv:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            raw = pd.read_csv(file)
            missing = [c for c in names if c not in raw.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                return
            seq_df = raw[names].astype(float)
            row_df = seq_df.tail(1)
            st.caption(f"{len(seq_df)} rows")

    with tab_manual:
        row_df = manual_form(names)

    if st.button("Predict", type="primary", use_container_width=True):
        if row_df is None:
            st.warning("Add data first.")
            return

        lstm_ok = use_lstm and seq_df is not None and len(seq_df) >= SEQUENCE_LENGTH
        if use_lstm and not lstm_ok:
            st.warning(f"LSTM needs a CSV with at least {SEQUENCE_LENGTH} rows.")

        try:
            with st.spinner("Running models…" + (" (first run can take 1–2 min)" if lstm_ok else "")):
                bundle = get_models(lstm_ok)
                rf = bundle.predict_tabular(row_df, "Random Forest")
                xgb = bundle.predict_tabular(row_df, "XGBoost")
                lstm = bundle.predict_lstm(seq_df) if lstm_ok else None

            st.markdown("### Results")
            render_prediction("Random Forest", rf)
            render_prediction("XGBoost", xgb)
            if lstm is not None:
                render_prediction("LSTM", lstm)
                render_prediction("Ensemble", (rf + xgb + lstm) / 3, hero=True)
            else:
                render_prediction("Ensemble", (rf + xgb) / 2, hero=True)

        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
