"""Apple stock predictor — minimal Streamlit UI."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import streamlit as st

from model_utils import SEQUENCE_LENGTH, ModelBundle

st.set_page_config(
    page_title="Apple Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 42rem;}
    h1 {font-weight: 500; letter-spacing: -0.02em; font-size: 1.75rem; margin-bottom: 0.25rem;}
    .subtitle {color: #6b7280; font-size: 0.95rem; margin-bottom: 2rem;}
    .result-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 0.5rem 0;
        background: #fff;
    }
    .result-label {color: #6b7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em;}
    .result-value {font-size: 1.5rem; font-weight: 500; margin-top: 0.15rem;}
    .hero-value {font-size: 2.25rem; font-weight: 500; letter-spacing: -0.03em;}
    div[data-testid="stFileUploader"] section {padding: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models() -> ModelBundle:
    return ModelBundle()


def render_prediction(label: str, value: float, hero: bool = False) -> None:
    css_class = "hero-value" if hero else "result-value"
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">{label}</div>
            <div class="{css_class}">{value:.6f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def manual_features_form(names: list[str]) -> pd.DataFrame:
    st.caption("Enter 12 features (training scale). Defaults are 0.")
    cols = st.columns(3)
    values: dict[str, float] = {}
    for i, name in enumerate(names):
        with cols[i % 3]:
            values[name] = st.number_input(
                name.replace("_", " "),
                value=0.0,
                format="%.4f",
                label_visibility="visible",
                key=f"f_{name}",
            )
    return pd.DataFrame([values])


def run_predictions(bundle: ModelBundle, row: pd.DataFrame, seq: pd.DataFrame | None) -> None:
    st.markdown("### Results")

    rf = bundle.predict_tabular(row, "Random Forest")
    xgb = bundle.predict_tabular(row, "XGBoost")
    render_prediction("Random Forest", rf)

    lstm_ok = seq is not None and len(seq) >= SEQUENCE_LENGTH
    if lstm_ok:
        lstm = bundle.predict_lstm(seq)
        render_prediction("XGBoost", xgb)
        render_prediction("LSTM", lstm)
        avg = (rf + xgb + lstm) / 3
        render_prediction("Ensemble", avg, hero=True)
    else:
        render_prediction("XGBoost", xgb)
        avg = (rf + xgb) / 2
        render_prediction("Ensemble", avg, hero=True)
        st.caption(f"Upload a CSV with {SEQUENCE_LENGTH}+ rows to include LSTM.")


def main() -> None:
    st.markdown("# Apple predictor")
    st.markdown(
        '<p class="subtitle">Random Forest · XGBoost · LSTM</p>',
        unsafe_allow_html=True,
    )

    try:
        bundle = load_models()
    except Exception as exc:
        st.error(f"Models failed to load. {exc}")
        return

    names = bundle.feature_names
    tab_csv, tab_manual = st.tabs(["CSV upload", "Manual input"])

    row_df: pd.DataFrame | None = None
    seq_df: pd.DataFrame | None = None

    with tab_csv:
        st.caption(f"CSV needs columns: {', '.join(names)}")
        file = st.file_uploader("Upload", type=["csv"], label_visibility="collapsed")
        if file:
            raw = pd.read_csv(file)
            missing = [c for c in names if c not in raw.columns]
            if missing:
                st.error(f"Missing: {', '.join(missing)}")
                return
            seq_df = raw[names].astype(float)
            row_df = seq_df.tail(1)
            st.caption(f"{len(seq_df)} rows loaded")

    with tab_manual:
        row_df = manual_features_form(names)
        seq_df = None

    st.markdown("")
    if st.button("Predict", type="primary", use_container_width=True):
        if row_df is None:
            st.warning("Add data first.")
            return
        try:
            with st.spinner(""):
                run_predictions(bundle, row_df, seq_df)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
