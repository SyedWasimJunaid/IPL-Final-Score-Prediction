from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

from model_utils import FEATURES, overs_to_balls

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ipl_score_model.keras"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.joblib"
METRICS_PATH = ROOT / "outputs" / "evaluation_metrics.json"

st.set_page_config(page_title="IPL Score Prediction", page_icon="🏏", layout="centered")
st.title("IPL Score Prediction Using Deep Learning")
st.write(
    "Estimate the final innings score from the match situation so far. "
    "Predictions are estimates and are not guaranteed."
)


@st.cache_resource
def load_artifacts() -> tuple[tf.keras.Model, dict]:
    if not MODEL_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError(
            "Trained model files are missing. Run "
            "`python train.py` from the project folder first."
        )
    return (
        tf.keras.models.load_model(MODEL_PATH),
        joblib.load(PREPROCESSOR_PATH),
    )


try:
    model, artifacts = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

teams = artifacts["teams"]
venues = artifacts["venues"]
if METRICS_PATH.is_file():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    interval = metrics["score_range"]["absolute_residual_quantile"]
    interval_level = metrics["score_range"]["nominal_level"]
else:
    interval = float(artifacts["residual_quantile"])
    interval_level = float(artifacts["interval_level"])

with st.form("prediction_form"):
    batting_team = st.selectbox("Batting team", teams)
    bowling_options = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox("Bowling team", bowling_options)
    venue = st.selectbox("Venue", venues)
    overs = st.number_input(
        "Overs completed (cricket notation, e.g. 12.3)",
        min_value=0.0,
        max_value=19.6,
        value=10.0,
        step=0.1,
    )
    current_runs = st.number_input(
        "Current runs", min_value=0, max_value=500, value=80, step=1
    )
    wickets = st.number_input("Wickets lost", min_value=0, max_value=10, value=2, step=1)
    runs_last_5 = st.number_input(
        "Runs in the last five overs", min_value=0, max_value=500, value=40, step=1
    )
    wickets_last_5 = st.number_input(
        "Wickets lost in the last five overs", min_value=0, max_value=10, value=1, step=1
    )
    submitted = st.form_submit_button("Predict final score", type="primary")

if submitted:
    try:
        completed_balls = overs_to_balls(float(overs))
        if wickets_last_5 > wickets:
            raise ValueError("Recent wickets cannot exceed total wickets lost.")
        if runs_last_5 > current_runs:
            raise ValueError("Runs in the last five overs cannot exceed the current score.")
        row = pd.DataFrame(
            [
                {
                    "bat_team": batting_team,
                    "bowl_team": bowling_team,
                    "venue": venue,
                    "overs_balls": completed_balls,
                    "runs": current_runs,
                    "wickets": wickets,
                    "runs_last_5": runs_last_5,
                    "wickets_last_5": wickets_last_5,
                }
            ],
            columns=FEATURES,
        )
        transformed = artifacts["preprocessor"].transform(row).astype("float32")
        prediction = float(model.predict(transformed, verbose=0).reshape(-1)[0])
        prediction = max(prediction, float(current_runs))
        lower = max(int(current_runs), round(prediction - interval))
        upper = round(prediction + interval)
        st.subheader(f"Predicted Final Score: {round(prediction)}")
        st.write(
            f"Approximate score range: **{lower}–{upper}** "
            f"(based on the {interval_level:.0%} absolute-residual quantile on held-out matches; "
            "not a guarantee)."
        )
        st.caption(
            "The model was trained on historical data through 2017, so team, venue, "
            "and playing-condition changes may affect its estimates."
        )
    except ValueError as error:
        st.error(str(error))

if METRICS_PATH.is_file():
    with st.expander("Held-out evaluation"):
        neural_metrics = metrics["neural_network"]
        baseline_metrics = metrics["baseline"]
        st.write(f"Neural network MAE: {neural_metrics['mae']:.2f} runs")
        st.write(f"Training-mean baseline MAE: {baseline_metrics['mae']:.2f} runs")
        st.write(
            "These metrics are computed on chronologically later matches that were "
            "kept out of training and validation."
        )