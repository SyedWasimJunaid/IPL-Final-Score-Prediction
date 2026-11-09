from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

CATEGORICAL_FEATURES = ["bat_team", "bowl_team", "venue"]
NUMERIC_FEATURES = [
    "overs_balls",
    "runs",
    "wickets",
    "runs_last_5",
    "wickets_last_5",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "total"


def canonicalize_team_names(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().replace(TEAM_ALIASES)


def overs_to_balls(overs: float) -> int:
    """Convert cricket over notation (for example 19.3) to completed balls."""
    if not np.isfinite(overs):
        raise ValueError("Overs must use cricket notation between 0.0 and 20.0.")
    whole_overs = int(np.floor(overs))
    decimal_balls = (overs - whole_overs) * 10
    balls_in_over = int(round(decimal_balls))
    if (
        whole_overs < 0
        or not np.isclose(decimal_balls, balls_in_over)
        or balls_in_over < 0
        or balls_in_over > 6
        or whole_overs > 20
        or (whole_overs == 20 and balls_in_over != 0)
    ):
        raise ValueError("Overs must use cricket notation between 0.0 and 20.0.")
    return whole_overs * 6 + balls_in_over
