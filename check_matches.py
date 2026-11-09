
import pandas as pd
from pathlib import Path

# Load the dataset
dataset_path = Path(__file__).resolve().parent / "dataset" / "ipl_data.csv"
ipl = pd.read_csv(dataset_path)

print("\n===== UNIQUE TEAMS =====")
print("Batting teams:", sorted(ipl["bat_team"].unique()))
print("\nBowling teams:", sorted(ipl["bowl_team"].unique()))

print("\n===== MATCH INSPECTION =====")
print("Number of matches:", ipl["mid"].nunique())
print("Rows in first match:", len(ipl[ipl["mid"] == 1]))

print("\n===== TOTAL SCORE CHECK =====")
print(
    ipl.groupby("mid")["total"]
    .nunique()
    .value_counts()
    .sort_index()
)

print("\n===== OVERS RANGE =====")
print("Minimum overs:", ipl["overs"].min())
print("Maximum overs:", ipl["overs"].max())

print("\n===== SAMPLE MATCH PROGRESSION =====")
print(
    ipl[ipl["mid"] == 1][
        ["overs", "runs", "wickets", "runs_last_5", "total"]
    ].tail(10).to_string(index=False)
)
