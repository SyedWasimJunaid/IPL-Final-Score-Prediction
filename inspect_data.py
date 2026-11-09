
import pandas as pd
from pathlib import Path

# Resolve the dataset path relative to this script so it works from any directory
base_dir = Path(__file__).resolve().parent
dataset_path = base_dir / "dataset" / "ipl_data.csv"

if not dataset_path.exists():
    print("ERROR: Dataset not found!")
    print(f"Looked for: {dataset_path}")
    print("Check the filename and the dataset folder.")
else:
    # Load the CSV file
    ipl = pd.read_csv(dataset_path)

    print("\n===== IPL DATASET INSPECTION =====")

    # Dataset dimensions
    print("\nDataset shape (rows, columns):")
    print(ipl.shape)

    # Column names
    print("\nColumn names:")
    print(ipl.columns.tolist())

    # Match and category coverage
    print("\nUnique matches:", ipl["mid"].nunique())
    print("\nBatting teams:")
    print(sorted(ipl["bat_team"].dropna().unique().tolist()))
    print("\nBowling teams:")
    print(sorted(ipl["bowl_team"].dropna().unique().tolist()))
    print("\nUnique venues:", ipl["venue"].nunique())
    print(sorted(ipl["venue"].dropna().unique().tolist()))
    print("\nExact duplicate rows:", ipl.duplicated().sum())
    print(
        "Matches with inconsistent final totals:",
        ipl.groupby("mid")["total"].nunique().gt(1).sum(),
    )

    # First five rows
    print("\nFirst five rows:")
    print(ipl.head().to_string(index=False))

    # Missing values
    print("\nMissing values per column:")
    print(ipl.isnull().sum().to_string())

    # Data types
    print("\nColumn data types:")
    print(ipl.dtypes.to_string())

    print("\n===== INSPECTION COMPLETE =====")
